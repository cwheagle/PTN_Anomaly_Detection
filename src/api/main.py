import os
import sys
import asyncio
import json
import traceback
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Body, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import threading

# 프로젝트 루트 디렉토리를 path에 추가
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
os.chdir(root_dir) # 작업 디렉토리를 프로젝트 루트로 강제 변경

from src.api.alarm_tracker import plan_alarm_events
from src.data.db_connector import DBConnector
from src.data.data_collector import DataCollector
from src.models.trainer import Trainer
from src.models import registry as model_registry
from src.pipeline.drift_monitor import DriftMonitor
from src.models import promotion_gate
from src.pipeline import alerting, retrain_policy
from src.pipeline.retrain_policy import load_retrain_policy
from src.data import train_window
from src.config import PATHS, MODEL_CONFIG, API_VERSION, REDIS_CONFIG, RETENTION_DAYS
import redis

# 글로벌 객체
db = DBConnector()
collector = DataCollector()
event_queues = set() # SSE 클라이언트들을 위한 큐 집합
# 학습 진행 상태 상세 추적
training_status = {
    "traffic": {"is_training": False, "current_epoch": 0, "total_epochs": 0, "loss": 0, "val_loss": None, "last_error": None, "success_msg": None}, 
    "optical": {"is_training": False, "current_epoch": 0, "total_epochs": 0, "loss": 0, "val_loss": None, "last_error": None, "success_msg": None}
} 
active_trainers = {} # 현재 실행 중인 Trainer 인스턴스 (중지용)
drift_monitor = DriftMonitor()
last_drift_result = None # 가장 최근의 Drift Check 결과 캐싱
redis_client = redis.Redis(**REDIS_CONFIG) # Redis Client for Pub/Sub

async def broadcast_alarm(alarm_data: dict):
    """모든 연결된 SSE 클라이언트에게 알람 전송 (전송 성공 여부 반환)"""
    if not event_queues:
        print(f"[SSE] No active clients connected. Skipping broadcast.")
        return False
    
    print(f"[SSE] Broadcasting to {len(event_queues)} clients...")
    message = json.dumps(alarm_data)
    for queue in event_queues:
        await queue.put(message)
    return True

# 글로벌 알람 상태 관리: {(ip, slot, port): 마지막으로 ALARM 을 보낸 occur_date}
active_alarms_state = {}
# 지속 중인 CRITICAL 의 재알림 간격(분). 15=매 스텝(기존 동작), 60=1시간마다, 0=진입 시 1회만
ALARM_RENOTIFY_MINUTES = int(os.getenv("ALARM_RENOTIFY_MINUTES", "15"))

async def alarm_callback(anomalies_df):
    """Consumer 웹훅 콜백: 포트 단위로 알람 발생/해제를 판단 (중복 방지 포함)

    payload에 포함된 포트의 상태만 평가한다. (Consumer는 포트 1건씩 전송하므로,
    payload에 없는 포트를 '복구'로 간주하면 다른 포트의 알람이 잘못 해제된다.)
    """
    events = plan_alarm_events(anomalies_df.to_dict(orient="records"), active_alarms_state, ALARM_RENOTIFY_MINUTES)

    for event, key, occur in events:
        if event["type"] == "ALARM":
            # 실제 전송에 성공했을 때만 상태 업데이트 (클라이언트가 없으면 다음에 재시도)
            if await broadcast_alarm(event):
                active_alarms_state[key] = occur
                print(f"[SSE] Successfully broadcasted ALARM: {key}")
            else:
                print(f"[SSE] Retrying ALARM later (No clients): {key}")
        else:
            await broadcast_alarm(event)
            active_alarms_state.pop(key, None)
            print(f"[SSE] Broadcasted CLEAR: {key}")


def _model_dir(ft: str) -> str:
    return os.path.dirname(PATHS[ft]['model'])


def _saved_training_config(ft: str) -> dict:
    """활성 모델이 학습 때 쓴 설정(epochs, lr 등)을 재학습에 재사용"""
    _, meta_path = get_active_model_info(ft)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                return json.load(f).get("config", {})
        except Exception as e:
            print(f"[*] [Drift] Failed to load previous config for {ft}: {e}")
    return {}


def _launch_thread(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()


def handle_drift(result: dict, trigger_source: str, launcher=_launch_thread) -> dict:
    """드리프트 판정 결과를 재학습 결정으로 연결 (일일 유지보수와 /api/drift/check 가 공유).

    트랙마다 지속성 카운트를 갱신하고 `retrain_policy.decide` 로 none/notify/train 을 정한다.
    train 이면 후보 학습을 시작한다 (launcher 로 스레드/BackgroundTasks 선택). result 에 `retrain`,
    `auto_retrain_triggered` 를 덧붙이고 트랙별 결정을 반환한다.
    """
    policy = load_retrain_policy()
    now = datetime.now()
    decisions = {}
    for ft in ('traffic', 'optical'):
        track_result = result.get(ft)
        if not isinstance(track_result, dict):
            continue
        model_dir = _model_dir(ft)
        state = retrain_policy.record_check(retrain_policy.load_state(model_dir, ft), track_result, now)
        pending = retrain_policy.pending_drift_candidate(model_registry.load(model_dir, ft))
        decision = retrain_policy.decide(track_result, state, now, policy, pending_candidate=pending,
                                         is_training=training_status.get(ft, {}).get("is_training", False), track=ft)
        if decision.action == "train":
            # 쿨다운·지속성 초기화는 여기가 아니라 Trainer 시작 시점에 기록 (수집 단계 중단은 쿨다운을 걸지 않음)
            training_status[ft]["is_training"] = True        # 파이프라인 시작 전 중복 트리거 방지
            print(f"[*] [Drift] ({trigger_source}) Triggering candidate retraining for {ft}: {decision.reason}")
            launcher(run_training_pipeline, ft, _saved_training_config(ft), None, "drift")
        else:
            print(f"[*] [Drift] ({trigger_source}) {ft}: {decision.action} — {decision.reason}")
        retrain_policy.save_state(model_dir, ft, state)
        decisions[ft] = {"action": decision.action, "reason": decision.reason}
    result["retrain"] = decisions
    result["auto_retrain_triggered"] = any(d["action"] == "train" for d in decisions.values())
    return decisions

maintenance_scheduler = BackgroundScheduler()

def run_daily_maintenance():
    """매일 새벽 3시에 실행되는 자동 유지보수 (DB 정리 및 데이터 드리프트 검사)"""
    print("[*] [Maintenance] Running daily maintenance tasks...")
    try:
        # 1. 오래된 데이터 정리
        db.delete_old_data(RETENTION_DAYS)
        print(f"[*] [Maintenance] Cleaned up DB data older than {RETENTION_DAYS} days.")
        
        # 2. 데이터 드리프트 검사
        global last_drift_result
        result = drift_monitor.check_drift()
        last_drift_result = result
        
        if result.get("status") != "error":
            handle_drift(result, "maintenance")
    except Exception as e:
        print(f"[!] [Maintenance] Error during daily maintenance: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 생명주기 관리"""
    print("[*] PTN Anomaly Detection API Service Initializing...")
    maintenance_scheduler.add_job(run_daily_maintenance, 'cron', hour=3, minute=0)
    maintenance_scheduler.start()
    print("[*] Daily maintenance scheduler started (Runs at 03:00 AM).")
    yield
    print("[*] Service shutting down...")
    maintenance_scheduler.shutdown()

app = FastAPI(
    title="PTN Anomaly Detection API",
    description="REST API for PTN Anomaly Detection & Predictive Maintenance",
    lifespan=lifespan
)

# API 버전 및 커스텀 헤더 미들웨어
@app.middleware("http")
async def add_api_version_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-API-Version"] = API_VERSION
    return response

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"service": "PTN Anomaly Detection API", "status": "online"}

# --- 모델 관리 API ---

def get_active_model_info(ft: str):
    p = PATHS.get(ft, {})
    if not p: return "", ""
    base_model = p.get('model', '')
    if not base_model: return "", ""
    
    model_dir = os.path.dirname(base_model)
    registry_path = os.path.join(model_dir, f"{ft}_registry.json")
    
    actual_model = base_model
    actual_meta = base_model.replace('.pth', '.json')
    
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            active_ver = registry.get("active_version")
            if active_ver:
                for v in registry.get("versions", []):
                    if v["version"] == active_ver:
                        if "model_path" in v:
                            actual_model = os.path.join(model_dir, v["model_path"])
                        if "config_path" in v:
                            actual_meta = os.path.join(model_dir, v["config_path"])
                        break
        except:
            pass
    return actual_model, actual_meta

@app.get("/api/model/status")
async def get_model_status():
    """현재 모델들의 학습 상태, 훈련 설정, 추론 설정을 구분하여 조회"""
    status = {}
    for ft in ['traffic', 'optical']:
        actual_model, meta_path = get_active_model_info(ft)
        if not actual_model:
            continue
        
        # 기본 구조 정의
        info = {
            "exists": os.path.exists(actual_model),
            "training": training_status.get(ft, {"is_training": False}),
            "last_trained": None,
            "samples_used": 0,
            "active_version": model_registry.list_versions(os.path.dirname(PATHS[ft]['model']), ft)["active_version"],
            "candidate_versions": [v["version"] for v in model_registry.list_versions(os.path.dirname(PATHS[ft]['model']), ft)["versions"]
                                   if v["status"] == "candidate"],
            # 추론 설정 (실시간 수정 가능 항목만 노출)
            "inference_config": {
                "threshold": MODEL_CONFIG.get('threshold', 0.1),
                "slope_threshold": MODEL_CONFIG.get('slope_threshold', 3.0)
            },
            # 훈련 설정 (학습 시 수정 가능 항목만 노출)
            "training_config": {
                "epochs": MODEL_CONFIG['epochs'],
                "learning_rate": MODEL_CONFIG['learning_rate'],
                "batch_size": MODEL_CONFIG['batch_size'],
                "threshold_percentile": MODEL_CONFIG['threshold_percentile'],
                "patience": MODEL_CONFIG['patience']
            }
        }
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    info["last_trained"] = meta.get("trained_at")
                    info["samples_used"] = meta.get("samples_used", 0)
                    
                    # 파일에 저장된 값으로 오버라이드
                    saved_config = meta.get("config", {})
                    for key in info["training_config"]:
                        if key in saved_config:
                            info["training_config"][key] = saved_config[key]
                    
                    # 추론 설정 업데이트
                    info["inference_config"]["threshold"] = meta.get("threshold", info["inference_config"]["threshold"])
                    # slope_threshold 등도 config 내부에 저장되어 있을 수 있음
                    for key in info["inference_config"]:
                        if key in saved_config:
                            info["inference_config"][key] = saved_config[key]

            except:
                pass
        status[ft] = info
    return status



def _fetch_raw_for_gate(ft: str, start, end):
    return (db.fetch_traffic if ft == 'traffic' else db.fetch_optical)(start, end)


def run_gate_for(ft: str, version: str, now: datetime = None) -> dict:
    """후보 `version` 에 승격 게이트를 실행하고 결과를 레지스트리에 기록 (게이트 실패는 ERROR 결과로 기록되며 예외로 올라가지 않음)"""
    policy = load_retrain_policy()
    window = train_window.plan_window(now or datetime.now(), policy)
    result = promotion_gate.run_gate(_model_dir(ft), ft, version, window, policy, _fetch_raw_for_gate,
                                     alerting.load_policy())
    gate = result.to_dict()
    model_registry.set_gate(_model_dir(ft), ft, version, gate)
    return gate


def _resolve_alert_policy(ft: str, preset: str = None):
    """학습 결과(후보)와 짝으로 저장할 알람 정책(메타 dict). preset 이 있으면 그 프리셋,
    없으면 **활성 모델의 정책을 승계**한다(드리프트 재학습 포함 — 학습 설정 재사용과 같은 방식). 활성 모델에도 없으면 None(기본 정책)."""
    if preset:
        return alerting.policy_to_meta(alerting.get_preset(preset), preset)
    reg = model_registry.load(_model_dir(ft), ft)
    active = model_registry.find(reg, reg.get("active_version")) if reg.get("active_version") else None
    return (active or {}).get("alert_policy")


def run_training_pipeline(ft: str, training_config: dict, date_params: dict = None, trigger: str = "manual",
                          exclude_suspect: bool = True, alert_policy: str = None):
    """백그라운드 학습 실행 (스레드에서 실행되어 이벤트 루프 차단 방지)

    date_params 가 없으면 **포트 분할 모드**: 최근까지 포함한 구간 [T-train_days, T] 을 포트 단위 홀드아웃으로
    학습/검증에 나눈다. 있으면 사용자가 지정한 날짜 분할(기존 UI 호환).
    exclude_suspect: 자기 알람 이력·규칙으로 찾은 장애 의심 구간을 학습에서 제외 (의심 비율이 한도를 넘으면 학습 중단).
    결과는 후보로만 저장되며, 드리프트 트리거(trigger='drift')의 결과는 재학습 상태 파일에도 기록된다.
    alert_policy: 후보와 짝으로 저장할 알람 정책 프리셋 이름. 생략하면 활성 모델의 정책을 승계한다.
    """
    print(f"[*] [BG] Starting training pipeline for {ft} (trigger={trigger})...")
    policy = load_retrain_policy()
    started = datetime.now()
    outcome, trained_version = "failed", None
    trainer_started = False   # 수집을 통과해 Trainer 가 시작됐는가 (드리프트 상태 기록 방식이 갈림)

    # 상태 초기화
    training_status[ft] = {
        "is_training": True, 
        "current_epoch": 0, 
        "total_epochs": training_config.get('epochs', MODEL_CONFIG['epochs']), 
        "loss": 0, 
        "val_loss": None,
        "last_error": None,
        "success_msg": None,
        "suspect_fraction": None,
        "gate_status": None,
    }
    
    def on_progress(epoch, total, loss, val_loss):
        training_status[ft].update({
            "current_epoch": epoch,
            "total_epochs": total,
            "loss": loss,
            "val_loss": val_loss
        })

    # Trainer 생성 및 중지 관리를 위한 래퍼
    active_trainers[ft] = {"stop_requested": False, "trainers": []}

    try:
        # 데이터 수집 전 중지 요청 확인
        if active_trainers[ft]["stop_requested"]:
            print(f"[*] [BG] {ft} training cancelled before data collection.")
            outcome = "stopped"
            return

        # 데이터 수집 (요청된 ft만 수집)
        collect_kwargs = {"suspect_policy": policy if exclude_suspect else None}
        if date_params:
            window_info = {"mode": "date", **date_params}
            collect_kwargs.update(train_start=date_params['train_start'], train_end=date_params['train_end'],
                                  test_start=date_params['test_start'], test_end=date_params['test_end'])
        else:
            window = train_window.plan_window(started, policy)
            window_info = {"mode": "port_split", "start": window.start.strftime('%Y-%m-%d %H:%M:%S'),
                           "end": window.end.strftime('%Y-%m-%d %H:%M:%S'),
                           "val_port_fraction": policy.val_port_fraction, "salt": policy.split_salt}
            collect_kwargs.update(train_start=window.start, train_end=window.end,
                                  val_port_fraction=policy.val_port_fraction, split_salt=policy.split_salt)
        results = collector.collect_and_save(
            feature_type=ft, stop_checker=lambda: active_trainers[ft]["stop_requested"], **collect_kwargs)
        
        # 데이터 수집 후 중지 요청 확인
        if active_trainers[ft]["stop_requested"]:
            print(f"[*] [BG] {ft} training cancelled after data collection.")
            outcome = "stopped"
            return

        track_result = (results or {}).get(ft, {})
        suspect_stats = track_result.get("suspect_stats")
        if suspect_stats:
            training_status[ft]["suspect_fraction"] = suspect_stats["fraction"]

        # 의심 구간 비율이 한도를 넘으면 중단 (대규모 장애 중에는 재학습하지 않는다)
        if "skipped" in track_result:
            err_msg = (f"학습 중단: 장애 의심 구간이 학습 데이터의 {suspect_stats['fraction']:.1%} "
                       f"(한도 {policy.max_excluded_fraction:.0%}). 장애가 해소된 뒤 다시 시도하세요.")
            training_status[ft]["last_error"] = err_msg
            outcome = f"skipped: {track_result['skipped']}"
            print(f"[!] [BG] {ft} training skipped: {err_msg}")
            return

        # 데이터 수집 결과 검증
        min_samples = MODEL_CONFIG.get('window_size', 12)
        if ft not in (results or {}) or track_result['train'] < min_samples:
            count = track_result.get('train', 0)
            err_msg = f"Insufficient train data: {count} datas found. (Min required: {min_samples})"
            training_status[ft]["last_error"] = err_msg
            outcome = "failed: insufficient data"
            print(f"[!] [BG] {ft} training failed: {err_msg}")
            return

        # 학습 실행
        # activate=False: 학습 결과는 '후보'로만 저장. 사람이 승격(/api/model/promote)해야 Consumer 에 반영된다.
        # (활성 모델이 아직 없는 최초 학습만 자동 활성화)
        if trigger == "drift":
            # 후보를 만들기 시작한 시점: 쿨다운 시작 + 지속성 초기화 (이후 학습 예외·게이트 ERROR 도 쿨다운 적용)
            model_dir = _model_dir(ft)
            retrain_policy.save_state(model_dir, ft, retrain_policy.record_training(
                retrain_policy.load_state(model_dir, ft), datetime.now(), outcome="started"))
        trainer_started = True
        trainer = Trainer(feature_type=ft, config_override=training_config, progress_callback=on_progress, activate=False,
                          trigger=trigger, window_info=window_info, suspect_stats=suspect_stats,
                          alert_policy=_resolve_alert_policy(ft, alert_policy))
        active_trainers[ft]["trainers"] = [trainer]
        
        if active_trainers[ft]["stop_requested"]:
            outcome = "stopped"
            return
        success = trainer.train()
        early_stopped = trainer.early_stopped
        reload_fts = [ft]

        if success:
            msg = "Model training complete."
            if early_stopped:
                msg = f"Training finished early at epoch {training_status[ft]['current_epoch']} (Optimal weights saved)."
            
            training_status[ft]["candidate_version"] = trainer.version
            trained_version, outcome = trainer.version, "candidate"

            # 승격 게이트: 결과를 후보에 기록해 두고(승격 API 가 사용), 실행이 실패해도 학습 결과는 후보로 남긴다
            gate = None
            try:
                gate = run_gate_for(ft, trainer.version, started)
                training_status[ft]["gate_status"] = gate["status"]
                print(f"[*] [BG] {ft} {trainer.version} gate: {gate['status']}")
            except Exception as e:
                training_status[ft]["gate_status"] = "ERROR"
                print(f"[!] [BG] {ft} gate could not be recorded: {e}")
            if trainer.activated:
                # 최초 학습(활성 모델 없음): 즉시 반영
                outcome = "activated"
                msg += f" ({trainer.version} activated; reload broadcasted to Consumers via Redis)"
                redis_client.publish('ptn_control', json.dumps({'action': 'reload', 'track': ft}))
            elif trigger == "drift" and policy.mode == "auto" and gate and gate["status"] == "PASS":
                # auto 모드: 드리프트 재학습 결과가 게이트(PASS)를 통과하면 자동 활성화
                try:
                    model_registry.promote(_model_dir(ft), ft, trainer.version, reason="auto-gate")
                    _broadcast_model_reload(ft)
                    outcome = "auto-promoted"
                    msg += f" Gate PASS — {trainer.version} auto-promoted (mode=auto); reload broadcasted."
                except model_registry.RegistryError as e:
                    msg += f" Gate PASS but auto-promotion failed: {e}. Saved as candidate {trainer.version}."
            else:
                msg += f" Saved as candidate {trainer.version} (NOT active). Review and promote it in Model Management to deploy."
                
            training_status[ft]["success_msg"] = msg
            print(f"[*] [BG] {ft.capitalize()} {msg}")
        else:
            if training_status[ft]["last_error"] is None:
                training_status[ft]["last_error"] = "Model training failed or was stopped."
            print(f"[!] [BG] {ft.capitalize()} model training failed.")
    except Exception as e:
        full_error = traceback.format_exc()
        err_msg = f"Runtime Error: {str(e)}"
        training_status[ft]["last_error"] = err_msg
        outcome = f"failed: {e}"
        print(f"[!] [BG] Training Error: {err_msg}")
        print(full_error)
    finally:
        training_status[ft]["is_training"] = False
        if ft in active_trainers:
            del active_trainers[ft]
        if trigger == "drift":
            try:
                model_dir = _model_dir(ft)
                state, now = retrain_policy.load_state(model_dir, ft), datetime.now()
                if trainer_started or outcome == "stopped":
                    state = retrain_policy.record_training(state, now, trained_version, outcome)
                else:   # 수집 단계 중단(의심 비율 초과)·수집 오류: 쿨다운 없이 24h 재시도 보류만 기록
                    state = retrain_policy.record_skip(state, now, outcome)
                retrain_policy.save_state(model_dir, ft, state)
            except Exception as e:
                print(f"[!] [BG] Failed to record retrain state: {e}")

@app.post("/api/model/train")
async def train_model(
    background_tasks: BackgroundTasks,
    ft: str = Query(..., regex="^(traffic|optical)$"),
    training_config: dict = Body({}),
    train_start: str = Query(None),
    train_end: str = Query(None),
    test_start: str = Query(None),
    test_end: str = Query(None),
    exclude_suspect: bool = Query(True),
    alert_policy: str = Query(None)
):
    """모델 후보 학습 시작.

    날짜를 하나도 지정하지 않으면 최근 train_days 일을 포트 단위 홀드아웃으로 나눠 학습(권장 기본).
    날짜를 지정하면 기존처럼 날짜 분할(누락된 값은 이전 기본값으로 채움).
    exclude_suspect=false 이면 장애 의심 구간 제외를 끈다.
    alert_policy: 후보와 짝으로 저장할 알람 정책 프리셋(default / precision). 생략하면 활성 모델의 정책을 승계한다.
    """
    if alert_policy:
        try:
            alerting.get_preset(alert_policy)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    if training_status.get(ft, {}).get("is_training"):
        raise HTTPException(status_code=409, detail=f"{ft} training is already in progress.")

    if any((train_start, train_end, test_start, test_end)):
        now = datetime.now()
        date_params = {
            'train_start': train_start or (now - timedelta(days=37)).strftime('%Y-%m-%d'),
            'train_end': train_end or (now - timedelta(days=7)).strftime('%Y-%m-%d'),
            'test_start': test_start or (now - timedelta(days=7)).strftime('%Y-%m-%d'),
            'test_end': test_end or now.strftime('%Y-%m-%d'),
        }
        range_info = date_params
    else:
        policy = load_retrain_policy()
        date_params = None
        window = train_window.plan_window(datetime.now(), policy)
        range_info = {'mode': 'port_split', 'start': window.start.strftime('%Y-%m-%d %H:%M:%S'),
                      'end': window.end.strftime('%Y-%m-%d %H:%M:%S'), 'val_port_fraction': policy.val_port_fraction}
    # 백그라운드 작업이 시작되기 전의 연타도 막기 위해 즉시 학습 중으로 표시
    training_status[ft]["is_training"] = True
    background_tasks.add_task(run_training_pipeline, ft, training_config, date_params, "manual", exclude_suspect, alert_policy)
    return {
        "status": "started", 
        "message": f"{ft} model training task queued.",
        "range": range_info,
        "exclude_suspect": exclude_suspect,
        "alert_policy": alert_policy,
    }

def _broadcast_model_reload(ft: str):
    """승격/롤백 후 Consumer 들이 새 활성 모델을 읽도록 Redis 로 알림 (실패해도 API 응답에는 영향 없음)"""
    try:
        redis_client.publish('ptn_control', json.dumps({'action': 'reload', 'track': ft}))
    except Exception as e:
        print(f"[!] [Registry] Failed to broadcast model reload: {e}")


@app.get("/api/model/versions")
async def get_model_versions(ft: str = Query(None, pattern="^(traffic|optical)$")):
    """트랙별 모델 버전 목록 (active / candidate / retired). ft 를 생략하면 두 트랙 모두."""
    fts = [ft] if ft else ['traffic', 'optical']
    return {t: model_registry.list_versions(_model_dir(t), t) for t in fts}


@app.post("/api/model/promote")
async def promote_model(ft: str = Query(..., pattern="^(traffic|optical)$"), version: str = Query(...),
                        force: bool = Query(False)):
    """후보(또는 이전 버전)를 활성화. 현재 모델과 크게 어긋나면(임계치/검증 손실) force=true 없이는 409."""
    try:
        result = model_registry.promote(_model_dir(ft), ft, version, force=force)
    except model_registry.PromotionWarning as e:
        raise HTTPException(status_code=409, detail={"message": "승격 검증에서 문제가 발견되었습니다. 확인 후 force=true 로 다시 요청하세요.",
                                                     "warnings": e.warnings, "checks": e.checks})
    except model_registry.VersionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except model_registry.RegistryError as e:
        raise HTTPException(status_code=409, detail=str(e))
    _broadcast_model_reload(ft)
    return {"status": "success", "track": ft, **result}


@app.post("/api/model/policy")
async def create_policy_version(ft: str = Query(..., pattern="^(traffic|optical)$"), version: str = Query(...),
                                preset: str = Query(...)):
    """재학습 없이 같은 가중치에 알람 정책만 바꾼 **새 후보 버전**을 만들고 게이트를 실행한다 (활성 버전은 불변).
    승격·롤백은 기존 절차를 따른다."""
    try:
        policy = alerting.get_preset(preset)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if training_status.get(ft, {}).get("is_training"):
        raise HTTPException(status_code=409, detail=f"{ft} training is in progress.")
    try:
        entry = model_registry.create_policy_version(_model_dir(ft), ft, version, alerting.policy_to_meta(policy, preset))
    except model_registry.VersionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except model_registry.RegistryError as e:
        raise HTTPException(status_code=409, detail=str(e))
    gate = await asyncio.to_thread(run_gate_for, ft, entry["version"])
    return {"status": "success", "track": ft, "version": entry["version"], "derived_from": version,
            "alert_policy": preset, "gate": gate}


@app.post("/api/model/gate")
async def run_model_gate(ft: str = Query(..., pattern="^(traffic|optical)$"), version: str = Query(...)):
    """후보 모델의 승격 게이트를 (재)실행하고 결과를 레지스트리에 기록. 학습 중이면 409."""
    if training_status.get(ft, {}).get("is_training"):
        raise HTTPException(status_code=409, detail=f"{ft} training is in progress.")
    if model_registry.find(model_registry.load(_model_dir(ft), ft), version) is None:
        raise HTTPException(status_code=404, detail=f"버전 '{version}' 이(가) 없습니다.")
    gate = await asyncio.to_thread(run_gate_for, ft, version)
    return {"status": "success", "track": ft, "version": version, "gate": gate}


@app.post("/api/model/rollback")
async def rollback_model(ft: str = Query(..., pattern="^(traffic|optical)$")):
    """직전에 활성이었던 버전으로 되돌림"""
    try:
        result = model_registry.rollback(_model_dir(ft), ft)
    except model_registry.RegistryError as e:
        raise HTTPException(status_code=409, detail=str(e))
    _broadcast_model_reload(ft)
    return {"status": "success", "track": ft, **result}


@app.post("/api/model/train/stop")
async def stop_training(ft: str = Query(..., regex="^(traffic|optical)$")):
    """현재 진행 중인 모델 학습 강제 중지"""
    task_info = active_trainers.get(ft)
    if not task_info:
        return {"status": "ignored", "message": f"No active training task found for {ft}."}
    
    task_info["stop_requested"] = True
    for trainer in task_info["trainers"]:
        trainer.stop()
        
    training_status[ft]["last_error"] = "Training stopped by user."
    return {"status": "success", "message": f"Stop request sent to {ft} trainer."}

# --- 데이터 및 스케줄러 API ---

@app.get("/api/anomalies")
async def get_anomalies(
    severity_min: int = Query(0),
    severity_max: int = Query(3),
    rising_only: bool = Query(False)
):
    conn = db.get_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB connection failed")
    try:
        where_clauses = [
            "occur_date = (SELECT MAX(occur_date) FROM anomaly_detection)",
            "occur_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR)"
        ]
        where_clauses.append(f"alarm_level BETWEEN {severity_min} AND {severity_max}")
        if rising_only: where_clauses.append("slope_label = 'RISING'")
        
        query = f"""
            SELECT occur_date, ip_addr, cid as slot_id, lid as port_id, 
                   severity, alarm_level, alarm_label, slope, slope_label, 
                   ttf_minutes, expected_fatal_time, anomaly_reason,
                   is_traffic_anomaly, is_optical_anomaly, rca_diagnosis, rca_action, feature_contribution
            FROM anomaly_detection
            WHERE {" AND ".join(where_clauses)}
            ORDER BY severity DESC, slope DESC
        """
        df = pd.read_sql(query, conn)
        if not df.empty:
            for col in ['occur_date', 'expected_fatal_time']:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
        return df.replace({np.nan: None}).to_dict(orient="records")
    finally:
        conn.close()

@app.get("/api/anomalies/history")
async def get_anomaly_history(ip_addr: str, slot_id: int, port_id: int, days: int = 1):
    conn = db.get_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB connection failed")
    try:
        query = """
            SELECT occur_date, tx_packet, rx_packet, error_packet, tx_avg_power, rx_avg_power, 
                    anomaly_score, threshold, severity, alarm_level, alarm_label, anomaly_reason,
                    traffic_score, traffic_threshold, traffic_severity,
                    optical_score, optical_threshold, optical_severity,
                    is_traffic_anomaly, is_optical_anomaly,
                    rca_diagnosis, rca_action, feature_contribution
            FROM anomaly_detection
            WHERE ip_addr = %s AND cid = %s AND lid = %s
              AND occur_date >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ORDER BY occur_date ASC
        """
        df = pd.read_sql(query, conn, params=(ip_addr, slot_id, port_id, days))
        if not df.empty:
            df['occur_date'] = pd.to_datetime(df['occur_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return df.replace({np.nan: None}).to_dict(orient="records")
    finally:
        conn.close()

@app.get("/api/stream/alarms")
async def stream_alarms(request: Request):
    queue = asyncio.Queue()
    event_queues.add(queue)
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected(): break
                data = await queue.get()
                yield f"event: alarm\ndata: {data}\n\n"
        finally:
            event_queues.remove(queue)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/internal/alarm")
async def trigger_sse_alarm(data: list = Body(...)):
    """Consumer에서 추론 완료 후 발생한 알람을 전달받아 SSE로 브로드캐스트하는 내부 엔드포인트"""
    try:
        if not data:
            return {"status": "ok"}
        df = pd.DataFrame(data)
        await alarm_callback(df)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- [Phase 9] Data Drift & Auto-Retraining API ---

@app.get("/api/drift/status")
async def get_drift_status():
    """최근 수행된 Data Drift 감지 결과와 트랙별 재학습 상태(지속 일수/쿨다운)를 반환"""
    policy = load_retrain_policy()
    now = datetime.now()
    return {
        "status": "success",
        "last_result": last_drift_result,
        "retrain_state": {ft: retrain_policy.describe_state(retrain_policy.load_state(_model_dir(ft), ft), now, policy)
                          for ft in ('traffic', 'optical')},
    }

@app.post("/api/drift/check")
async def check_drift_and_retrain(background_tasks: BackgroundTasks):
    """수동으로 Drift 감지를 즉시 실행. 재학습 여부는 retrain_policy 가 결정 (지속성·쿨다운·mode 적용)"""
    global last_drift_result

    print("[*] [Drift] Running Data Drift check...")
    result = drift_monitor.check_drift()

    if "status" in result and result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    handle_drift(result, "manual", launcher=background_tasks.add_task)
    last_drift_result = result
    return result

# --- [Phase 8] RCA Rules API ---

def _publish_rules_reload():
    """룰 변경을 Consumer들에 전파 (Redis Pub/Sub). 실패해도 API 응답에는 영향 없음."""
    try:
        redis_client.publish('ptn_control', json.dumps({'action': 'reload_rules'}))
    except Exception as e:
        print(f"[!] [RCA] Failed to broadcast rules reload: {e}")

@app.get("/api/rca/rules")
async def get_rca_rules():
    """현재 등록된 RCA 도메인 룰 테이블 반환"""
    try:
        from src.rca.rule_engine import RCAEngine
        engine = RCAEngine()
        rules = engine.get_rules()
        return {"count": len(rules), "rules": rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rca/rules")
async def add_rca_rule(rule: dict = Body(...)):
    """새 RCA 룰 추가"""
    required = {"id", "track", "priority", "diagnosis", "action"}
    missing = required - rule.keys()
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")
    
    if rule.get("track") not in ("traffic", "optical", "integrated"):
        raise HTTPException(status_code=400, detail="Invalid track. Must be 'traffic', 'optical' or 'integrated'.")
    
    from src.rca.rule_engine import validate_rule
    issues = validate_rule(rule)
    if issues:
        raise HTTPException(status_code=400, detail=f"Invalid rule: {'; '.join(issues)}")

    try:
        from src.rca.rule_engine import RCAEngine
        engine = RCAEngine()
        success = engine.add_rule(rule)
        
        if success:
            _publish_rules_reload()
            return {"status": "success", "message": f"Rule '{rule['id']}' added.", "rule": rule}
        else:
            raise HTTPException(status_code=409, detail=f"Rule ID '{rule.get('id')}' already exists or save failed.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/rca/rules/{rule_id}")
async def delete_rca_rule(rule_id: str):
    """특정 RCA 룰 삭제"""
    import json, os
    from src.rca.rule_engine import DEFAULT_RULES_PATH
    try:
        if not os.path.exists(DEFAULT_RULES_PATH):
            raise HTTPException(status_code=404, detail="Rules file not found.")
        with open(DEFAULT_RULES_PATH, "r", encoding="utf-8") as f:
            rules = json.load(f)
        new_rules = [r for r in rules if r.get("id") != rule_id]
        if len(new_rules) == len(rules):
            raise HTTPException(status_code=404, detail=f"Rule ID '{rule_id}' not found.")
        with open(DEFAULT_RULES_PATH, "w", encoding="utf-8") as f:
            json.dump(new_rules, f, ensure_ascii=False, indent=2)
        _publish_rules_reload()
        return {"status": "success", "message": f"Rule '{rule_id}' deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
