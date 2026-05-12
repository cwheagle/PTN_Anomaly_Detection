import os
import sys
import asyncio
import json
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Body, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# 프로젝트 루트 디렉토리를 path에 추가
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
os.chdir(root_dir) # 작업 디렉토리를 프로젝트 루트로 강제 변경

from src.pipeline.scheduler import PTNAnomalyScheduler
from src.data.db_connector import DBConnector
from src.data.data_collector import DataCollector
from src.models.trainer import Trainer
from src.config import PATHS, MODEL_CONFIG, SEVERITY_CONFIG

# 글로벌 객체
scheduler_instance = None
db = DBConnector()
collector = DataCollector()
event_queues = set() # SSE 클라이언트들을 위한 큐 집합

async def broadcast_alarm(alarm_data: dict):
    """모든 연결된 SSE 클라이언트에게 알람 전송"""
    if not event_queues:
        return
    
    message = json.dumps(alarm_data)
    for queue in event_queues:
        await queue.put(message)

def alarm_callback(anomalies_df):
    """스케줄러에서 호출할 콜백 함수: Critical 알람 필터링 및 브로드캐스트"""
    criticals = anomalies_df[anomalies_df['alarm_label'] == 'CRITICAL']
    if not criticals.empty:
        loop = asyncio.get_event_loop()
        for _, row in criticals.iterrows():
            alarm_info = {
                "event_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "ip_addr": row['ip_addr'],
                "slot_id": int(row['cid']),
                "port_id": int(row['lid']),
                "severity": row['alarm_label'],
                "reason": row['anomaly_reason']
            }
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_alarm(alarm_info), loop)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 생명주기 관리"""
    global scheduler_instance
    print("[*] PTN Anomaly Detection Service Initializing...")
    
    scheduler_instance = PTNAnomalyScheduler()
    if hasattr(scheduler_instance, 'set_callback'):
        scheduler_instance.set_callback(alarm_callback)
    
    scheduler_instance.start()
    yield
    
    print("[*] Service shutting down...")
    if scheduler_instance:
        scheduler_instance.scheduler.shutdown()

app = FastAPI(
    title="PTN Anomaly Detection API",
    description="REST API for PTN Anomaly Detection & Predictive Maintenance",
    lifespan=lifespan
)

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

@app.get("/api/model/status")
async def get_model_status():
    """현재 모델들의 학습 상태, 훈련 설정, 추론 설정을 구분하여 조회"""
    status = {}
    for ft in ['traffic', 'optical']:
        p = PATHS[ft]
        meta_path = p['model'].replace('.pth', '.json')
        
        # 기본 구조 정의
        info = {
            "exists": os.path.exists(p['model']),
            "last_trained": None,
            "samples_used": 0,
            # 추론 설정 (실시간 수정 가능)
            "inference_config": {
                "threshold": MODEL_CONFIG.get('threshold', 0.1),
                "slope_threshold": SEVERITY_CONFIG.get('slope_threshold', 1.0),
                "rul_target": SEVERITY_CONFIG.get('rul_target', 90.0)
            },
            # 훈련 설정 (학습 시에만 적용)
            "training_config": {
                "epochs": MODEL_CONFIG['epochs'],
                "learning_rate": MODEL_CONFIG['learning_rate'],
                "batch_size": MODEL_CONFIG['batch_size'],
                "threshold_percentile": MODEL_CONFIG['threshold_percentile'],
                "window_size": MODEL_CONFIG['window_size']
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

@app.post("/api/model/inference-config")
async def update_inference_config(ft: str = Query(..., regex="^(traffic|optical)$"), settings: dict = Body(...)):
    """모델 재학습 없이 추론 설정(임계치 등)만 즉시 업데이트"""
    p = PATHS[ft]
    meta_path = p['model'].replace('.pth', '.json')
    
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"No model found for {ft}. Train first.")
        
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # threshold는 최상위와 config 내부 모두 업데이트 (호환성)
        if "threshold" in settings:
            meta["threshold"] = settings["threshold"]
            if "config" not in meta: meta["config"] = {}
            meta["config"]["threshold"] = settings["threshold"]
            
        # 기타 설정들 config에 반영
        for key, value in settings.items():
            if key != "threshold":
                if "config" not in meta: meta["config"] = {}
                meta["config"][key] = value
            
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)
        return {"status": "success", "updated_config": settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_training_pipeline(ft: str, training_config: dict, date_params: dict):
    """백그라운드 학습 실행 (명시적인 훈련 설정 및 날짜 범위 사용)"""
    print(f"[*] [BG] Starting training pipeline for {ft}...")
    try:
        # 데이터 수집 (날짜 파라미터 기반)
        collector.collect_and_save(
            train_start=date_params['train_start'],
            train_end=date_params['train_end'],
            test_start=date_params['test_start'],
            test_end=date_params['test_end']
        )
        
        # Trainer 생성 시 주입된 training_config가 우선 적용됨
        trainer = Trainer(feature_type=ft, config_override=training_config)
        if trainer.train():
            print(f"[*] [BG] {ft.capitalize()} model training complete.")
        else:
            print(f"[!] [BG] {ft.capitalize()} model training failed.")
    except Exception as e:
        print(f"[!] [BG] Training Error: {e}")

@app.post("/api/model/train")
async def train_model(
    background_tasks: BackgroundTasks,
    ft: str = Query(..., regex="^(traffic|optical)$"),
    training_config: dict = Body({}),
    train_start: str = Query(None),
    train_end: str = Query(None),
    test_start: str = Query(None),
    test_end: str = Query(None)
):
    """사용자가 지정한 훈련 설정 및 날짜 범위를 기반으로 모델 재학습 시작"""
    # 날짜 파라미터가 없으면 기본값(최근 37일/7일) 생성
    now = datetime.now()
    if not train_start: train_start = (now - timedelta(days=37)).strftime('%Y-%m-%d')
    if not train_end:   train_end   = (now - timedelta(days=7)).strftime('%Y-%m-%d')
    if not test_start:  test_start  = (now - timedelta(days=7)).strftime('%Y-%m-%d')
    if not test_end:    test_end    = now.strftime('%Y-%m-%d')

    date_params = {
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end
    }
    background_tasks.add_task(run_training_pipeline, ft, training_config, date_params)
    return {
        "status": "started", 
        "message": f"{ft} model training task queued.",
        "range": date_params
    }

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
        where_clauses = ["occur_date = (SELECT MAX(occur_date) FROM anomaly_detection)"]
        where_clauses.append(f"alarm_level BETWEEN {severity_min} AND {severity_max}")
        if rising_only: where_clauses.append("slope_label = 'RISING'")
        
        query = f"""
            SELECT occur_date, ip_addr, cid as slot_id, lid as port_id, 
                   severity, alarm_level, alarm_label, slope, slope_label, 
                   ttf_minutes, expected_fatal_time, anomaly_reason
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
            SELECT occur_date, tx_packet, rx_packet, error_packet, 
                   tx_avg_power, rx_avg_power, anomaly_score, severity, threshold
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

@app.get("/api/scheduler/status")
async def get_scheduler_status():
    if not scheduler_instance: return {"status": "stopped", "next_run_time": None}
    state = scheduler_instance.scheduler.state
    status_str = "running" if state == 1 else "stopped"
    next_run = None
    if status_str == "running":
        jobs = scheduler_instance.scheduler.get_jobs()
        if jobs and jobs[0].next_run_time:
            next_run = jobs[0].next_run_time.strftime('%Y-%m-%d %H:%M:%S')
    return {"status": status_str, "next_run_time": next_run}

@app.post("/api/scheduler/status")
async def control_scheduler(data: dict = Body(...)):
    action = data.get("action")
    if not scheduler_instance: raise HTTPException(status_code=500, detail="Scheduler not ready")
    if action == "stop": scheduler_instance.scheduler.pause()
    elif action == "start":
        scheduler_instance.restart()
        if scheduler_instance.scheduler.state == 2: scheduler_instance.scheduler.resume()
    return await get_scheduler_status()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
