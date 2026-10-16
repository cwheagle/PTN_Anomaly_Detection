"""
P1-2 판정 모델 재학습 실행기 (설계서 9.7.5a, 개정 6 — 사용자 결정 Q15: 개발 기간 [S, J0) 안에서만 학습)

- 입력은 DB 가 아니라 스냅샷의 정상 ETH 행 중 [S, end) 만이다(어댑터가 반열림을 강제). 기존 `DataCollector.collect_and_save` 를 그대로 호출하고
  DB 대신 스냅샷 어댑터를 주입한다(`src` 수정 없음). 자기 알람 이력은 빈 값, 의심 구간은 모델과 무관한 규칙 B 만, 광은 빈 프레임(트래픽 트랙만).
- `MODEL_CONFIG` 그대로(`config_override` 없음, RECIPE 단축 없음), 포트 홀드아웃 10%, `train_seed=0`, 판정 모델은 1회 학습.
- 모델은 `validation/runs/p1_2/models/<run_id>/` 에 저장하고 `PATHS` 는 학습 중에만 바꿨다가 복구한다. 운영 `models/` 파일 해시를 전후로 비교한다.
- 학습 실패(의심 비율 초과·시퀀스 0·홀드아웃 0·비유한 손실)는 기준 완화·재설정 없이 '판정 모델 없음 → AI 판정 불가'다.
  기술적 실패(예외)만 같은 시드로 다시 실행할 수 있고 재실행 횟수를 기록한다.
"""
import contextlib
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

TRACK = "traffic"
TRAIN_SEED = 0
EXPLORE_TRAIN_DATA_DAYS = 21            # 탐색 모델: [S, X), X = 21번째 데이터일의 다음 날 (설계 R5, 데이터를 보지 않고 정함)
KEYS = ["ip_addr", "cid", "lid"]
MODELS_SUBDIR = "models"


class TrainFailure(RuntimeError):
    """학습 규칙 실패 — 판정 모델 없음(AI 판정 불가). 기준을 완화하거나 설정을 바꿔 다시 학습하지 않는다."""


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(directory):
    """폴더 안 파일들의 {상대 경로: sha256}. 폴더가 없으면 {}."""
    out = {}
    if not os.path.isdir(directory):
        return out
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            p = os.path.join(root, f)
            out[os.path.relpath(p, directory)] = _sha_file(p)
    return out


class SnapshotDB:
    """DataCollector 가 쓰는 DB 인터페이스의 스냅샷 판. `[start, end)` 밖 행은 요청 범위와 상관없이 돌려주지 않는다(반열림 강제)."""

    def __init__(self, eth, start, end):
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        df = eth.copy()
        df["occur_date"] = pd.to_datetime(df["bin_end"] if "bin_end" in df.columns else df["occur_date"])
        df["ip_addr"] = df["ip_addr"].astype(str).str.strip()
        for c in ("tx_packet", "rx_packet", "error_packet"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.drop_duplicates(subset=KEYS + ["occur_date"], keep="last")               # 재생·운영과 같은 마지막 행 유지
        df = df[(df["occur_date"] >= self.start) & (df["occur_date"] < self.end)]
        self.traffic = df[["occur_date", "ip_addr", "cid", "lid", "tx_packet", "rx_packet", "error_packet"]].sort_values(KEYS + ["occur_date"]).reset_index(drop=True)

    def fetch_traffic(self, start, end, stop_checker=None):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        t = self.traffic
        return t[(t["occur_date"] >= max(s, self.start)) & (t["occur_date"] < min(e, self.end))].copy()

    def fetch_optical(self, start, end, stop_checker=None):
        return pd.DataFrame(columns=["occur_date", "ip_addr", "cid", "lid", "tx_avg_power", "rx_avg_power"])

    def fetch_alarm_rows(self, start, end, min_level=1):
        return pd.DataFrame(columns=["occur_date"] + KEYS)                                  # 자기 알람 이력(A)은 빈 값으로 고정


@contextlib.contextmanager
def use_models_dir(models_dir, track=TRACK):
    """`PATHS[track]` 를 models_dir 로 잠깐 바꾼다 (끝나면 복구, lessons #16)."""
    from src.config import PATHS
    saved = dict(PATHS[track])
    PATHS[track] = {"model": os.path.join(models_dir, f"{track}_ae.pth"), "scaler": os.path.join(models_dir, f"{track}_scaler.joblib")}
    try:
        yield
    finally:
        PATHS[track] = saved


def training_window(ctx, kind):
    """(start, end, 설명). judge: [S, J0). explore: [S, X), X = 21번째 데이터일의 다음 날."""
    p = ctx.periods
    if not p.get("ok"):
        raise TrainFailure(p.get("reason") or "기간을 정할 수 없어 학습할 수 없음")
    if kind == "judge":
        return p["S"], p["J0"], "judge"
    if kind == "explore":
        x = ctx.data_days[EXPLORE_TRAIN_DATA_DAYS - 1] + pd.Timedelta(days=1)
        return p["S"], x, "explore"
    raise ValueError(f"kind 는 judge | explore: {kind}")


def describe_csv(path):
    df = pd.read_csv(path, parse_dates=["occur_date"])
    return {"path": os.path.basename(path), "sha256": _sha_file(path), "rows": int(len(df)),
            "ports": int(df[KEYS].drop_duplicates().shape[0]) if len(df) else 0,
            "data_days": int(df["occur_date"].dt.normalize().nunique()) if len(df) else 0,
            "min_occur_date": str(df["occur_date"].min()) if len(df) else None, "max_occur_date": str(df["occur_date"].max()) if len(df) else None}


def collect(ctx, start, end, out_dir):
    """스냅샷에서 [start, end) 학습/검증 CSV 를 만든다 (운영과 같은 DataCollector, 규칙 B 만, 자기 알람 빈 값). 반환 collect 결과 dict."""
    from src.data.data_collector import DataCollector
    from src.pipeline.retrain_policy import RetrainPolicy
    policy = RetrainPolicy()
    collector = DataCollector.__new__(DataCollector)               # __init__ 은 DB 풀을 만들므로 건너뛰고 스냅샷 어댑터를 주입
    collector.db = SnapshotDB(ctx.eth, start, end)
    results = collector.collect_and_save(train_start=pd.Timestamp(start), train_end=pd.Timestamp(end), feature_type=TRACK,
                                         output_dir=out_dir, val_port_fraction=policy.val_port_fraction, split_salt=policy.split_salt,
                                         suspect_policy=policy)
    return results.get(TRACK, {}), policy


def _check_collect(res):
    if "skipped" in res:
        raise TrainFailure(f"학습 데이터 없음: {res['skipped']} (의심 비율 한도 초과 등) — 판정 모델 없음, AI 판정 불가")
    if not res.get("train") or not res.get("test"):
        raise TrainFailure(f"학습/검증 시퀀스 또는 홀드아웃 포트가 0: {res} — 판정 모델 없음, AI 판정 불가")


def _finite(*vals):
    return all(v is not None and isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def train_model(ctx, kind="judge", run_id=None, seed=TRAIN_SEED):
    """판정/탐색 모델 1회 학습. 반환 train_run dict (status: completed)."""
    import torch
    from src.config import MODEL_CONFIG, PATHS
    from src.models.trainer import Trainer

    start, end, role = training_window(ctx, kind)
    run_id = run_id or f"{role}_seed{seed}"
    out_dir = os.path.join(ctx.runs_dir, MODELS_SUBDIR, run_id)
    prior = os.path.join(out_dir, "train_run.json")
    retries = 0
    if os.path.exists(out_dir):
        old = json.load(open(prior, encoding="utf-8")) if os.path.exists(prior) else {}
        if old.get("status") != "technical_failure":
            raise TrainFailure(f"이미 있는 모델 폴더입니다(덮어쓰지 않음, 재학습·다른 시드 금지): {out_dir} (status={old.get('status')})")
        retries = int(old.get("retries", 0)) + 1
        shutil.rmtree(out_dir)                                       # 기술적 실패의 같은 시드 재실행: 부분 산출물 정리
    os.makedirs(out_dir)
    ctx.record(f"train-model[{role}]", start, end, "dev")            # 학습 구간은 개발 기간 안 (end ≤ J0)
    op_dir = os.path.dirname(PATHS[TRACK]["model"])
    op_before = hash_dir(op_dir)
    run = {"status": "started", "role": role, "run_id": run_id, "train_seed": seed, "retries": retries, "started_at": str(datetime.now()),
           "device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__, "python": platform.python_version(),
           "model_config_sha": hashlib.sha256(json.dumps(MODEL_CONFIG, sort_keys=True, default=str).encode()).hexdigest(),
           "snapshot_manifest_sha": _sha_file(os.path.join(ctx.run_dir, "manifest.json")), "snapshot": os.path.basename(ctx.run_dir),
           "suspect_alarm_history": "empty (규칙 B 만)", "alert_policy": None, "operational_models_dir": op_dir}
    window = {"mode": "port_split", "start": str(pd.Timestamp(start)), "end": str(pd.Timestamp(end)), "end_exclusive": True, "source": "snapshot",
              "snapshot": os.path.basename(ctx.run_dir)}
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        res, policy = collect(ctx, start, end, os.path.join(out_dir, "train_data"))
        run["val_port_fraction"], run["split_salt"] = policy.val_port_fraction, policy.split_salt
        _check_collect(res)
        window.update(val_port_fraction=policy.val_port_fraction, salt=policy.split_salt)
        tr = os.path.join(out_dir, "train_data", f"{TRACK}_train.csv")
        va = os.path.join(out_dir, "train_data", f"{TRACK}_test.csv")
        run["train_csv"], run["val_csv"] = describe_csv(tr), describe_csv(va)
        if pd.Timestamp(run["train_csv"]["max_occur_date"]) >= pd.Timestamp(end) or pd.Timestamp(run["val_csv"]["max_occur_date"]) >= pd.Timestamp(end):
            raise TrainFailure("학습/검증 CSV 에 학습 구간 끝 이후의 행이 있음 — 중단")
        run["holdout_ports"] = sorted(map(list, pd.read_csv(va)[KEYS].drop_duplicates().itertuples(index=False, name=None)))
        run["suspect_stats"] = res.get("suspect_stats")
        saved = dict(PATHS[TRACK])
        try:
            PATHS[TRACK] = {"model": os.path.join(out_dir, f"{TRACK}_ae.pth"), "scaler": os.path.join(out_dir, f"{TRACK}_scaler.joblib")}
            trainer = Trainer(TRACK, config_override=None, activate=True, trigger="manual", window_info=window,
                              suspect_stats=res.get("suspect_stats"), alert_policy=None)
            ok = trainer.train(train_path=tr, val_path=va)
        finally:
            PATHS[TRACK] = saved
        if not ok:
            raise TrainFailure("Trainer.train 이 False 를 반환 — 판정 모델 없음, AI 판정 불가")
        info = model_info_from_dir(out_dir)
        meta = json.load(open(info["meta_path"], encoding="utf-8"))
        if not _finite(meta.get("threshold"), meta.get("final_val_loss")):
            raise TrainFailure(f"손실/임계치가 유한하지 않음 (threshold={meta.get('threshold')}, val_loss={meta.get('final_val_loss')})")
        run.update(status="completed", version=info["version"], threshold=meta["threshold"], final_val_loss=meta["final_val_loss"], trained_at=meta.get("trained_at"),
                   training_window=meta.get("training_window"), model_sha=info["model_sha"], scaler_sha=info["scaler_sha"], meta_sha=info["meta_sha"])
    except TrainFailure as e:
        run.update(status="failed_rule", error=str(e), note="판정 모델 없음 — AI 판정 불가(9.7.4). 기준 완화·설정 변경·다른 시드 재학습 금지")
        _finish(run, out_dir, op_dir, op_before)
        raise
    except BaseException as e:
        run.update(status="technical_failure", error=repr(e), note="데이터와 무관한 예외면 같은 시드로 다시 실행 가능(재실행 횟수 기록)")
        _finish(run, out_dir, op_dir, op_before)
        raise
    _finish(run, out_dir, op_dir, op_before)
    return run


def _finish(run, out_dir, op_dir, op_before):
    after = hash_dir(op_dir)
    run["operational_models_unchanged"] = (after == op_before)
    run["operational_models_files"] = len(after)
    run["finished_at"] = str(datetime.now())
    with open(os.path.join(out_dir, "train_run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=1, default=str)
    if not run["operational_models_unchanged"]:
        raise RuntimeError("운영 models/ 의 파일 해시가 학습 전후로 다릅니다 — 즉시 중단 (운영 모델은 바뀌면 안 됨)")


def model_info_from_dir(models_dir, track=TRACK):
    """모델 폴더(자체 레지스트리)의 활성 버전 메타·해시. 잠금·판정 전 재대조용."""
    reg_path = os.path.join(models_dir, f"{track}_registry.json")
    reg = json.load(open(reg_path, encoding="utf-8"))
    ver = reg["active_version"]
    ent = next(v for v in reg["versions"] if v["version"] == ver)
    meta_path = os.path.join(models_dir, ent["config_path"])
    meta = json.load(open(meta_path, encoding="utf-8"))
    mp, sp = os.path.join(models_dir, ent["model_path"]), os.path.join(models_dir, ent["scaler_path"])
    out = {"track": track, "version": ver, "trained_at": meta.get("trained_at"), "training_window": meta.get("training_window"),
           "alert_policy": meta.get("alert_policy"), "meta_path": meta_path, "model_sha": _sha_file(mp), "scaler_sha": _sha_file(sp),
           "meta_sha": _sha_file(meta_path), "models_dir": models_dir}
    tr = os.path.join(models_dir, "train_run.json")
    if os.path.exists(tr):
        run = json.load(open(tr, encoding="utf-8"))
        out.update(train_run_sha=_sha_file(tr), train_csv_sha=(run.get("train_csv") or {}).get("sha256"), val_csv_sha=(run.get("val_csv") or {}).get("sha256"),
                   snapshot_manifest_sha=run.get("snapshot_manifest_sha"), role=run.get("role"), train_status=run.get("status"))
    return out


def verify_reproducible(ctx, models_dir, info=None):
    """R6: 학습 CSV 최대 occur_date < J0, 스냅샷 [S, J0) 에서 같은 절차로 다시 만든 학습/검증 CSV 해시가 기록과 같아야 한다, 스냅샷 매니페스트 해시 일치.
    통과하면 None, 아니면 사유 문자열 리스트."""
    info = info or model_info_from_dir(models_dir)
    run = json.load(open(os.path.join(models_dir, "train_run.json"), encoding="utf-8"))
    problems = []
    if run.get("status") != "completed":
        return [f"train_run status={run.get('status')}"]
    j0 = pd.Timestamp(ctx.periods["J0"])
    start, end = pd.Timestamp(run["training_window"]["start"]), pd.Timestamp(run["training_window"]["end"])
    if end > j0:
        problems.append(f"학습 구간 끝 {end} > J0 {j0}")
    if pd.Timestamp(run["train_csv"]["max_occur_date"]) >= j0:
        problems.append("학습 CSV 의 최대 occur_date ≥ J0")
    if run.get("snapshot_manifest_sha") != _sha_file(os.path.join(ctx.run_dir, "manifest.json")):
        problems.append("스냅샷 매니페스트 해시가 학습 때와 다름")
    tmp = tempfile.mkdtemp(prefix="p1_2_rebuild_")
    try:
        collect(ctx, start, end, tmp)
        for kind, name in (("train_csv", f"{TRACK}_train.csv"), ("val_csv", f"{TRACK}_test.csv")):
            if _sha_file(os.path.join(tmp, name)) != run[kind]["sha256"]:
                problems.append(f"{kind} 재현 해시 불일치 (같은 절차로 다시 만든 CSV 가 다름)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return problems
