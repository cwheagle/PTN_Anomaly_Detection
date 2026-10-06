"""
알람 정책 튜닝 (재학습 없이): 저장된 점수에 정책을 적용해 detect() 와 같은 알람을 빠르게 재계산

모델 추론(느림)은 한 번만 하고 포트·시각별 (mse, 과거 평균, 과거 표준편차)를 저장해 두면,
임계치 배율 / σ 배수 / 댐프닝 횟수를 바꿔 가며 알람과 평가 지표를 수 초 만에 다시 계산할 수 있다.

운영 로직(src/pipeline/alerting.py)의 함수를 그대로 호출하고 detect() 의 처리 순서를 따른다:
  트랙별 임계치/심각도/플래그 -> 트랙 병합(outer) -> 우세 트랙의 심각도 -> 등급 -> 포트별 댐프닝
tests/pipeline/test_tuning_equivalence.py 가 detect() 와의 일치를 검증한다.
"""
import itertools
import os
import pickle
import time

import numpy as np
import pandas as pd

from validation.evaluation.baselines import BASELINES
from validation.evaluation.metrics import summarize
from validation.simulator.scenario_generator import load
from src.pipeline.alerting import AlertPolicy, alarms_from_track_scores
from src.pipeline.alerting import _track_frame as _alerting_track_frame

KEY = ["ip_addr", "cid", "lid"]


def compute_track_scores(detector, traffic, optical):
    """모델 추론을 한 번 수행해 트랙별 원시 점수를 반환: {'traffic': (df, 전역 임계치), 'optical': (...)}"""
    out = {}
    for ft, df in (("traffic", traffic), ("optical", optical)):
        if df is not None and len(df):
            scores, th = detector.track_scores(df, ft)
            if scores is not None:
                out[ft] = (scores, th)
    return out


# 알람 계산은 운영·승격 게이트와 같은 코드를 쓰도록 src/pipeline/alerting.py 로 이동했다 (기존 이름으로 재노출).
_track_frame = _alerting_track_frame
simulate_alarms = alarms_from_track_scores


def policy_grid(threshold_scales=(1.0,), sigma_ks=(3.0,), damping_presets=None, severity_decays=(0.5,)):
    """정책 후보 격자. damping_presets: {이름: {등급: 횟수}}"""
    damping_presets = damping_presets or {"default(3/2/1)": {1: 3, 2: 2, 3: 1}}
    for scale, k, (name, steps), dec in itertools.product(threshold_scales, sigma_ks, damping_presets.items(), severity_decays):
        yield {"threshold_scale": scale, "sigma_k": k, "damping": name, "severity_decay": dec,
               "policy": AlertPolicy(threshold_scale=scale, sigma_k=k, severity_decay=dec, dampening_steps=dict(steps))}


DAMPING_PRESETS = {
    "none(1/1/1)": {1: 1, 2: 1, 3: 1},
    "light(2/1/1)": {1: 2, 2: 1, 3: 1},
    "default(3/2/1)": {1: 3, 2: 2, 3: 1},
    "strict(4/3/2)": {1: 4, 2: 3, 3: 2},
    "heavy(6/4/3)": {1: 6, 2: 4, 3: 3},
    "heavy2(8/6/4)": {1: 8, 2: 6, 3: 4},
    "critical-only(99/99/1)": {1: 99, 2: 99, 3: 1},      # MINOR/MAJOR 알람 없이 CRITICAL 만
}


RUNS = os.path.join("validation", "runs")


def get_scores(dataset, models_dir, tag):
    """데이터셋의 트랙별 점수를 캐시에서 읽거나 모델 추론으로 계산 (dataset 은 validation/runs 아래 폴더 이름)"""
    d = os.path.join(RUNS, dataset)
    cache = os.path.join(d, f"scores_{tag}.pkl")
    data = load(d)
    if os.path.exists(cache):
        return data, pickle.load(open(cache, "rb"))
    if models_dir:
        from src.config import PATHS
        for ft in ("traffic", "optical"):
            PATHS[ft] = {"model": os.path.join(models_dir, f"{ft}_ae.pth"),
                         "scaler": os.path.join(models_dir, f"{ft}_scaler.joblib")}
    from src.pipeline.inference import AnomalyDetector
    t = time.time()
    scores = compute_track_scores(AnomalyDetector(), data["traffic"], data["optical"])
    print(f"[*] {dataset}: 모델 추론 {time.time() - t:.0f}s (점수 캐시 저장)")
    pickle.dump(scores, open(cache, "wb"))
    return data, scores


def evaluate_alarms(sim, data, extra_cols=None):
    """simulate_alarms 결과(또는 같은 형식의 알람 프레임)를 정답과 조인해 이벤트/스텝 지표 계산"""
    ev = sim.merge(data["labels"][KEY + ["occur_date", "state", "episode_id", "scenario", "nuisance"]],
                   on=KEY + ["occur_date"], how="inner")
    if "severity" in ev.columns:
        ev = ev.rename(columns={"severity": "score"})
    return summarize(ev, data["episodes"])


def evaluate_policy(data, scores, policy):
    return evaluate_alarms(simulate_alarms(scores, policy), data)


def select_policy(df, min_early=0.70, max_fp=0.02):
    """조기 탐지율 >= min_early 이고 오탐 <= max_fp 인 정책 중 이벤트 F1 이 최대인 행을 반환 (없으면 None)"""
    ok = df[(df["early_detection_rate"] >= min_early) & (df["false_incidents_per_port_day"] <= max_fp)]
    return None if ok.empty else ok.sort_values("event_f1", ascending=False).iloc[0]


def baseline_alarms(data, rows, name):
    """기준선 알람을 AI 가 채점한 스텝(rows)과 같은 스텝에서 계산"""
    raw = data["traffic"].merge(data["optical"], on=KEY + ["occur_date"], how="outer")
    b = BASELINES[name](raw)
    merged = rows[KEY + ["occur_date"]].merge(pd.concat([raw[KEY + ["occur_date"]], b], axis=1),
                                              on=KEY + ["occur_date"], how="left")
    return pd.DataFrame({**{k: merged[k] for k in KEY + ["occur_date"]},
                         "alarm": merged["alarm"].fillna(False).astype(bool), "severity": merged["score"].fillna(0.0)})
