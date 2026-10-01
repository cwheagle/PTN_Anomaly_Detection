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

from validation.evaluation.metrics import summarize
from validation.simulator.scenario_generator import load
from src.pipeline.alerting import (AlertPolicy, dampening_step, dynamic_threshold, level_from_severity,
                                   severity_from_ratio)

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


def _track_frame(scores, global_th, policy, ft):
    df = scores.copy()
    th = dynamic_threshold(df["past_mean"].values, df["past_std"].values, global_th, policy)
    df[f"{ft}_flag"] = df["mse"].values > th
    df[f"{ft}_sev"] = severity_from_ratio(df["mse"].values, th, policy)
    return df[KEY + ["occur_date", f"{ft}_flag", f"{ft}_sev"]]


def simulate_alarms(track_scores, policy: AlertPolicy):
    """
    정책을 적용해 최종 알람을 계산.
    Returns: DataFrame(KEY, occur_date, alarm, alarm_raw, severity, level)
      alarm     : 댐프닝까지 통과한 최종 알람 (detect() 의 is_anomaly)
      alarm_raw : 댐프닝 전 (트랙 플래그 or)
    """
    frames = [_track_frame(sc, th, policy, ft) for ft, (sc, th) in track_scores.items()]
    final = frames[0]
    for f in frames[1:]:
        final = final.merge(f, on=KEY + ["occur_date"], how="outer")
    for ft in ("traffic", "optical"):
        if f"{ft}_flag" not in final:
            final[f"{ft}_flag"], final[f"{ft}_sev"] = False, 0.0
        final[f"{ft}_flag"] = final[f"{ft}_flag"].fillna(False).astype(bool)
        final[f"{ft}_sev"] = final[f"{ft}_sev"].fillna(0.0)

    final = final.sort_values(KEY + ["occur_date"]).reset_index(drop=True)
    final["severity"] = np.maximum(final["traffic_sev"].values, final["optical_sev"].values)   # 우세 트랙의 심각도
    final["level"] = level_from_severity(final["severity"].values)
    final["alarm_raw"] = final["traffic_flag"].values | final["optical_flag"].values

    levels, raw = final["level"].values, final["alarm_raw"].values
    gid = final.groupby(KEY, sort=False).ngroup().values
    alarm = np.zeros(len(final), dtype=bool)
    start = 0
    for i in range(1, len(final) + 1):
        if i == len(final) or gid[i] != gid[start]:             # 포트가 바뀌면 상태 초기화 (포트별 독립 상태)
            state = {"level": 0, "count": 0}
            for j in range(start, i):
                lv = int(levels[j])
                passed = dampening_step(state, lv, policy)
                alarm[j] = lv > 0 and passed and bool(raw[j])
            start = i
    final["alarm"] = alarm
    return final[KEY + ["occur_date", "alarm", "alarm_raw", "severity", "level"]]


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
