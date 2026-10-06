"""
알람 판정 로직 (임계치 / 심각도 / 댐프닝) — 순수 함수

운영 추론(AnomalyDetector)과 튜닝 도구(validation)가 **같은 코드**를 쓰도록 분리했다.
튜닝에서 찾은 값이 그대로 운영에 적용되며, 기본값(AlertPolicy())은 분리 이전의 동작과 같다.

파이프라인
  1) dynamic_threshold : 최종 임계치 = max(μ + k·σ (과거 구간), 전역 임계치 × scale)
  2) severity_from_ratio: ratio = score / 임계치 -> 0~100 심각도 (1.0 에서 50)
  3) level_from_severity: 심각도 -> 경보 등급 (MINOR 50 / MAJOR 70 / CRITICAL 90)
  4) dampening_step    : 등급별 연속 횟수 요건을 충족해야 알람을 통과 (알람 피로도 억제)
"""
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# 경보 등급 하한 (SEVERITY_CONFIG 와 동일: MINOR 50 / MAJOR 70 / CRITICAL 90)
LEVEL_MIN_SEVERITY = {1: 50.0, 2: 70.0, 3: 90.0}
LEVEL_LABEL = {0: "NORMAL", 1: "MINOR", 2: "MAJOR", 3: "CRITICAL"}
KEY = ["ip_addr", "cid", "lid"]
STEP = pd.Timedelta(minutes=15)


@dataclass(frozen=True)
class AlertPolicy:
    """알람 판정 정책. 기본값은 기존(하드코딩) 동작과 같다."""
    sigma_k: float = 3.0                 # 동적 임계치 = 과거 평균 + k * 표준편차
    dyn_cap: float = 1.2                 # 과거 평균이 이미 전역 임계치를 넘으면 동적 임계치 상한 = 전역 임계치 * dyn_cap
    threshold_scale: float = 1.0         # 전역 임계치 배율 (percentile 조정과 같은 효과를 재학습 없이 시험)
    severity_decay: float = 0.5          # 임계치 초과 시 심각도 상승 속도 (50 + 50 * (1 - exp(-decay * (ratio - 1))))
    # 등급별 알람 통과에 필요한 '연속' 횟수. 기존: CRITICAL 1(즉시) / MAJOR 2 / MINOR 3
    dampening_steps: dict = field(default_factory=lambda: {1: 3, 2: 2, 3: 1})

    def with_(self, **kw):
        return replace(self, **kw)


def load_policy():
    """src/config.py 의 ALERT_POLICY(선택)로 기본 정책을 덮어쓴다. 없으면 기본값."""
    try:
        from src import config
        override = getattr(config, "ALERT_POLICY", None) or {}
    except Exception:
        override = {}
    steps = override.get("dampening_steps")
    kw = {k: v for k, v in override.items() if k != "dampening_steps" and k in AlertPolicy.__dataclass_fields__}
    policy = AlertPolicy(**kw)
    if steps:
        policy = policy.with_(dampening_steps={int(k): int(v) for k, v in steps.items()})
    return policy


def dynamic_threshold(past_mean, past_std, global_th, policy: AlertPolicy):
    """최종 임계치 (벡터). 전역 임계치(학습 percentile) × scale 이 하한이다."""
    min_th = global_th * policy.threshold_scale
    dyn = past_mean + policy.sigma_k * past_std
    # 과거부터 이미 오류가 지속되어 past_mean 이 하한을 넘으면, 비정상을 '새로운 정상'으로 학습하지 않도록 상한을 둠
    dyn = np.where(past_mean > min_th, np.minimum(dyn, min_th * policy.dyn_cap), dyn)
    return np.maximum(dyn, min_th)


def severity_from_ratio(score, threshold, policy: AlertPolicy):
    """심각도 0~100 (벡터). ratio=1 에서 50, 이후 지수적으로 100 에 수렴."""
    th_safe = np.where(threshold <= 0, 1e-9, threshold)
    ratio = score / th_safe
    sev = np.where(ratio <= 1.0, ratio * 50.0,
                   50.0 + 50.0 * (1 - np.exp(-policy.severity_decay * (ratio - 1.0))))
    return np.where(threshold <= 0, 0.0, sev)


def level_from_severity(severity):
    """경보 등급 (벡터): 0 정상 / 1 MINOR / 2 MAJOR / 3 CRITICAL"""
    sev = np.asarray(severity, dtype=float)
    return np.select([sev >= LEVEL_MIN_SEVERITY[3], sev >= LEVEL_MIN_SEVERITY[2], sev >= LEVEL_MIN_SEVERITY[1]],
                     [3, 2, 1], default=0)


def dampening_step(state: dict, level: int, policy: AlertPolicy):
    """
    포트 1개의 한 스텝 댐프닝. state = {'level': int, 'count': int} 를 갱신한다.
    Returns: (passed: bool). level 0 이면 상태를 리셋하고 False.
    """
    if level == 0:
        state["level"], state["count"] = 0, 0
        return False
    if level >= state["level"]:
        state["level"] = level
        state["count"] += 1
    else:
        state["level"] = level      # 등급이 내려가면 새로 카운트 시작
        state["count"] = 1
    need = policy.dampening_steps.get(state["level"], 1)
    return state["count"] >= need


# ─────────────────────────────────────────────
# 저장된 점수 -> 알람 (튜닝 도구 · 승격 게이트 공용)
# ─────────────────────────────────────────────
def _track_frame(scores, global_th, policy, ft):
    df = scores.copy()
    th = dynamic_threshold(df["past_mean"].values, df["past_std"].values, global_th, policy)
    df[f"{ft}_flag"] = df["mse"].values > th
    df[f"{ft}_sev"] = severity_from_ratio(df["mse"].values, th, policy)
    return df[KEY + ["occur_date", f"{ft}_flag", f"{ft}_sev"]]


def alarms_from_track_scores(track_scores, policy: AlertPolicy):
    """
    트랙별 원시 점수(`AnomalyDetector.track_scores`)에 정책을 적용해 최종 알람을 계산.
    detect() 와 같은 순서(트랙별 임계치/심각도/플래그 -> 트랙 병합 -> 우세 트랙의 심각도 -> 등급 -> 포트별 댐프닝)를
    따르며, 일치는 tests/pipeline/test_alerting.py · test_tuning_equivalence.py 가 검증한다.

    track_scores: {'traffic': (df[occur_date, ip_addr, cid, lid, mse, past_mean, past_std], 전역 임계치), 'optical': ...}
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


def count_incidents(alarms: pd.DataFrame, gap_steps: int = 4) -> int:
    """포트별로 연속된 알람(간격 gap_steps 이하)을 하나의 인시던트로 묶어 개수를 센다 (알람 폭주를 1건으로 계수).
    alarms: KEY + occur_date + alarm(bool) 컬럼. 평가 도구의 make_incidents 와 같은 묶음 규칙."""
    a = alarms[alarms["alarm"].astype(bool)].sort_values(KEY + ["occur_date"])
    if a.empty:
        return 0
    gap = a.groupby(KEY)["occur_date"].diff()
    return int((gap.isna() | (gap > STEP * gap_steps)).sum())
