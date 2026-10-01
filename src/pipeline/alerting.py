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

# 경보 등급 하한 (SEVERITY_CONFIG 와 동일: MINOR 50 / MAJOR 70 / CRITICAL 90)
LEVEL_MIN_SEVERITY = {1: 50.0, 2: 70.0, 3: 90.0}
LEVEL_LABEL = {0: "NORMAL", 1: "MINOR", 2: "MAJOR", 3: "CRITICAL"}


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
