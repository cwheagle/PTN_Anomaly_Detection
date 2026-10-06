"""
알람 판정 로직(src/pipeline/alerting.py) 검증: 임계치 / 심각도 / 등급 / 댐프닝 / 정책

실행 방법:
  pytest tests/pipeline/test_alerting.py -v
"""
import numpy as np
import pytest

from src.pipeline.alerting import (AlertPolicy, dampening_step, dynamic_threshold, level_from_severity,
                                   load_policy, severity_from_ratio)

P = AlertPolicy()


def test_default_policy_matches_previous_hardcoded_behavior():
    assert (P.sigma_k, P.dyn_cap, P.threshold_scale, P.severity_decay) == (3.0, 1.2, 1.0, 0.5)
    assert P.dampening_steps == {1: 3, 2: 2, 3: 1}                 # MINOR 3회 / MAJOR 2회 / CRITICAL 즉시


def test_dynamic_threshold_floor_is_global_threshold():
    """과거 변동이 없으면 전역 임계치가 최종 임계치"""
    th = dynamic_threshold(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.3, P)
    assert np.allclose(th, 0.3)


def test_dynamic_threshold_rises_with_recent_variability():
    th = dynamic_threshold(np.array([0.1]), np.array([0.2]), 0.3, P)      # 0.1 + 3*0.2 = 0.7
    assert th[0] == pytest.approx(0.7)


def test_dynamic_threshold_is_capped_when_past_already_abnormal():
    """과거부터 이미 오류가 지속(past_mean > 전역 임계치)이면 '새로운 정상'이 되지 않도록 상한 1.2배"""
    th = dynamic_threshold(np.array([0.5]), np.array([0.5]), 0.3, P)      # 0.5+1.5=2.0 -> 0.3*1.2
    assert th[0] == pytest.approx(0.36)


def test_threshold_scale_and_sigma_k_change_the_threshold():
    base = dynamic_threshold(np.array([0.1]), np.array([0.2]), 0.3, P)[0]
    assert dynamic_threshold(np.array([0.0]), np.array([0.0]), 0.3, P.with_(threshold_scale=2.0))[0] == pytest.approx(0.6)
    assert dynamic_threshold(np.array([0.1]), np.array([0.2]), 0.3, P.with_(sigma_k=2.0))[0] < base


def test_severity_curve_anchor_points():
    sev = severity_from_ratio(np.array([0.0, 0.5, 1.0, 2.0, 4.0, 4.3, 20.0]), np.ones(7), P)
    assert sev[0] == 0 and sev[1] == 25 and sev[2] == 50              # 임계치에서 정확히 50
    assert 69 < sev[3] < 70                                           # ratio 2.0 -> 69.7 (MAJOR 경계는 ratio 2.02)
    assert sev[4] < 90 <= sev[5]                                      # CRITICAL 경계는 ratio 4.22: 4.0 미만 / 4.3 이상
    assert sev[6] < 100.0001


def test_severity_zero_threshold_is_safe():
    assert severity_from_ratio(np.array([1.0]), np.array([0.0]), P)[0] == 0.0


def test_level_boundaries():
    assert list(level_from_severity([0, 49.99, 50, 69.99, 70, 89.99, 90, 100])) == [0, 0, 1, 1, 2, 2, 3, 3]


def _run(levels, policy=P):
    state = {"level": 0, "count": 0}
    return [dampening_step(state, lv, policy) for lv in levels]


def test_dampening_critical_is_immediate_major_needs_two_minor_needs_three():
    assert _run([3]) == [True]
    assert _run([2, 2, 2]) == [False, True, True]
    assert _run([1, 1, 1, 1]) == [False, False, True, True]


def test_dampening_resets_on_normal_and_restarts_when_level_drops():
    assert _run([2, 2, 0, 2]) == [False, True, False, False]           # 정상(0)으로 돌아가면 카운트 리셋
    assert _run([3, 1, 1, 1]) == [True, False, False, True]            # 등급이 내려가면 새로 카운트


def test_dampening_policy_can_be_relaxed_or_tightened():
    relaxed = P.with_(dampening_steps={1: 1, 2: 1, 3: 1})
    strict = P.with_(dampening_steps={1: 4, 2: 3, 3: 2})
    assert _run([1], relaxed) == [True]
    assert _run([3, 3], strict) == [False, True]


def test_load_policy_defaults_without_override(monkeypatch):
    import src.config as cfg
    monkeypatch.delattr(cfg, "ALERT_POLICY", raising=False)
    assert load_policy() == AlertPolicy()


def test_load_policy_applies_config_override(monkeypatch):
    """src/config.py 의 ALERT_POLICY(config.py.example 의 precision 프리셋)가 정책으로 반영되어야 함"""
    import src.config as cfg
    monkeypatch.setattr(cfg, "ALERT_POLICY", {"threshold_scale": 3.0, "sigma_k": 2.0,
                                              "dampening_steps": {1: 6, 2: 4, 3: 3}}, raising=False)
    p = load_policy()
    assert (p.threshold_scale, p.sigma_k) == (3.0, 2.0)
    assert p.dampening_steps == {1: 6, 2: 4, 3: 3}
    assert p.dyn_cap == 1.2                                  # 지정하지 않은 값은 기본값 유지


def test_load_policy_ignores_unknown_keys(monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(cfg, "ALERT_POLICY", {"no_such_option": 1, "sigma_k": 2.5}, raising=False)
    assert load_policy().sigma_k == 2.5


# ──────────────────────────────────────────────
# P1-1: 저장된 점수 -> 알람 (튜닝 도구 · 승격 게이트 공용)
# ──────────────────────────────────────────────
KEY = ["ip_addr", "cid", "lid", "occur_date"]


def test_alarms_from_track_scores_matches_detect_with_real_inference(tiny_env, tiny_scenario):
    """T-A1: 소형 모델로 detect() 와 alarms_from_track_scores 의 알람/등급이 한 건도 다르지 않음 (models/ 없이도 검증)"""
    from src.pipeline.alerting import alarms_from_track_scores
    from src.pipeline.inference import AnomalyDetector

    for policy in (AlertPolicy(), AlertPolicy(threshold_scale=0.6, dampening_steps={1: 1, 2: 1, 3: 1}),
                   AlertPolicy(sigma_k=2.0, dampening_steps={1: 6, 2: 4, 3: 3})):
        det = AnomalyDetector(policy=policy)
        res = det.detect(df_traffic=tiny_scenario["traffic"], df_optical=tiny_scenario["optical"], latest_only=False)
        scores = {}
        for ft, df in (("traffic", tiny_scenario["traffic"]), ("optical", tiny_scenario["optical"])):
            sc, th = det.track_scores(df, ft)
            scores[ft] = (sc, th)
        sim = alarms_from_track_scores(scores, policy)
        m = res[KEY + ["is_anomaly", "alarm_level"]].merge(sim, on=KEY, how="outer", indicator=True)
        assert (m["_merge"] == "both").all()
        assert (m["is_anomaly"].astype(bool) == m["alarm"]).all()
        assert (m["alarm_level"] == m["level"].where(m["alarm"], 0)).all()
        assert int(sim["alarm"].sum()) >= 0


def test_validation_tuning_reexports_the_same_functions():
    """tuning 도구는 운영과 같은 함수를 쓴다 (lessons #31) — 복제본이 아니라 동일 객체"""
    from src.pipeline import alerting
    from validation.evaluation import tuning
    assert tuning.simulate_alarms is alerting.alarms_from_track_scores
    assert tuning._track_frame is alerting._track_frame


def _synthetic_scores(seed=0, ports=6, n=300):
    import pandas as pd
    rng = np.random.default_rng(seed)
    parts = []
    for p in range(ports):
        mse = rng.gamma(2.0, 0.05, n)
        mse[100:115] *= 10
        mse[200:203] *= 20
        s = pd.Series(mse)
        parts.append(pd.DataFrame({
            "occur_date": pd.date_range("2026-03-01", periods=n, freq="15min"), "ip_addr": "1.1.1.1", "cid": 0, "lid": p,
            "mse": mse, "past_mean": s.rolling(11, min_periods=1).mean().shift(1).fillna(0.1).values,
            "past_std": s.rolling(11, min_periods=1).std().shift(1).fillna(0.05).values}))
    return {"traffic": (pd.concat(parts, ignore_index=True), 0.2)}


# 이동 전(git HEAD 의 validation/evaluation/tuning.py simulate_alarms)에서 계산한 값
GOLDEN_SYNTH = {"default": 62, "relaxed": 151, "strict": 20}


def test_alarms_from_track_scores_golden_for_synthetic_scores():
    """이동 전 simulate_alarms 의 출력을 고정한 골든: 합성 점수(시드 고정)로 정책별 알람 수가 이동 전과 같아야 함"""
    from src.pipeline.alerting import alarms_from_track_scores
    counts = {name: int(alarms_from_track_scores(_synthetic_scores(), pol).alarm.sum())
              for name, pol in (("default", AlertPolicy()), ("relaxed", AlertPolicy(dampening_steps={1: 1, 2: 1, 3: 1})),
                                ("strict", AlertPolicy(dampening_steps={1: 6, 2: 4, 3: 3})))}
    assert counts == GOLDEN_SYNTH


def test_count_incidents_groups_consecutive_alarms_per_port():
    import pandas as pd
    from src.pipeline.alerting import count_incidents
    t = pd.date_range("2026-03-01", periods=40, freq="15min")
    flags = np.zeros(40, dtype=bool)
    flags[[2, 3, 4, 8, 20]] = True               # 2~4, 8(간격 4 이하 -> 같은 인시던트), 20(분리)
    a = pd.DataFrame({"ip_addr": "1.1.1.1", "cid": 0, "lid": 1, "occur_date": t, "alarm": flags})
    b = a.assign(lid=2)
    assert count_incidents(a) == 2
    assert count_incidents(pd.concat([a, b])) == 4                 # 포트별로 따로 센다
    assert count_incidents(a.assign(alarm=False)) == 0


def test_count_incidents_equals_evaluation_make_incidents():
    """T-A2: 게이트가 쓰는 개수 == 평가 도구의 make_incidents 개수"""
    from src.pipeline.alerting import alarms_from_track_scores, count_incidents
    from validation.evaluation.metrics import make_incidents
    for seed in (0, 1, 2):
        sim = alarms_from_track_scores(_synthetic_scores(seed), AlertPolicy(dampening_steps={1: 1, 2: 1, 3: 1}))
        ev = sim.assign(nuisance=False)
        inc, _ = make_incidents(ev)
        assert count_incidents(sim) == len(inc) > 0


# ──────────────────────────────────────────────
# T-P3-A1/A2: 모델별 정책 프리셋과 메타 왕복 (P1-3)
# ──────────────────────────────────────────────
def test_presets_have_the_documented_values():
    """T-P3-A1: precision = 오탐 억제형(performance_report 3~5장), default = 코드 기본값"""
    from src.pipeline.alerting import POLICY_PRESETS, get_preset
    p = get_preset("precision")
    assert (p.threshold_scale, p.sigma_k, p.dampening_steps) == (3.0, 2.0, {1: 6, 2: 4, 3: 3})
    assert (p.dyn_cap, p.severity_decay) == (1.2, 0.5)                       # 건드리지 않는 값은 기본 그대로
    d = get_preset("default")
    assert d == AlertPolicy() and (d.threshold_scale, d.sigma_k, d.dampening_steps) == (1.0, 3.0, {1: 3, 2: 2, 3: 1})
    assert set(POLICY_PRESETS) == {"default", "precision"}


def test_unknown_preset_name_is_an_error():
    from src.pipeline.alerting import get_preset
    with pytest.raises(ValueError, match="fast"):
        get_preset("fast")


def test_policy_meta_roundtrip_and_string_keys_restored():
    """T-P3-A2: 메타(JSON 이라 댐프닝 키가 문자열)로 저장했다 읽어도 같은 정책"""
    import json
    from src.pipeline.alerting import get_preset, policy_from_meta, policy_to_meta
    meta = policy_to_meta(get_preset("precision"))
    assert meta == {"preset": "precision", "sigma_k": 2.0, "dyn_cap": 1.2, "threshold_scale": 3.0,
                    "severity_decay": 0.5, "dampening_steps": {"1": 6, "2": 4, "3": 3}}
    restored = policy_from_meta(json.loads(json.dumps(meta)))
    assert restored == get_preset("precision") and restored.dampening_steps == {1: 6, 2: 4, 3: 3}
    custom = AlertPolicy(sigma_k=2.5, dampening_steps={1: 2, 2: 2, 3: 1})
    assert policy_to_meta(custom)["preset"] == "custom"
    assert policy_from_meta(policy_to_meta(custom)) == custom


def test_policy_from_meta_missing_or_partial():
    from src.pipeline.alerting import policy_from_meta
    assert policy_from_meta(None) is None and policy_from_meta({}) is None            # 정책 없음 = 모델이 지정하지 않음
    partial = policy_from_meta({"preset": "precision", "sigma_k": 2.5})               # 빠진 값은 프리셋에서
    assert (partial.sigma_k, partial.threshold_scale, partial.dampening_steps) == (2.5, 3.0, {1: 6, 2: 4, 3: 3})
