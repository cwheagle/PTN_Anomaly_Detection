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
