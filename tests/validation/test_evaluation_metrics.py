"""
test_evaluation_metrics.py — 평가 체계(시나리오 생성기, 지표, 베이스라인) 검증

평가 코드가 틀리면 모델 성능 판단이 틀어지므로, 손으로 계산한 정답이 있는 작은 사례로 검증한다.

실행 방법:
  python -m pytest tests/test_evaluation_metrics.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

from validation.evaluation.baselines import always_alarm, fixed_threshold, rolling_rule
from validation.evaluation.metrics import event_metrics, make_incidents, step_metrics
from validation.simulator.scenario_generator import ScenarioConfig, generate

T0 = pd.Timestamp("2026-01-05 00:00:00")
KEY = {"ip_addr": "1.1.1.1", "cid": 0, "lid": 1}


def t(step):
    return T0 + pd.Timedelta(minutes=15 * step)


def make_ev(n, alarm_steps, ep_range=None, nuisance_steps=()):
    """단일 포트 n 스텝. ep_range=(start, ramp_end, plateau_end) 스텝 구간에 장애 정답"""
    state = np.zeros(n, dtype=int)
    if ep_range:
        s, f, e = ep_range
        state[s:f] = 1
        state[f:e + 1] = 2
    alarm = np.zeros(n, dtype=bool)
    alarm[list(alarm_steps)] = True
    nuis = np.zeros(n, dtype=int)
    nuis[list(nuisance_steps)] = 1
    return pd.DataFrame({
        "occur_date": [t(i) for i in range(n)], **{k: [v] * n for k, v in KEY.items()},
        "alarm": alarm, "score": alarm.astype(float), "state": state,
        "episode_id": np.where(state > 0, 0, -1), "scenario": np.where(state > 0, "crc_error", ""),
        "nuisance": nuis,
    })


def make_episodes(start, fail, end):
    return pd.DataFrame([{"episode_id": 0, **KEY, "scenario": "crc_error", "magnitude": 1.0,
                          "t_start": t(start), "t_fail": t(fail), "t_end": t(end)}])


EP = (10, 20, 25)   # start=10, 열화 10~19 (램프), 20~25 지속(장애), t_fail = step 20


# ──────────────────────────────────────────────
# 이벤트 지표 (손계산 검증)
# ──────────────────────────────────────────────

def test_early_detection_and_lead_time():
    """failure(step 20) 5스텝 전인 step 15에 첫 알람 -> 조기 탐지, 리드 75분"""
    ev = make_ev(60, alarm_steps=[15, 16, 17], ep_range=EP)
    m = event_metrics(ev, make_episodes(*EP))

    assert m["episode_detection_rate"] == 1.0
    assert m["early_detection_rate"] == 1.0
    assert m["lead_time_median_min"] == 75.0
    assert m["false_incidents"] == 0
    assert m["event_f1"] == 1.0


def test_late_only_detection_is_not_early():
    """failure 이후(step 22)에야 알람 -> 탐지는 되었지만 예지는 아님"""
    m = event_metrics(make_ev(60, [22], ep_range=EP), make_episodes(*EP))

    assert m["episode_detection_rate"] == 1.0
    assert m["early_detection_rate"] == 0.0
    assert m["late_only_rate"] == 1.0
    assert m["lead_time_median_min"] is None


def test_missed_episode_and_false_incident():
    """에피소드는 놓치고(알람 없음), 멀리 떨어진 step 45에 알람 -> 미탐 1 + 오탐 1"""
    m = event_metrics(make_ev(60, [45], ep_range=EP), make_episodes(*EP))

    assert m["missed_rate"] == 1.0
    assert m["false_incidents"] == 1
    assert m["incident_precision"] == 0.0
    assert m["event_f1"] == 0.0


def test_alarm_long_after_recovery_is_false_not_credited():
    """복구(step 25) + 허용(4스텝) 이후의 알람은 정답으로 인정하지 않는다 (기존 168h 관대 창의 문제)"""
    m = event_metrics(make_ev(80, [31], ep_range=EP), make_episodes(*EP))   # 25+4=29 이후
    assert m["episode_detection_rate"] == 0.0
    assert m["false_incidents"] == 1


def test_alarm_burst_counted_as_single_incident():
    ev = make_ev(60, alarm_steps=[40, 41, 42, 43], ep_range=EP)
    m = event_metrics(ev, make_episodes(*EP))
    assert m["false_incidents"] == 1                       # 4건의 연속 알람 = 1 인시던트


def test_false_incident_on_nuisance_is_reported_separately():
    ev = make_ev(60, alarm_steps=[45], ep_range=EP, nuisance_steps=[45])
    m = event_metrics(ev, make_episodes(*EP))
    assert m["false_incidents"] == 1 and m["false_incidents_on_nuisance"] == 1


def test_always_alarm_cannot_hide_false_alarms():
    """회귀: 알람이 계속 켜져 있으면 에피소드와 겹치는 하나의 인시던트가 되어 오탐이 0으로 보이던 맹점"""
    ev = make_ev(200, alarm_steps=range(200), ep_range=EP)
    m = event_metrics(ev, make_episodes(*EP))

    assert m["episode_detection_rate"] == 1.0
    assert m["false_incidents"] >= 1                       # 활성 구간 밖 알람이 오탐으로 잡혀야 함
    assert m["false_alarm_step_ratio"] > 0.9               # 정상 구간의 대부분이 알람 상태


def test_by_scenario_breakdown():
    m = event_metrics(make_ev(60, [15], ep_range=EP), make_episodes(*EP))
    assert m["by_scenario"]["crc_error"]["early"] == 1.0


# ──────────────────────────────────────────────
# 스텝 지표 / 인시던트 묶기
# ──────────────────────────────────────────────

def test_step_metrics_hand_calculation():
    """장애 16스텝(10~25) 중 알람 8스텝 적중, 정상 구간 알람 2스텝"""
    ev = make_ev(60, alarm_steps=list(range(12, 20)) + [40, 41], ep_range=EP)
    s = step_metrics(ev)

    assert s["precision"] == pytest.approx(8 / 10)
    assert s["recall"] == pytest.approx(8 / 16)
    assert s["prevalence"] == pytest.approx(16 / 60)
    assert s["false_positive_rate"] == pytest.approx(2 / 44)
    assert s["recall_plateau"] == 0.0 and s["recall_ramp"] == pytest.approx(8 / 10)


def test_make_incidents_gap_rule():
    ev = make_ev(60, alarm_steps=[10, 11, 13, 30], ep_range=EP)   # 11->13 간격 2스텝(<=4)이므로 한 묶음
    inc, _ = make_incidents(ev, gap_steps=4)
    assert list(inc["n_alarms"]) == [3, 1]


# ──────────────────────────────────────────────
# 시나리오 생성기
# ──────────────────────────────────────────────

SMALL = dict(nodes=1, ports_per_node=4, days=10)


def test_generator_is_reproducible_by_seed():
    a, b = generate(ScenarioConfig(seed=3, **SMALL)), generate(ScenarioConfig(seed=3, **SMALL))
    c = generate(ScenarioConfig(seed=4, **SMALL))
    assert all(a[k].equals(b[k]) for k in a)
    assert not a["traffic"].equals(c["traffic"])


def test_generator_labels_match_episodes():
    d = generate(ScenarioConfig(seed=5, mean_gap_days=2.0, **SMALL))
    labels, eps = d["labels"], d["episodes"]
    assert len(eps) > 0
    for r in eps.itertuples():
        seg = labels[(labels.ip_addr == r.ip_addr) & (labels.cid == r.cid) & (labels.lid == r.lid)
                     & (labels.occur_date >= r.t_start) & (labels.occur_date <= r.t_end)]
        assert (seg["state"] > 0).all() and (seg["episode_id"] == r.episode_id).all()
        assert (seg[seg.occur_date < r.t_fail]["state"] == 1).all()
        assert (seg[seg.occur_date >= r.t_fail]["state"] == 2).all()


def test_generator_episodes_do_not_overlap_per_port():
    eps = generate(ScenarioConfig(seed=6, mean_gap_days=1.5, **SMALL))["episodes"]
    for _, g in eps.sort_values("t_start").groupby(["ip_addr", "cid", "lid"]):
        assert (g["t_start"].values[1:] > g["t_end"].values[:-1]).all()


def test_generator_fault_signal_present_in_data():
    """optical 에피소드의 지속 구간에서 rx 전력이 포트 정상 중앙값보다 낮아야 함"""
    d = generate(ScenarioConfig(seed=8, mean_gap_days=2.0, **SMALL))
    o, labels = d["optical"], d["labels"]
    ev = o.merge(labels, on=["occur_date", "ip_addr", "cid", "lid"])
    ev = ev.merge(d["episodes"][["episode_id", "magnitude"]], on="episode_id", how="left")
    base = ev[ev.state == 0].groupby(["ip_addr", "cid", "lid"])["rx_avg_power"].median().rename("base")
    ev = ev.join(base, on=["ip_addr", "cid", "lid"])
    opt = ev[(ev.scenario == "optical_degradation") & (ev.state == 2)]
    assert len(opt) > 0
    assert (opt["base"] - opt["rx_avg_power"] > 2.0).mean() > 0.9


def test_generator_prevalence_is_low():
    d = generate(ScenarioConfig(seed=7, nodes=3, days=14))
    assert 0.01 < (d["labels"]["state"] > 0).mean() < 0.12     # 기존 시뮬레이터는 ~65%였음


# ──────────────────────────────────────────────
# 베이스라인
# ──────────────────────────────────────────────

def _raw(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "occur_date": [t(i) for i in range(n)], **{k: [v] * n for k, v in KEY.items()},
        "tx_packet": rng.integers(90000, 110000, n), "rx_packet": rng.integers(90000, 110000, n),
        "error_packet": 0, "tx_avg_power": -3.0, "rx_avg_power": -6.0 + rng.normal(0, 0.1, n),
    })


def test_always_alarm_is_all_true():
    assert always_alarm(_raw())["alarm"].all()


def test_fixed_threshold_flags_large_errors_and_deep_rx_drop_only():
    raw = _raw()
    raw.loc[50, "error_packet"] = 500
    raw.loc[60, "rx_avg_power"] = -20.0
    raw.loc[70, "error_packet"] = 3           # 정상 범위의 산발 에러는 무시
    a = fixed_threshold(raw)["alarm"]
    assert a[50] and a[60] and not a[70] and a.sum() == 2


def test_rolling_rule_detects_relative_rx_drop_and_ignores_stable_port():
    raw = _raw()
    assert not rolling_rule(raw)["alarm"].any()            # 안정적인 포트: 알람 없음
    raw.loc[150:, "rx_avg_power"] -= 5.0                   # 5 dB 하락 (고정 임계 -15 에는 안 걸리는 크기)
    r = rolling_rule(raw)["alarm"]
    assert r[150] and not fixed_threshold(raw)["alarm"][150]
