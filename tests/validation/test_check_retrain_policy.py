"""validation/cli/check_retrain_policy.py 핵심 함수 테스트 (P1-1 D)"""
import numpy as np
import pandas as pd
import pytest

from src.data import train_window
from src.pipeline.retrain_policy import RetrainPolicy
from validation.cli import check_retrain_policy as crp


@pytest.fixture(scope="module")
def data():
    return crp.make_data(seed=7, nodes=2, days=10)


def test_make_data_is_deterministic(data):
    again = crp.make_data(seed=7, nodes=2, days=10)
    pd.testing.assert_frame_equal(data["labels"], again["labels"])


def test_explicit_intervals_cover_all_fault_steps(data):
    pol = RetrainPolicy()
    iv = crp.explicit_intervals(data, pol)
    c = crp.coverage(data, {"traffic": iv, "optical": iv}, pol)
    assert c["fault_steps"] > 0
    assert c["fault_excl"] == pytest.approx(1.0)
    assert (iv["source"] == "explicit").all()


def test_no_intervals_excludes_nothing(data):
    pol = RetrainPolicy()
    empty = train_window._empty_intervals()
    c = crp.coverage(data, {"traffic": empty, "optical": empty}, pol)
    assert c["fault_excl"] == 0.0 and c["total_excl"] == 0.0
    assert not crp.passes(c)


def test_rule_intervals_are_per_track_and_catch_crc_and_optical(data):
    pol = RetrainPolicy()
    iv = crp.rule_intervals(data, pol)
    assert set(iv) == {"traffic", "optical"}
    c = crp.coverage(data, iv, pol)
    assert c["fault_excl_optical"] >= 0.9           # 광 열화는 규칙으로 잡힘
    assert c["nuisance_excl"] <= crp.MAX_NUISANCE    # 정상 교란은 남겨야 함


def test_passes_thresholds():
    ok = {"fault_excl": 0.95, "nuisance_excl": 0.05, "total_excl": 0.10}
    assert crp.passes(ok)
    assert not crp.passes({**ok, "fault_excl": 0.89})
    assert not crp.passes({**ok, "nuisance_excl": 0.11})
    assert not crp.passes({**ok, "total_excl": 0.20})


def test_select_rule_params_prefers_passing_and_reports_failure(data):
    base = RetrainPolicy()
    grid = {"rule_error_ge": [20.0], "rule_rx_drop_db": [3.0], "rule_traffic_ratio": [0.15], "rule_min_consecutive": [4]}
    params, ok, results = crp.select_rule_params([data], base, grid)
    assert params == {k: v[0] for k, v in grid.items()} and len(results) == 1
    assert ok == results[0]["all_pass"]
    # 불가능한 기준이 아니라 후보가 여러 개일 때 하나를 고르고 결과 수가 격자 크기와 같다
    grid2 = {**grid, "rule_min_consecutive": [3, 6]}
    p2, _, r2 = crp.select_rule_params([data], base, grid2)
    assert len(r2) == 2 and p2["rule_min_consecutive"] in (3, 6)


def test_evaluate_methods_without_alarms_only_reports_b_and_explicit(data):
    res = crp.evaluate_methods(data, RetrainPolicy())
    assert set(res) == {"B 규칙 단독", "C 명시 구간(정답)"}


def test_evaluate_methods_with_alarms_adds_a_and_union(data):
    L = data["labels"]
    alarms = L[L.state == 2][["occur_date", "ip_addr", "cid", "lid"]]
    res = crp.evaluate_methods(data, RetrainPolicy(), alarms)
    assert {"A 알람 단독", "A∪B"} <= set(res)
    assert res["A∪B"]["fault_excl"] >= res["B 규칙 단독"]["fault_excl"]
