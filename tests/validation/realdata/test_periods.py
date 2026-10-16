"""U5/U6: 데이터 일 기반 기간 분할과 판정 구간 봉인 (설계서 9.7, 픽스처 F-9)"""
import json
import os

import pandas as pd
import pytest

from validation.realdata import periods as pr

D0 = pd.Timestamp("2026-07-10")


def days(n, skip=()):
    out, i = [], 0
    while len(out) < n:
        d = D0 + pd.Timedelta(days=i)
        if i not in skip:
            out.append(d)
        i += 1
    return out


def test_57_data_days_gives_j0_after_28th_data_day_with_gaps():
    ds = days(57, skip={3, 4, 10, 11})                      # 중간에 데이터 없는 날(장비 꺼짐)이 있어도 데이터 일로 센다
    p = pr.compute_periods(ds)
    assert p["ok"] and p["judge_data_days"] == 29 and p["data_days"] == 57
    assert p["J0"] == ds[27] + pd.Timedelta(days=1) and p["D"] == ds[-1] + pd.Timedelta(days=1) and p["S"] == ds[0]
    assert p["alternative"] == "fixed_split_v1"


@pytest.mark.parametrize("n,ok", [(42, False), (43, True), (28, False), (0, False)])
def test_judge_needs_at_least_15_data_days(n, ok):
    p = pr.compute_periods(days(n))
    assert p["ok"] is ok
    if not ok:
        assert "판정 실행 불가" in p["reason"] or "데이터 일 0" in p["reason"]
    else:
        assert p["judge_data_days"] == 15


def test_data_days_from_eth_uses_bin_end_date():
    eth = pd.DataFrame({"bin_end": ["2026-07-10 00:15:00", "2026-07-10 23:45:00", "2026-07-11 00:00:00", "2026-07-13 12:00:00"]})
    assert [d.strftime("%m-%d") for d in pr.data_days_from_eth(eth)] == ["07-10", "07-11", "07-13"]
    assert pr.data_days_from_eth(pd.DataFrame()) == []


def test_range_bounds_half_open():
    p = pr.compute_periods(days(57))
    s, e = pr.range_bounds(p, "dev")
    s2, e2 = pr.range_bounds(p, "judge")
    assert (s, e) == (p["S"], p["J0"]) and (s2, e2) == (p["J0"], p["D"]) and e == s2
    with pytest.raises(pr.PeriodError):
        pr.range_bounds(p, "all")
    with pytest.raises(pr.PeriodError):
        pr.range_bounds(pr.compute_periods(days(10)), "dev")


def test_unsealed_guard_rejects_judge_overlap_without_lock(tmp_path):
    p = pr.compute_periods(days(57))
    runs = str(tmp_path)
    pr.resolve_range("dev", p, runs, "alarms")                                       # 개발 기간은 항상 허용
    with pytest.raises(pr.PeriodError, match="잠금"):
        pr.resolve_range("judge", p, runs, "alarms")
    j0 = p["J0"]
    pr.resolve_range(f"{(j0 - pd.Timedelta(days=5)):%Y-%m-%d}..{j0:%Y-%m-%d}", p, runs, "alarms")        # 끝이 J0 와 같으면(반열림) 겹치지 않음
    with pytest.raises(pr.PeriodError):
        pr.resolve_range(f"{(j0 - pd.Timedelta(days=5)):%Y-%m-%d} 00:00..{(j0 + pd.Timedelta(hours=1)):%Y-%m-%d %H:%M}", p, runs, "alarms")
    with pytest.raises(pr.PeriodError):                                              # 개발 기간 전체 + 판정 일부 (all 같은 우회)
        pr.resolve_range(f"{p['S']:%Y-%m-%d}..{p['D']:%Y-%m-%d}", p, runs, "count-only")
    with pytest.raises(pr.PeriodError, match="형식"):
        pr.resolve_range("last-week", p, runs, "alarms")


def test_lock_file_opens_judge_range(tmp_path):
    p = pr.compute_periods(days(57))
    lp = pr.lock_path(str(tmp_path), p)
    assert lp.endswith(f"locks/judge_20260710_{p['D']:%Y%m%d}.json")
    os.makedirs(os.path.dirname(lp))
    json.dump({"alternative": "fixed_split_v1"}, open(lp, "w"))
    s, e = pr.resolve_range("judge", p, str(tmp_path), "alarms")
    assert (s, e) == (p["J0"], p["D"]) and pr.load_lock(str(tmp_path), p)["alternative"] == "fixed_split_v1"


def test_provisional_j0_from_28_data_days_seals_but_cannot_lock(tmp_path):
    p30 = pr.compute_periods(days(30))
    assert not p30["ok"] and p30["provisional"] and p30["J0"] == days(28)[-1] + pd.Timedelta(days=1)
    with pytest.raises(pr.PeriodError, match="잠금"):                                           # ok=False 여도 [J0, D) 분석은 거부 (F-14)
        pr.resolve_range(f"{p30['J0']:%Y-%m-%d}..{p30['D']:%Y-%m-%d}", p30, str(tmp_path), "alarms")
    with pytest.raises(pr.PeriodError, match="판정 실행 불가|판정 기간"):
        pr.range_bounds(p30, "judge")
    assert pr.range_bounds(p30, "dev") == (p30["S"], p30["J0"])
    p45 = pr.compute_periods(days(45))
    assert p45["ok"] and p45["J0"] == p30["J0"]                                                    # 데이터가 뒤로 늘어도 28번째 데이터일은 같다


def test_under_28_data_days_is_explore_only(tmp_path):
    p = pr.compute_periods(days(20))
    assert p["explore_only"] and not p["ok"] and "J0" not in p
    for spec in ("dev", "judge"):
        with pytest.raises(pr.PeriodError, match="explore"):
            pr.resolve_range(spec, p, str(tmp_path), "alarms")
    s, e = pr.resolve_range("2026-07-12..2026-07-20", p, str(tmp_path), "alarms")
    assert pr.analysis_purpose(p, s, e) == "explore"
    assert pr.analysis_purpose(pr.compute_periods(days(57)), D0, D0 + pd.Timedelta(days=3)) == "dev"


def test_ledger_is_append_only_and_conflicts_cover_other_snapshots_and_judge_runs(tmp_path):
    runs = str(tmp_path)
    p = pr.compute_periods(days(57))
    assert pr.ledger_conflicts(runs, p) is None                                                     # 장부 없음
    pr.append_ledger(runs, "alarms --summary", "dev", D0, p["J0"], p, "r1")
    assert pr.ledger_conflicts(runs, p) == []                                                       # 개발 범위만 → 충돌 없음
    other = pr.compute_periods(days(30))                                                            # 다른 스냅샷(S 같고 D 짧음)의 explore 성격 기록
    pr.append_ledger(runs, "replay", "dev", p["J0"] - pd.Timedelta(days=1), p["J0"] + pd.Timedelta(days=1), other, "r0")
    pr.append_ledger(runs, "report", "judge", p["J0"], p["D"], p, "r1")                             # 이전 판정 실행
    conf = pr.ledger_conflicts(runs, p)
    assert [c["run"] for c in conf] == ["r0", "r1"] and len(pr.read_ledger(runs)) == 3
    # 경계: 끝이 J0 와 같으면(반열림) 충돌 아님
    pr.append_ledger(runs, "x", "dev", D0, p["J0"], p, "r2")
    assert len(pr.ledger_conflicts(runs, p)) == 2


# ---- 9.7.5 판정 모델 학습 구간 검사 / 잠금 / 1회 표식 -------------------------------------------------
J0 = pd.Timestamp("2026-08-22")


@pytest.mark.parametrize("meta,ok", [
    ({"trained_at": "2026-08-22 00:00:00"}, True),                                                   # (a) 정확히 J0 → 통과
    ({"trained_at": "2026-08-22 00:00:01"}, False),                                                  # (a) 초과 + 학습 구간 없음 → 거부
    ({"trained_at": "2026-09-18 10:07:08", "training_window": {"mode": "port_split", "start": "2026-07-10", "end": "2026-08-22"}}, True),     # (b)
    ({"trained_at": "2026-09-18", "training_window": {"mode": "port_split", "start": "2026-07-10", "end": "2026-08-22 00:00:01"}}, False),    # (c) > J0
    ({"trained_at": "2026-09-18", "training_window": {"mode": "date", "train_end": "2026-08-01", "test_end": "2026-08-22"}}, True),          # 날짜 모드
    ({"trained_at": "2026-09-18", "training_window": {"mode": "date", "train_end": "2026-08-01", "test_end": "2026-09-01"}}, False),
    ({"trained_at": "2026-09-18", "training_window": {"mode": "date"}}, False),                                                              # 끝 시각 없음
    ({"trained_at": "2026-09-18", "training_window": {"end": "garbage"}}, False),                                                           # 파싱 불가
    ({"trained_at": "garbage"}, False), ({}, False), (None, False)])
def test_training_window_check_fail_closed(meta, ok):
    assert pr.check_training_window(meta, J0)[0] is ok


def test_stable_hash_is_order_independent_and_changes_with_content():
    assert pr.stable_hash({"a": 1, "b": [1, 2]}) == pr.stable_hash({"b": [1, 2], "a": 1})
    assert pr.stable_hash({"a": 1}) != pr.stable_hash({"a": 2})


def test_lock_is_exclusive_and_run_marker_is_one_shot(tmp_path):
    p = pr.compute_periods(days(57))
    path = pr.create_lock(str(tmp_path), p, {"x": 1})
    assert json.load(open(path))["x"] == 1
    with pytest.raises(pr.PeriodError, match="이미 잠긴"):
        pr.create_lock(str(tmp_path), p, {"x": 2})
    assert json.load(open(path))["x"] == 1
    pr.claim_judge_run(str(tmp_path), p, {"a": 1})
    with pytest.raises(pr.PeriodError, match="1회"):
        pr.claim_judge_run(str(tmp_path), p, {"a": 2})
