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


def test_guard_inactive_when_periods_unavailable(tmp_path):
    p = pr.compute_periods(days(30))                                                 # 판정 구간 자체가 없음 → 봉인 대상 없음
    pr.require_unsealed(D0, D0 + pd.Timedelta(days=100), p, str(tmp_path), "x")
