"""U1: 테이블 분류·범위·프로필·격리 판정·결과 테이블 충돌·plan (설계서 4.2~4.4, 4.9)"""
from datetime import datetime, timedelta

import pytest

from tests.validation.realdata.fakes import (ETH_COLS, ALARM_COLS, SIM_ETH_COLS, FakeServer, alarm_row, eth_row, make_db)
from validation.realdata.tables import (
    ALARM, ANOMALY_TABLE_COLUMNS, ETH, OPTIC, ProfileError, build_plan, build_where, check_result_table,
    classify_table, format_plan, is_trusted, list_tables, missing_tables, needed_tables, resolve_filters, tz_warning)

D = datetime


@pytest.mark.parametrize("name,kind,start,daily", [
    ("cowptn_noti_pm_2026_10_01_09", ETH, D(2026, 10, 1, 9), False),
    ("cowptn_noti_pm_optic_power_2026_10_01_09", OPTIC, D(2026, 10, 1, 9), False),
    ("cowptn_noti_alarm_2026_10_01", ALARM, D(2026, 10, 1), True),
    ("cowptn_noti_alarm_2026_10_01_23", ALARM, D(2026, 10, 1, 23), False),
])
def test_classify_ok(name, kind, start, daily):
    t = classify_table(name, 5)
    assert (t.kind, t.start, t.daily, t.rows_est) == (kind, start, daily, 5)


@pytest.mark.parametrize("name", [
    "cowptn_noti_pm_2026_10_01", "cowptn_noti_pm_2026_10_01_09_extra", "cowptn_noti_pm_optic_2026_10_01_09",
    "cowptn_noti_pm_2026_13_01_09", "cowptn_noti_pm_2026_10_01_09`; DROP TABLE x", "anomaly_detection", None,
    "cowptn_noti_pm_2026_10_01_25",
])
def test_classify_rejects(name):
    assert classify_table(name) is None


def test_optic_name_is_not_classified_as_eth():
    assert classify_table("cowptn_noti_pm_optic_power_2026_10_01_09").kind == OPTIC


def _tabs(kind_prefix, hours):
    return [classify_table(f"{kind_prefix}{h.strftime('%Y_%m_%d_%H')}") for h in hours]


def test_needed_tables_window_perf_and_alarm():
    base = D(2026, 10, 5)
    eth = _tabs("cowptn_noti_pm_", [base + timedelta(hours=i) for i in range(-3, 80)])
    sel = needed_tables(eth, ETH, D(2026, 10, 5, 0, 0), D(2026, 10, 5, 2, 30))   # floor(end−1s)=02:00, +1h 포함
    assert [t.start.hour for t in sel] == [23, 0, 1, 2, 3] and sel[0].start.day == 4
    alarm_h = _tabs("cowptn_noti_alarm_", [base + timedelta(hours=i) for i in range(-80, 10)])
    sel = needed_tables(alarm_h, ALARM, D(2026, 10, 5), D(2026, 10, 5, 1))
    assert sel[0].start == D(2026, 10, 2) and sel[-1].start == D(2026, 10, 5, 1)       # lookback 72h


def test_needed_tables_daily_alarm_overlap():
    days = [classify_table(f"cowptn_noti_alarm_2026_10_{d:02d}") for d in range(1, 10)]
    sel = needed_tables(days, ALARM, D(2026, 10, 6, 5), D(2026, 10, 6, 9))
    assert [t.start.day for t in sel] == [3, 4, 5, 6]          # 72h 전(10-03 05:00)부터 겹치는 일 테이블


def test_missing_tables():
    present = _tabs("cowptn_noti_pm_", [D(2026, 10, 1, 0), D(2026, 10, 1, 2)])
    assert missing_tables(present, ETH, D(2026, 10, 1, 0), D(2026, 10, 1, 3)) == [D(2026, 10, 1, 1)]
    daily = [classify_table("cowptn_noti_alarm_2026_10_01"), classify_table("cowptn_noti_alarm_2026_10_03")]
    assert missing_tables(daily, ALARM, D(2026, 10, 1), D(2026, 10, 4)) == [D(2026, 10, 2)]


def test_list_tables_filters_by_regex_not_like():
    srv = FakeServer(tables={
        "cowptn_noti_pm_2026_10_01_00": {"cols": ["id"], "rows": []},
        "cowptn_noti_pm_optic_power_2026_10_01_00": {"cols": ["id"], "rows": []},
        "cowptn_noti_pm_junk": {"cols": ["id"], "rows": []},
    })
    t = list_tables(make_db(srv))
    assert [x.name for x in t[ETH]] == ["cowptn_noti_pm_2026_10_01_00"]
    assert [x.name for x in t[OPTIC]] == ["cowptn_noti_pm_optic_power_2026_10_01_00"]


# ---- 4.3a 격리 판정 --------------------------------------------------------------
def test_is_trusted_hourly_boundaries():
    t = classify_table("cowptn_noti_pm_2026_10_01_10")      # 덮는 구간 [10:00, 11:00] → 정상 [08:30, 12:30]
    assert is_trusted(t, D(2026, 10, 1, 8, 30)) and is_trusted(t, D(2026, 10, 1, 12, 30))
    assert not is_trusted(t, D(2026, 10, 1, 8, 29, 59)) and not is_trusted(t, D(2026, 10, 1, 12, 30, 1))
    assert not is_trusted(t, D(2012, 1, 1)) and not is_trusted(t, D(2027, 10, 1, 10)) and not is_trusted(t, None)


def test_is_trusted_daily_and_release():
    t = classify_table("cowptn_noti_alarm_2026_10_01")      # [10-01 00:00, 10-02 00:00] ± 1.5h
    assert is_trusted(t, D(2026, 9, 30, 22, 30)) and not is_trusted(t, D(2026, 9, 30, 22, 29))
    assert is_trusted(t, D(2026, 10, 2, 1, 30)) and not is_trusted(t, D(2026, 10, 2, 1, 31))
    assert not is_trusted(t, D(2026, 10, 1, 5), release_date=D(2026, 10, 1, 4), is_alarm=True)
    assert is_trusted(t, D(2026, 10, 1, 5), release_date=None, is_alarm=True)
    assert is_trusted(t, D(2026, 10, 1, 5), release_date=D(2026, 10, 1, 5), is_alarm=True)


# ---- 4.4 프로필 ------------------------------------------------------------------
def test_resolve_filters_real_requires_columns():
    assert resolve_filters(ETH, set(ETH_COLS), "real") == ([("signal_type", 6), ("time_type", 1)], [])
    with pytest.raises(ProfileError, match="time_type"):
        resolve_filters(ETH, set(SIM_ETH_COLS), "real")
    assert resolve_filters(OPTIC, {"signal_type", "time_type"}, "real")[0] == [("signal_type", 44), ("time_type", 1)]
    assert resolve_filters(ALARM, set(ALARM_COLS), "real") == ([], [])


def test_resolve_filters_sim_omits_missing_and_records():
    assert resolve_filters(ETH, set(SIM_ETH_COLS), "sim") == ([("signal_type", 6)], ["time_type"])
    assert resolve_filters(OPTIC, {"id"}, "sim") == ([], ["signal_type", "time_type"])


def test_build_where_binds_values_only():
    w, p = build_where([("signal_type", 6), ("time_type", 1)], ["a", "b"])
    assert w == "`signal_type` = %s AND `time_type` = %s AND `ip_addr` IN (%s, %s)" and p == [6, 1, "a", "b"]
    assert build_where([], None) == ("1=1", [])


# ---- 4.9 ------------------------------------------------------------------------
def test_result_table_check_three_states():
    assert check_result_table(make_db(FakeServer()))["status"] == "none"
    ok = FakeServer(tables={"anomaly_detection": {"cols": sorted(ANOMALY_TABLE_COLUMNS) + ["extra"], "rows": []}})
    assert check_result_table(make_db(ok))["status"] == "operational"
    bad = FakeServer(tables={"anomaly_detection": {"cols": ["id", "foo"], "rows": []}})
    r = check_result_table(make_db(bad))
    assert r["status"] == "conflict_suspected" and "운영 서비스를 이 DB 로 기동하지 말 것" in r["warning"]


def test_tz_warning():
    now = D(2026, 10, 16, 12)
    assert tz_warning(D(2026, 10, 16, 3), now) and tz_warning(D(2026, 10, 16, 2), now)
    assert tz_warning(D(2026, 10, 16, 11), now) is None and tz_warning(None, now) is None


# ---- plan ------------------------------------------------------------------------
def _plan_server(tz_shift_hours=0):
    base = D(2026, 10, 1, 10) - timedelta(hours=tz_shift_hours)
    rows = [eth_row(1, base + timedelta(minutes=15)), eth_row(2, base + timedelta(minutes=30), sig=7)]
    tabs = {
        "cowptn_noti_pm_2026_10_01_10": {"cols": ETH_COLS, "rows": rows},
        "cowptn_noti_pm_2026_10_01_12": {"cols": ETH_COLS, "rows": [eth_row(3, D(2026, 10, 1, 12, 15))]},
        "cowptn_noti_pm_optic_power_2026_10_01_10": {"cols": ["id", "signal_type", "time_type", "occur_date"], "rows": []},
        "cowptn_noti_alarm_2026_10_01": {"cols": ALARM_COLS, "rows": [alarm_row(1, D(2026, 10, 1, 9))]},
    }
    return FakeServer(tables=tabs)


def test_plan_reports_tables_missing_distribution_and_does_not_extract_rows():
    srv = _plan_server()
    plan = build_plan(make_db(srv), D(2026, 10, 1, 10), D(2026, 10, 1, 13), now=D(2026, 10, 1, 13))
    eth = plan["kinds"][ETH]
    assert eth["tables_total"] == 2 and eth["missing"] == ["2026-10-01 11:00:00"]
    groups = {tuple(sorted(g.items())) for g in eth["signal_time_type"][0]["groups"]}
    assert (("n", 1), ("signal_type", 7), ("time_type", 1)) in groups               # 섞인 signal_type 이 수치로 드러남
    assert plan["kinds"][ALARM]["name_format"] == "daily"
    assert plan["result_table"]["status"] == "none"
    assert plan["tz_warning"] is None
    for q in srv.statements():                                                     # 행 추출 쿼리 없음 (집계만)
        assert "`id` >" not in q and "LIMIT" not in q


def test_plan_tz_warning_and_format():
    srv = _plan_server(tz_shift_hours=0)
    plan = build_plan(make_db(srv), D(2026, 10, 1, 10), D(2026, 10, 1, 13), now=D(2026, 10, 1, 21, 30))
    assert plan["tz_warning"]
    assert "시간대 의심" in format_plan(plan)


def test_plan_profile_error_surfaces_for_real_but_not_sim():
    tabs = {"cowptn_noti_pm_2026_10_01_10": {"cols": SIM_ETH_COLS, "rows": []}}
    real = build_plan(make_db(FakeServer(tables=tabs)), D(2026, 10, 1, 10), D(2026, 10, 1, 11), kinds=(ETH,), now=D(2026, 10, 1, 11))
    assert "time_type" in real["kinds"][ETH]["profile_error"]
    sim = build_plan(make_db(FakeServer(tables=tabs)), D(2026, 10, 1, 10), D(2026, 10, 1, 11), kinds=(ETH,), schema="sim", now=D(2026, 10, 1, 11))
    assert sim["kinds"][ETH]["profile_error"] is None and sim["kinds"][ETH]["filters_omitted"] == ["time_type"]


def test_plan_warns_on_conflicting_result_table_and_mixed_alarm_names():
    srv = _plan_server()
    srv.tables["anomaly_detection"] = {"cols": ["id", "x"], "rows": []}
    srv.tables["cowptn_noti_alarm_2026_10_01_05"] = {"cols": ALARM_COLS, "rows": []}
    plan = build_plan(make_db(srv), D(2026, 10, 1, 10), D(2026, 10, 1, 13), now=D(2026, 10, 1, 13))
    assert plan["kinds"][ALARM]["name_format"] == "mixed"
    assert any("이름 충돌 의심" in w for w in plan["warnings"]) and any("섞여" in w for w in plan["warnings"])
