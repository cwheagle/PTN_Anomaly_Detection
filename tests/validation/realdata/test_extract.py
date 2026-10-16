"""U2: extract — 키셋 청크·상한·격리·매핑·스냅샷·증분 (설계서 4.3a~4.7, 5장)"""
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from tests.validation.realdata.fakes import (ALARM_COLS, ETH_COLS, SIM_ETH_COLS, FakeServer, alarm_row, eth_row, make_db)
from validation.realdata.extract import ExtractError, extract, load_ports_csv, load_snapshot
from validation.realdata.tables import ALARM, ETH, OPTIC, ProfileError

D = datetime
H10 = "cowptn_noti_pm_2026_10_01_10"
OPT10 = "cowptn_noti_pm_optic_power_2026_10_01_10"


def _eth(rows, name=H10, cols=ETH_COLS):
    return {name: {"cols": cols, "rows": rows}}


def _run(srv, tmp_path, start=D(2026, 10, 1, 10), end=D(2026, 10, 1, 11), kinds=(ETH,), **kw):
    return extract(make_db(srv), str(tmp_path / "run"), start, end, kinds, **kw)


def test_keyset_chunks_without_offset_and_without_occur_date_in_sql(tmp_path):
    rows = [eth_row(i, D(2026, 10, 1, 10, 0) + timedelta(minutes=15 * (i % 4)), lid=i) for i in range(1, 6)]
    srv = FakeServer(tables=_eth(rows))
    m = _run(srv, tmp_path, chunk=2)
    sel = [(q, p) for q, p in srv.log if q.startswith("SELECT") and f"FROM `{H10}`" in q and "COUNT" not in q]
    assert [p[-2] for _, p in sel] == [0, 2, 4]                                  # id > 0, 2, 4 로 키셋 전진
    assert all("OFFSET" not in q.upper() and "occur_date" not in q.split(" WHERE ")[1] for q, _ in sel)   # WHERE 에 occur_date 없음
    assert all("`signal_type` = %s AND `time_type` = %s" in q for q, _ in sel)
    assert m["complete"] and m["stats"][ETH]["fetched"] == 5 and m["total_rows"] == 5
    assert m["tz"] == "Asia/Seoul (naive)" and m["database"] == "ems"


def test_column_mapping_matches_operational_fetch_traffic(tmp_path):
    rows = [eth_row(1, D(2026, 10, 1, 10, 15), tx=11, rx=22, err=3)]
    _run(FakeServer(tables=_eth(rows)), tmp_path)
    df = load_snapshot(str(tmp_path / "run"), ETH)
    assert list(df[["tx_packet", "rx_packet", "error_packet"]].iloc[0]) == [11, 22, 3]      # es/ses/bbe_in_error
    assert "es" not in df.columns and df["ip_addr"].iloc[0] == "10.0.0.1"
    assert df["address"].iloc[0] == "A-S.1 -P1"                                              # 공백 제거


def test_quarantine_boundaries_and_garbage_years(tmp_path):
    rows = [
        eth_row(1, D(2026, 10, 1, 10, 15)),
        eth_row(2, D(2026, 10, 1, 8, 30), lid=2),          # 경계 안 (H−1.5h) — 단 기간 필터 [10:00, 11:00) 밖 → out_of_range
        eth_row(3, D(2026, 10, 1, 8, 29, 59), lid=3),      # 격리
        eth_row(4, D(2026, 10, 1, 12, 30, 1), lid=4),      # 격리
        eth_row(5, D(2012, 5, 5, 10), lid=5), eth_row(6, D(2027, 1, 1, 10), lid=6), eth_row(7, None, lid=7),
    ]
    m = _run(FakeServer(tables=_eth(rows)), tmp_path)
    st = m["stats"][ETH]
    assert (st["fetched"], st["kept"], st["quarantined"], st["out_of_range"]) == (7, 1, 5, 1)
    snap = load_snapshot(str(tmp_path / "run"), ETH)
    assert list(snap["lid"]) == [1]                                                  # 격리 행은 분석 프레임에 없음
    q = pd.read_csv(tmp_path / "run" / "quarantine" / "eth_round1.csv.gz", keep_default_na=False)
    assert sorted(q["lid"]) == [3, 4, 5, 6, 7] and set(q["quarantine_reason"]) == {"range", "null"}
    assert st["quarantine_year_distribution"] == {"2012": 1, "2027": 1, "2026": 2}
    assert len(st["quarantine_samples"]) == 5


def test_quarantine_heavy_device_and_ratios(tmp_path):
    rows = [eth_row(1, D(2026, 10, 1, 10, 15), ip="1.1.1.1"),
            eth_row(2, D(2012, 1, 1), ip="2.2.2.2", lid=2), eth_row(3, D(2012, 1, 1), ip="2.2.2.2", lid=3),
            eth_row(4, D(2026, 10, 1, 10, 30), ip="2.2.2.2", lid=4)]
    st = _run(FakeServer(tables=_eth(rows)), tmp_path)["stats"][ETH]
    assert st["quarantine_heavy"] == ["2.2.2.2"] and st["quarantine_by_device"]["2.2.2.2"] == [2, 3]
    assert st["quarantine_by_system_type"]["31"] == [2, 4]


def test_alarm_quarantine_release_before_occur_and_daily_table(tmp_path):
    name = "cowptn_noti_alarm_2026_10_01"
    rows = [alarm_row(1, D(2026, 10, 1, 9), rel=D(2026, 10, 1, 9, 5)),
            alarm_row(2, D(2026, 10, 1, 9), rel=D(2026, 10, 1, 8), cid=2),            # release < occur → 격리
            alarm_row(3, D(2026, 10, 1, 9, 30), rel=None, cid=3),                    # 진행 중 → 유지
            alarm_row(4, D(2026, 10, 3, 9), rel=None, cid=4),                        # 일 테이블 ±1.5h 밖 → 격리
            alarm_row(5, D(2026, 9, 30, 22, 40), rel=D(2026, 10, 1, 9), cid=5)]       # 테이블 날짜 −1.5h 안, 기간 안
    srv = FakeServer(tables={name: {"cols": ALARM_COLS, "rows": rows}})
    m = _run(srv, tmp_path, start=D(2026, 10, 1, 8), end=D(2026, 10, 1, 12), kinds=(ALARM,))
    st = m["stats"][ALARM]
    assert (st["kept"], st["quarantined"]) == (3, 2) and st["lookback_boundary_alarms"] == 1
    snap = load_snapshot(str(tmp_path / "run"), ALARM)
    assert sorted(snap["cid"]) == [1, 3, 5] and snap["ip_addr"].eq("10.0.0.1").all()
    assert snap.loc[snap["cid"] == 3, "release_date"].isna().all()


def test_alarm_range_filter_excludes_released_before_start_and_after_end(tmp_path):
    name = "cowptn_noti_alarm_2026_10_01"
    rows = [alarm_row(1, D(2026, 10, 1, 1), rel=D(2026, 10, 1, 2)),                 # start 전에 해제 → 제외
            alarm_row(2, D(2026, 10, 1, 1), rel=D(2026, 10, 1, 9), cid=2),          # start 에 걸침 → 포함
            alarm_row(3, D(2026, 10, 1, 13), rel=None, cid=3)]                     # end 이후 발생 → 제외
    m = _run(FakeServer(tables={name: {"cols": ALARM_COLS, "rows": rows}}), tmp_path,
             start=D(2026, 10, 1, 8), end=D(2026, 10, 1, 12), kinds=(ALARM,))
    assert list(load_snapshot(str(tmp_path / "run"), ALARM)["cid"]) == [2] and m["stats"][ALARM]["out_of_range"] == 2


def test_rounding_shift_duplicates_over_limit_and_value_ranges(tmp_path):
    rows = [eth_row(1, D(2026, 10, 1, 10, 16), tx=5), eth_row(2, D(2026, 10, 1, 10, 15), tx=2e9, lid=2),
            eth_row(3, D(2026, 10, 1, 10, 29), tx=7), eth_row(4, D(2026, 10, 1, 10, 30), tx=8)]       # id 3,4 → 같은 bin 10:30
    st = _run(FakeServer(tables=_eth(rows)), tmp_path)["stats"][ETH]
    assert st["rounding_shifted"] == 2 and st["duplicates"] == 1 and st["over_1e9"] == 1
    assert st["value_ranges"]["tx_packet"] == [5.0, 2e9]


def test_filtered_out_rows_counted(tmp_path):
    rows = [eth_row(1, D(2026, 10, 1, 10, 15)), eth_row(2, D(2026, 10, 1, 10, 15), sig=7, lid=2),
            eth_row(3, D(2026, 10, 1, 10, 15), tt=2, lid=3)]
    m = _run(FakeServer(tables=_eth(rows)), tmp_path)
    assert m["stats"][ETH]["filtered_out"] == 2 and m["tables"][ETH][H10]["filtered_out"] == 2
    assert m["stats"][ETH]["fetched"] == 1


def test_profile_real_stops_and_sim_omits_filter_with_record(tmp_path):
    srv = FakeServer(tables=_eth([{"id": 1, "occur_date": D(2026, 10, 1, 10, 15), "ip_addr": "1.1.1.1", "cid": 1,
                                   "lid": 1, "signal_type": 6, "es": 1, "ses": 2, "bbe_in_error": 3}], cols=SIM_ETH_COLS))
    with pytest.raises(ProfileError):
        _run(srv, tmp_path)
    m = extract(make_db(srv), str(tmp_path / "sim"), D(2026, 10, 1, 10), D(2026, 10, 1, 11), (ETH,), schema="sim")
    assert m["filters_omitted"] == {ETH: ["time_type"]} and m["stats"][ETH]["kept"] == 1


def test_optic_zero_rows_is_skip_not_error(tmp_path):
    srv = FakeServer(tables={**_eth([eth_row(1, D(2026, 10, 1, 10, 15))]),
                             OPT10: {"cols": ["id", "ip_addr", "cid", "lid", "signal_type", "time_type", "occur_date",
                                              "rx_avg_power", "tx_avg_power"], "rows": []}})
    m = _run(srv, tmp_path, kinds=(ETH, OPTIC))
    assert m["optic"]["status"] == "SKIP" and m["complete"] and m["stats"][ETH]["kept"] == 1


def test_missing_hour_tables_recorded(tmp_path):
    tabs = _eth([eth_row(1, D(2026, 10, 1, 10, 15))])
    tabs["cowptn_noti_pm_2026_10_01_12"] = {"cols": ETH_COLS, "rows": []}
    m = _run(FakeServer(tables=tabs), tmp_path, end=D(2026, 10, 1, 13))
    assert m["missing_tables"][ETH] == ["2026-10-01 11:00:00"]


def test_max_rows_precheck_refuses_before_reading(tmp_path):
    tabs = _eth([eth_row(1, D(2026, 10, 1, 10, 15))])
    tabs[H10]["rows_est"] = 5_000
    srv = FakeServer(tables=tabs)
    with pytest.raises(ExtractError, match="--max-rows"):
        _run(srv, tmp_path, max_rows=1_000)
    assert not any("`id` >" in q for q in srv.statements())


def test_max_rows_abort_marks_incomplete(tmp_path):
    rows = [eth_row(i, D(2026, 10, 1, 10, 15), lid=i) for i in range(1, 8)]
    tabs = _eth(rows)
    tabs[H10]["rows_est"] = 0                                                     # 추정치는 낮고 실제는 많은 경우
    m = _run(FakeServer(tables=tabs), tmp_path, max_rows=4, chunk=2)
    assert not m["complete"] and m["abort_reason"] == "max_rows" and "range_end" not in m
    assert all(not d["complete"] for d in m["days"][ETH].values())
    assert json.load(open(tmp_path / "run" / "manifest.json", encoding="utf-8"))["complete"] is False


def test_stop_checker_aborts(tmp_path):
    m = _run(FakeServer(tables=_eth([eth_row(1, D(2026, 10, 1, 10, 15))])), tmp_path, stop_checker=lambda: True)
    assert not m["complete"] and m["abort_reason"] == "stopped"


def test_ports_filter_sql_and_python(tmp_path):
    rows = [eth_row(1, D(2026, 10, 1, 10, 15), ip="1.1.1.1", cid=1, lid=1), eth_row(2, D(2026, 10, 1, 10, 15), ip="1.1.1.1", cid=1, lid=2),
            eth_row(3, D(2026, 10, 1, 10, 15), ip="2.2.2.2", cid=1, lid=1), eth_row(4, D(2026, 10, 1, 10, 15), ip="3.3.3.3")]
    p = tmp_path / "ports.csv"
    p.write_text("ip_addr,cid,lid\n1.1.1.1,1,1\n2.2.2.2,,\n", encoding="utf-8")
    srv = FakeServer(tables=_eth(rows))
    m = _run(srv, tmp_path, ports=load_ports_csv(str(p)))
    snap = load_snapshot(str(tmp_path / "run"), ETH)
    assert sorted(zip(snap["ip_addr"], snap["lid"])) == [("1.1.1.1", 1), ("2.2.2.2", 1)]
    assert any("`ip_addr` IN (%s, %s)" in q for q in srv.statements())
    assert m["ports_filter"] == {"devices": 2, "rows": 2} and m["stats"][ETH]["filtered_out_unknown"]


def test_alarm_ports_keep_device_level_rows_of_selected_devices(tmp_path):
    name = "cowptn_noti_alarm_2026_10_01"
    rows = [alarm_row(1, D(2026, 10, 1, 9), rel=None, ip="1.1.1.1", cid=1, lid=1),
            alarm_row(2, D(2026, 10, 1, 9), rel=None, ip="1.1.1.1", cid=0, lid=0),         # 장비 단위: 선택 장비면 유지
            alarm_row(3, D(2026, 10, 1, 9), rel=None, ip="1.1.1.1", cid=2, lid=2),         # 다른 포트: 제외
            alarm_row(4, D(2026, 10, 1, 9), rel=None, ip="9.9.9.9", cid=0, lid=0)]         # 다른 장비: 제외
    ports = pd.DataFrame({"ip_addr": ["1.1.1.1"], "cid": [1.0], "lid": [1.0]})
    _run(FakeServer(tables={name: {"cols": ALARM_COLS, "rows": rows}}), tmp_path, start=D(2026, 10, 1, 8),
         end=D(2026, 10, 1, 12), kinds=(ALARM,), ports=ports)
    assert sorted(load_snapshot(str(tmp_path / "run"), ALARM)["id"]) == [1, 2]


def test_existing_run_dir_is_not_overwritten(tmp_path):
    srv = FakeServer(tables=_eth([eth_row(1, D(2026, 10, 1, 10, 15))]))
    _run(srv, tmp_path)
    with pytest.raises(ExtractError, match="덮어쓰지"):
        _run(srv, tmp_path)


# ---- 증분 추출 ---------------------------------------------------------------------
def _hour_tables(spec):
    return {f"cowptn_noti_pm_2026_10_{d:02d}_{h:02d}": {"cols": ETH_COLS, "rows": rows, "rows_est": 0}
            for (d, h), rows in spec.items()}


def test_append_merges_incomplete_boundary_day_and_accumulates_rounds(tmp_path):
    run = str(tmp_path / "run")
    srv = FakeServer(tables=_hour_tables({
        (1, 10): [eth_row(1, D(2026, 10, 1, 10, 15)), eth_row(2, D(2026, 10, 1, 10, 30), lid=2)],
        (1, 20): [eth_row(10, D(2026, 10, 1, 20, 15), lid=10)],
        (2, 9): [eth_row(11, D(2026, 10, 2, 9, 15), lid=11)]}))
    m1 = extract(make_db(srv), run, D(2026, 10, 1), D(2026, 10, 1, 12), (ETH,))
    assert m1["days"][ETH]["2026-10-01"]["complete"] is False and m1["total_rows"] == 2
    m2 = extract(make_db(srv), run, D(2026, 10, 1, 5), D(2026, 10, 2), (ETH,), append=True)   # start 인자는 무시, 기존 끝부터
    assert m2["rounds"][1]["range"] == ["2026-10-01 12:00:00", "2026-10-02 00:00:00"]
    assert m2["range_start"] == "2026-10-01 00:00:00" and m2["range_end"] == "2026-10-02 00:00:00"
    assert sorted(load_snapshot(run, ETH)["lid"]) == [1, 2, 10]                              # 불완전 파일에 병합
    assert m2["days"][ETH]["2026-10-01"]["rows"] == 3 and m2["days"][ETH]["2026-10-01"]["complete"] is True
    assert len(m2["rounds"]) == 2 and m2["rounds"][1]["new_data_days"] == 0 and m2["total_rows"] == 3
    m3 = extract(make_db(srv), run, D(2026, 10, 2), D(2026, 10, 3), (ETH,), append=True)
    assert sorted(load_snapshot(run, ETH)["lid"]) == [1, 2, 10, 11] and m3["rounds"][2]["new_data_days"] == 1
    assert [r["range"][0] for r in m3["rounds"]] == ["2026-10-01 00:00:00", "2026-10-01 12:00:00", "2026-10-02 00:00:00"]


def test_append_does_not_overwrite_complete_day_file(tmp_path):
    run = str(tmp_path / "run")
    name = "cowptn_noti_alarm_2026_10_01"
    srv = FakeServer(tables={name: {"cols": ALARM_COLS, "rows": [alarm_row(1, D(2026, 10, 1, 9), rel=D(2026, 10, 3, 5))]}})
    m1 = extract(make_db(srv), run, D(2026, 10, 1), D(2026, 10, 2), (ALARM,))
    assert m1["days"][ALARM]["2026-10-01"]["complete"] is True
    path = os.path.join(run, "raw", "alarm", "2026-10-01.csv.gz")
    before, mtime = open(path, "rb").read(), os.path.getmtime(path)
    m2 = extract(make_db(srv), run, D(2026, 10, 2), D(2026, 10, 4), (ALARM,), append=True)    # lookback 으로 같은 알람을 다시 읽음
    assert open(path, "rb").read() == before and os.path.getmtime(path) == mtime
    assert m2["stats"][ALARM]["skipped_existing_day_rows"] == 1


def test_ongoing_alarm_day_stays_incomplete_and_release_update_is_merged(tmp_path):
    run = str(tmp_path / "run")
    name = "cowptn_noti_alarm_2026_10_01"
    srv = FakeServer(tables={name: {"cols": ALARM_COLS, "rows": [alarm_row(1, D(2026, 10, 1, 9), rel=None)]}})
    m1 = extract(make_db(srv), run, D(2026, 10, 1), D(2026, 10, 2), (ALARM,))
    assert m1["days"][ALARM]["2026-10-01"]["complete"] is False                          # 진행 중 알람 → 불완전
    srv.tables[name]["rows"][0]["release_date"] = D(2026, 10, 2, 3)                      # 이후 해제됨
    m2 = extract(make_db(srv), run, D(2026, 10, 2), D(2026, 10, 3), (ALARM,), append=True)
    snap = load_snapshot(run, ALARM)
    assert len(snap) == 1 and str(snap["release_date"].iloc[0]).startswith("2026-10-02 03:00")   # 해제 시각 반영, 중복 없음
    assert m2["days"][ALARM]["2026-10-01"]["complete"] is True


def test_append_requires_complete_existing_run(tmp_path):
    srv = FakeServer(tables=_hour_tables({(1, 10): [eth_row(1, D(2026, 10, 1, 10, 15)), eth_row(2, D(2026, 10, 1, 10, 30), lid=2)]}))
    run = str(tmp_path / "run")
    extract(make_db(srv), run, D(2026, 10, 1, 10), D(2026, 10, 1, 11), (ETH,), max_rows=1, chunk=1)
    with pytest.raises(ExtractError, match="불완전"):
        extract(make_db(srv), run, D(2026, 10, 1, 11), D(2026, 10, 2), (ETH,), append=True)
    with pytest.raises(ExtractError, match="manifest"):
        extract(make_db(srv), str(tmp_path / "none"), D(2026, 10, 1), D(2026, 10, 2), (ETH,), append=True)


def test_append_end_must_advance(tmp_path):
    srv = FakeServer(tables=_hour_tables({(1, 10): [eth_row(1, D(2026, 10, 1, 10, 15))]}))
    run = str(tmp_path / "run")
    extract(make_db(srv), run, D(2026, 10, 1, 10), D(2026, 10, 1, 11), (ETH,))
    with pytest.raises(ExtractError, match="이후"):
        extract(make_db(srv), run, D(2026, 10, 1, 10), D(2026, 10, 1, 11), (ETH,), append=True)
