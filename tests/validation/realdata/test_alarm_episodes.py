"""U5: 알람 정답 처리 F1~F8, 요약, 건수 (설계서 8장, 픽스처 F-2·F-3·F-4)"""
import numpy as np
import pandas as pd
import pytest

from validation.realdata import alarm_episodes as ae
from validation.realdata.alarm_names import AlarmNames, TargetMap, DEFAULT_TARGET_MAP

D_END = pd.Timestamp("2026-10-13")
T = pd.Timestamp("2026-09-01 10:00")
TM = TargetMap.load(DEFAULT_TARGET_MAP)


def names():
    # code = alarm_type - 367. 합성 표: 인덱스로 이름 지정
    t = ["N%d" % i for i in range(256)]
    spec = {1: "ETH-LINK-FAIL", 2: "OPT-LOS", 3: "ETH-NO-RX-TRAFFIC", 4: "ETH-ERR", 5: "BOOTING", 6: "UNIT-FAIL", 7: "LINK-FAIL-MANUAL",
            8: "HI-PWR", 9: "CLI-FCS-SD", 10: "ETH-LINK-FAIL", 11: "FAN-A-1-FAIL"}
    for i, n in spec.items():
        t[i] = n
    return AlarmNames({"code_offset": 367, "sys_id": {"31": "CE9001", "27": "CE9003"}, "tables": {"CE9001": t},
                       "models_without_table": ["CE9003"]})


A = {n: 367 + i for i, n in {1: "ETH-LINK-FAIL", 2: "OPT-LOS", 3: "ETH-NO-RX-TRAFFIC", 4: "ETH-ERR", 5: "BOOTING", 6: "UNIT-FAIL",
                             7: "LINK-FAIL-MANUAL", 8: "HI-PWR", 9: "CLI-FCS-SD", 11: "FAN-A-1-FAIL"}.items()}


def alarm(name, occ, rel, ip="1.1.1.1", cid=3, lid=1, sev=3, st=31):
    return {"ip_addr": ip, "cid": cid, "lid": lid, "system_type": st, "alarm_type": A[name] if name in A else name,
            "severity_type": sev, "occur_date": occ, "release_date": rel, "address": "S.2 -P1"}


def prep(rows, d_end=D_END):
    return ae.prepare_alarms(pd.DataFrame(rows), names(), TM, d_end)


def episodes(rows, **kw):
    a = prep(rows)
    return ae.build_episodes(a[a["row_class"] == "port"], TM, D_END, **kw)


def m(x):
    return T + pd.Timedelta(minutes=x)


def test_f2_flicker_30_alarms_10_to_14_seconds_merge_into_one_episode():
    rows = [alarm("ETH-LINK-FAIL", T + pd.Timedelta(seconds=12 * i), T + pd.Timedelta(seconds=12 * i + 11)) for i in range(30)]
    ep = episodes(rows)
    assert len(ep) == 1 and ep["n_alarms"].iloc[0] == 30
    assert ep["start"].iloc[0] == T and ep["end"].iloc[0] == T + pd.Timedelta(seconds=12 * 29 + 11)       # 첫 발생 ~ 마지막 해제
    assert bool(ep["is_short"].iloc[0]) is True and bool(ep["ongoing"].iloc[0]) is False


def test_f1_merge_boundary_60_minutes():
    ok = episodes([alarm("ETH-LINK-FAIL", m(0), m(20)), alarm("ETH-LINK-FAIL", m(80), m(100))])            # 다음 occur = 이전 끝 + 60 → 병합
    assert len(ok) == 1 and ok["end"].iloc[0] == m(100)
    split = episodes([alarm("ETH-LINK-FAIL", m(0), m(20)), alarm("ETH-LINK-FAIL", m(81), m(100))])
    assert len(split) == 2
    # 병합 끝은 이전 알람들의 최대 해제 시각 (짧은 알람이 긴 알람 안에 있어도 끝이 줄지 않음)
    nest = episodes([alarm("ETH-LINK-FAIL", m(0), m(200)), alarm("ETH-LINK-FAIL", m(10), m(20)), alarm("ETH-LINK-FAIL", m(250), m(260))])
    assert len(nest) == 1 and nest["end"].iloc[0] == m(260)


def test_f1_separate_by_port_and_name():
    ep = episodes([alarm("ETH-LINK-FAIL", m(0), m(30)), alarm("ETH-LINK-FAIL", m(0), m(30), lid=2), alarm("OPT-LOS", m(0), m(30))])
    assert len(ep) == 3


@pytest.mark.parametrize("dur,short", [(14.9, True), (15, False), (0, True)])
def test_f2_min_duration_boundary(dur, short):
    ep = episodes([alarm("ETH-LINK-FAIL", m(0), m(0) + pd.Timedelta(minutes=dur))])
    assert bool(ep["is_short"].iloc[0]) is short


def test_f3_ongoing_release_null_and_after_d_cut_at_d():
    ep = episodes([alarm("ETH-LINK-FAIL", D_END - pd.Timedelta(hours=2), None),
                   alarm("ETH-ERR", D_END - pd.Timedelta(hours=3), D_END + pd.Timedelta(days=2))])         # D 이후 해제 → ongoing, 끝=D
    assert set(ep["ongoing"]) == {True} and set(ep["end"]) == {D_END}
    assert sorted(ep["duration_min"]) == [120, 180]
    same = episodes([alarm("ETH-ERR", D_END - pd.Timedelta(hours=3), D_END)])                              # 정확히 D 에 해제 → 해제됨
    assert bool(same["ongoing"].iloc[0]) is False


def test_f3_alarm_occurring_after_d_is_dropped():
    assert prep([alarm("ETH-ERR", D_END + pd.Timedelta(hours=1), None)]).empty


def test_f7_severity_columns_and_other_values():
    ep = episodes([alarm("ETH-ERR", m(0), m(5), sev=2), alarm("ETH-ERR", m(10), m(15), sev=3), alarm("ETH-ERR", m(20), m(25), sev=4),
                   alarm("ETH-ERR", m(30), m(35), sev=7)])
    r = ep.iloc[0]
    assert (r["n_CRITICAL"], r["n_MAJOR"], r["n_MINOR"], r["n_other"]) == (1, 1, 1, 1)
    assert ae.severity_label(7) == "기타-7" and ae.severity_label(None) == "기타-NA"
    assert "ETH-ERR" == ep["alarm_name"].iloc[0]                                                          # 심각도로 거르지 않음


def test_f4_simultaneous_alarms_become_one_traffic_event_and_optic_gets_no_event():
    rows = [alarm("ETH-LINK-FAIL", m(0), m(60)), alarm("OPT-LOS", m(2), m(60)), alarm("ETH-NO-RX-TRAFFIC", m(5), m(60))]
    ev = ae.build_target_events(episodes(rows))
    one = ev[(ev["target_type"] == "트래픽단절") & (ev["grade"] == "1차")]
    assert len(one) == 1 and one["t0"].iloc[0] == m(0) and one["alarm_names"].iloc[0] == "ETH-LINK-FAIL;ETH-NO-RX-TRAFFIC;OPT-LOS"
    assert one["n_episodes"].iloc[0] == 3
    assert set(ev["target_type"]) == {"트래픽단절"}                                                        # 광관찰은 타겟 이벤트가 아님


def test_f4_merge_within_60_minutes_and_split_beyond():
    a = ae.build_target_events(episodes([alarm("ETH-LINK-FAIL", m(0), m(30)), alarm("ETH-NO-RX-TRAFFIC", m(90), m(120))]))
    assert len(a) == 1                                                                                    # 끝 30 + 60 = 90 → 병합
    b = ae.build_target_events(episodes([alarm("ETH-LINK-FAIL", m(0), m(30)), alarm("ETH-NO-RX-TRAFFIC", m(91), m(120))]))
    assert len(b) == 2


def test_f4_short_episodes_excluded_and_crc_event_and_grades_separate():
    ev = ae.build_target_events(episodes([alarm("ETH-LINK-FAIL", m(0), m(10)), alarm("ETH-ERR", m(0), m(30)),
                                          alarm("CLI-FCS-SD", m(200), m(230)), alarm("ETH-LINK-FAIL", m(400), m(440), lid=2)]))
    assert set(zip(ev["target_type"], ev["grade"], ev["lid"])) == {("CRC증가", "1차", 1), ("트래픽단절", "1차", 2)}


def events_with(rows, maint=None):
    a = prep(rows)
    ev = ae.build_target_events(ae.build_episodes(a[a["row_class"] == "port"], TM, D_END))
    return ae.mark_work_and_hw(ev, a, maint)


def test_f6_work_alarm_scopes_and_window_boundary():
    base = alarm("ETH-LINK-FAIL", m(120), m(180))
    r = events_with([base, alarm("BOOTING", m(60), m(65), cid=256, lid=255)])                  # 장비 단위, t0 − 60분 정확히 → 포함
    assert bool(r["work_suspect"].iloc[0]) and not bool(r["is_judge"].iloc[0]) and r["work_source"].iloc[0].startswith("alarm:BOOTING")
    assert not events_with([base, alarm("BOOTING", m(59), m(65), cid=256, lid=255)])["work_suspect"].iloc[0]       # 61분 전은 아님
    assert events_with([base, alarm("BOOTING", m(180), m(190), cid=256, lid=255)])["work_suspect"].iloc[0]          # t0 + 60분 정확히 → 포함
    assert not events_with([base, alarm("BOOTING", m(181), m(190), cid=256, lid=255)])["work_suspect"].iloc[0]      # t0 + 61분 → 밖


def test_f6_scope_port_slot_device_and_other_name():
    base = alarm("ETH-LINK-FAIL", m(120), m(180), cid=3, lid=1)
    port_same = events_with([base, alarm("LINK-FAIL-MANUAL", m(125), m(130), cid=3, lid=1)])
    port_other = events_with([base, alarm("LINK-FAIL-MANUAL", m(125), m(130), cid=3, lid=2)])
    slot = events_with([base, alarm("LINK-FAIL-MANUAL", m(125), m(130), cid=3, lid=0)])
    slot_other = events_with([base, alarm("LINK-FAIL-MANUAL", m(125), m(130), cid=4, lid=0)])
    other_ip = events_with([base, alarm("LINK-FAIL-MANUAL", m(125), m(130), ip="9.9.9.9", cid=256, lid=255)])
    not_work = events_with([base, alarm("ETH-ERR", m(125), m(130), cid=3, lid=1)])
    assert [bool(x["work_suspect"].iloc[0]) for x in (port_same, port_other, slot, slot_other, other_ip)] == [True, False, True, False, False]
    assert not_work["work_suspect"].iloc[0] == False and bool(not_work["is_judge"].iloc[0])


def test_f6_maintenance_csv_union_and_crc_events_not_flagged(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("start,end,ip_addr,cid,lid,note\n2026-09-01 11:30,2026-09-01 13:00,1.1.1.1,3,,점검\n", encoding="utf-8")
    mt = ae.load_maintenance(str(p))
    r = events_with([alarm("ETH-LINK-FAIL", m(120), m(180))], mt)
    assert r["work_suspect"].iloc[0] and r["work_source"].iloc[0] == "csv"
    crc = events_with([alarm("ETH-ERR", m(100), m(140)), alarm("BOOTING", m(100), m(105), cid=256, lid=255)])
    assert crc[crc["target_type"] == "CRC증가"]["work_suspect"].iloc[0] == False                     # F6 은 트래픽단절만


def test_f8_hw_coincident_is_mark_only_and_keeps_event_in_denominator():
    base = alarm("ETH-LINK-FAIL", m(120), m(180))
    r = events_with([base, alarm("UNIT-FAIL", m(105), m(110), cid=256, lid=255)])                   # 장비 단위 HW, −15분
    assert bool(r["hw_coincident"].iloc[0]) and r["hw_names"].iloc[0] == "UNIT-FAIL" and bool(r["is_judge"].iloc[0])
    assert not events_with([base, alarm("UNIT-FAIL", m(104), m(110), cid=256, lid=255)])["hw_coincident"].iloc[0]
    assert events_with([base, alarm("FAN-A-1-FAIL", m(130), m(140), cid=5, lid=1)])["hw_coincident"].iloc[0]  # 다른 포트의 FAN-* 도 같은 장비
    assert not events_with([base, alarm("UNIT-FAIL", m(110), m(115), ip="9.9.9.9")])["hw_coincident"].iloc[0]


def test_summary_no_table_share_unclassified_and_unmapped_counts():
    rows = [alarm("ETH-LINK-FAIL", m(0), m(30)), alarm("BOOTING", m(0), m(5), cid=256, lid=255), alarm(367 + 50, m(0), m(10)),
            alarm(999, m(0), m(10), ip="2.2.2.2", st=27), alarm(999, m(1), m(10), ip="2.2.2.2", st=27), alarm(5, m(1), m(10), ip="3.3.3.3", st=77)]
    a = prep(rows)
    ep = ae.build_episodes(a[a["row_class"] == "port"], TM, D_END)
    ev = ae.mark_work_and_hw(ae.build_target_events(ep), a)
    s = ae.summarize_alarms(a, ep, ev)
    assert s["no_table"]["devices_without_table"] == 2 and s["no_table"]["devices"] == 3 and "67%" in s["first_line"]
    assert "비중 ≥ 1/3" in s["first_line"]
    assert [r["alarm_name"] for r in s["unclassified_top"]] == ["N50"]
    assert s["work_alarms"] == {"BOOTING": 1} and s["row_classes"]["device"] == 1
    un = ae.unmapped_alarm_counts(a)
    assert set(un["model"]) == {"CE9003", "UNKNOWN"} and un.set_index("model").loc["CE9003", "count"].sum() == 2
    assert s["target_events"] == {"트래픽단절/1차": 1}


def test_merge_sensitivity_reference_counts():
    a = prep([alarm("ETH-LINK-FAIL", m(0), m(5)), alarm("ETH-LINK-FAIL", m(30), m(35)), alarm("ETH-LINK-FAIL", m(250), m(255))])
    s = ae.merge_sensitivity(a, TM, D_END, windows=(15, 60, 240))
    assert s == {15: 3, 60: 2, 240: 1}


def test_count_judge_events_weekly_types_and_rate():
    t0 = [pd.Timestamp(x) for x in ("2026-09-01", "2026-09-02", "2026-09-09", "2026-09-10")]          # 화,수 / 화,목 (월요일 시작 주)
    ev = pd.DataFrame({"t0": t0, "target_type": ["트래픽단절", "CRC증가", "트래픽단절", "트래픽단절"], "grade": ["1차"] * 4,
                       "is_judge": [True, True, True, False], "work_suspect": [False, False, False, True]})
    days = [pd.Timestamp("2026-09-01") + pd.Timedelta(days=i) for i in range(10)]
    r = ae.count_judge_events(ev, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-11"), days)
    assert r["events"] == 3 and r["by_type"] == {"CRC증가": 1, "트래픽단절": 2} and r["weekly"] == {"2026-08-31": 2, "2026-09-07": 1}
    assert r["data_days"] == 10 and r["r_per_data_day"] == pytest.approx(0.3) and r["excluded_work_suspect"] == 1
    half = ae.count_judge_events(ev, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-03"), days)
    assert half["events"] == 2 and half["data_days"] == 2
    assert np.isnan(ae.count_judge_events(ev.iloc[0:0], pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-03"), [])["r_per_data_day"])
