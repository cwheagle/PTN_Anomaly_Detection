"""U6: 재생 입력·베이스라인(광 조건 제거)·C1/C3/C4b/탐지 지연/C6/겹침 (설계서 9.2~9.6, 픽스처 F-1·F-11)"""
import numpy as np
import pandas as pd
import pytest

from validation.evaluation.baselines import fixed_threshold, rolling_rule
from validation.realdata import replay as rp
from validation.realdata import report as rpt

T = pd.Timestamp("2026-09-01 00:00")
KEY = ["ip_addr", "cid", "lid"]


def m(minutes, base=T):
    return base + pd.Timedelta(minutes=minutes)


# ---- 베이스라인 use_optical ----------------------------------------------------------------
def raw_with_optics(n=150):
    t = pd.date_range("2026-09-01", periods=n, freq="15min")
    return pd.DataFrame({"occur_date": t, "ip_addr": "a", "cid": 1, "lid": 1, "tx_packet": 100, "rx_packet": 100, "error_packet": 0,
                         "tx_avg_power": -5.0, "rx_avg_power": np.where(np.arange(n) >= 120, -20.0, -8.0)})


def test_use_optical_default_is_unchanged_behavior():
    raw = raw_with_optics()
    assert fixed_threshold(raw)["alarm"].iloc[130] and rolling_rule(raw)["alarm"].iloc[130]          # 광 조건이 기본으로 켜져 있음
    pd.testing.assert_frame_equal(fixed_threshold(raw), fixed_threshold(raw, use_optical=True))
    pd.testing.assert_frame_equal(rolling_rule(raw), rolling_rule(raw, use_optical=True))


def test_use_optical_false_ignores_power_and_works_without_power_columns():
    raw = raw_with_optics()
    assert not fixed_threshold(raw, use_optical=False)["alarm"].any() and not rolling_rule(raw, use_optical=False)["alarm"].any()
    bare = raw.drop(columns=["tx_avg_power", "rx_avg_power"])
    bare.loc[140, "error_packet"] = 150
    f, r = fixed_threshold(bare, use_optical=False), rolling_rule(bare, use_optical=False)
    assert f["alarm"].iloc[140] and r["alarm"].iloc[140] and f["alarm"].sum() == 1
    bare.loc[100:110, ["tx_packet", "rx_packet"]] = 1                                                   # 트래픽 급감 조건은 남아 있음
    assert rolling_rule(bare, use_optical=False)["alarm"].iloc[105]


# ---- 재생 입력 ----------------------------------------------------------------------------
def eth(n_days=6, ports=((1, 1),), err=None):
    rows = []
    for c, l in ports:
        for i in range(1, 96 * n_days + 1):
            t = m(15 * i)
            rows.append({"ip_addr": "1.1.1.1", "cid": c, "lid": l, "tx_packet": 100, "rx_packet": 100,
                         "error_packet": (err or {}).get(t, 0), "occur_date": t, "bin_end": t})
    return pd.DataFrame(rows)


def test_traffic_frame_window_dedupe_and_columns():
    e = eth()
    e = pd.concat([e, e.iloc[[10]].assign(error_packet=7)])                         # 중복: 마지막 행 유지
    tf = rp.traffic_frame(e, T + pd.Timedelta(days=3), T + pd.Timedelta(days=4))
    assert tf["occur_date"].min() == T + pd.Timedelta(days=1) and tf["occur_date"].max() < T + pd.Timedelta(days=4)     # 2일 앞부터
    assert list(tf.columns) == ["occur_date", "ip_addr", "cid", "lid", "tx_packet", "rx_packet", "error_packet"]
    dup = rp.traffic_frame(e, T, T + pd.Timedelta(days=1), warmup_days=0)
    assert len(dup) == 95 and dup.loc[dup["occur_date"] == m(15 * 11), "error_packet"].iloc[0] == 7


class FakeDetector:
    """연속 required_rows(=3) 이상 행이 있는 슬롯에만 점수를 내고 error>0 이면 알람."""

    def detect(self, df_traffic, df_optical, latest_only):
        assert df_optical is None and latest_only is False
        d = df_traffic.sort_values(KEY + ["occur_date"]).copy()
        d["_n"] = d.groupby(KEY).cumcount()
        d = d[d["_n"] >= 3]
        return d.assign(is_anomaly=d["error_packet"] > 0, severity=np.where(d["error_packet"] > 100, 90.0, 55.0),
                        traffic_score=d["error_packet"].astype(float))[["occur_date", "ip_addr", "cid", "lid", "is_anomaly", "severity", "traffic_score"]]


def test_replay_cuts_warmup_and_reports_playable_slots_per_method():
    s, e = T + pd.Timedelta(days=3), T + pd.Timedelta(days=4)
    r = rp.replay(eth(err={s + pd.Timedelta(hours=5): 150}), s, e, detector=FakeDetector())
    assert all(f["occur_date"].min() >= s and f["occur_date"].max() < e for f in r["frames"].values())
    assert r["normal_rows"] == 96 and r["slots"]["ai"]["playable"] == 96 and r["slots"]["fixed"]["ratio"] == 1.0
    assert r["frames"]["ai"]["alarm"].sum() == 1 and r["frames"]["fixed"]["alarm"].sum() == 1 and r["frames"]["rolling"]["alarm"].sum() == 1


def test_playable_ratio_reflects_gaps_for_ai_and_rolling():
    e = eth(n_days=1)
    e = e[(e["occur_date"].dt.hour < 2) | (e["occur_date"].dt.hour >= 8)]            # 2~8시 공백: 공백을 메우지 않는다
    r = rp.replay(e, T, T + pd.Timedelta(days=1), detector=FakeDetector(), warmup_days=0)
    assert r["slots"]["ai"]["playable"] == r["normal_rows"] - 3                       # 워밍업 3행
    assert r["slots"]["rolling"]["playable"] == r["normal_rows"] - rp.ROLLING_MIN_PERIODS


def test_incidents_gap_of_60_minutes_and_port_days():
    t = [m(0), m(15), m(75), m(150)]                                                  # 15→75 는 60분 간격(같은 인시던트), 150 은 별도
    f = pd.DataFrame({"ip_addr": "a", "cid": 1, "lid": 1, "occur_date": t, "alarm": True, "severity": [55.0, 90.0, 55.0, 55.0]})
    inc, _ = rp.incidents_of(f)
    assert len(inc) == 2 and inc["severity_max"].tolist() == [90.0, 55.0]
    assert rp.port_days(f) == 1 and rp.port_days(f.iloc[0:0]) == 0


# ---- C3 -----------------------------------------------------------------------------------
def perf_for(slots_before, key=("a", 1, 1), t0=m(1000), extra=()):
    """t0 앞 6시간에 slots_before 개의 슬롯 (가까운 쪽부터) + t0 이후 몇 개."""
    ts = [t0 - pd.Timedelta(minutes=15 * (i + 1)) for i in range(slots_before)] + list(extra)
    return rpt.PerfIndex(pd.DataFrame({"ip_addr": key[0], "cid": key[1], "lid": key[2], "bin_end": ts}))


def ev_frame(t0s, key=("a", 1, 1), target="트래픽단절", grade="1차"):
    return pd.DataFrame([{"ip_addr": key[0], "cid": key[1], "lid": key[2], "target_type": target, "grade": grade, "t0": t} for t in t0s])


def frames_with_alarms(times, key=("a", 1, 1), method_alarms=None):
    base = pd.DataFrame({"ip_addr": [key[0]] * len(times), "cid": key[1], "lid": key[2], "occur_date": times, "alarm": True})
    empty = base.iloc[0:0]
    return {"ai": base, "fixed": empty, "rolling": empty, **(method_alarms or {})}


def test_c3_lead_window_boundaries_and_cumulative_buckets():
    t0 = m(1000)
    perf = perf_for(24, t0=t0)
    cases = {"exactly 24h before": t0 - pd.Timedelta(hours=24), "15min before": t0 - pd.Timedelta(minutes=15),
             "10min before": t0 - pd.Timedelta(minutes=10), "24h+1s before": t0 - pd.Timedelta(hours=24, seconds=1),
             "at t0": t0, "after": t0 + pd.Timedelta(minutes=30)}
    lead = {}
    for name, ts in cases.items():
        rows = rpt.c3_events(ev_frame([t0]), frames_with_alarms([ts]), perf)
        lead[name] = rows["ai_lead_min"].iloc[0]
    assert lead["exactly 24h before"] == 1440 and lead["15min before"] == 15
    assert all(np.isnan(lead[k]) for k in ("10min before", "24h+1s before", "at t0", "after"))
    # 가장 이른 알람이 리드를 정한다
    rows = rpt.c3_events(ev_frame([t0]), frames_with_alarms([t0 - pd.Timedelta(minutes=30), t0 - pd.Timedelta(hours=5)]), perf)
    assert rows["ai_lead_min"].iloc[0] == 300


def test_c3_summary_cumulative_lead_ratios_and_bucket_counts():
    leads = [20, 59, 60, 179, 180, 359, 360, 1440, np.nan, np.nan]
    rows = pd.DataFrame({"status": "ok", "ai_lead_min": leads, "fixed_lead_min": np.nan, "rolling_lead_min": np.nan})
    s = rpt.c3_summary(pd.concat([rows] * 3, ignore_index=True), None)           # 30 이벤트 (판정 가능)
    ai = s["methods"]["ai"]
    assert s["included"] == 30 and not s["hold"] and ai["detected"] == 24 and ai["rate"] == pytest.approx(0.8)
    assert (ai["lead_ge_1h"], ai["lead_ge_3h"], ai["lead_ge_6h"]) == (pytest.approx(18 / 30), pytest.approx(12 / 30), pytest.approx(6 / 30))   # 분모 = 전체 이벤트, 누적
    assert ai["buckets"] == {"[15,60)min": 6, "[60,180)min": 6, "[180,360)min": 6, "[360,1440]min": 6}      # 24h(1440) 는 마지막 닫힌 구간에 포함


def test_c3_holds_when_fewer_than_20_events_or_join_not_verified():
    rows = pd.DataFrame({"status": ["ok"] * 19, "ai_lead_min": 30.0, "fixed_lead_min": np.nan, "rolling_lead_min": np.nan})
    assert rpt.c3_summary(rows, None)["hold"] is True
    rows20 = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    assert rpt.c3_summary(rows20, None)["hold"] is False
    s = rpt.c3_summary(rows20, "join_status = failed (6.3)")
    assert s["hold"] and "join_status" in s["hold_reasons"][0]
    assert rpt.c4b(s)["status"] == "hold"
    ok = rpt.c3_summary(pd.concat([rows20.assign(fixed_lead_min=np.nan)]), None)
    assert rpt.c4b(ok) == {"status": "ok", "ai_gt_fixed": True, "ai_rate": 1.0, "fixed_rate": 0.0}


@pytest.mark.parametrize("slots,status", [(12, "ok"), (11, "perf_gap"), (24, "ok"), (0, "perf_gap")])
def test_c3_perf_gap_boundary_50_percent_of_24_slots(slots, status):
    t0 = m(1000)
    perf = perf_for(slots, t0=t0) if slots else rpt.PerfIndex(pd.DataFrame({"ip_addr": ["a"], "cid": [1], "lid": [1], "bin_end": [t0 + pd.Timedelta(hours=9)]}))
    assert rpt.c3_events(ev_frame([t0]), frames_with_alarms([]), perf)["status"].iloc[0] == status


def test_c3_no_port_when_key_not_in_performance_and_key_shift_is_explicit():
    t0 = m(1000)
    perf = perf_for(24, key=("a", 1, 1), t0=t0)
    ev = ev_frame([t0], key=("a", 2, 1))                                           # 알람 키가 성능 키와 +1 어긋난 구성
    assert rpt.c3_events(ev, frames_with_alarms([]), perf)["status"].iloc[0] == "no_port"
    assert rpt.c3_events(ev, frames_with_alarms([], key=("a", 1, 1)), perf, shift=-1)["status"].iloc[0] == "ok"   # 진단용 명시 보정만


# ---- 탐지 지연 (F-1 순간 단절) ------------------------------------------------------------------
D_END = m(100000)


def delay_row(t0, alarm_times, perf_extra=(), d_end=D_END):
    perf = rpt.PerfIndex(pd.DataFrame({"ip_addr": "a", "cid": 1, "lid": 1, "bin_end": [t0 + pd.Timedelta(minutes=15 * i) for i in range(-8, 30)] + list(perf_extra)}))
    return rpt.delay_events(ev_frame([t0]), frames_with_alarms(alarm_times), perf, d_end).iloc[0]


def test_bin_end_on_boundary_goes_to_next_bin():
    t_off = pd.Timestamp("2026-09-01 10:07")
    assert rpt.bin_end(t_off) == pd.Timestamp("2026-09-01 10:15")
    assert rpt.bin_end(pd.Timestamp("2026-09-01 10:15")) == pd.Timestamp("2026-09-01 10:30")      # 정각 경계 t0 → 다음 구간


def test_delay_wall_clock_and_step_values():
    t0 = pd.Timestamp("2026-09-01 10:07")                                         # bin_end = 10:15
    r = delay_row(t0, [pd.Timestamp("2026-09-01 10:15")])
    assert r["ai_wall_min"] == 8 and r["ai_step"] == 0 and r["ai_pre15"] == False
    r = delay_row(t0, [pd.Timestamp("2026-09-01 10:45")])
    assert r["ai_wall_min"] == 38 and r["ai_step"] == 2
    r = delay_row(t0, [pd.Timestamp("2026-09-01 16:07")])                          # t0 + 6h 정확히 → 포함
    assert r["ai_step"] == 23
    r = delay_row(t0, [pd.Timestamp("2026-09-01 16:08")])                          # 6h 초과 → 미탐
    assert np.isnan(r["ai_wall_min"]) and r["status"] == "ok"


def test_delay_pre15_alarm_counts_as_detection_with_step_zero():
    t0 = pd.Timestamp("2026-09-01 10:07")
    r = delay_row(t0, [pd.Timestamp("2026-09-01 10:00")])                          # [t0−15분, bin_end) 의 알람
    assert r["ai_pre15"] == True and r["ai_step"] == 0 and r["ai_wall_min"] == 8
    r = delay_row(t0, [pd.Timestamp("2026-09-01 09:45")])                          # t0−15분 이전은 C3 선행 창 몫 (지연에서 탐지 아님)
    assert np.isnan(r["ai_wall_min"])


def test_delay_exclusions_truncated_and_perf_gap_with_boundaries():
    t0 = pd.Timestamp("2026-09-01 10:07")
    perf_none_after = rpt.PerfIndex(pd.DataFrame({"ip_addr": "a", "cid": 1, "lid": 1, "bin_end": [t0 - pd.Timedelta(hours=1)]}))
    assert rpt.delay_events(ev_frame([t0]), frames_with_alarms([]), perf_none_after, D_END)["status"].iloc[0] == "perf_gap"
    edge = rpt.PerfIndex(pd.DataFrame({"ip_addr": "a", "cid": 1, "lid": 1, "bin_end": [t0 + pd.Timedelta(hours=1)]}))                  # [bin_end, t0+1h] 닫힌 구간 끝
    assert rpt.delay_events(ev_frame([t0]), frames_with_alarms([]), edge, D_END)["status"].iloc[0] == "ok"
    assert delay_row(t0, [], d_end=t0 + pd.Timedelta(hours=6))["status"] == "ok"                                                         # t0+6h == D → 포함
    assert delay_row(t0, [], d_end=t0 + pd.Timedelta(hours=6) - pd.Timedelta(seconds=1))["status"] == "truncated"


def test_delay_summary_lists_individually_below_10_and_aggregates_from_10():
    rows = pd.DataFrame({"status": "ok", "ai_wall_min": [8.0, 38.0, np.nan] + [20.0] * 7, "fixed_wall_min": np.nan, "rolling_wall_min": np.nan,
                         "ai_step": [0, 2, np.nan] + [1] * 7, "fixed_step": np.nan, "rolling_step": np.nan,
                         "ai_pre15": False, "fixed_pre15": False, "rolling_pre15": False})
    s = rpt.delay_summary(rows.iloc[:9], "failed")
    assert s["list_individually"] and "individual" in s and "median_wall_min" not in s["methods"]["ai"]
    s = rpt.delay_summary(rows, "verified")
    ai = s["methods"]["ai"]
    assert not s["list_individually"] and ai["median_wall_min"] == 20.0 and ai["le_30min"] == pytest.approx(0.8) and ai["missed_rate"] == pytest.approx(0.1)
    assert ai["steps"] == {"0": 1, "1": 7, "2": 1, "3+": 0}


# ---- C6 -----------------------------------------------------------------------------------
def daily_frames(counts, start=T):
    """날짜별 인시던트 수 (인시던트 = 서로 떨어진 알람 하나씩)."""
    rows = []
    for d, n in enumerate(counts):
        for k in range(n):
            rows.append({"ip_addr": "a", "cid": 1, "lid": k + 1, "occur_date": start + pd.Timedelta(days=d, hours=1), "alarm": True})
    f = pd.DataFrame(rows)
    return {"ai": f, "fixed": f.iloc[0:0], "rolling": f.iloc[0:0]}


def test_c6_hold_rules_and_band():
    days = [T + pd.Timedelta(days=i) for i in range(20)]
    stable = rpt.c6(daily_frames([10] * 18 + [4, 25]), days, T, T + pd.Timedelta(days=20))
    ai = stable["methods"]["ai"]
    assert ai["median"] == 10 and ai["out_of_band_days"] == 2 and ai["pass"] is True and not ai["hold"]            # 4<5, 25>20 → 2일(기준 ≤ 2)
    bad = rpt.c6(daily_frames([10] * 17 + [4, 25, 30]), days, T, T + pd.Timedelta(days=20))["methods"]["ai"]
    assert bad["out_of_band_days"] == 3 and bad["pass"] is False
    low = rpt.c6(daily_frames([5] * 20), days, T, T + pd.Timedelta(days=20))["methods"]["ai"]
    assert low["hold"] and "중앙값" in low["hold_reasons"][0] and low["pass"] is None
    few = rpt.c6(daily_frames([10] * 9), days[:9], T, T + pd.Timedelta(days=9))["methods"]["ai"]
    assert few["hold"] and "망 이벤트 없는 날 9" in few["hold_reasons"][0]


def test_c6_network_event_days_removed_from_pass_judgement_and_non_data_days_not_zero():
    days = [T + pd.Timedelta(days=i) for i in range(20)]
    ev_days = [days[18], days[19]]
    r = rpt.c6(daily_frames([10] * 18 + [4, 25]), days, T, T + pd.Timedelta(days=20), network_event_days=ev_days)
    ai = r["methods"]["ai"]
    assert r["network_event_days"] == 2 and ai["non_event_days"] == 18 and ai["out_of_band_days"] == 0
    gappy = [d for i, d in enumerate(days) if i % 2 == 0]                                                      # 데이터 없는 날은 0 이 아니라 제외
    g = rpt.c6(daily_frames([10] * 20), gappy, T, T + pd.Timedelta(days=20))
    assert g["data_days"] == 10 and set(g["methods"]["ai"]["daily"]) == {str(d.date()) for d in gappy}


# ---- 겹침·태깅 -----------------------------------------------------------------------------------
def test_overlap_and_tagging():
    ai = pd.DataFrame({"ip_addr": "a", "cid": 1, "lid": [1, 1, 2], "occur_date": [m(0), m(300), m(0)], "alarm": True})
    rr = pd.DataFrame({"ip_addr": "a", "cid": 1, "lid": [1, 1, 3], "occur_date": [m(0), m(15), m(0)], "alarm": True})
    fr = {"ai": ai, "fixed": ai.iloc[0:0], "rolling": rr}
    o = rpt.overlap(fr)
    assert o == {"ai_incidents": 3, "ai_overlapping_rolling": 1, "ai_only": 2, "rolling_incidents": 2, "rolling_overlapping_ai": 1, "rolling_only": 1}
    ep = pd.DataFrame({"ip_addr": ["a"], "cid": [1], "lid": [1], "start": [m(30)], "end": [m(90)], "target_grades": ["트래픽단절/1차"]})
    tags = rpt.tag_ai_incidents(fr, ep)
    assert tags["incidents"] == 3 and tags["tags"]["트래픽단절/1차"] == 1 and tags["tags"]["EMS 에피소드 없음"] == 2
