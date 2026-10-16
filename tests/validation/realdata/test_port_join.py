"""U4: 포트 조인·행 분류·J1~J4·판정·M1 (설계서 6장, 픽스처 F-5)"""
import numpy as np
import pandas as pd
import pytest

from validation.realdata import port_join as pj

T0 = pd.Timestamp("2026-10-01 00:00")


def eth_frame(ports, hours=24, zero=None, ip="1.1.1.1", system_type=12, addr_fmt="CA1-UT-S.{c} -P{l}", shift=0):
    """ports: [(cid, lid)], zero: {(cid,lid): (start, end)} 구간(구간 끝 T 기준 (T−15분, T] 가 [start,end] 에 포함)에서 rx=0."""
    rows = []
    for (c, l) in ports:
        for i in range(1, hours * 4 + 1):
            t = T0 + pd.Timedelta(minutes=15 * i)
            rx = 100
            if zero and (c, l) in zero:
                s, e = zero[(c, l)]
                if t - pd.Timedelta(minutes=15) >= s and t <= e:
                    rx = 0
            rows.append({"ip_addr": ip, "cid": c, "lid": l, "system_type": system_type, "address": addr_fmt.format(c=c + shift, l=l),
                         "rx_packet": rx, "tx_packet": 10, "error_packet": 0, "occur_date": t, "bin_end": t})
    return pd.DataFrame(rows)


def episodes(rows):
    return pd.DataFrame(rows, columns=["ip_addr", "cid", "lid", "start", "end"])


def ts(h, m=0, d=0):
    return T0 + pd.Timedelta(days=d, hours=h, minutes=m)


# ---- 정규화·분류 ----------------------------------------------------------------------------

def rates(j4=1.0, j2=1.0, st=12):
    return {st: {"n": 100, "j4": j4, "j2": j2, "j2_pairs": 50}}


def test_f5a_verified_when_corrected_keys_are_consistent():
    eth = eth_frame([(1, 1), (1, 2), (2, 1)], zero={(1, 1): (ts(10), ts(12))})
    ep = episodes([["1.1.1.1", 1, 1, ts(10), ts(12)]] * 5)
    j3 = pj.j3_physical_match(ep, eth)
    assert j3["with_bins"] == 5 and j3["match_a"] == 1.0 and j3["match_c"] == 0.0
    assert j3["diag_alive"]["episodes"] == 5 and j3["diag_alive"]["match_a_alive"] == 1.0 and j3["diag_alive"]["match_c_alive"] == 0.0
    r = pj.judge_join(pj.j1_key_consistency(eth), j3, rates())
    assert r["status"] == "verified" and r["reasons"] == [] and r["note"] == "키 보정은 데이터 일관성으로만 확인, 현장 대조 미실시"



def test_f5b_failed_when_j3_direction_is_wrong_and_no_auto_switch_to_address():
    # 알람 키가 정상 포트(cid=3)를 가리키고 rx=0 인 포트(cid=1)는 대조군: 방향(a > c) 위반. address 참고 조인(d)은 맞는 포트를 본다.
    eth = eth_frame([(1, 1), (3, 1)], zero={(1, 1): (ts(10), ts(12))}, addr_fmt="CA1-S.{c} -P{l}")
    eth.loc[eth["cid"] == 1, "address"] = "CA1-S.3 -P1"; eth.loc[eth["cid"] == 3, "address"] = "CA1-S.1 -P1"
    ep = episodes([["1.1.1.1", 3, 1, ts(10), ts(12)]] * 5)
    ep["addr_port"] = "S.3-P1"
    j3 = pj.j3_physical_match(ep, eth)
    assert j3["match_a"] == 0.0 and j3["match_d"] == 1.0
    r = pj.judge_join(pj.j1_key_consistency(eth), j3, rates())
    assert r["status"] == "failed" and r["stop"] is True and "방향 불일치" in " ".join(r["reasons"]) and r["address_join_diag"] == 1.0



def test_j3_insufficient_flag_does_not_block_when_other_checks_pass():
    eth = eth_frame([(1, 1), (1, 2)], zero={(1, 1): (ts(10), ts(12))})
    j3 = pj.j3_physical_match(episodes([["1.1.1.1", 1, 1, ts(10), ts(12)]] * 4), eth)
    r = pj.judge_join(pj.j1_key_consistency(eth), j3, rates())
    assert r["status"] == "verified" and r["flags"] == ["j3_insufficient"]                        # 대상 < 5 → 막지 않고 표시
    assert pj.judge_join(pj.j1_key_consistency(eth), j3, {})["status"] == "unverified"             # 오프셋 표에 모델이 없으면 검증 불가


def test_missing_rows_count_as_match_and_are_reported_separately():
    eth = eth_frame([(1, 1), (1, 2)])
    eth = eth[~((eth["cid"] == 1) & (eth["lid"] == 1) & (eth["bin_end"] > ts(10)) & (eth["bin_end"] <= ts(11)))]   # 1시간 행 없음
    j3 = pj.j3_physical_match(episodes([["1.1.1.1", 1, 1, ts(10), ts(12)]]), eth)
    assert j3["match_a"] == pytest.approx(4 / 8) and j3["a_missing"] == pytest.approx(0.5) and j3["a_zero"] == 0.0


def test_episode_truncated_to_24h_and_short_episode_skipped():
    eth = eth_frame([(1, 1), (1, 2)], hours=24 * 3, zero={(1, 1): (ts(0, 0), ts(24))})            # 첫 24시간만 rx=0
    j3 = pj.j3_physical_match(episodes([["1.1.1.1", 1, 1, ts(0), ts(0, 0, d=3)]]), eth)
    assert j3["match_a"] == 1.0                                                                    # 72시간 에피소드도 앞 24시간만 본다
    short = pj.j3_physical_match(episodes([["1.1.1.1", 1, 1, ts(10), ts(10, 29)]]), eth)
    assert short["episodes"] == 0                                                                  # 30분 미만 제외
    assert pj.j3_physical_match(episodes([["1.1.1.1", 1, 1, ts(10), ts(10, 30)]]), eth)["episodes"] == 1


def test_key_not_in_snapshot_is_not_counted_as_match():
    eth = eth_frame([(1, 1), (1, 2)])
    j3 = pj.j3_physical_match(episodes([["1.1.1.1", 9, 9, ts(10), ts(12)]]), eth)
    assert j3["not_in_snapshot"] == 1 and j3["with_bins"] == 0



def test_control_is_deterministic_and_same_device_only():
    ports = [(c, 1) for c in range(1, 8)]
    eth = pd.concat([eth_frame(ports, zero={(1, 1): (ts(10), ts(12))}), eth_frame([(1, 1)], ip="2.2.2.2", zero={(1, 1): (ts(10), ts(12))})])
    ep = episodes([["1.1.1.1", 1, 1, ts(10), ts(12)]])
    a = pj.j3_physical_match(ep, eth)["detail"]["c"].iloc[0]
    b = pj.j3_physical_match(ep, eth)["detail"]["c"].iloc[0]
    assert a == b == 0.0                                           # 다른 장비(2.2.2.2)의 rx=0 포트는 대조군이 될 수 없음
    single = pj.j3_physical_match(ep, eth_frame([(1, 1)], zero={(1, 1): (ts(10), ts(12))}))
    assert single["n_control"] == 0 and np.isnan(single["match_c"]) and single["diag_alive"]["episodes"] == 0        # 대조군 없음 → 방향 판단 불가
    assert pj.judge_join({"one_to_one_ratio": 1.0}, single, rates())["flags"] == ["j3_insufficient"]


def test_by_model_breakdown():
    e1 = eth_frame([(1, 1), (1, 2)], zero={(1, 1): (ts(10), ts(12))}, system_type=12)
    e2 = eth_frame([(1, 1), (1, 2)], ip="2.2.2.2", system_type=31)
    ep = episodes([["1.1.1.1", 1, 1, ts(10), ts(12)], ["2.2.2.2", 1, 1, ts(10), ts(12)]])
    bm = pj.j3_physical_match(ep, pd.concat([e1, e2]))["by_model"]
    assert bm["12"]["match_a"] == 1.0 and bm["31"]["match_a"] == 0.0


# ---- 판정 경계값 (사전 고정 기준) ----------------------------------------------------------------
def J(a, c, d=float("nan"), n=5):
    return {"with_bins": n, "match_a": a, "match_c": c, "match_d": d}




def test_judge_does_not_switch_to_address_join_when_it_would_pass():
    j3 = {"diag_alive": {"episodes": 0}, "match_d": 0.95}
    r = pj.judge_join({"one_to_one_ratio": 1.0}, j3, rates(j4=0.5))
    assert r["status"] == "failed" and r["stop"] is True and r["address_join_diag"] == 0.95


# ---- J2 / J4 -------------------------------------------------------------------------------
def test_j2_systematic_slot_shift_by_model():
    eth = eth_frame([(1, 1), (1, 2)], system_type=12)
    al = pd.DataFrame({"ip_addr": ["1.1.1.1"] * 4, "system_type": [12] * 4, "cid": [1, 1, 0, 1], "lid": [1, 2, 0, 0],
                       "address": ["S.3 -P1  ", "S.3 -P2  ", "S.3  ", "S.1 -P0"], "p5_cid": [1, 1, 0, 1], "p5_lid": [1, 2, 0, 0]})
    j2 = pj.j2_diagnosis(pj.classify_alarm_rows(al), eth)
    m = j2["by_model"][0]
    assert m["pairs"] == 2 and m["diff_rate"] == 1.0 and m["slot_shift_mode"] == 2
    assert j2["device_alarm_address_shape"]["device/slot-only"] == 1 and j2["p5_same_ratio"]["12"] == 1.0
    assert len(j2["diff_samples"]) == 2



def test_j4_coverage_denominator_is_devices_present_in_performance():
    eth = eth_frame([(1, 1), (1, 2)])
    al = pd.DataFrame({"ip_addr": ["1.1.1.1", "1.1.1.1", "9.9.9.9", "1.1.1.1"], "system_type": [12, 12, 31, 12], "cid": [1, 5, 1, 0], "lid": [1, 5, 1, 0],
                       "alarm_type": [1, 2, 3, 4], "occur_date": [ts(1)] * 4, "address": ["S.1 -P1"] * 4})
    r = pj.j4_coverage(pj.classify_alarm_rows(al), eth)
    assert r["port_alarms"] == 3 and r["port_alarms_in_perf_devices"] == 2 and r["no_perf_device"] == 1 and r["no_perf_device_by_model"] == {"31": 1}
    assert r["matched_ratio"] == pytest.approx(1 / 2)                                              # 분모: 성능 장비의 포트 알람만
    assert len(r["unmatched_samples"]) == 1


# ---- M1 ------------------------------------------------------------------------------------

def test_manual_check_uses_corrected_key_and_stops_on_single_mismatch():
    ok = pd.DataFrame({"ip_addr": ["a", "b", "c"], "system_type": [12, 31, 31], "perf_cid": [1, 2, 3], "perf_lid": [1, 1, 1],
                       "alarm_cid": [2, 3, 4], "alarm_lid": [1, 1, 1]})                          # 알람 cid = 성능 cid + 1 (k=1)
    r = pj.check_manual(ok)
    assert not r["stop"] and r["warnings"] == []
    bad = ok.copy(); bad.loc[2, "alarm_cid"] = 5
    assert pj.check_manual(bad)["stop"] and len(pj.check_manual(bad)["mismatches"]) == 1
    assert pj.check_manual(ok, offsets={31: 2})["stop"]                                           # 모델 31 의 k 가 2 로 정해졌다면 불일치
    assert len(pj.check_manual(ok.iloc[:2].assign(system_type=12))["warnings"]) == 2
    with pytest.raises(ValueError):
        pj.check_manual(ok.drop(columns=["alarm_lid"]))


# ---- F-13 행 분류 (설계 6.1 개정 4, 규칙 1~10) ---------------------------------------------------------------
F13 = [  # (address, cid, lid, 기대 규칙, 기대 분류)
    ("SYSTEM", 256, 255, "dev_system", "device"), ("system ", None, None, "dev_system", "device"),                     # 순서: SYSTEM + NULL 키 → device
    ("x", 256, 255, "dev_256_255", "device"), ("S.1 -P1", 256, 3, "dev_partial", "device"), ("S.1 -P1", 3, 255, "dev_partial", "device"),
    ("S.1 -P1", 0, 5, "dev_cid0", "device"), ("S.1 -P1", None, 1, "bad_key", "bad_key"), ("S.1 -P1", 2, -1, "bad_key", "bad_key"),
    ("S.1 -P1", 3, 0, "slot_lid0", "slot"), ("S.6", 6, 1, "slot_addr", "slot"),
    ("PAPA.A-FAN-7", 3, 1, "unit_addr", "unit"), ("MAIN-P1", 4, 1, "port_main", "port"), ("main-p2  ", 4, 2, "port_main", "port"),
    ("MAIN-P9  PW-5021", 4, 9, "service_main", "service"), ("MAIN-P3", 4, 1, "main_lid_mismatch", "unit"),
    ("S.2 -P1  ", 3, 1, "port", "port"), ("", 3, 2, "port_noaddr", "port"), (None, 3, 3, "port_noaddr", "port")]


def f13_frame():
    return pd.DataFrame({"ip_addr": "a", "system_type": 12, "address": [r[0] for r in F13], "cid": [r[1] for r in F13], "lid": [r[2] for r in F13],
                         "alarm_type": 1})


def test_f13_f17_every_rule_has_a_row_and_rule_order_is_applied():
    out = pj.classify_alarm_rows(f13_frame())
    assert list(out["row_rule"]) == [r[3] for r in F13] and list(out["row_class"]) == [r[4] for r in F13]
    assert set(out["row_rule"]) == set(pj.RULE_IDS) and pj.RULE_IDS[0] == "dev_system" and pj.RULE_IDS[-1] == "port_noaddr"


def test_unit_and_other_non_port_rows_never_become_episodes_or_target_events():
    from validation.realdata import alarm_episodes as ae
    from tests.validation.realdata.test_alarm_episodes import TM, A, names
    rows = [{"ip_addr": "1.1.1.1", "cid": 3, "lid": 1, "system_type": 31, "alarm_type": A["ETH-LINK-FAIL"], "severity_type": 3,
             "occur_date": pd.Timestamp("2026-09-01 10:00"), "release_date": pd.Timestamp("2026-09-01 11:00"), "address": ad}
            for ad in ("PAPA.A-FAN-7", "S.6", "SYSTEM", "S.2 -P1")]
    a = ae.prepare_alarms(pd.DataFrame(rows), names(), TM, pd.Timestamp("2026-10-13"))
    assert sorted(a["row_class"]) == ["device", "port", "slot", "unit"]
    ep = ae.build_episodes(a[a["row_class"] == "port"], TM, pd.Timestamp("2026-10-13"))
    assert len(ep) == 1                                                                         # 포트 알람 1행만


def test_row_rule_report_fixed_tables_and_perf_key_counts():
    n = len(F13)
    al = f13_frame().assign(system_type=[12] * 8 + [31] * (n - 8))
    cls = pj.classify_alarm_rows(al)
    eth = eth_frame([(6, 1), (3, 1), (4, 1), (4, 9)], hours=1, ip="a")                           # 성능 키 (6,1) (3,1) (4,1) (4,9) 존재
    rep = pj.row_rule_report(cls, eth)
    assert rep["version"] == "row_class_v4" and rep["rule_order"] == list(pj.RULE_IDS)
    assert list(rep["by_model_rule"]["12"].keys()) == list(pj.RULE_IDS) and rep["by_model_rule"]["12"]["dev_system"] == 2
    assert sum(sum(v.values()) for v in rep["by_model_rule"].values()) == n
    assert rep["reclassified_vs_rev1"] == 9                                                      # 행 0~4(device 로), S.6(slot), FAN(unit), service_main, main_lid_mismatch
    w = rep["with_perf_key"]
    assert w["slot_addr"]["31"] == {"rows": 1, "with_perf_key": 1} and w["unit_addr"]["31"] == {"rows": 1, "with_perf_key": 1}
    assert w["service_main"]["31"] == {"rows": 1, "with_perf_key": 1} and w["main_lid_mismatch"]["31"] == {"rows": 1, "with_perf_key": 1}
    txt = pj.format_row_rule_report(rep)
    assert txt.splitlines()[0].startswith("[행 분류 row_class_v4]") and "model 12:" in txt and "unit_addr 중 성능 키 있음" in txt and "service_main 중 성능 키 있음" in txt
    assert "S.n-Pm 뒤에 토큰이 남는" in txt
    j2 = pj.j2_diagnosis(cls, eth)
    assert j2["row_class_rules"]["version"] == "row_class_v4"


def test_trailing_tokens_after_port_are_reported_by_model():
    df = pd.DataFrame({"ip_addr": "a", "system_type": [12, 12, 31, 31], "cid": 3, "lid": [1, 2, 1, 1], "alarm_type": 1,
                       "address": ["S.2 -P1", "S.2 -P2 PW-5021", "CA-S.1 -P1 LSP-9", "S.1 -P1"]})
    r = pj.trailing_tokens_report(pj.classify_alarm_rows(df))
    assert r["total"] == 2 and r["by_model"] == {"12": 1, "31": 1} and r["rows_checked"] == 4
    assert r["top_tokens"]["12"] == {"PW-5021": 1} and r["samples"]["31"][0]["trailing"] == "LSP-9"


def test_main_regex_rules_7a_7s_7b_and_addr_port_extension():
    df = pd.DataFrame({"ip_addr": "a", "system_type": 9, "cid": 2, "lid": [1, 1, 3, 2, 4], "alarm_type": 1,
                       "address": ["MAIN-P1", "MAIN-P1 PW-5011", "MAIN-P1", " main-p2 ", "MAIN-PX"]})
    out = pj.classify_alarm_rows(df)
    assert list(out["row_rule"]) == ["port_main", "service_main", "main_lid_mismatch", "port_main", "unit_addr"]
    assert pj.norm_port("CA-UT7080-MAIN-P1") == "MAIN-P1" and pj.norm_port("S.2 -P1  ") == "S.2-P1" and pj.norm_port("MAIN-P3 X") == "MAIN-P3"
    assert pj.norm_port_base("MAIN-P1") == ""                                                    # 규칙 8 의 '빈 값' 판단은 확장 전 기준


def test_rev1_comparison_counts_rows_whose_class_changed():
    df = pd.DataFrame({"ip_addr": "a", "system_type": 1, "address": ["SYSTEM", "S.6", "PAPA.A-FAN-7", "S.2 -P1", "S.1 -P1"],
                       "cid": [256, 6, 3, 3, 0], "lid": [255, 1, 1, 1, 0], "alarm_type": 1})
    rep = pj.row_rule_report(pj.classify_alarm_rows(df))
    assert rep["reclassified_vs_rev1"] == 3                      # SYSTEM(port→device)·S.6(port→slot)·FAN(port→unit)
