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
@pytest.mark.parametrize("addr,exp", [
    ("CA216-UT7320-S.1 -P1", "S.1-P1"), ("S.2 -P1  ", "S.2-P1"), ("S. 3 - P 12", "S.3-P12"), ("A-S.1-P1 S.2 -P4", "S.2-P4"),
    ("CA-S.4", "S.4"), ("nothing", ""), ("", ""), (None, ""), (float("nan"), "")])
def test_norm_port(addr, exp):
    assert pj.norm_port(addr) == exp


def test_classify_alarm_rows():
    df = pd.DataFrame({"ip_addr": [" 1.1.1.1 "] * 7, "cid": [1, 1, 0, 0, None, -1, 2], "lid": [1, 0, 0, 3, 1, 1, None]})
    out = pj.classify_alarm_rows(df)
    assert list(out["row_class"]) == ["port", "slot", "device", "device", "bad_key", "bad_key", "bad_key"]
    assert out["ip_addr"].iloc[0] == "1.1.1.1"


def test_classify_real_data_device_and_slot_forms():
    df = pd.DataFrame({"ip_addr": ["a"] * 5, "cid": [256, 3, 6, 3, 2], "lid": [255, 1, 1, 1, 2],
                       "address": ["SYSTEM", "S.2 -P1  ", "S.6", "PAPA.A-FAN-7", "S.1 -P2"]})
    assert list(pj.classify_alarm_rows(df)["row_class"]) == ["device", "port", "slot", "port", "port"]


def test_cid_offset_scan_shows_shift_without_changing_keys():
    eth = eth_frame([(1, 1), (1, 2), (2, 1)], hours=1)
    al = pd.DataFrame({"ip_addr": ["1.1.1.1"] * 3, "system_type": [12] * 3, "cid": [2, 2, 3], "lid": [1, 2, 1], "address": ["S.1 -P1"] * 3})
    scan = pj.cid_offset_scan(pj.classify_alarm_rows(al), eth)
    assert scan[0]["all"] == pytest.approx(1 / 3) and scan[-1]["all"] == 1.0 and scan[-1]["by_model"]["12"] == 1.0
    off = pj.cid_slot_offsets(eth)
    assert off == {"12": {"0": 3 * 4}}


def test_p5_same_and_addr_port_columns():
    df = pd.DataFrame({"ip_addr": ["a", "a"], "cid": [1, 2], "lid": [1, 1], "p5_cid": [1, 9], "p5_lid": [1, 1], "address": ["S.2 -P1  ", ""]})
    out = pj.add_diag_columns(pj.classify_alarm_rows(df), alarm=True)
    assert list(out["p5_same"]) == [True, False] and list(out["addr_port"]) == ["S.2-P1", ""]


# ---- J1 ------------------------------------------------------------------------------------
def test_j1_one_to_one_and_violations():
    ok = eth_frame([(1, 1), (1, 2), (2, 1)], hours=1)
    r = pj.j1_key_consistency(ok)
    assert r["one_to_one_ratio"] == 1.0 and r["not_one_to_one_count"] == 0 and r["optic_eth_overlap"] is None
    bad = pd.concat([ok, eth_frame([(5, 5)], hours=1, addr_fmt="X-S.1 -P1")])             # 같은 addr_port S.1-P1 이 두 키에
    r = pj.j1_key_consistency(bad)
    assert r["not_one_to_one_count"] == 2 and r["one_to_one_ratio"] == pytest.approx(2 / 4)
    two_addr = ok.copy()
    two_addr.loc[two_addr.index[0], "address"] = "CA1-UT-S.9 -P9"                          # 한 키가 addr_port 둘
    assert pj.j1_key_consistency(two_addr)["not_one_to_one_count"] >= 1


def test_j1_optic_overlap_and_no_address():
    eth = eth_frame([(1, 1), (1, 2)], hours=1)
    optic = eth_frame([(1, 1), (7, 7)], hours=1)
    assert pj.j1_key_consistency(eth, optic)["optic_eth_overlap"] == 0.5
    no = eth.assign(address="")
    r = pj.j1_key_consistency(no)
    assert r["no_address"] == 2 and np.isnan(r["one_to_one_ratio"])


# ---- J3 ------------------------------------------------------------------------------------
def test_grid_bins_alignment_and_containment():
    assert list(pj._grid_bins(ts(10), ts(12))) == [ts(10, 15) + pd.Timedelta(minutes=15 * i) for i in range(8)]   # 10:15 ~ 12:00
    assert list(pj._grid_bins(ts(10, 7), ts(10, 50))) == [ts(10, 30), ts(10, 45)]                                # 완전 포함만
    assert len(pj._grid_bins(ts(10, 1), ts(10, 14))) == 0


def test_f5a_verified_when_keys_match():
    eth = eth_frame([(1, 1), (1, 2), (2, 1)], zero={(1, 1): (ts(10), ts(12))})
    ep = episodes([["1.1.1.1", 1, 1, ts(10), ts(12)]] * 5)
    j3 = pj.j3_physical_match(ep, eth)
    assert j3["with_bins"] == 5 and j3["match_a"] == 1.0 and j3["match_c"] == 0.0 and j3["a_zero"] == 1.0 and j3["a_missing"] == 0.0
    r = pj.judge_join(pj.j1_key_consistency(eth), j3)
    assert r["status"] == "verified" and r["reasons"] == []


def test_f5b_failed_when_alarm_key_is_shifted_and_no_auto_switch_to_address():
    # 성능 쪽 address 슬롯이 키보다 2 어긋난 구성: 알람의 cid=3 이 실제로는 성능 cid=1 포트(rx=0)와 같은 물리 포트라고 가정한 address 조인(d)
    eth = eth_frame([(1, 1), (3, 1)], zero={(1, 1): (ts(10), ts(12))}, addr_fmt="CA1-S.{c} -P{l}")
    eth.loc[eth["cid"] == 1, "address"] = "CA1-S.3 -P1"; eth.loc[eth["cid"] == 3, "address"] = "CA1-S.1 -P1"          # 키와 address 가 엇갈림
    ep = episodes([["1.1.1.1", 3, 1, ts(10), ts(12)]] * 5)
    ep["addr_port"] = "S.3-P1"                                    # 알람 address 는 S.3-P1 (= 성능 cid=1 의 address)
    j3 = pj.j3_physical_match(ep, eth)
    assert j3["match_a"] == 0.0 and j3["match_d"] == 1.0           # 키 조인은 정상 포트(cid=3)를 봐서 0, address 조인은 1
    r = pj.judge_join(pj.j1_key_consistency(eth), j3)
    assert r["status"] == "failed" and r["stop"] is True and r["address_join_would_pass"] is False    # 대조군도 rx=0 이라 d−c < 0.5
    assert "J3 일치율" in " ".join(r["reasons"])


def test_unverified_when_fewer_than_five_episodes():
    eth = eth_frame([(1, 1), (1, 2)], zero={(1, 1): (ts(10), ts(12))})
    j3 = pj.j3_physical_match(episodes([["1.1.1.1", 1, 1, ts(10), ts(12)]] * 4), eth)
    r = pj.judge_join(pj.j1_key_consistency(eth), j3)
    assert r["status"] == "unverified" and "판정 불가" in r["reasons"][0]


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
    assert single["n_control"] == 0 and np.isnan(single["match_c"])
    assert pj.judge_join({"one_to_one_ratio": 1.0}, {**single, "with_bins": 9})["status"] == "unverified"


def test_by_model_breakdown():
    e1 = eth_frame([(1, 1), (1, 2)], zero={(1, 1): (ts(10), ts(12))}, system_type=12)
    e2 = eth_frame([(1, 1), (1, 2)], ip="2.2.2.2", system_type=31)
    ep = episodes([["1.1.1.1", 1, 1, ts(10), ts(12)], ["2.2.2.2", 1, 1, ts(10), ts(12)]])
    bm = pj.j3_physical_match(ep, pd.concat([e1, e2]))["by_model"]
    assert bm["12"]["match_a"] == 1.0 and bm["31"]["match_a"] == 0.0


# ---- 판정 경계값 (사전 고정 기준) ----------------------------------------------------------------
def J(a, c, d=float("nan"), n=5):
    return {"with_bins": n, "match_a": a, "match_c": c, "match_d": d}


@pytest.mark.parametrize("r1,a,c,status", [
    (0.95, 0.8, 0.3, "verified"),                  # 정확히 경계 (a − c = 0.5)
    (0.9499, 0.8, 0.0, "failed"), (0.95, 0.7999, 0.0, "failed"), (0.95, 0.8, 0.3001, "failed"),
    (1.0, 1.0, 0.0, "verified")])
def test_judge_boundaries(r1, a, c, status):
    assert pj.judge_join({"one_to_one_ratio": r1}, J(a, c))["status"] == status


def test_judge_does_not_switch_to_address_join_when_it_would_pass():
    r = pj.judge_join({"one_to_one_ratio": 1.0}, J(0.1, 0.0, d=0.95))
    assert r["status"] == "failed" and r["address_join_would_pass"] is True and r["stop"] is True


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


def test_j4_coverage_by_model_and_samples():
    eth = eth_frame([(1, 1), (1, 2)])
    al = pd.DataFrame({"ip_addr": ["1.1.1.1", "1.1.1.1", "9.9.9.9", "1.1.1.1"], "system_type": [12, 12, 31, 12], "cid": [1, 5, 1, 0], "lid": [1, 5, 1, 0],
                       "alarm_type": [1, 2, 3, 4], "occur_date": [ts(1)] * 4})
    r = pj.j4_coverage(pj.classify_alarm_rows(al), eth)
    assert r["port_alarms"] == 3 and r["matched_ratio"] == pytest.approx(1 / 3) and r["device_coverage"] == pytest.approx(2 / 3)
    assert len(r["unmatched_samples"]) == 2


# ---- M1 ------------------------------------------------------------------------------------
def test_manual_check_stops_on_single_mismatch():
    ok = pd.DataFrame({"ip_addr": ["a", "b", "c"], "system_type": [12, 31, 31], "perf_cid": [1, 2, 3], "perf_lid": [1, 1, 1],
                       "alarm_cid": [1, 2, 3], "alarm_lid": [1, 1, 1]})
    r = pj.check_manual(ok)
    assert not r["stop"] and r["warnings"] == []
    bad = ok.copy(); bad.loc[2, "alarm_cid"] = 4
    r = pj.check_manual(bad)
    assert r["stop"] and len(r["mismatches"]) == 1
    few = pj.check_manual(ok.iloc[:2].assign(system_type=12))
    assert len(few["warnings"]) == 2
    with pytest.raises(ValueError):
        pj.check_manual(ok.drop(columns=["alarm_lid"]))
