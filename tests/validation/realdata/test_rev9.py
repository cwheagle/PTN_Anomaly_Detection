"""개정 9: J4 분모(규칙 9·9s), J4m(MAIN-Pn 주소 기반), 9s service_port 조건부 포함 (설계 6.1·6.3·8.7, F-18)"""
import json

import numpy as np
import pandas as pd
import pytest

from tests.validation.realdata.test_alarm_episodes import TM, names
from tests.validation.realdata.test_port_join import eth_frame
from validation.realdata import alarm_episodes as ae
from validation.realdata import commands as cmds
from validation.realdata import port_join as pj


def alarms(rows):
    """rows: (system_type, alarm cid, lid, address). ip 1.1.1.1."""
    return pd.DataFrame([{"ip_addr": "1.1.1.1", "system_type": st, "cid": c, "lid": l, "address": ad, "alarm_type": 1, "occur_date": pd.Timestamp("2026-09-01")}
                         for st, c, l, ad in rows])


def prepared(rows):
    return pj.apply_offsets(pj.classify_alarm_rows(alarms(rows)))


# ---- 9s 분류 ------------------------------------------------------------------------------------
@pytest.mark.parametrize("addr,rule,stype", [("S.2 -P1 PW-1234", "service_port", "PW"), ("S.2 -P1  tp_MA_CA_01(P)", "service_port", "tp_P"),
                                            ("S.2 -P1  tp_MA_CA_01(W)", "service_port", "tp_W"), ("S.2 -P1  etc", "service_port", "other"),
                                            ("S.2 -P1  ", "port", ""), ("S.2 -P1", "port", "")])
def test_rule_9s_service_port_and_suffix_types(addr, rule, stype):
    out = pj.classify_alarm_rows(alarms([(12, 3, 1, addr)]))
    assert out["row_rule"].iloc[0] == rule and out["suffix_type"].iloc[0] == stype
    assert out["row_class"].iloc[0] == ("service" if rule == "service_port" else "port") and pj.ROW_CLASS_VERSION == "row_class_v5"
    assert pj.RULE_IDS.index("service_port") < pj.RULE_IDS.index("port")                          # 9s 가 9 보다 먼저


# ---- J4: 분모 = 규칙 9·9s (F-18) -----------------------------------------------------------------
def f18():
    """모델 X: 규칙 9 알람 10건(9건 성능 키 있음) + 성능 없는 MAIN-P3~P8 7a 알람 20건 (+ 규칙 10 2건)."""
    eth = eth_frame([(c, 1) for c in range(1, 10)] + [(1, 1), (1, 2)], hours=1, ip="1.1.1.1", addr_fmt="CA-S.{c} -P{l}")
    eth = pd.concat([eth, eth_frame([(1, 1), (1, 2)], hours=1, ip="1.1.1.1").assign(address=["CA-UT7080-MAIN-P1" if l == 1 else "CA-UT7080-MAIN-P2" for l in
                                                                                       eth_frame([(1, 1), (1, 2)], hours=1)["lid"]])]).drop_duplicates()
    rows = [(5, c + 1, 1, f"CA-S.{c} -P1") for c in range(1, 10)] + [(5, 99, 1, "CA-S.99 -P1")]            # 규칙 9 10건 중 9건 매칭 (알람 cid = 성능 + 1)
    rows += [(5, 2, lid, f"MAIN-P{lid}") for lid in range(3, 9) for _ in range(3)] + [(5, 2, 3, "MAIN-P3")] * 2   # 7a 20건 (성능 없는 MAIN-P3~P8)
    rows += [(5, 4, 1, ""), (5, 77, 1, "")]                                                                # 규칙 10 2건
    return prepared(rows), eth


def test_j4_denominator_is_rule_9_and_9s_only_and_7a_is_reported_separately():
    a, eth = f18()
    assert (a["row_rule"] == "port_main").sum() == 20
    j4 = pj.j4_coverage(a, eth)
    assert j4["rules"] == ["port", "service_port"] and j4["port_alarms"] == 10 and j4["port_alarms_in_perf_devices"] == 10
    assert j4["matched_ratio"] == pytest.approx(0.9)                                                         # 7a 는 분모 제외
    assert j4["rule10_ref"]["rows_in_perf_devices"] == 2 and j4["rule10_ref"]["matched_ratio"] == pytest.approx(0.5) and "참고" in j4["rule10_ref"]["note"]


def test_j4_includes_9s_service_port_rows_in_denominator():
    eth = eth_frame([(1, 1)], hours=1, ip="1.1.1.1", addr_fmt="CA-S.{c} -P{l}")
    a = prepared([(5, 2, 1, "CA-S.1 -P1"), (5, 2, 1, "CA-S.1 -P1 PW-1234"), (5, 9, 1, "CA-S.9 -P1 tp_x(P)")])
    j4 = pj.j4_coverage(a, eth)
    assert j4["port_alarms"] == 3 and j4["matched_ratio"] == pytest.approx(2 / 3)                            # 9s(승격 전 service 클래스)도 키 검사 대상, cid_j 로 조인


def test_scan_and_model_rates_use_rules_9_and_9s():
    a, eth = f18()
    scan = pj.offset_scan(a, eth)
    assert scan[5]["n_alarms"] == 10 and scan[5]["ks"][1]["j4"] == pytest.approx(0.9)
    assert pj.model_join_rates(a, eth)[5]["j4"] == pytest.approx(0.9)


# ---- J4m ------------------------------------------------------------------------------------------
def test_j4m_distinguishes_no_perf_from_key_mismatch_and_excludes_models_over_02():
    a, eth = f18()
    m = pj.j4m(a, eth)["by_model"]["5"]
    # 성능 주소에 MAIN-P1·P2 만 있음: 7a 의 MAIN-P3~P8 은 주소로 확인된 성능 없음 → port_no_perf, 키가 맞는지는 판단 대상 아님
    assert m["port_no_perf"] == 20 and m["matched"] == 0 and m["key_mismatch"] == 0 and m["excluded"] is False
    assert np.isnan(m["mismatch_ratio"])
    # 성능에 MAIN-P1(cid 1, lid 1), MAIN-P2(cid 1, lid 2): 알람 cid 2 → cid_j 1
    ok = prepared([(5, 2, 1, "MAIN-P1"), (5, 2, 2, "MAIN-P2")])
    r = pj.j4m(ok, eth)
    assert r["by_model"]["5"]["matched"] == 2 and r["excluded_models"] == []
    bad = prepared([(5, 2, 1, "MAIN-P1")] * 2 + [(5, 6, 1, "MAIN-P1")] * 1)                                    # 키가 다른(cid_j 5) MAIN-P1 알람 1/3
    rb = pj.j4m(bad, eth)["by_model"]["5"]
    assert rb["matched"] == 2 and rb["key_mismatch"] == 1 and rb["mismatch_ratio"] == pytest.approx(1 / 3) and rb["excluded"] is True
    edge = prepared([(5, 2, 1, "MAIN-P1")] * 4 + [(5, 6, 1, "MAIN-P1")] * 1)                                   # 정확히 0.2 → 넘지 않음
    assert pj.j4m(edge, eth)["by_model"]["5"]["mismatch_ratio"] == pytest.approx(0.2) and pj.j4m(edge, eth)["excluded_models"] == []


def test_join_stays_verified_when_only_7a_has_no_perf(tmp_path):
    a, eth = f18()
    rates = {5: {"j4": pj.model_join_rates(a, eth)[5]["j4"], "j2": 1.0, "j2_pairs": 9}}
    j1 = pj.j1_key_consistency(eth)
    r = pj.judge_join({**j1, "one_to_one_ratio": 1.0}, {"diag_alive": {"episodes": 0}}, rates)
    assert r["status"] == "verified"                                                                          # 개정 9: 성능 없는 MAIN-P3~P8 이 조인 전체를 failed 로 만들지 않는다


# ---- 7a 이벤트 제외 (판정 구간 해석) ----------------------------------------------------------------------
def test_events_carry_rules_so_7a_events_can_be_excluded():
    raw = alarms([(31, 2, 1, "MAIN-P1"), (31, 3, 1, "S.2 -P1")]).assign(alarm_type=367 + 1, severity_type=3, release_date=pd.Timestamp("2026-09-01 01:00"))
    a = ae.prepare_alarms(raw, names(), TM, pd.Timestamp("2026-10-13"))
    ep = ae.build_episodes(a[a["row_class"] == "port"], TM, pd.Timestamp("2026-10-13"))
    assert sorted(ep["rules"]) == ["port", "port_main"]
    ev = ae.build_target_events(ep)
    assert sorted(ev["rules"]) == ["port", "port_main"]
    keep = ev[~ev["rules"].str.contains("port_main")]                                                       # 7a 이벤트만 제외하는 필터
    assert len(keep) == 1 and keep["rules"].iloc[0] == "port"


# ---- 9s 조건부 포함 측정 (그룹별, (P) 분리) -----------------------------------------------------------------
def svc_alarms_and_perf(spec):
    """spec: [(suffix address, 개수, 직후 공백 개수)] 알람은 8시간 간격, 포트 (cid_j 2, lid 1)."""
    nm = names()
    nm.tables["CE9001"][12] = "PW-LOC"
    rows, perf = [], []
    base = pd.Timestamp("2026-09-01")
    i = 0
    for addr, n, zero in spec:
        for j in range(n):
            t = base + pd.Timedelta(hours=8 * i)
            i += 1
            rows.append({"ip_addr": "1.1.1.1", "cid": 3, "lid": 1, "system_type": 31, "alarm_type": 367 + 12, "severity_type": 3, "occur_date": t,
                         "release_date": t + pd.Timedelta(minutes=40), "address": addr})
            slots = [t - pd.Timedelta(minutes=15 * k) for k in range(1, 25)]
            if j >= zero:
                slots += [t + pd.Timedelta(minutes=15 * k) for k in range(1, 5)]
            perf += [{"ip_addr": "1.1.1.1", "cid": 2, "lid": 1, "bin_end": s, "occur_date": s} for s in slots]
    return ae.prepare_alarms(pd.DataFrame(rows), nm, TM, pd.Timestamp("2026-10-13")), pd.DataFrame(perf)


def test_9s_groups_are_measured_separately_with_p_suffix_split():
    a, perf = svc_alarms_and_perf([("S.2 -P1  PW-5021", 40, 30), ("S.2 -P1  tp_A_B_01(W)", 20, 20), ("S.2 -P1  tp_A_B_01(P)", 35, 5), ("MAIN-P1 PW-5011", 31, 31)])
    m = ae.measure_service_main(a, perf, TM)
    g = m["groups"]
    assert g["service_port_nonP"]["n"] == 60 and g["service_port_nonP"]["ratio"] == pytest.approx(50 / 60) and g["service_port_nonP"]["include"] is True
    assert g["service_port_P"]["n"] == 35 and g["service_port_P"]["ratio"] == pytest.approx(5 / 35) and g["service_port_P"]["include"] is False        # (P) 따로 판단 → 보조
    assert g["service_main"]["n"] == 31 and g["service_main"]["include"] is True and m["n"] == 31          # 최상위 값은 service_main 그룹 (이전 호환)
    assert m["service_port_all"]["n"] == 95 and m["include_map"] == {"service_main": True, "service_port_nonP": True, "service_port_P": False}
    assert {k: v["n"] for k, v in m["by_suffix"].items()} == {"PW": 40, "tp_W": 20, "tp_P": 35}
    assert m["by_suffix"]["tp_W"]["ratio"] == 1.0 and m["by_suffix"]["PW"]["ratio"] == pytest.approx(30 / 40)


def test_9s_group_with_n_below_30_stays_aux():
    a, perf = svc_alarms_and_perf([("S.2 -P1  PW-5021", 29, 29)])
    m = ae.measure_service_main(a, perf, TM)
    assert m["groups"]["service_port_nonP"]["n"] == 29 and m["groups"]["service_port_nonP"]["include"] is False and m["include_map"]["service_port_nonP"] is False


def test_included_9s_rows_are_promoted_to_port_with_corrected_key_others_stay_service():
    nm = names()
    nm.tables["CE9001"][12] = "PW-LOC"
    raw = alarms([(31, 3, 1, "S.2 -P1  PW-5021"), (31, 3, 1, "S.2 -P1  tp_x(P)"), (31, 3, 1, "S.2 -P1  tp_y(W)")]).assign(
        alarm_type=367 + 12, severity_type=3, release_date=pd.Timestamp("2026-09-01 01:00"))
    off = ae.prepare_alarms(raw, nm, TM, pd.Timestamp("2026-10-13"))
    assert set(off["row_class"]) == {"service"} and set(off["cid"]) == {3}
    on = ae.prepare_alarms(raw, nm, TM, pd.Timestamp("2026-10-13"), None, {"service_port_nonP": True, "service_port_P": False})
    by = dict(zip(on["suffix_type"], zip(on["row_class"], on["cid"])))
    assert by["PW"] == ("port", 2) and by["tp_W"] == ("port", 2) and by["tp_P"] == ("service", 3)             # (P) 는 따로 보조


def test_context_reads_include_map_and_old_single_flag(tmp_path):
    svc = tmp_path / "svc.json"
    svc.write_text(json.dumps({"include": True}), encoding="utf-8")                                             # 개정 8 형식
    assert cmds.Context._load_service_include(str(svc)) == {"service_main": True, "service_port_nonP": False, "service_port_P": False}
    svc.write_text(json.dumps({"include_map": {"service_port_nonP": True}}), encoding="utf-8")
    assert cmds.Context._load_service_include(str(svc)) == {"service_main": False, "service_port_nonP": True, "service_port_P": False}
    assert cmds.Context._load_service_include(str(tmp_path / "none.json")) == {g: False for g in ae.SERVICE_GROUPS}
