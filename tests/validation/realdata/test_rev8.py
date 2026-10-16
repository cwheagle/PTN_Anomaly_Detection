"""개정 8: target_map_v2, service_main 조건부 포함, 8.7 순환 고정 문구, perf_gap 첫 줄, ETH-LINK-FAIL 단독 (설계 7.4·8.7)"""
import csv
import json
import os

import numpy as np
import pandas as pd
import pytest

from tests.validation.realdata.test_alarm_episodes import A, TM, names
from tests.validation.realdata.test_lock_judge import GOOD, ctx_of, locked  # noqa: F401
from tests.validation.realdata.test_replay_report import FakeDetector
from tests.validation.realdata.test_snapshot_commands import run  # noqa: F401  (fixture)
from validation.realdata import alarm_episodes as ae
from validation.realdata import commands as cmds
from validation.realdata import periods as pr
from validation.realdata import report as rpt
from validation.realdata.alarm_names import DEFAULT_TARGET_MAP, TARGET_MAP_VERSION, TargetMap

T = pd.Timestamp("2026-09-01 12:00")


# ---- 7.4 target_map_v2 -------------------------------------------------------------------------
def test_target_map_v2_primary_and_aux_rows_and_no_pw_lsp_globs():
    tm = TargetMap.load(DEFAULT_TARGET_MAP)
    assert TARGET_MAP_VERSION == "target_map_v2"
    for n in ("OPT-LOS", "DCC-FAIL", "ETH-LINK-FAIL", "PW-LOC", "LSP-LOC", "ETH-NO-RX-TRAFFIC", "ETH-NO-TX-TRAFFIC", "CLI-LINK-FAIL", "CLI-LINK-LOS",
              "OTU-ETH-LINKDOWN", "OTU-ETH-LF", "OTU-ETH-RF", "CLI-OPT-LOS"):
        assert ("트래픽단절", "1차", ) in [(t, g) for t, g, _ in tm.classify(n)], n
    for n in ("TRK-CONN-MIS", "AUTONEGO-MISMATCH", "LINK-FAIL-LLCF", "ALM-ID-254", "PW-CSF", "PW-RDI", "LSP-RDI"):
        assert [(t, g) for t, g, _ in tm.classify(n)] == [("트래픽단절", "보조")], n
    assert {t for t, _, _ in tm.classify("OPT-LOS")} == {"트래픽단절", "광관찰"}                      # 광관찰 행은 그대로
    assert tm.classify("LSP-ERR")[0][:2] == ("해당없음", "보조") and tm.classify("PW-OV")[0][0] == "해당없음"       # 그 밖의 PW-·LSP- 이름은 5.2 정확한 이름 행
    assert all("*" not in p or not (p.startswith("PW-") or p.startswith("LSP-")) for p in list(tm.exact) + list(tm.globs))      # PW-*·LSP-* 글롭 없음
    rows = list(csv.DictReader(open(DEFAULT_TARGET_MAP, encoding="utf-8")))
    changed = {r["pattern"] for r in rows if r["source"] == "rev8-2026-10-16"}
    assert changed == {"DCC-FAIL", "PW-LOC", "LSP-LOC", "TRK-CONN-MIS", "ALM-ID-254", "PW-CSF", "PW-RDI", "LSP-RDI"}


def test_target_map_loads_against_real_names_without_ambiguity_or_pw_lsp_rule():
    # 개정 7 의 'PW-·LSP- 이름의 단절 매핑 금지' 로드 검사는 개정 8 에서 취소됨 — PW-LOC·LSP-LOC 가 단절로 매핑돼도 로드 오류가 아님
    TargetMap.load(DEFAULT_TARGET_MAP)
    tm = TargetMap([{"pattern": "PW-LOC", "target_type": "트래픽단절", "grade": "1차", "source": "x"}])
    assert tm.classify("PW-LOC")[0][0] == "트래픽단절"


# ---- 설정 해시 ----------------------------------------------------------------------------------
def test_config_hash_includes_map_version_and_service_inclusion(run, tmp_path):
    c = locked(run)
    parts = pr.load_lock(run["runs"], c.periods)["config_parts"]
    assert parts["target_map_version"] == "target_map_v2" and len(parts["target_map_sha"]) == 64
    assert parts["service_inclusion"]["include"] == {"service_main": False, "service_port_nonP": False, "service_port_P": False} and parts["service_inclusion"]["rule"] == [30, 0.5, 12]
    svc = tmp_path / "svc.json"
    svc.write_text(json.dumps({"include": True}), encoding="utf-8")
    c2 = cmds.Context(str(run["rd"]), run["runs"], names_path=run["names"], service_json=str(svc))
    with pytest.raises(pr.PeriodError, match="설정 해시"):
        cmds.cmd_report(c2, "judge", "judge", detector=FakeDetector())


# ---- service_main 조건부 포함 (n ≥ 30 · 비율 ≥ 50%) --------------------------------------------------
def svc_setup(n, zero_after, name="PW-LOC"):
    """service_main 알람 n 건: 앞 zero_after 건은 알람 직후 1시간 성능 행이 없다. 포트 (cid_j=2, lid=1), 알람 cid 3."""
    from tests.validation.realdata.test_port_join import eth_frame
    base = pd.Timestamp("2026-09-01 00:00")
    code = 367 + 12
    nm = names()
    nm.tables["CE9001"][12] = name
    alarm_rows, perf_rows = [], []
    for i in range(n):
        t = base + pd.Timedelta(hours=8 * i)                                  # 알람마다 8시간 간격(서로 독립)
        alarm_rows.append({"ip_addr": "1.1.1.1", "cid": 3, "lid": 1, "system_type": 31, "alarm_type": code, "severity_type": 3,
                           "occur_date": t, "release_date": t + pd.Timedelta(minutes=40), "address": "MAIN-P1 PW-5021"})
        slots = [t - pd.Timedelta(minutes=15 * k) for k in range(1, 25)]       # 직전 6h 정상 수집
        if i >= zero_after:
            slots += [t + pd.Timedelta(minutes=15 * k) for k in range(1, 5)]   # 직후 1h 행 있음
        perf_rows += [{"ip_addr": "1.1.1.1", "cid": 2, "lid": 1, "bin_end": s, "occur_date": s} for s in slots]
    a = ae.prepare_alarms(pd.DataFrame(alarm_rows), nm, TM, pd.Timestamp("2026-10-13"))
    return a, pd.DataFrame(perf_rows)


@pytest.mark.parametrize("n,zero,include", [(30, 15, True), (30, 14, False), (29, 29, False), (40, 40, True), (30, 30, True)])
def test_service_main_inclusion_rule_boundaries(n, zero, include):
    a, perf = svc_setup(n, zero)
    assert (a["row_rule"] == "service_main").all() and (a["row_class"] == "service").all()
    m = ae.measure_service_main(a, perf, TM)
    assert m["n"] == n and m["include"] is include and m["rule"] == "n >= 30 and ratio >= 0.5"
    assert m["by_name"]["PW-LOC"]["n"] == n and m["decision"].startswith("포함" if include else "보조")


def test_service_main_non_primary_names_are_not_measured_or_included():
    a, perf = svc_setup(40, 40, name="PW-OV")                               # 5.2 해당없음 이름
    assert ae.measure_service_main(a, perf, TM)["n"] == 0


def test_included_service_rows_become_port_alarms_with_corrected_key_else_stay_service():
    a_off, _ = svc_setup(3, 3)
    a_on = ae.prepare_alarms(pd.DataFrame([{**r} for r in a_off[["ip_addr", "cid_raw", "lid", "system_type", "alarm_type", "severity_type", "occur_date", "release_date", "address"]]
                                           .rename(columns={"cid_raw": "cid"}).to_dict("records")]), names_with("PW-LOC"), TM, pd.Timestamp("2026-10-13"), None, True)
    assert set(a_on["row_class"]) == {"port"} and set(a_on["cid"]) == {2} and set(a_on["row_rule"]) == {"service_main"}
    ep = ae.build_episodes(a_on[a_on["row_class"] == "port"], TM, pd.Timestamp("2026-10-13"))
    assert len(ep) == 3                                                                           # 8시간 간격 3건 → 에피소드 3개
    assert set(a_off["row_class"]) == {"service"} and set(a_off["cid"]) == {3}                     # 포함하지 않으면 service(참고) 로 남고 키는 원 값


def names_with(name):
    nm = names()
    nm.tables["CE9001"][12] = name
    return nm


# ---- 8.7 고정 문구·지연 표 첫 줄·ETH-LINK-FAIL 단독 ------------------------------------------------------
def ev(names_, status_n=()):
    return pd.DataFrame({"status": list(status_n), "alarm_names": list(names_)})


def test_c3_summary_counts_eth_link_fail_only_and_has_perf_gap_note():
    rows = pd.DataFrame({"status": ["ok", "ok", "perf_gap", "ok"], "alarm_names": ["ETH-LINK-FAIL", "ETH-LINK-FAIL;OPT-LOS", "ETH-LINK-FAIL", "OPT-LOS"],
                         "ai_lead_min": 30.0, "fixed_lead_min": np.nan, "rolling_lead_min": np.nan})
    s = rpt.c3_summary(rows, None)
    assert s["eth_link_fail_only"]["total"] == 2 and s["eth_link_fail_only"]["included"] == 1 and rpt.PERF_GAP_NOTE in s["notes"]
    assert "단절 정답 알람은 개발 구간에서 '알람 직후 성능 데이터가 사라지는 비율'로 골랐다" in rpt.PERF_GAP_NOTE
    assert "부분적으로 같은 현상을 본다" in rpt.PERF_GAP_NOTE


def test_delay_summary_first_line_is_perf_gap_ratio_and_counts_eth_link_fail_only():
    cols = {"ai_wall_min": np.nan, "fixed_wall_min": np.nan, "rolling_wall_min": np.nan, "ai_step": np.nan, "fixed_step": np.nan, "rolling_step": np.nan,
            "ai_pre15": False, "fixed_pre15": False, "rolling_pre15": False}
    dr = pd.DataFrame([{"status": s, "alarm_names": n, **cols} for s, n in
                       (("perf_gap", "OPT-LOS"), ("perf_gap", "OPT-LOS"), ("perf_gap", "ETH-LINK-FAIL"), ("ok", "ETH-LINK-FAIL"))])
    s = rpt.delay_summary(dr, "verified")
    assert s["first_line"].startswith("perf_gap 3/4 (75.0%)") and s["perf_gap_ratio"] == 0.75
    assert s["eth_link_fail_only"] == {"total": 2, "included": 1} and rpt.PERF_GAP_NOTE in s["notes"]
    assert list(s)[0] == "first_line"                                                                       # 표의 첫 줄


def test_report_header_and_join_check_carry_fixed_notes(run, capsys):
    c = ctx_of(run)
    res = cmds.cmd_join_check(c, "dev")
    assert res["target_map_version"] == "target_map_v2" and res["j3_note"] == rpt.PERF_GAP_NOTE and "service_main" in res
    rep = cmds.cmd_report(c, "dev", "dev", detector=FakeDetector())
    notes = rep["header"]["fixed_notes"]
    assert rpt.PERF_GAP_NOTE in notes and "키 보정은 데이터 일관성으로만 확인, 현장 대조 미실시" in notes
    assert rep["delay"]["first_line"].startswith("perf_gap") and rep["c3"]["합계"]["notes"] == [rpt.PERF_GAP_NOTE]


def test_write_offsets_also_writes_service_main_inclusion_json(run, tmp_path):
    svc = tmp_path / "svc.json"
    off = tmp_path / "off.csv"
    c = cmds.Context(str(run["rd"]), run["runs"], names_path=run["names"], offset_csv=str(off), service_json=str(svc))
    res = cmds.cmd_join_check(c, "dev", write_offsets=True)
    d = json.load(open(svc, encoding="utf-8"))
    assert d["include"] is False and d["n"] == 0 and d["range"] and res["service_main"]["decision"].startswith("보조")
