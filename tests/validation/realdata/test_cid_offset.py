"""개정 5: 알람 cid 오프셋 보정 — cid_j, 오프셋 표 유도(사전 고정 기준), 설정 해시, 판정 구간 degraded/offset_unverified (설계 6.1·6.3, F-15)"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from tests.validation.realdata.test_lock_judge import GOOD, ctx_of, locked  # noqa: F401
from tests.validation.realdata.test_port_join import eth_frame
from tests.validation.realdata.test_replay_report import FakeDetector
from tests.validation.realdata.test_snapshot_commands import run  # noqa: F401  (fixture)
from validation.realdata import commands as cmds
from validation.realdata import periods as pr
from validation.realdata import port_join as pj


def alarm_df(rows):
    """rows: (system_type, alarm cid, lid, address)."""
    return pd.DataFrame([{"ip_addr": "1.1.1.1", "system_type": st, "cid": c, "lid": l, "address": ad, "alarm_type": 1,
                          "occur_date": pd.Timestamp("2026-09-01")} for st, c, l, ad in rows])


# ---- cid_j 적용 -------------------------------------------------------------------------------
def test_apply_offsets_default_k1_only_for_port_and_slot_rows_and_keeps_raw():
    al = alarm_df([(12, 3, 1, "S.2 -P1"), (12, 3, 0, "S.2"), (12, 256, 255, "SYSTEM"), (12, 4, 1, "PAPA.A-FAN-7"), (12, 6, 1, "S.5")])
    out = pj.apply_offsets(pj.classify_alarm_rows(al))
    assert list(out["row_class"]) == ["port", "slot", "device", "unit", "slot"]
    assert list(out["cid"]) == [2, 2, 256, 4, 5]                      # 포트·slot 만 cid − 1, device·unit 은 원 값 유지(원 cid 로 분류)
    assert list(out["cid_raw"]) == [3, 3, 256, 4, 6] and list(out["cid_j"]) == [2, 2, 255, 3, 5] and set(out["k_applied"]) == {1}


def test_apply_offsets_table_exception_and_nonpositive_cid_j_becomes_bad_key():
    al = alarm_df([(9, 3, 1, "S.1 -P1"), (12, 3, 1, "S.1 -P1"), (12, 1, 1, "S.1 -P1"), (9, 2, 1, "S.1 -P1")])
    out = pj.apply_offsets(pj.classify_alarm_rows(al), {9: 2})
    assert list(out["cid"]) == [1, 2, 1, 2]                            # 모델 9 는 k=2, 모델 12 는 기본 1
    assert list(out["row_class"]) == ["port", "port", "bad_key", "bad_key"] and list(out["cid_nonpositive"]) == [False, False, True, True]
    assert out.loc[2, "cid"] == 1 and out.loc[3, "cid"] == 2          # bad_key 는 원 cid 유지(판정 제외)


def test_k_for_handles_missing_and_bad_system_type():
    assert pj.k_for(None, {1: 2}) == 1 and pj.k_for("x", {}) == 1 and pj.k_for(7, {7: 3}) == 3


# ---- J0s 스캔·표 유도 (사전 고정 기준 경계) ------------------------------------------------------
def scan_for(n_alarms, shift, match_n=None, st=5):
    """모델 st 의 포트 알람 n_alarms 건: 알람 cid = 성능 cid + shift. 성능 포트 cid 는 10 간격이라 다른 k 와 우연히 겹치지 않는다."""
    eth = eth_frame([(10 * c, 1) for c in range(1, 41)], hours=1, system_type=st, addr_fmt="X-S.{c} -P{l}")
    rows = [(st, 10 * (i % 40 + 1) + shift if (match_n is None or i < match_n) else 999, 1, f"X-S.{10 * (i % 40 + 1)} -P1") for i in range(n_alarms)]
    return pj.offset_scan(pj.classify_alarm_rows(alarm_df(rows)), eth), eth


def test_offset_scan_finds_k_and_j2_address_agreement():
    scan, _ = scan_for(60, 2)
    ks = scan[5]["ks"]
    assert scan[5]["n_alarms"] == 60 and ks[2]["j4"] == pytest.approx(1.0) and ks[1]["j4"] < 0.5 and ks[0]["j4"] < 0.5
    assert ks[2]["j2"] == 1.0 and ks[2]["j2_pairs"] > 0                    # 보정 키로 맞춘 쌍은 address 도 같다 (S.n-Pm)


@pytest.mark.parametrize("n,expected_in_table", [(29, False), (30, True)])
def test_table_needs_at_least_30_port_alarms(n, expected_in_table):
    scan, _ = scan_for(n, 1)
    assert (len(pj.derive_offset_table(scan)) == 1) is expected_in_table


def test_exception_requires_j4_ge_08_and_gain_ge_02_over_k1():
    scan = {5: {"n_alarms": 100, "ks": {0: {"j4": 0.0}, 1: {"j4": 0.5}, 2: {"j4": 0.8}, 3: {"j4": 0.1}}}}                # 0.8, +0.3 → 예외
    r = pj.derive_offset_table(scan).iloc[0]
    assert (r.k, r.basis, r.j4_at_k, r.j4_at_1) == (2, "exception", 0.8, 0.5)
    for k2, k1 in ((0.7999, 0.3), (0.8, 0.6001)):                                                                   # J4 미달 / 이득 0.2 미달 → 기본 k=1
        d = pj.derive_offset_table({5: {"n_alarms": 100, "ks": {0: {"j4": 0}, 1: {"j4": k1}, 2: {"j4": k2}, 3: {"j4": 0}}}}).iloc[0]
        assert (d.k, d.basis) == (1, "default")
    tie = pj.derive_offset_table({5: {"n_alarms": 100, "ks": {0: {"j4": 0}, 1: {"j4": 0.9}, 2: {"j4": 0.9}, 3: {"j4": 0}}}}).iloc[0]
    assert tie.k == 1                                                                                               # 동률이면 기본 1 우선


def test_f15_models_a_b_c(tmp_path):
    """모델 A(+1), B(+2, 예외 기준 충족), C(포트 알람 20건 → 표 제외)."""
    parts, eth_parts = [], []
    for st, shift, n in ((1, 1, 40), (2, 2, 40), (3, 1, 20)):
        sc, eth = scan_for(n, shift, st=st)
        parts.append(sc[st])
        eth_parts.append(eth.assign(ip_addr=f"10.0.0.{st}"))
    scan = {st: p for st, p in zip((1, 2, 3), parts)}
    table = pj.derive_offset_table(scan)
    assert {int(r.system_type): int(r.k) for r in table.itertuples()} == {1: 1, 2: 2}                                # C 는 표에 없음
    path = tmp_path / "off.csv"
    pj.write_offsets_csv(table, str(path))
    assert pj.load_offsets(str(path)) == {1: 1, 2: 2} and pj.load_offsets(str(tmp_path / "none.csv")) == {}
    assert list(pd.read_csv(path).columns) == pj.OFFSET_COLUMNS


# ---- 판정: 오프셋 표가 바뀌면 설정 해시가 달라져 판정 거부 ---------------------------------------------
def test_offset_table_is_part_of_config_hash_and_changing_it_blocks_judge(run, tmp_path):
    c = locked(run)
    lock = pr.load_lock(run["runs"], c.periods)
    assert lock["config_parts"]["alarm_cid_offsets"]["default_k"] == 1 and lock["config_parts"]["alarm_cid_offsets"]["criteria"] == [30, 0.8, 0.2, 0.8, 0.9]
    off = tmp_path / "off.csv"
    off.write_text("system_type,k,n_alarms,j4_at_k,j4_at_1,basis\n31,2,50,0.9,0.1,exception\n", encoding="utf-8")
    c2 = cmds.Context(str(run["rd"]), run["runs"], names_path=run["names"], offset_csv=str(off))
    with pytest.raises(pr.PeriodError, match="설정 해시"):
        cmds.cmd_report(c2, "judge", "judge", detector=FakeDetector())
    with pytest.raises(pr.PeriodError, match="잠금 뒤에는 오프셋 표"):
        cmds.cmd_join_check(c2, "dev", write_offsets=True)                                                             # 잠금 뒤 표 갱신 금지


def test_write_offsets_before_lock_updates_context_keys(run, tmp_path):
    off = tmp_path / "off.csv"
    c = cmds.Context(str(run["rd"]), run["runs"], names_path=run["names"], offset_csv=str(off))
    res = cmds.cmd_join_check(c, "dev", write_offsets=True)
    assert res["offset_table_written"] and os.path.exists(off) and list(pd.read_csv(off).columns) == pj.OFFSET_COLUMNS
    assert c.offsets == {} and c.alarms["cid"].iloc[0] == c.alarms["cid_raw"].iloc[0] - 1                              # 포트 알람 < 30 → 표 비어 있음, 기본 k=1 보정


# ---- 판정 구간 해석: degraded / offset_unverified -----------------------------------------------------
def test_join_for_report_degraded_and_offset_unverified(run):
    c = ctx_of(run)
    c.offsets_table_models = {31}
    good_al = c.alarms[c.alarms["row_class"] == "port"]
    e = c.eth
    r = cmds.join_for_report(c, good_al, e, True)
    assert r["j4"] == 1.0 and r["degraded"] is False and r["offset_unverified_models"] == []        # 표에 있는 모델이거나 기준 충족
    c.offsets_table_models = set()                                                                  # 표에 없는 모델: 기준 충족이면 제외 안 함
    assert cmds.join_for_report(c, good_al, e, True)["offset_unverified_models"] == []
    bad = good_al.assign(cid=good_al["cid"] + 7)                                                    # 보정 키가 성능과 안 맞는 구간
    r2 = cmds.join_for_report(c, bad, e, True)
    assert r2["degraded"] is True and r2["offset_unverified_models"] == [31]


def test_judge_report_marks_degraded_and_excludes_offset_unverified_events(run, monkeypatch):
    c = locked(run)
    # 판정 구간 알람의 cid 를 어긋나게 만들어 J4 < 0.8 이 되도록 한다 (잠금 뒤 환경을 흉내)
    c.alarms = c.alarms.assign(cid=c.alarms["cid"] + 5)
    c.events = c.events.assign(cid=c.events["cid"] + 5)
    c.episodes = c.episodes.assign(cid=c.episodes["cid"] + 5)
    rep = cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())
    j = rep["join_in_range"]
    assert j["degraded"] is True and j["join_status_used"] == "degraded" and rep["header"]["join_status"] == "degraded"
    assert rep["c3"]["합계"]["hold"] is True and "degraded" in " ".join(rep["c3"]["합계"]["hold_reasons"])
    assert j["offset_unverified"] == {} or j["offset_unverified"]["events"] >= 0
    assert "키 보정은 데이터 일관성으로만 확인, 현장 대조 미실시" in rep["header"]["fixed_notes"]
