"""P1-2 잠금 전 reviewer 점검 수정: 고정 문구(A), service 한정 출력(B), 코드 상수 해시(C), tool_version(D), lock --models-dir 필수(E)"""
import json
import types

import pandas as pd
import pytest

from tests.validation.realdata.test_lock_judge import ctx_of, locked  # noqa: F401
from tests.validation.realdata.test_replay_report import FakeDetector
from tests.validation.realdata.test_snapshot_commands import run  # noqa: F401  (fixture)
from validation.realdata import commands as cmds
from validation.realdata import periods as pr
from validation.realdata import report as rpt

SVC = {"groups": {"service_main": {"n": 203, "ratio": 0.635, "include": True}, "service_port_nonP": {"n": 247, "ratio": 0.644, "include": True},
                  "service_port_P": {"n": 95, "ratio": 0.853, "include": True}},
       "by_suffix": {"PW": {"n": 229, "ratio": 0.672, "include": True, "by_name": {"PW-LOC": {}}},
                     "tp_W": {"n": 18, "ratio": 0.2778, "include": False, "by_name": {"LSP-LOC": {}}},
                     "tp_P": {"n": 95, "ratio": 0.853, "include": True, "by_name": {"LSP-LOC": {}}}},
       "range": ["a", "b"]}


# ---- A: 고정 문구 -------------------------------------------------------------------------------
def test_fixed_notes_have_rev9_adjustment_always_and_explore_note_in_judge(run):
    c = locked(run)
    dev = cmds.cmd_report(c, "dev", "dev", detector=FakeDetector())["header"]["fixed_notes"]
    assert rpt.REV9_NOTE in dev and rpt.EXPLORE_NOTE not in dev
    assert "개발 구간 사후 조정" in rpt.REV9_NOTE and "0.8" in rpt.REV9_NOTE and "열지 않았" in rpt.REV9_NOTE
    jd = cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())["header"]
    assert rpt.REV9_NOTE in jd["fixed_notes"] and rpt.EXPLORE_NOTE in jd["fixed_notes"]
    assert jd["explore_reference"]["used_for_selection"] is False and jd["explore_reference"]["available"] is False      # 탐색 산출물이 없으면 없다고 적는다


def test_explore_reference_is_display_only_and_not_in_config_hash(run, tmp_path):
    c = ctx_of(run)
    h0 = pr.stable_hash(cmds.config_parts(c, "verified"))
    d = tmp_path / "models" / "explore_seed0"
    d.mkdir(parents=True)
    (d / "train_run.json").write_text(json.dumps({"threshold": 1.1, "final_val_loss": 0.06, "training_window": {"end": "2026-08-11 00:00:00"}}), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    jd = str(c.periods["J0"])[:10]
    (out / f"report_dev_2026-08-11_{jd}.json").write_text(json.dumps({"header": {"range": ["x", "y"]}, "c1_raw": {"ai": {"incidents": 86, "port_days": 795, "incidents_per_port_day": 0.108}},
                                                                    "c3": {"합계": {"hold": True, "events_total": 7, "included": 5, "hold_reasons": ["r"]}}}), encoding="utf-8")
    fake = types.SimpleNamespace(runs_dir=str(tmp_path), out_dir=str(out), periods=c.periods)
    ref = cmds.explore_reference(fake)
    assert ref["available"] and ref["used_for_selection"] is False and ref["train_run"]["threshold"] == 1.1 and len(ref["train_run"]["sha256"]) == 64
    assert ref["dev_report"]["c1_raw"]["ai"]["incidents"] == 86 and ref["dev_report"]["c3_total"]["included"] == 5
    assert pr.stable_hash(cmds.config_parts(c, "verified")) == h0                                  # 표기 전용: 해시 불변


# ---- B: service 한정 ----------------------------------------------------------------------------
def test_service_inclusion_header_states_tp_w_limit_from_json(tmp_path):
    p = tmp_path / "svc.json"
    p.write_text(json.dumps(SVC), encoding="utf-8")
    fake = types.SimpleNamespace(service_json=str(p), service_include={"service_main": True, "service_port_nonP": True, "service_port_P": True})
    h = cmds.service_inclusion_header(fake)
    assert h["include_map"]["service_port_nonP"] is True and h["groups"]["service_port_nonP"]["n"] == 247 and h["by_suffix"]["tp_W"]["n"] == 18
    assert len(h["limits"]) == 1 and "tp_W" in h["limits"][0] and "LSP-LOC" in h["limits"][0] and "n=18" in h["limits"][0] and "27.8%" in h["limits"][0]
    fake2 = types.SimpleNamespace(service_json=str(p), service_include={"service_main": True, "service_port_nonP": False, "service_port_P": True})
    assert cmds.service_inclusion_header(fake2)["limits"] == []                                    # 그룹이 제외면 한정 문구 불필요
    missing = cmds.service_inclusion_header(types.SimpleNamespace(service_json=str(tmp_path / "none.json"), service_include={}))
    assert missing["available"] is False


def test_report_header_carries_service_inclusion(run):
    h = cmds.cmd_report(ctx_of(run), "dev", "dev", detector=FakeDetector())["header"]
    assert set(h["service_inclusion"]["include_map"]) == {"service_main", "service_port_nonP", "service_port_P"}


# ---- C: 코드 상수 해시 --------------------------------------------------------------------------
@pytest.mark.parametrize("mod,name,val", [(rpt, "C3_WINDOW", pd.Timedelta(hours=12)), (rpt, "DELAY_WINDOW", pd.Timedelta(hours=3)), (rpt, "C6_BAND", (0.4, 2.5)),
                                          (rpt, "LEAD_BUCKETS", ((15, 30),)), (rpt, "C3_PERF_LOOKBACK", pd.Timedelta(hours=3)), (rpt, "DELAY_PERF_WINDOW", pd.Timedelta(hours=2))])
def test_code_constants_change_config_hash(run, monkeypatch, mod, name, val):
    c = ctx_of(run)
    h0 = pr.stable_hash(cmds.config_parts(c, "verified"))
    monkeypatch.setattr(mod, name, val)
    assert pr.stable_hash(cmds.config_parts(c, "verified")) != h0


def test_replay_and_port_join_constants_change_hash_and_perf_gap_read_from_constants(run, monkeypatch):
    from validation.realdata import port_join as pj
    from validation.realdata import replay as rp
    c = ctx_of(run)
    parts = cmds.config_parts(c, "verified")
    assert parts["perf_gap"]["c3_lookback_h"] == 6 and parts["perf_gap"]["delay_window_h"] == 1                  # 현재 값은 그대로
    h0 = pr.stable_hash(parts)
    for mod, name, val in ((rp, "WARMUP_DAYS", 3), (rp, "ROLLING_MIN_PERIODS", 12), (pj, "J4M_MAX_MISMATCH", 0.3)):
        with monkeypatch.context() as m:
            m.setattr(mod, name, val)
            assert pr.stable_hash(cmds.config_parts(c, "verified")) != h0, name


# ---- D: tool_version ----------------------------------------------------------------------------
def test_lock_records_tool_version_and_judge_warns_on_mismatch(run, monkeypatch, capsys):
    monkeypatch.setattr(cmds, "_tool_version", lambda: "aaa1111")
    c = locked(run)
    lock = pr.load_lock(run["runs"], c.periods)
    assert lock["tool_version"] == "aaa1111"
    capsys.readouterr()
    monkeypatch.setattr(cmds, "_tool_version", lambda: "bbb2222")
    rep = cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())
    assert "aaa1111" in capsys.readouterr().err                                                    # 경고만, 거부 아님
    assert rep["header"]["tool_version"] == "bbb2222" and rep["header"]["tool_version_at_lock"] == "aaa1111"


# ---- E: lock 은 --models-dir 필수 ---------------------------------------------------------------
def test_lock_without_models_dir_is_refused_and_writes_nothing(run):
    c = ctx_of(run)
    cmds.cmd_join_check(c, "dev")
    with pytest.raises(pr.PeriodError, match="--models-dir"):
        cmds.cmd_lock(c)
    assert pr.load_lock(run["runs"], c.periods) is None
