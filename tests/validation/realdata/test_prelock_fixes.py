"""P1-2 잠금 전 reviewer 점검 수정: 고정 문구(A), service 한정 출력(B), 코드 상수 해시(C), tool_version(D), lock --models-dir 필수(E)"""
import json
import types

import pandas as pd
import pytest

from tests.validation.realdata.test_lock_judge import GOOD, ctx_of, locked  # noqa: F401
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
def test_fixed_notes_dev_and_judge_and_explore_not_run(run):
    c = locked(run)
    dev = cmds.cmd_report(c, "dev", "dev", detector=FakeDetector())["header"]["fixed_notes"]
    assert rpt.REV9_NOTE in dev and rpt.EXPLORE_NOTE not in dev and rpt.LSP_NOTE not in dev
    for k in ("J4m", "9s", "7a", "사후 조정", "0.8", "0.773"):
        assert k in rpt.REV9_NOTE
    jd = cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())["header"]
    assert rpt.REV9_NOTE in jd["fixed_notes"] and rpt.LSP_NOTE in jd["fixed_notes"] and rpt.EXPLORE_NOT_RUN_NOTE in jd["fixed_notes"] and rpt.EXPLORE_NOTE not in jd["fixed_notes"]
    assert jd["explore_refs"] is None and "LSP-LOC" in rpt.LSP_NOTE and "n 18" in rpt.LSP_NOTE and "27.8%" in rpt.LSP_NOTE


def _make_explore(run, tmp_runs):
    d = tmp_runs / "models" / "explore_seed0"
    d.mkdir(parents=True)
    (d / "train_run.json").write_text(json.dumps({"threshold": 1.1, "training_window": {"end": "2026-08-11 00:00:00"}}), encoding="utf-8")
    return d


def test_explore_refs_recorded_in_lock_outside_config_hash_and_judge_never_opens_them(run):
    import os
    c = ctx_of(run)
    h0 = pr.stable_hash(cmds.config_parts(c, "verified"))
    d = _make_explore(run, __import__("pathlib").Path(run["runs"]))
    cmds.cmd_join_check(c, "dev")
    cmds.cmd_lock(c, model_infos=json.loads(json.dumps(GOOD)))
    lock = pr.load_lock(run["runs"], c.periods)
    refs = lock["explore_refs"]
    assert refs[0]["path"].endswith("train_run.json") and len(refs[0]["sha256"]) == 64 and "explore_refs" not in lock["config_parts"]
    assert lock["report_notes"]["explore"] == rpt.EXPLORE_NOTE
    assert lock["config_hash"] == pr.stable_hash(cmds.config_parts(c, lock["join_status"])) and pr.stable_hash(cmds.config_parts(c, "verified")) == h0
    ok = cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())["header"]
    assert rpt.EXPLORE_NOTE in ok["fixed_notes"] and ok["explore_refs"] == refs
    (d / "train_run.json").write_text("{손상", encoding="utf-8")                                    # 손상·삭제해도 판정 보고는 성공 (파일을 열지 않음)
    os.remove(d / "train_run.json")
    c2 = ctx_of(run)
    assert pr.load_lock(run["runs"], c2.periods)["explore_refs"] == refs


def test_judge_report_does_not_open_explore_files_nor_carry_explore_numbers(run, monkeypatch):
    import builtins
    c = ctx_of(run)
    d = _make_explore(run, __import__("pathlib").Path(run["runs"]))
    cmds.cmd_join_check(c, "dev")
    cmds.cmd_lock(c, model_infos=json.loads(json.dumps(GOOD)))
    real_open = builtins.open
    opened = []

    def spy(f, *a, **k):
        opened.append(str(f))
        return real_open(f, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    (d / "train_run.json").write_text("{손상", encoding="utf-8")
    rep = cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())
    assert not [f for f in opened if "explore_seed0" in f]
    assert "threshold" not in json.dumps(rep["header"], default=str)             # 탐색 수치(임계치 등)가 판정 출력에 섞이지 않음


def test_judge_refused_when_locked_report_notes_change(run, monkeypatch):
    c = locked(run)
    monkeypatch.setattr(rpt, "REV9_NOTE", rpt.REV9_NOTE + " (수정)")
    with pytest.raises(pr.PeriodError, match="고정 문구"):
        cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())


def test_lsp_loc_event_counts_split_nonP():
    t = pd.Timestamp("2026-09-01 10:00")
    ev = pd.DataFrame([{"t0": t, "end": t + pd.Timedelta(hours=1), "is_judge": True, "alarm_names": "LSP-LOC", "ip_addr": "1.1.1.1", "cid": 1, "lid": 1},
                       {"t0": t, "end": t + pd.Timedelta(hours=1), "is_judge": True, "alarm_names": "LSP-LOC", "ip_addr": "1.1.1.1", "cid": 2, "lid": 1},
                       {"t0": t, "end": t + pd.Timedelta(hours=1), "is_judge": True, "alarm_names": "PW-LOC", "ip_addr": "1.1.1.1", "cid": 3, "lid": 1}])
    al = pd.DataFrame([{"occur_date": t + pd.Timedelta(minutes=5), "ip_addr": "1.1.1.1", "cid": 1, "lid": 1, "alarm_name": "LSP-LOC", "row_rule": "service_port", "suffix_type": "tp_W"},
                       {"occur_date": t + pd.Timedelta(minutes=5), "ip_addr": "1.1.1.1", "cid": 2, "lid": 1, "alarm_name": "LSP-LOC", "row_rule": "service_port", "suffix_type": "tp_P"}])
    r = cmds.lsp_loc_event_counts(ev, al, t - pd.Timedelta(days=1), t + pd.Timedelta(days=1))
    assert r == {"lsp_loc_judge_events": 2, "lsp_loc_service_port_nonP_judge_events": 1}


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


def test_lock_refuses_models_dir_without_train_run_json_and_model_infos_outside_tests(run, tmp_path, monkeypatch):
    c = ctx_of(run)
    cmds.cmd_join_check(c, "dev")
    ops = tmp_path / "models"                                                                       # 운영 models/ 모양 (train_run.json 없음)
    ops.mkdir()
    (ops / "traffic_registry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(pr.PeriodError, match="train_run.json"):
        cmds.cmd_lock(c, models_dir=str(ops))
    monkeypatch.setattr(cmds, "_ALLOW_MODEL_INFOS_FOR_TESTS", False)                                # 운영 기본값: model_infos 우회 불가
    with pytest.raises(pr.PeriodError, match="테스트 전용"):
        cmds.cmd_lock(c, model_infos=json.loads(json.dumps(GOOD)))
    assert pr.load_lock(run["runs"], c.periods) is None
