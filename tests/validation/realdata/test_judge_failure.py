"""개정 4 ②: 판정 실행 실패 처리 — 표식 전 로드, phase(claimed→computing→completed), aborted_<n>, retry-judge 1회 (설계 9.7.6, F-14)"""
import json
import os
import shutil

import pandas as pd
import pytest

from tests.validation.realdata.test_lock_judge import GOOD, ctx_of, locked  # noqa: F401
from tests.validation.realdata.test_replay_report import FakeDetector
from tests.validation.realdata.test_snapshot_commands import run  # noqa: F401  (fixture)
from validation.realdata import commands as cmds
from validation.realdata import periods as pr
from validation.realdata import replay as rp
from validation.realdata.alarm_names import DEFAULT_TARGET_MAP


def marker(run, c, suffix=None):
    p = pr.run_marker_path(run["runs"], c.periods) if suffix is None else pr.lock_path(run["runs"], c.periods).replace(".json", f".{suffix}.json")
    return json.load(open(p)) if os.path.exists(p) else None


class Boom(FakeDetector):
    def detect(self, *a, **k):
        raise RuntimeError("detect failed")


def judge(c, det):
    return cmds.cmd_report(c, "judge", "judge", detector=det)


def test_completed_marker_has_phase_and_output_hashes_and_blocks_rerun(run):
    c = locked(run)
    judge(c, FakeDetector())
    m = marker(run, c)
    assert m["phase"] == "completed" and m["attempt"] == 1 and m["tool_commit"] and len(m["outputs"]) == 5
    assert all(len(h) == 64 for h in m["outputs"].values()) and any(f.startswith("report_judge_judge") for f in m["outputs"])
    with pytest.raises(pr.PeriodError, match="phase=completed.*다시 실행하지 않습니다"):
        judge(c, FakeDetector())
    with pytest.raises(pr.PeriodError, match="완료된 판정"):
        pr.retry_judge(run["runs"], c.periods, "x", "rev", "abc")


def test_model_load_failure_happens_before_marker_and_ledger(run, monkeypatch):
    base = {"track": "traffic", "version": "v9", "trained_at": "2020-01-01 00:00:00", "training_window": None, "alert_policy": None,
            "model_sha": "m", "scaler_sha": "s", "meta_sha": "j"}
    monkeypatch.setattr(cmds, "model_info", lambda ft, v=None: dict(base))
    c = ctx_of(run)
    cmds.cmd_join_check(c, "dev")
    cmds.cmd_lock(c)
    n_ledger = len(pr.read_ledger(run["runs"]))

    def bad_build(versions=None, policy=None):
        raise RuntimeError("model file corrupted")

    monkeypatch.setattr(rp, "build_detector", bad_build)
    with pytest.raises(RuntimeError, match="corrupted"):
        cmds.cmd_report(c, "judge", "judge")
    assert marker(run, c) is None and len(pr.read_ledger(run["runs"])) == n_ledger                  # 표식·장부 없음 → 그냥 다시 실행
    monkeypatch.setattr(rp, "build_detector", lambda versions=None, policy=None: FakeDetector())
    assert cmds.cmd_report(c, "judge", "judge")["header"]["purpose"] == "judge"
    assert marker(run, c)["phase"] == "completed"


@pytest.mark.parametrize("exc", [RuntimeError("disk"), KeyboardInterrupt()])
def test_claimed_phase_failure_renames_marker_to_aborted_and_allows_rerun(run, monkeypatch, exc):
    c = locked(run)
    real = rp.traffic_frame
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise exc
        return real(*a, **k)

    monkeypatch.setattr(rp, "traffic_frame", flaky)
    with pytest.raises(type(exc)):
        judge(c, FakeDetector())
    assert marker(run, c) is None
    ab = marker(run, c, "aborted_1")
    assert ab["phase"] == "aborted" and ab["error"] and ab["config_hash"]                              # 삭제가 아니라 이름 변경, 오류 포함
    rep = judge(c, FakeDetector())                                                                      # 자동 재실행 허용
    assert rep["header"]["purpose"] == "judge" and marker(run, c)["phase"] == "completed"
    assert len(pr.count_variants(run["runs"], c.periods, "aborted")) == 1


def test_computing_failure_is_not_auto_reverted_and_retry_judge_allows_exactly_one_retry(run):
    c = locked(run)
    with pytest.raises(RuntimeError, match="detect failed"):
        judge(c, Boom())
    assert marker(run, c)["phase"] == "computing"
    with pytest.raises(pr.PeriodError, match="retry-judge"):                                            # 자동 되돌림 없음
        judge(c, FakeDetector())
    # 사유·검토자 없이는 거부
    with pytest.raises(pr.PeriodError, match="--reason"):
        cmds.cmd_retry_judge(c, "", "", detector=FakeDetector())
    dest = cmds.cmd_retry_judge(c, "detector 버그 수정", "reviewer-a", detector=FakeDetector())
    f1 = json.load(open(dest))
    assert dest.endswith(".failed_1.json") and f1["phase"] == "failed" and f1["reason"] == "detector 버그 수정" and f1["reviewed_by"] == "reviewer-a"
    assert f1["commit_before"] and f1["commit_after"] and isinstance(f1["partial_outputs"], list)
    assert marker(run, c) is None
    rep = judge(c, FakeDetector())
    assert rep["header"]["purpose"] == "judge" and marker(run, c)["phase"] == "completed" and marker(run, c)["attempt"] == 2


def test_second_computing_failure_exhausts_the_judge(run):
    c = locked(run)
    with pytest.raises(RuntimeError):
        judge(c, Boom())
    cmds.cmd_retry_judge(c, "r", "rev", detector=FakeDetector())
    with pytest.raises(RuntimeError):
        judge(c, Boom())
    with pytest.raises(pr.PeriodError, match="소진"):
        cmds.cmd_retry_judge(c, "r2", "rev", detector=FakeDetector())
    with pytest.raises(pr.PeriodError, match="이미 있습니다"):
        judge(c, FakeDetector())


def test_retry_judge_refused_when_config_hash_changed_or_nothing_to_retry(run, tmp_path):
    c = locked(run)
    with pytest.raises(pr.PeriodError, match="표식이 없습니다"):
        cmds.cmd_retry_judge(c, "r", "rev", detector=FakeDetector())
    with pytest.raises(RuntimeError):
        judge(c, Boom())
    changed = tmp_path / "map.csv"
    shutil.copy(DEFAULT_TARGET_MAP, changed)
    with open(changed, "a", encoding="utf-8") as f:
        f.write("NEW-ALARM,제외,사용안함,D\n")                                                           # 실패 사이에 분류/매핑을 바꾸는 시도
    c2 = ctx_of(run, map_path=str(changed))
    with pytest.raises(pr.PeriodError, match="설정 해시"):
        cmds.cmd_retry_judge(c2, "r", "rev", detector=FakeDetector())
    assert marker(run, c)["phase"] == "computing" and not pr.count_variants(run["runs"], c.periods, "failed")


def test_cli_has_retry_judge_action():
    from validation.cli import real_data_check as cli
    a = cli.build_parser().parse_args(["periods", "--run", "r", "retry-judge", "--reason", "x", "--reviewed-by", "y"])
    assert a.action == "retry-judge" and a.reason == "x" and a.reviewed_by == "y"
