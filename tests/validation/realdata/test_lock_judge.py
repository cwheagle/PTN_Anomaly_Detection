"""U6: 판정 기간 잠금과 판정 실행 거부 규칙 (설계서 9.7.3~9.7.6, lessons #38 — 1회 규칙은 코드로 강제)"""
import json
import os
import shutil

import pandas as pd
import pytest

from tests.validation.realdata.test_replay_report import FakeDetector
from tests.validation.realdata.test_snapshot_commands import run  # noqa: F401  (fixture)
from validation.realdata import commands as cmds
from validation.realdata import periods as pr
from validation.realdata.alarm_names import DEFAULT_TARGET_MAP

GOOD = {"traffic": {"version": "v9", "trained_at": "2020-01-01 00:00:00", "training_window": None, "alert_policy": None}}
BAD = {"traffic": {"version": "v1", "trained_at": "2999-01-01 00:00:00", "training_window": None, "alert_policy": None}}


def ctx_of(run, map_path=DEFAULT_TARGET_MAP):
    return cmds.Context(str(run["rd"]), run["runs"], names_path=run["names"], map_path=map_path)


def locked(run, monkeypatch=None):
    c = ctx_of(run)
    cmds.cmd_join_check(c, "dev")
    cmds.cmd_lock(c, model_infos=json.loads(json.dumps(GOOD)))
    return c


def test_lock_requires_dev_join_check_result(run):
    with pytest.raises(pr.PeriodError, match="join-check"):
        cmds.cmd_lock(ctx_of(run), model_infos=json.loads(json.dumps(GOOD)))


def test_lock_refuses_model_with_unprovable_training_window_and_writes_nothing(run):
    c = ctx_of(run)
    cmds.cmd_join_check(c, "dev")
    with pytest.raises(pr.PeriodError, match="학습 구간 검사 실패"):
        cmds.cmd_lock(c, model_infos=json.loads(json.dumps(BAD)))
    assert pr.load_lock(run["runs"], c.periods) is None


def test_lock_payload_is_exclusive_and_records_frozen_values(run):
    c = locked(run)
    lock = pr.load_lock(run["runs"], c.periods)
    assert lock["alternative"] == "fixed_split_v1" and lock["models"]["traffic"]["version"] == "v9"
    assert lock["J0"].startswith("2026-08-07") and lock["judge_data_days"] == 23 and lock["join_status"] in ("unverified", "failed", "verified")
    assert lock["config_hash"] and lock["policy_hash"] and "과거 데이터" in lock["exposure_note"]
    assert lock["c3_hold_expected"] is True and lock["dev_event_rate_r"] == pytest.approx(1 / 28)
    with pytest.raises(pr.PeriodError, match="이미 잠긴"):
        cmds.cmd_lock(c, model_infos=json.loads(json.dumps(GOOD)))


def test_judge_before_lock_and_wrong_ranges_are_refused(run):
    c = ctx_of(run)
    with pytest.raises(pr.PeriodError, match="잠금"):
        cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())
    c = locked(run)
    with pytest.raises(pr.PeriodError, match="judge 만"):
        cmds.cmd_report(c, "dev", "judge", detector=FakeDetector())
    with pytest.raises(pr.PeriodError, match="purpose=dev"):                                    # 잠금 뒤에도 dev 라벨로 판정 구간을 다시 볼 수 없음
        cmds.cmd_report(c, "judge", "dev", detector=FakeDetector())
    with pytest.raises(pr.PeriodError, match="진단 전용"):
        cmds.cmd_report(c, "judge", "judge", detector=FakeDetector(), key_shift=-1)
    with pytest.raises(pr.PeriodError, match="동결"):
        cmds.cmd_report(c, "judge", "judge", versions={"traffic": "v1"}, detector=FakeDetector())
    assert not os.path.exists(pr.run_marker_path(run["runs"], c.periods))                       # 거부는 측정 전, 표식도 남기지 않음


def test_judge_runs_exactly_once_and_uses_frozen_join_status(run):
    c = locked(run)
    lock = pr.load_lock(run["runs"], c.periods)
    rep = cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())
    h = rep["header"]
    assert h["purpose"] == "judge" and h["join_status"] == lock["join_status"] and h["model_versions"] == {"traffic": "v9"}
    assert any("과거 데이터" in n for n in h["fixed_notes"]) and rep["c3"]["합계"]["hold"] is True
    assert pd.Timestamp(h["range"][0]) == c.periods["J0"] and pd.Timestamp(h["range"][1]) == c.periods["D"]
    assert os.path.exists(pr.run_marker_path(run["runs"], c.periods))
    with pytest.raises(pr.PeriodError, match="1회"):
        cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())


def test_judge_refused_when_config_hash_changed_and_marker_not_consumed(run, tmp_path):
    locked(run)
    changed = tmp_path / "map.csv"
    shutil.copy(DEFAULT_TARGET_MAP, changed)
    with open(changed, "a", encoding="utf-8") as f:
        f.write("NEW-ALARM,제외,사용안함,D\n")                                                   # 잠금 뒤 매핑 변경
    c2 = ctx_of(run, map_path=str(changed))
    with pytest.raises(pr.PeriodError, match="설정 해시"):
        cmds.cmd_report(c2, "judge", "judge", detector=FakeDetector())
    assert not os.path.exists(pr.run_marker_path(run["runs"], c2.periods))


def test_judge_refused_when_locked_model_fails_window_check_at_run_time(run):
    c = locked(run)
    lp = pr.lock_path(run["runs"], c.periods)
    lock = json.load(open(lp))
    lock["models"]["traffic"]["trained_at"] = "2999-01-01 00:00:00"                                # 잠금 파일 변조 시나리오
    json.dump(lock, open(lp, "w"))
    with pytest.raises(pr.PeriodError, match="학습 구간 검사"):
        cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())


def test_dev_report_runs_with_fake_detector_and_writes_outputs(run):
    c = ctx_of(run)
    cmds.cmd_join_check(c, "dev")
    rep = cmds.cmd_report(c, "dev", "dev", detector=FakeDetector())
    assert rep["header"]["purpose"] == "dev" and rep["header"]["baselines_use_optical"] is False
    assert not any("과거 데이터" in n for n in rep["header"]["fixed_notes"])
    out = os.listdir(run["rd"] / "out")
    assert any(f.startswith("report_dev_dev") and f.endswith(".json") for f in out) and any(f.startswith("replay_dev_dev") for f in out)
    assert set(rep["c1_raw"]) == {"ai", "fixed", "rolling"} and "ai_incident_tags" in rep


def test_judge_rejects_policy_argument_and_checks_registry_meta_hash(run, monkeypatch):
    c = locked(run)
    with pytest.raises(pr.PeriodError, match="--policy"):
        cmds.cmd_report(c, "judge", "judge", policy=object(), detector=FakeDetector())
    assert not os.path.exists(pr.run_marker_path(run["runs"], c.periods))
    # 실제 로드 경로(detector=None): 레지스트리 모델 메타의 정책/학습 시각이 잠금과 다르면 거부
    monkeypatch.setattr(cmds, "model_info", lambda ft, v=None: {"track": ft, "version": v, "trained_at": "2020-01-01 00:00:00",
                                                                 "training_window": None, "alert_policy": {"sigma_k": 9.0}})
    with pytest.raises(pr.PeriodError, match="alert_policy"):
        cmds.cmd_report(c, "judge", "judge")
    assert not os.path.exists(pr.run_marker_path(run["runs"], c.periods))


@pytest.mark.parametrize("field", ["model_sha", "scaler_sha", "meta_sha", "trained_at"])
def test_judge_rejects_changed_model_files_after_lock(run, monkeypatch, field):
    base = {"track": "traffic", "version": "v9", "trained_at": "2020-01-01 00:00:00", "training_window": None, "alert_policy": None,
            "model_sha": "m1", "scaler_sha": "s1", "meta_sha": "j1"}
    monkeypatch.setattr(cmds, "model_info", lambda ft, v=None: dict(base))
    c = ctx_of(run)
    cmds.cmd_join_check(c, "dev")
    cmds.cmd_lock(c)                                             # 잠금 시점의 해시 기록
    assert pr.load_lock(run["runs"], c.periods)["models"]["traffic"]["model_sha"] == "m1"
    monkeypatch.setattr(cmds, "model_info", lambda ft, v=None: {**base, field: "changed"})          # 같은 버전 문자열인데 파일이 바뀜
    with pytest.raises(pr.PeriodError, match=field):
        cmds.cmd_report(c, "judge", "judge")
    assert not os.path.exists(pr.run_marker_path(run["runs"], c.periods))                           # 측정 전 거부, 표식 소모 없음
    monkeypatch.setattr(cmds, "model_info", lambda ft, v=None: dict(base))
    # 해시가 같으면 통과 (표식은 실제 실행에서만 생성)
    cmds._verify_judge_preconditions(c, "judge", c.periods["J0"], c.periods["D"], None)
    assert os.path.exists(pr.run_marker_path(run["runs"], c.periods))
