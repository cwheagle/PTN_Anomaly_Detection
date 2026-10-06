"""
retrain_policy 검증 (P1-1): 설정 로드, decide() 규칙표, 지속성/쿨다운 상태, 상태 파일
"""
import json
from datetime import datetime, timedelta

import pytest

from src.pipeline import retrain_policy as rp
from src.pipeline.retrain_policy import RetrainPolicy, decide, record_check, record_skip, record_training

NOW = datetime(2026, 10, 4, 3, 0)


def _tr(kind="widespread", **kw):
    return {"status": {"widespread": "drifted", "localized": "localized"}.get(kind, "normal"), "kind": kind,
            "baseline_legacy": False, "localized_ports": 3, **kw}


def _state(streak=3, last_trigger=None):
    days = [(NOW - timedelta(days=i)).strftime("%Y-%m-%d") for i in reversed(range(streak))]
    return {"consecutive_widespread": days, "last_trigger_at": last_trigger, "last_outcome": None, "last_version": None,
            "last_skipped_at": None}


# T-P1: decide() 규칙표 — 위에서부터 첫 번째 해당
@pytest.mark.parametrize("status", ["error", "warming_up", "insufficient_data", "no_baseline"])
def test_decide_unjudgeable_status_is_none(status):
    d = decide({"status": status, "message": "m"}, _state(), NOW, RetrainPolicy())
    assert d.action == "none" and status in d.reason


def test_decide_normal_is_none():
    assert decide(_tr("none"), _state(), NOW, RetrainPolicy()).action == "none"


def test_decide_localized_notifies_without_training_even_with_long_streak():
    d = decide(_tr("localized"), _state(streak=5), NOW, RetrainPolicy(mode="auto"))
    assert d.action == "notify" and "국소" in d.reason and "3포트" in d.reason


def test_decide_legacy_baseline_only_notifies():
    d = decide(_tr(baseline_legacy=True), _state(), NOW, RetrainPolicy())
    assert d.action == "notify" and "구버전" in d.reason


def test_decide_mode_off_only_notifies():
    assert decide(_tr(), _state(), NOW, RetrainPolicy(mode="off")).action == "notify"


@pytest.mark.parametrize("streak,expected", [(0, "notify"), (1, "notify"), (2, "notify"), (3, "train"), (4, "train")])
def test_decide_requires_persistence(streak, expected):
    d = decide(_tr(), _state(streak=streak), NOW, RetrainPolicy())
    assert d.action == expected
    if expected == "notify":
        assert f"{streak}/3" in d.reason


@pytest.mark.parametrize("mode", ["candidate", "auto"])
def test_decide_trains_in_candidate_and_auto_modes(mode):
    assert decide(_tr(), _state(), NOW, RetrainPolicy(mode=mode)).action == "train"


def test_decide_cooldown_blocks_until_elapsed():
    last = (NOW - timedelta(hours=31)).strftime("%Y-%m-%d %H:%M:%S")
    d = decide(_tr(), _state(last_trigger=last), NOW, RetrainPolicy(cooldown_hours=72))
    assert d.action == "notify" and "쿨다운 41h" in d.reason
    old = (NOW - timedelta(hours=73)).strftime("%Y-%m-%d %H:%M:%S")
    assert decide(_tr(), _state(last_trigger=old), NOW, RetrainPolicy(cooldown_hours=72)).action == "train"


def test_decide_pending_candidate_blocks_retraining():
    d = decide(_tr(), _state(), NOW, RetrainPolicy(), pending_candidate="v5")
    assert d.action == "notify" and "v5" in d.reason


def test_decide_already_training_blocks():
    assert decide(_tr(), _state(), NOW, RetrainPolicy(), is_training=True).action == "notify"


# T-P2: 같은 날 중복 체크는 지속성 1회
def test_record_check_counts_one_per_day_and_requires_consecutive_days():
    s = {}
    for hour in (3, 9, 15):
        s = record_check(s, _tr(), NOW.replace(hour=hour))
    assert s["consecutive_widespread"] == ["2026-10-04"]
    s = record_check(s, _tr(), NOW + timedelta(days=1))
    assert len(s["consecutive_widespread"]) == 2
    s = record_check(s, _tr(), NOW + timedelta(days=3))                 # 하루 건너뜀: 연속 아님 -> 새로 시작
    assert s["consecutive_widespread"] == ["2026-10-07"]


def test_record_check_resets_when_judged_not_widespread_but_keeps_when_unjudgeable():
    s = _state(streak=2)
    assert record_check(s, {"status": "warming_up"}, NOW)["consecutive_widespread"] == s["consecutive_widespread"]
    assert record_check(s, {"status": "error"}, NOW)["consecutive_widespread"] == s["consecutive_widespread"]
    assert record_check(s, _tr("none"), NOW)["consecutive_widespread"] == []
    assert record_check(s, _tr("localized"), NOW)["consecutive_widespread"] == []


def test_record_check_does_not_mutate_input():
    s = _state(streak=1)
    before = json.dumps(s)
    record_check(s, _tr("none"), NOW)
    assert json.dumps(s) == before


def test_record_training_started_sets_cooldown_and_resets_streak_then_outcome_updates():
    s = record_training(_state(streak=3), NOW, outcome="started")
    assert s["last_trigger_at"] == "2026-10-04 03:00:00" and s["consecutive_widespread"] == []
    s2 = record_training(s, NOW + timedelta(hours=1), version="v4", outcome="candidate")
    assert s2["last_trigger_at"] == s["last_trigger_at"] and s2["last_outcome"] == "candidate" and s2["last_version"] == "v4"


def test_decide_after_trigger_enters_cooldown_end_to_end():
    s = record_training(_state(), NOW, outcome="started")
    s = record_check(s, _tr(), NOW + timedelta(days=1))
    assert decide(_tr(), s, NOW + timedelta(days=1), RetrainPolicy()).action == "notify"        # 연속 1/3


# T-P3: 상태 파일
def test_state_roundtrip_atomic_and_no_tmp_left(tmp_path):
    s = record_training({}, NOW, "v2", "candidate")
    rp.save_state(str(tmp_path), "traffic", s)
    assert rp.load_state(str(tmp_path), "traffic") == s
    assert [p.name for p in tmp_path.iterdir()] == ["traffic_retrain_state.json"]


def test_state_missing_or_corrupt_file_loads_empty(tmp_path):
    assert rp.load_state(str(tmp_path), "traffic")["consecutive_widespread"] == []
    (tmp_path / "traffic_retrain_state.json").write_text("{not json")
    assert rp.load_state(str(tmp_path), "traffic") == rp._empty_state()
    (tmp_path / "optical_retrain_state.json").write_text("[1,2]")
    assert rp.load_state(str(tmp_path), "optical") == rp._empty_state()


# T-P4: 설정 로드
def test_env_mode_overrides_config_and_invalid_falls_back_with_warning(monkeypatch, caplog):
    import src.config as config
    monkeypatch.setattr(config, "RETRAIN_POLICY", {"mode": "off", "train_days": 14, "bogus": 1}, raising=False)
    monkeypatch.delenv("DRIFT_RETRAIN_MODE", raising=False)
    p = rp.load_retrain_policy()
    assert p.mode == "off" and p.train_days == 14                       # config 적용, 모르는 키는 무시

    monkeypatch.setenv("DRIFT_RETRAIN_MODE", "auto")
    assert rp.load_retrain_policy().mode == "auto"                      # 환경변수 우선

    monkeypatch.setenv("DRIFT_RETRAIN_MODE", "yolo")
    with caplog.at_level("WARNING"):
        assert rp.load_retrain_policy().mode == "candidate"
    assert "yolo" in caplog.text


def test_defaults_match_design_decisions(monkeypatch):
    import src.config as config
    monkeypatch.delenv("DRIFT_RETRAIN_MODE", raising=False)
    monkeypatch.setattr(config, "RETRAIN_POLICY", {}, raising=False)
    p = rp.load_retrain_policy()
    assert (p.mode, p.train_days, p.drift_persist_checks, p.cooldown_hours, p.gate_max_incidents_per_port_day) == \
           ("candidate", 28, 3, 72, 0.05)


def test_pending_drift_candidate_derived_from_registry():
    reg = {"active_version": "v2", "history": [{"version": "v1"}, {"version": "v2"}], "versions": [
        {"version": "v1", "trigger": "manual"}, {"version": "v2", "trigger": "drift"},
        {"version": "v3", "trigger": "manual"}, {"version": "v4", "trigger": "drift"}]}
    assert rp.pending_drift_candidate(reg) == "v4"
    reg["history"].append({"version": "v4"})                              # 승격됨 -> 대기 아님
    reg["active_version"] = "v4"
    assert rp.pending_drift_candidate(reg) is None
    reg2 = {"active_version": "v5", "history": [{"version": "v5"}], "versions": [
        {"version": "v4", "trigger": "drift"}, {"version": "v5", "trigger": "manual"}]}
    assert rp.pending_drift_candidate(reg2) is None                      # 활성보다 오래된 후보는 대기 아님


# T-P1 (발견 A): 중단·수집 오류는 쿨다운 없이 24h 백오프
SKIP_OUTCOME = "skipped: suspect_fraction=0.324"


def test_record_skip_keeps_streak_sets_backoff_and_no_cooldown():
    s = record_skip(_state(streak=3), NOW, SKIP_OUTCOME)
    assert s["last_trigger_at"] is None and len(s["consecutive_widespread"]) == 3
    assert s["last_skipped_at"] == "2026-10-04 03:00:00" and s["last_outcome"] == SKIP_OUTCOME
    assert rp.cooldown_remaining_hours(s, NOW, RetrainPolicy()) == 0.0


def test_skip_within_24h_notifies_with_reason_and_remaining_hours():
    s = record_skip(_state(streak=3), NOW, SKIP_OUTCOME)
    d = decide(_tr(), s, NOW + timedelta(hours=9), RetrainPolicy())
    assert d.action == "notify" and "재학습 보류" in d.reason and "32.4% > 20%" in d.reason and "15h 후 재시도" in d.reason


def test_skip_backoff_ends_after_24h_then_trains():
    s = record_skip(_state(streak=3), NOW, SKIP_OUTCOME)
    assert decide(_tr(), s, NOW + timedelta(hours=23, minutes=59), RetrainPolicy()).action == "notify"
    assert decide(_tr(), s, NOW + timedelta(hours=24), RetrainPolicy()).action == "train"


def test_collection_error_skip_same_as_suspect_skip_and_generic_reason():
    s = record_skip(_state(streak=3), NOW, "failed: DB down")
    d = decide(_tr(), s, NOW + timedelta(hours=1), RetrainPolicy())
    assert d.action == "notify" and "DB down" in d.reason
    assert decide(_tr(), s, NOW + timedelta(hours=25), RetrainPolicy()).action == "train"


def test_streak_survives_skip_and_next_day_check_still_counts():
    s = record_skip(_state(streak=3), NOW, SKIP_OUTCOME)
    s = record_check(s, _tr(), NOW + timedelta(days=1))
    assert len(s["consecutive_widespread"]) == 4                   # 지속성이 이어짐
    assert decide(_tr(), s, NOW + timedelta(days=1), RetrainPolicy()).action == "train"


def test_trainer_start_then_failure_applies_cooldown_and_resets_streak():
    s = record_skip(_state(streak=3), NOW - timedelta(days=2), SKIP_OUTCOME)
    s = record_training(s, NOW, outcome="started")
    s = record_training(s, NOW + timedelta(minutes=5), outcome="failed: OOM")
    assert s["last_trigger_at"] and s["consecutive_widespread"] == [] and s["last_skipped_at"] is None
    assert rp.cooldown_remaining_hours(s, NOW + timedelta(hours=1), RetrainPolicy()) > 0


def test_cooldown_has_priority_over_skip_backoff_and_describe_state_exposes_backoff():
    s = record_skip(_state(streak=3, last_trigger="2026-10-04 02:00:00"), NOW, SKIP_OUTCOME)
    assert "쿨다운" in decide(_tr(), s, NOW, RetrainPolicy()).reason
    info = rp.describe_state(record_skip(_state(), NOW, SKIP_OUTCOME), NOW + timedelta(hours=4), RetrainPolicy())
    assert info["skip_backoff_remaining_hours"] == 20.0 and info["last_skipped_at"]
