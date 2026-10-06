"""
드리프트 -> 재학습 결정 -> 후보 학습 파이프라인 API 검증 (P1-1)

DB/모델 없이: 드리프트 판정·수집기·Trainer 를 가짜로 대체하고, 결정/상태/파이프라인 배선만 확인한다.
"""
import json
from datetime import datetime, timedelta

import pytest

from src.pipeline import retrain_policy as rp

TODAY = datetime.now().strftime("%Y-%m-%d")


def _day(offset):
    return (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")


def _widespread(**kw):
    return {"status": "drifted", "kind": "widespread", "mean_mse": 0.12, "baseline_mse": 0.05, "drift_ratio": 2.4,
            "median_ratio": 2.4, "drifted_port_fraction": 0.8, "ports": 40, "top_ports": [], "baseline_legacy": False, **kw}


def _normal():
    return {"status": "normal", "kind": "none", "mean_mse": 0.05, "baseline_mse": 0.05, "drift_ratio": 1.0,
            "baseline_legacy": False}


@pytest.fixture
def env(tmp_path, monkeypatch, api_main):
    main = api_main
    for ft in ("traffic", "optical"):
        monkeypatch.setitem(main.PATHS, ft, {"model": str(tmp_path / f"{ft}_ae.pth"),
                                             "scaler": str(tmp_path / f"{ft}_scaler.joblib")})
        main.training_status[ft]["is_training"] = False
    monkeypatch.delenv("DRIFT_RETRAIN_MODE", raising=False)
    monkeypatch.setattr(main, "last_drift_result", None)
    launched = []
    orig_pipeline = main.run_training_pipeline
    monkeypatch.setattr(main, "run_training_pipeline", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(main, "_saved_training_config", lambda ft: {"epochs": 3})
    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    class Env:
        pass
    e = Env()
    e.main, e.tmp, e.client, e.launched, e.monkeypatch = main, tmp_path, client, launched, monkeypatch
    e.orig_pipeline = orig_pipeline
    e.set_drift = lambda result: monkeypatch.setattr(main.drift_monitor, "check_drift", lambda: result)
    e.state = lambda ft="traffic": rp.load_state(str(tmp_path), ft)
    return e


def _prime_streak(env, ft, days):
    st = rp._empty_state()
    st["consecutive_widespread"] = [_day(i) for i in reversed(range(1, days + 1))]      # 어제까지 days 일 연속
    rp.save_state(str(env.tmp), ft, st)


def test_check_response_keeps_backward_compatible_keys_and_adds_retrain(env):
    """T-E1"""
    env.set_drift({"traffic": _widespread(), "optical": _normal(), "drift_detected": True,
                   "drifted_tracks": ["traffic"], "timestamp": "2026-10-04 03:00:00"})
    r = env.client.post("/api/drift/check")
    assert r.status_code == 200
    body = r.json()
    for key in ("drift_detected", "drifted_tracks", "auto_retrain_triggered", "timestamp"):
        assert key in body
    assert body["traffic"]["status"] == "drifted" and "drift_ratio" in body["traffic"]
    assert body["retrain"]["traffic"]["action"] == "notify" and "1/3" in body["retrain"]["traffic"]["reason"]
    assert body["retrain"]["optical"]["action"] == "none"
    assert body["auto_retrain_triggered"] is False and env.launched == []


def test_first_widespread_checks_do_not_train_until_persistent(env):
    env.set_drift({"traffic": _widespread(), "optical": _normal(), "drift_detected": True, "drifted_tracks": ["traffic"]})
    for _ in range(3):                                                  # 같은 날 반복 호출은 1회로만 집계
        env.client.post("/api/drift/check")
    assert env.state()["consecutive_widespread"] == [TODAY] and env.launched == []


def test_persistent_widespread_starts_candidate_training_with_drift_trigger(env):
    """T-E2(candidate): 3일째에 후보 학습 시작, 쿨다운 기준 시각 기록, 활성 모델은 건드리지 않음"""
    _prime_streak(env, "traffic", 2)
    env.set_drift({"traffic": _widespread(), "optical": _normal(), "drift_detected": True, "drifted_tracks": ["traffic"]})
    body = env.client.post("/api/drift/check").json()
    assert body["retrain"]["traffic"]["action"] == "train" and body["auto_retrain_triggered"] is True
    assert env.launched == [("traffic", {"epochs": 3}, None, "drift")]
    st = env.state()            # 쿨다운·지속성 초기화는 Trainer 시작 시점에 기록 (발견 A) — 런칭만으로는 바뀌지 않음
    assert st["last_trigger_at"] is None and len(st["consecutive_widespread"]) == 3
    assert not (env.tmp / "traffic_registry.json").exists()                   # 레지스트리/활성 모델 불변
    assert env.main.training_status["traffic"]["is_training"] is True        # 중복 트리거 방지


def test_cooldown_prevents_second_training_after_trainer_started(env):
    _prime_streak(env, "traffic", 2)
    st = rp.record_training(env.state(), datetime.now(), outcome="started")       # Trainer 시작 기록
    st["consecutive_widespread"] = [_day(2), _day(1)]
    rp.save_state(str(env.tmp), "traffic", st)
    env.set_drift({"traffic": _widespread(), "optical": _normal(), "drift_detected": True, "drifted_tracks": ["traffic"]})
    body = env.client.post("/api/drift/check").json()
    assert body["retrain"]["traffic"]["action"] == "notify" and "쿨다운" in body["retrain"]["traffic"]["reason"]
    assert env.launched == []


def test_mode_off_never_trains(env):
    env.monkeypatch.setenv("DRIFT_RETRAIN_MODE", "off")
    _prime_streak(env, "traffic", 5)
    env.set_drift({"traffic": _widespread(), "optical": _normal(), "drift_detected": True, "drifted_tracks": ["traffic"]})
    body = env.client.post("/api/drift/check").json()
    assert body["retrain"]["traffic"]["action"] == "notify" and env.launched == []


def test_localized_anomaly_never_trains(env):
    _prime_streak(env, "traffic", 5)
    loc = {**_normal(), "status": "localized", "kind": "localized", "localized_ports": 2}
    env.set_drift({"traffic": loc, "optical": _normal(), "drift_detected": False, "drifted_tracks": []})
    body = env.client.post("/api/drift/check").json()
    assert body["retrain"]["traffic"]["action"] == "notify" and "국소" in body["retrain"]["traffic"]["reason"]
    assert env.launched == [] and env.state()["consecutive_widespread"] == []


def test_pending_drift_candidate_blocks_new_training(env):
    from src.models import registry
    for ver, trig, act in (("v1", "manual", True), ("v2", "drift", False)):
        entry = {"version": ver, "model_path": "m", "scaler_path": "s", "config_path": "c", "trigger": trig}
        with registry.transaction(str(env.tmp), "traffic") as reg:
            registry.add_version(reg, entry, act)
    _prime_streak(env, "traffic", 5)
    env.set_drift({"traffic": _widespread(), "optical": _normal(), "drift_detected": True, "drifted_tracks": ["traffic"]})
    body = env.client.post("/api/drift/check").json()
    assert "v2" in body["retrain"]["traffic"]["reason"] and env.launched == []


def test_check_error_returns_500_without_touching_state(env):
    env.set_drift({"status": "error", "message": "DB Connection failed"})
    assert env.client.post("/api/drift/check").status_code == 500
    assert not (env.tmp / "traffic_retrain_state.json").exists()


def test_drift_status_exposes_last_result_and_retrain_state(env):
    env.set_drift({"traffic": _widespread(), "optical": _normal(), "drift_detected": True, "drifted_tracks": ["traffic"]})
    env.client.post("/api/drift/check")
    s = env.client.get("/api/drift/status").json()
    assert s["last_result"]["retrain"]["traffic"]["action"] == "notify"
    t = s["retrain_state"]["traffic"]
    assert t["streak_days"] == 1 and t["persist_required"] == 3 and t["mode"] == "candidate"
    assert t["cooldown_remaining_hours"] == 0


def test_maintenance_and_endpoint_share_the_same_handler(env):
    """F4: 일일 유지보수도 같은 handle_drift 를 거친다 (중복 트리거 코드 제거)"""
    _prime_streak(env, "traffic", 2)
    result = {"traffic": _widespread(), "optical": _normal(), "drift_detected": True, "drifted_tracks": ["traffic"]}
    env.monkeypatch.setattr(env.main.drift_monitor, "check_drift", lambda: result)
    env.monkeypatch.setattr(env.main.db, "delete_old_data", lambda days: None)
    env.main.run_daily_maintenance()
    import time
    for _ in range(50):
        if env.launched:
            break
        time.sleep(0.02)
    assert env.launched == [("traffic", {"epochs": 3}, None, "drift")]
    assert env.main.last_drift_result["retrain"]["traffic"]["action"] == "train"


# ──────────────────────────────────────────────
# run_training_pipeline 배선: 수집 모드 / 제외 정책 / 메타데이터 / 상태 기록
# ──────────────────────────────────────────────
class _FakeCollector:
    def __init__(self, result):
        self.result, self.calls = result, []

    def collect_and_save(self, **kw):
        self.calls.append(kw)
        return self.result


class _FakeTrainer:
    instances = []

    def __init__(self, **kw):
        self.kw, self.early_stopped, self.activated, self.version = kw, False, False, "v7"
        _FakeTrainer.instances.append(self)

    def train(self):
        return True


@pytest.fixture
def pipe(env):
    m = env.main
    env.monkeypatch.setattr(m, "run_training_pipeline", env.orig_pipeline)        # 실제 파이프라인을 되살림
    _FakeTrainer.instances.clear()
    env.published = []
    env.monkeypatch.setattr(m.redis_client, "publish", lambda ch, msg: env.published.append(json.loads(msg)))
    env.monkeypatch.setattr(m, "Trainer", _FakeTrainer)
    env.set_collector = lambda res: env.monkeypatch.setattr(m, "collector", _FakeCollector(res))
    return env


OK_RESULT = {"traffic": {"train": 5000, "test": 500,
                         "suspect_stats": {"rows": 5500, "excluded_rows": 100, "fraction": 0.018, "dropped_ports": [],
                                           "by_source": {"alarm": 100}}}}


def test_default_training_uses_port_split_window_with_suspect_exclusion(pipe):
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {"epochs": 3}, None, "manual")
    call = pipe.main.collector.calls[0]
    pol = rp.load_retrain_policy()
    assert call["val_port_fraction"] == pol.val_port_fraction and call["split_salt"] == pol.split_salt
    assert call["suspect_policy"] is not None and call["feature_type"] == "traffic"
    assert call["train_end"] - call["train_start"] == timedelta(days=pol.train_days)         # 최근까지 포함한 28일
    assert abs((datetime.now() - call["train_end"]).total_seconds()) < 60
    assert "test_start" not in call
    t = _FakeTrainer.instances[0].kw
    assert t["trigger"] == "manual" and t["activate"] is False and t["window_info"]["mode"] == "port_split"
    assert t["suspect_stats"]["fraction"] == 0.018
    st = pipe.main.training_status["traffic"]
    assert st["is_training"] is False and st["candidate_version"] == "v7" and st["suspect_fraction"] == 0.018


def test_exclude_suspect_false_disables_automatic_exclusion(pipe):
    pipe.set_collector({"traffic": {"train": 5000, "test": 500}})
    pipe.main.run_training_pipeline("traffic", {}, None, "manual", False)
    assert pipe.main.collector.calls[0]["suspect_policy"] is None


def test_explicit_dates_keep_date_split_for_ui_compatibility(pipe):
    pipe.set_collector({"traffic": {"train": 5000, "test": 500}})
    dp = {"train_start": "2026-09-01", "train_end": "2026-09-20", "test_start": "2026-09-20", "test_end": "2026-09-27"}
    pipe.main.run_training_pipeline("traffic", {}, dp, "manual")
    call = pipe.main.collector.calls[0]
    assert call["test_start"] == "2026-09-20" and "val_port_fraction" not in call
    assert _FakeTrainer.instances[0].kw["window_info"]["mode"] == "date"


def test_suspect_fraction_over_limit_aborts_training_and_records_state_for_drift(pipe):
    stats = {"rows": 1000, "excluded_rows": 300, "fraction": 0.3, "dropped_ports": [], "by_source": {"rule": 300}}
    pipe.set_collector({"traffic": {"skipped": "suspect_fraction=0.300", "suspect_stats": stats}})
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    st = pipe.main.training_status["traffic"]
    assert _FakeTrainer.instances == []                                   # 학습하지 않음
    assert "30.0%" in st["last_error"] and st["is_training"] is False
    assert pipe.state()["last_outcome"].startswith("skipped: suspect_fraction")


def _primed_pipe_state(pipe):
    st = rp._empty_state()
    st["consecutive_widespread"] = [_day(2), _day(1), _day(0)]
    rp.save_state(str(pipe.tmp), "traffic", st)


def test_suspect_abort_for_drift_applies_no_cooldown_keeps_streak_and_backs_off_24h(pipe):
    """발견 A (T-P1 ①): 중단은 쿨다운 없음·지속성 유지·last_skipped_at 기록 -> 24h 보류 후 재시도"""
    _primed_pipe_state(pipe)
    pipe.set_collector({"traffic": {"skipped": "suspect_fraction=0.324",
                                    "suspect_stats": {"rows": 1000, "excluded_rows": 324, "fraction": 0.324}}})
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    st = pipe.state()
    assert st["last_trigger_at"] is None and len(st["consecutive_widespread"]) == 3 and st["last_skipped_at"]
    assert st["last_outcome"] == "skipped: suspect_fraction=0.324"
    pipe.main.training_status["traffic"]["is_training"] = False
    pipe.set_drift({"traffic": _widespread(), "optical": _normal(), "drift_detected": True, "drifted_tracks": ["traffic"]})
    body = pipe.client.post("/api/drift/check").json()
    assert body["retrain"]["traffic"]["action"] == "notify" and "재학습 보류" in body["retrain"]["traffic"]["reason"]
    assert "32.4%" in body["retrain"]["traffic"]["reason"]


def test_collection_error_for_drift_is_treated_like_skip(pipe):
    """T-P1 ②: 수집 중 예외(DB 실패 등)도 쿨다운 없이 백오프만"""
    _primed_pipe_state(pipe)

    class _Boom:
        def collect_and_save(self, **kw):
            raise RuntimeError("DB down")
    pipe.monkeypatch.setattr(pipe.main, "collector", _Boom())
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    st = pipe.state()
    assert st["last_trigger_at"] is None and len(st["consecutive_widespread"]) == 3 and st["last_skipped_at"]
    assert st["last_outcome"].startswith("failed: DB down") and _FakeTrainer.instances == []


def test_insufficient_data_for_drift_is_treated_like_skip(pipe):
    _primed_pipe_state(pipe)
    pipe.set_collector({"traffic": {"train": 3, "test": 1}})
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    st = pipe.state()
    assert st["last_trigger_at"] is None and st["last_skipped_at"] and len(st["consecutive_widespread"]) == 3


def test_failure_after_trainer_start_applies_cooldown_and_resets_streak(pipe):
    """T-P1 ③: Trainer 시작 뒤 실패는 쿨다운 적용·지속성 초기화 (백오프 아님)"""
    _primed_pipe_state(pipe)
    pipe.set_collector(OK_RESULT)

    class _Crash(_FakeTrainer):
        def train(self):
            raise RuntimeError("OOM")
    pipe.monkeypatch.setattr(pipe.main, "Trainer", _Crash)
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    st = pipe.state()
    assert st["last_trigger_at"] and st["consecutive_widespread"] == [] and st["last_skipped_at"] is None
    assert st["last_outcome"].startswith("failed: OOM")


def test_trainer_start_is_recorded_before_training_runs(pipe):
    _primed_pipe_state(pipe)
    pipe.set_collector(OK_RESULT)
    seen = {}

    class _Peek(_FakeTrainer):
        def train(self):
            seen.update(pipe.state())
            return True
    pipe.monkeypatch.setattr(pipe.main, "Trainer", _Peek)
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    assert seen["last_trigger_at"] and seen["consecutive_widespread"] == [] and seen["last_outcome"] == "started"


def test_insufficient_data_fails_cleanly(pipe):
    pipe.set_collector({"traffic": {"train": 3, "test": 1}})
    pipe.main.run_training_pipeline("traffic", {}, None, "manual")
    assert "Insufficient" in pipe.main.training_status["traffic"]["last_error"] and _FakeTrainer.instances == []


def test_drift_trigger_records_candidate_outcome_and_version(pipe):
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    st = pipe.state()
    assert st["last_outcome"] == "candidate" and st["last_version"] == "v7"
    assert _FakeTrainer.instances[0].kw["trigger"] == "drift"
    assert pipe.published == []                                           # 후보 저장: Consumer 리로드 없음


def test_manual_training_does_not_touch_retrain_state(pipe):
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, "manual")
    assert not (pipe.tmp / "traffic_retrain_state.json").exists()


def test_train_endpoint_defaults_to_port_split_and_passes_exclude_flag(env):
    env.client.post("/api/model/train?ft=traffic&exclude_suspect=false")
    assert env.launched[-1][2:] == (None, "manual", False, None)
    r = env.client.post("/api/model/train?ft=optical").json()
    assert r["range"]["mode"] == "port_split" and r["exclude_suspect"] is True


def test_train_endpoint_with_dates_uses_date_split(env):
    r = env.client.post("/api/model/train?ft=traffic&train_start=2026-09-01&train_end=2026-09-20").json()
    dp = env.launched[-1][2]
    assert dp["train_start"] == "2026-09-01" and dp["test_end"] and r["range"] == dp
    assert env.client.post("/api/model/train?ft=traffic").status_code == 409        # 학습 중 연타 방지


# ──────────────────────────────────────────────
# 승격 게이트 배선: 학습 직후 게이트 실행·기록, mode=auto 자동 승격
# ──────────────────────────────────────────────
def _seed_registry(env, with_candidate=True):
    """활성 v1 + (Fake Trainer 가 만들었다고 가정하는) 후보 v7 을 레지스트리에 준비"""
    from src.models import registry
    for ver, act in (("v1", True), ("v7", False)):
        if ver == "v7" and not with_candidate:
            continue
        entry = {"version": ver, "model_path": f"m_{ver}", "scaler_path": f"s_{ver}", "config_path": f"c_{ver}",
                 "threshold": 0.3, "final_val_loss": 0.02, "trigger": "drift" if ver == "v7" else "manual"}
        for k in ("model_path", "scaler_path", "config_path"):
            (env.tmp / entry[k]).write_text("x")
        with registry.transaction(str(env.tmp), "traffic") as reg:
            registry.add_version(reg, entry, act)


def _fake_gate(env, status, calls):
    from src.models import registry

    def run(ft, version, now=None):
        calls.append((ft, version, now))
        gate = {"status": status, "checks": [], "evaluated_at": "x", "data": {"against": "v1"}}
        registry.set_gate(str(env.tmp), ft, version, gate)
        return gate
    env.monkeypatch.setattr(env.main, "run_gate_for", run)


def _active(env):
    from src.models import registry
    return registry.load(str(env.tmp), "traffic")["active_version"]


def test_gate_runs_after_training_and_is_recorded_on_the_candidate(pipe):
    _seed_registry(pipe)
    calls = []
    _fake_gate(pipe, "WARN", calls)
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, "manual")
    assert calls and calls[0][:2] == ("traffic", "v7") and isinstance(calls[0][2], datetime)
    assert pipe.main.training_status["traffic"]["gate_status"] == "WARN"
    from src.models import registry
    assert registry.list_versions(str(pipe.tmp), "traffic")["versions"][1]["gate_status"] == "WARN"
    assert _active(pipe) == "v1" and pipe.published == []


def test_auto_mode_promotes_drift_candidate_when_gate_passes(pipe):
    """T-E2(auto): mode=auto & drift & PASS -> 승격 + Consumer reload publish, 이력 reason=auto-gate"""
    pipe.monkeypatch.setenv("DRIFT_RETRAIN_MODE", "auto")
    _seed_registry(pipe)
    _fake_gate(pipe, "PASS", [])
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    from src.models import registry
    reg = registry.load(str(pipe.tmp), "traffic")
    assert reg["active_version"] == "v7" and reg["history"][-1]["reason"] == "auto-gate"
    assert pipe.published == [{"action": "reload", "track": "traffic"}]
    assert pipe.state()["last_outcome"] == "auto-promoted"
    assert "auto-promoted" in pipe.main.training_status["traffic"]["success_msg"]


@pytest.mark.parametrize("mode,trigger,gate_status", [
    ("candidate", "drift", "PASS"),        # candidate 모드: 게이트 통과해도 사람이 승격
    ("off", "drift", "PASS"),
    ("auto", "manual", "PASS"),            # 수동 학습은 자동 승격 없음
    ("auto", "drift", "WARN"),             # PASS 가 아니면 자동 승격 안 함
    ("auto", "drift", "FAIL"),
    ("auto", "drift", "ERROR"),
])
def test_candidate_stays_unpromoted_unless_auto_drift_and_gate_pass(pipe, mode, trigger, gate_status):
    """T-E2(candidate 등): 활성 모델 불변, Consumer 리로드 없음"""
    pipe.monkeypatch.setenv("DRIFT_RETRAIN_MODE", mode)
    _seed_registry(pipe)
    _fake_gate(pipe, gate_status, [])
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, trigger)
    assert _active(pipe) == "v1" and pipe.published == []


def test_auto_mode_does_not_promote_when_g3_is_skipped_for_small_holdout(pipe):
    """T-E2(발견 B): 실제 gate.evaluate 결과(G3 SKIP -> 종합 WARN)를 쓰는 auto 모드는 승격하지 않는다"""
    from src.models import promotion_gate as pg, registry
    from src.pipeline.retrain_policy import RetrainPolicy
    pipe.monkeypatch.setenv("DRIFT_RETRAIN_MODE", "auto")
    _seed_registry(pipe)
    small = {"ports": 4, "incidents": 0, "port_days": 12, "incidents_per_port_day": 0.0}

    def run(ft, version, now=None):
        entry = {"version": version, "threshold": 0.3, "final_val_loss": 0.02}
        gate = pg.evaluate(entry, {"version": "v1", "threshold": 0.3, "final_val_loss": 0.02}, small, small, None,
                           [0.3, 0.3, 0.3], RetrainPolicy()).to_dict()
        registry.set_gate(str(pipe.tmp), ft, version, gate)
        return gate
    pipe.monkeypatch.setattr(pipe.main, "run_gate_for", run)
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    assert pipe.main.training_status["traffic"]["gate_status"] == "WARN"
    assert _active(pipe) == "v1" and pipe.published == []


def test_gate_failure_to_run_still_leaves_a_candidate(pipe):
    _seed_registry(pipe)

    def boom(ft, version, now=None):
        raise RuntimeError("gate crashed")
    pipe.monkeypatch.setattr(pipe.main, "run_gate_for", boom)
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    st = pipe.main.training_status["traffic"]
    assert st["gate_status"] == "ERROR" and st["candidate_version"] == "v7" and st["is_training"] is False
    assert pipe.state()["last_outcome"] == "candidate" and _active(pipe) == "v1"


def test_run_gate_for_records_result_and_uses_operating_alert_policy(pipe):
    """run_gate_for 배선: 레지스트리에 기록하고, 운영 알람 정책·홀드아웃 설정으로 promotion_gate.run_gate 호출"""
    from src.models import registry
    from src.models.promotion_gate import GateResult, GateCheck
    _seed_registry(pipe)
    seen = {}

    def fake_run_gate(model_dir, ft, version, window, policy, fetch_raw, alert_policy, **kw):
        seen.update(model_dir=model_dir, ft=ft, version=version, window=window, policy=policy, alert=alert_policy,
                    fetch=fetch_raw)
        return GateResult(status="PASS", checks=[GateCheck("G1", "PASS")], evaluated_at="t", data={"against": "v1"})
    pipe.monkeypatch.setattr(pipe.main.promotion_gate, "run_gate", fake_run_gate)
    now = datetime(2026, 10, 1, 12, 0)
    gate = pipe.main.run_gate_for("traffic", "v7", now)
    assert gate["status"] == "PASS"
    assert seen["window"].gate_end == now and seen["window"].gate_start == now - timedelta(days=3)
    from src.pipeline.alerting import load_policy
    assert seen["alert"] == load_policy() and seen["model_dir"] == str(pipe.tmp)
    assert registry.load(str(pipe.tmp), "traffic")["versions"][1]["gate"]["status"] == "PASS"


def test_gate_endpoint_reruns_gate_and_returns_409_while_training(env):
    from src.models import registry
    _seed_registry(env)
    calls = []
    _fake_gate(env, "PASS", calls)
    r = env.client.post("/api/model/gate?ft=traffic&version=v7")
    assert r.status_code == 200 and r.json()["gate"]["status"] == "PASS" and calls[0][:2] == ("traffic", "v7")
    assert env.client.post("/api/model/gate?ft=traffic&version=v99").status_code == 404
    env.main.training_status["traffic"]["is_training"] = True
    assert env.client.post("/api/model/gate?ft=traffic&version=v7").status_code == 409        # T-E3
    assert len(calls) == 1


def test_promote_409_detail_includes_gate_checks_and_force_overrides(env):
    from src.models import registry
    _seed_registry(env)
    registry.set_gate(str(env.tmp), "traffic", "v7", {"status": "FAIL", "data": {"against": "v1"}, "checks": [
        {"id": "G3", "status": "FAIL", "message": "알람 과다", "value": 0.2, "limit": 0.05}]})
    env.monkeypatch.setattr(env.main.redis_client, "publish", lambda *a: None)
    r = env.client.post("/api/model/promote?ft=traffic&version=v7")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["warnings"] == ["[G3] 알람 과다"] and detail["checks"][0]["id"] == "G3"
    assert env.client.post("/api/model/promote?ft=traffic&version=v7&force=true").status_code == 200


# T-P3-E1 (P1-3): 모델별 알람 정책 — 학습 결과(후보)에 정책이 짝으로 저장되고, 드리프트 재학습은 활성 모델의 정책을 승계
def _set_active_policy(env, meta):
    from src.models import registry
    _seed_registry(env, with_candidate=False)
    with registry.transaction(str(env.tmp), "traffic") as reg:
        registry.find(reg, "v1")["alert_policy"] = meta


def test_drift_retraining_inherits_the_active_models_alert_policy(pipe):
    meta = {"preset": "precision", "sigma_k": 2.0, "dyn_cap": 1.2, "threshold_scale": 3.0, "severity_decay": 0.5,
            "dampening_steps": {"1": 6, "2": 4, "3": 3}}
    _set_active_policy(pipe, meta)
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    assert _FakeTrainer.instances[0].kw["alert_policy"] == meta


def test_explicit_preset_overrides_inheritance_and_no_policy_means_none(pipe):
    _set_active_policy(pipe, {"preset": "precision", "sigma_k": 2.0, "threshold_scale": 3.0})
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, "manual", True, "default")
    assert _FakeTrainer.instances[0].kw["alert_policy"]["preset"] == "default"
    assert _FakeTrainer.instances[0].kw["alert_policy"]["threshold_scale"] == 1.0              # 프리셋의 실제 값이 저장됨

    _FakeTrainer.instances.clear()
    from src.models import registry
    with registry.transaction(str(pipe.tmp), "traffic") as reg:
        reg["versions"][0].pop("alert_policy")                                                 # 활성 모델에 정책 없음
    pipe.main.run_training_pipeline("traffic", {}, None, "manual")
    assert _FakeTrainer.instances[0].kw["alert_policy"] is None


def test_auto_mode_does_not_promote_when_g3_absolute_limit_is_exceeded(pipe):
    """T-P3-E3 (U4'): mode=auto 에서 G3 절대 상한(0.05) 초과(WARN) 후보는 자동 승격되지 않고 활성 불변·Consumer 리로드 없음.
    실제 gate.evaluate(C4') 결과를 사용한다 (홀드아웃 150포트·일 이상, 상대 기준은 통과)."""
    from src.models import promotion_gate as pg, registry
    from src.pipeline.retrain_policy import RetrainPolicy
    pipe.monkeypatch.setenv("DRIFT_RETRAIN_MODE", "auto")
    _seed_registry(pipe)
    stats = {"ports": 130, "incidents": 39, "port_days": 390, "incidents_per_port_day": 0.10}

    def run(ft, version, now=None):
        entry = {"version": version, "threshold": 0.3, "final_val_loss": 0.02}
        gate = pg.evaluate(entry, {"version": "v1", "threshold": 0.3, "final_val_loss": 0.02}, stats, stats, None,
                           [0.3, 0.3, 0.3], RetrainPolicy()).to_dict()
        registry.set_gate(str(pipe.tmp), ft, version, gate)
        return gate
    pipe.monkeypatch.setattr(pipe.main, "run_gate_for", run)
    pipe.set_collector(OK_RESULT)
    pipe.main.run_training_pipeline("traffic", {}, None, "drift")
    st = pipe.main.training_status["traffic"]
    g3 = {c["id"]: c for c in registry.find(registry.load(str(pipe.tmp), "traffic"), "v7")["gate"]["checks"]}["G3"]
    assert g3["status"] == "WARN" and st["gate_status"] == "WARN"
    assert _active(pipe) == "v1" and pipe.published == []
