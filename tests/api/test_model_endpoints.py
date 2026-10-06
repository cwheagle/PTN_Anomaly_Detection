"""
모델 버전 API 검증: 후보/승격/롤백 (실제 FastAPI 앱 + 임시 레지스트리)

main.py 는 import 시 MySQL 풀을 만들므로 conftest 의 api_main 픽스처(풀 목 대체)로 불러온다.

실행 방법:
  pytest tests/api/test_model_endpoints.py -v
"""
import json

import pytest

from src.models import registry as reg_mod


@pytest.fixture
def client(tmp_path, monkeypatch, api_main):
    main = api_main                                         # DB 없이 import (tests/api/conftest.py)
    from fastapi.testclient import TestClient

    # 임시 폴더를 모델 폴더로 사용: v1(활성, 정상), v2(후보, 임계치 비정상), v3(후보, 정상)
    def add(version, threshold, val, activate):
        entry = {"version": version, "trained_at": "2026-10-01 10:00:00",
                 "model_path": f"traffic_ae_{version}.pth", "scaler_path": f"traffic_scaler_{version}.joblib",
                 "config_path": f"traffic_ae_{version}.json", "threshold": threshold, "final_val_loss": val}
        for k in ("model_path", "scaler_path"):
            (tmp_path / entry[k]).write_text("x")
        (tmp_path / entry["config_path"]).write_text(json.dumps({"threshold": threshold}))      # 메타는 JSON (정책 전용 버전이 복사)
        with reg_mod.transaction(str(tmp_path), "traffic") as reg:
            reg_mod.add_version(reg, entry, activate)

    add("v1", 0.17, 0.02, True)
    add("v2", 9.0, 0.66, False)
    add("v3", 0.25, 0.03, False)
    monkeypatch.setitem(main.PATHS, "traffic", {"model": str(tmp_path / "traffic_ae.pth"),
                                                "scaler": str(tmp_path / "traffic_scaler.joblib")})
    published = []
    monkeypatch.setattr(main.redis_client, "publish", lambda ch, msg: published.append((ch, json.loads(msg))))
    c = TestClient(main.app)
    c.published = published
    c.model_dir = str(tmp_path)
    return c


def test_versions_lists_status_per_version(client):
    d = client.get("/api/model/versions?ft=traffic").json()["traffic"]
    assert d["active_version"] == "v1"
    assert {v["version"]: v["status"] for v in d["versions"]} == {"v1": "active", "v2": "candidate", "v3": "candidate"}


def test_status_endpoint_exposes_active_and_candidates(client):
    s = client.get("/api/model/status").json()["traffic"]
    assert s["active_version"] == "v1" and s["candidate_versions"] == ["v2", "v3"]


def test_promote_abnormal_candidate_is_rejected_with_warnings_and_not_broadcast(client):
    """회귀: 임계치 9.0 인 후보가 검증 없이 활성화되던 문제"""
    r = client.post("/api/model/promote?ft=traffic&version=v2")
    assert r.status_code == 409 and r.json()["detail"]["warnings"]
    assert client.get("/api/model/versions?ft=traffic").json()["traffic"]["active_version"] == "v1"
    assert client.published == []                              # 거부되었으니 Consumer 리로드도 없음


def test_promote_with_force_activates_and_broadcasts_reload(client):
    r = client.post("/api/model/promote?ft=traffic&version=v2&force=true")
    assert r.status_code == 200 and r.json()["previous"] == "v1" and r.json()["warnings"]
    assert client.published == [("ptn_control", {"action": "reload", "track": "traffic"})]


def test_promote_normal_candidate_then_rollback(client):
    assert client.post("/api/model/promote?ft=traffic&version=v3").json()["active"] == "v3"
    back = client.post("/api/model/rollback?ft=traffic").json()
    assert back["previous"] == "v3" and back["active"] == "v1"
    assert client.get("/api/model/versions?ft=traffic").json()["traffic"]["active_version"] == "v1"
    assert len(client.published) == 2                          # 승격 + 롤백 각각 리로드 알림


def _history(client):
    return [(h["version"], h["reason"]) for h in reg_mod.load(client.model_dir, "traffic")["history"]]


def test_promote_reason_is_recorded_in_history_and_distinct_from_rollback(client):
    assert client.post("/api/model/promote?ft=traffic&version=v3&reason=%20P1-3%20rehearsal%20").status_code == 200
    client.post("/api/model/rollback?ft=traffic")
    assert _history(client)[-2:] == [("v3", "P1-3 rehearsal"), ("v1", "rollback")]


def test_promote_without_or_blank_reason_defaults_to_manual(client):
    client.post("/api/model/promote?ft=traffic&version=v3")
    client.post("/api/model/rollback?ft=traffic")
    client.post("/api/model/promote?ft=traffic&version=v3&reason=%20%20")
    assert [r for _, r in _history(client)] == ["trained", "manual", "rollback", "manual"]


def test_promote_reason_is_recorded_with_force_and_too_long_is_rejected(client):
    assert client.post("/api/model/promote?ft=traffic&version=v2&force=true&reason=forced-ok").status_code == 200
    assert _history(client)[-1] == ("v2", "forced-ok")
    assert client.post("/api/model/promote?ft=traffic&version=v3&reason=" + "x" * 101).status_code == 422


def test_error_codes(client):
    assert client.post("/api/model/promote?ft=traffic&version=v9").status_code == 404
    assert client.post("/api/model/promote?ft=traffic&version=v1").status_code == 409     # 이미 활성
    assert client.post("/api/model/rollback?ft=traffic").status_code == 409               # 이전 버전 없음
    assert client.post("/api/model/promote?ft=bad&version=v1").status_code == 422        # 잘못된 트랙


# ──────────────────────────────────────────────
# T-P3-E1/E2: 모델별 알람 정책 API (P1-3)
# ──────────────────────────────────────────────
PRECISION_META = {"preset": "precision", "sigma_k": 2.0, "dyn_cap": 1.2, "threshold_scale": 3.0, "severity_decay": 0.5,
                  "dampening_steps": {"1": 6, "2": 4, "3": 3}}


@pytest.fixture
def launched(client, api_main, monkeypatch):
    calls = []
    monkeypatch.setattr(api_main, "run_training_pipeline", lambda *a, **k: calls.append(a))
    yield calls
    api_main.training_status["traffic"]["is_training"] = False       # /train 이 '학습 중'으로 표시한 채 파이프라인이 안 돌므로 정리


def test_train_passes_alert_policy_preset_to_pipeline(client, launched, api_main):
    """T-P3-E1: 프리셋 이름이 파이프라인에 전달됨 / 미지정이면 None(= 활성 모델 승계)"""
    assert client.post("/api/model/train?ft=traffic&alert_policy=precision").json()["alert_policy"] == "precision"
    api_main.training_status["traffic"]["is_training"] = False
    client.post("/api/model/train?ft=traffic")
    assert [a[-1] for a in launched] == ["precision", None]
    assert launched[0][:2] == ("traffic", {})


def test_train_rejects_unknown_preset_without_starting(client, launched):
    r = client.post("/api/model/train?ft=traffic&alert_policy=fast")
    assert r.status_code == 422 and "fast" in r.json()["detail"] and launched == []


def test_alert_policy_is_inherited_from_active_model_unless_preset_given(client, api_main):
    """T-P3-E1: 드리프트 재학습처럼 프리셋 없이 학습하면 활성 모델의 정책을 이어받음, 활성에도 없으면 None"""
    assert api_main._resolve_alert_policy("traffic") is None                                   # v1 에 정책 없음
    with reg_mod.transaction(str(api_main._model_dir("traffic")), "traffic") as reg:
        reg_mod.find(reg, "v1")["alert_policy"] = PRECISION_META
    assert api_main._resolve_alert_policy("traffic") == PRECISION_META                         # 승계
    assert api_main._resolve_alert_policy("traffic", "default")["preset"] == "default"         # 지정이 우선
    assert api_main._resolve_alert_policy("traffic", "precision") == PRECISION_META


@pytest.fixture
def gate_calls(client, api_main, monkeypatch):
    calls = []
    monkeypatch.setattr(api_main, "run_gate_for", lambda ft, v, now=None: calls.append((ft, v)) or {"status": "PASS", "checks": []})
    return calls


def test_policy_endpoint_creates_candidate_with_same_weights_and_runs_gate(client, api_main, gate_calls):
    """T-P3-E2: 정상 — 새 후보 버전(같은 가중치, 정책만 다름), 활성 불변, 게이트 실행"""
    r = client.post("/api/model/policy?ft=traffic&version=v3&preset=precision")
    assert r.status_code == 200
    body = r.json()
    assert (body["version"], body["derived_from"], body["alert_policy"]) == ("v4", "v3", "precision")
    assert gate_calls == [("traffic", "v4")]
    versions = {v["version"]: v for v in client.get("/api/model/versions?ft=traffic").json()["traffic"]["versions"]}
    assert versions["v4"]["status"] == "candidate" and versions["v4"]["alert_policy"] == PRECISION_META
    assert client.get("/api/model/versions?ft=traffic").json()["traffic"]["active_version"] == "v1"
    assert client.published == []                                                              # 활성 불변 -> Consumer 리로드 없음


def test_policy_endpoint_errors(client, api_main, gate_calls):
    assert client.post("/api/model/policy?ft=traffic&version=v3&preset=fast").status_code == 422
    assert client.post("/api/model/policy?ft=traffic&version=v9&preset=precision").status_code == 404
    api_main.training_status["traffic"]["is_training"] = True
    try:
        assert client.post("/api/model/policy?ft=traffic&version=v3&preset=precision").status_code == 409
    finally:
        api_main.training_status["traffic"]["is_training"] = False
    assert gate_calls == []                                                                    # 실패한 요청은 게이트를 돌리지 않음
    assert [v["version"] for v in client.get("/api/model/versions?ft=traffic").json()["traffic"]["versions"]] == ["v1", "v2", "v3"]
