"""
모델 버전 API 검증: 후보/승격/롤백 (실제 FastAPI 앱 + 임시 레지스트리)

main.py 는 import 시 MySQL 에 접속하므로 DB 가 없으면 건너뛴다.

실행 방법:
  pytest tests/api/test_model_endpoints.py -v
"""
import json

import pytest

from src.models import registry as reg_mod


@pytest.fixture
def client(tmp_path, monkeypatch):
    try:
        import src.api.main as main
    except Exception as e:                                  # DB 미접속 등
        pytest.skip(f"API 앱을 불러올 수 없음 (DB 필요): {str(e)[:60]}")
    from fastapi.testclient import TestClient

    # 임시 폴더를 모델 폴더로 사용: v1(활성, 정상), v2(후보, 임계치 비정상), v3(후보, 정상)
    def add(version, threshold, val, activate):
        entry = {"version": version, "trained_at": "2026-10-01 10:00:00",
                 "model_path": f"traffic_ae_{version}.pth", "scaler_path": f"traffic_scaler_{version}.joblib",
                 "config_path": f"traffic_ae_{version}.json", "threshold": threshold, "final_val_loss": val}
        for k in ("model_path", "scaler_path", "config_path"):
            (tmp_path / entry[k]).write_text("x")
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


def test_error_codes(client):
    assert client.post("/api/model/promote?ft=traffic&version=v9").status_code == 404
    assert client.post("/api/model/promote?ft=traffic&version=v1").status_code == 409     # 이미 활성
    assert client.post("/api/model/rollback?ft=traffic").status_code == 409               # 이전 버전 없음
    assert client.post("/api/model/promote?ft=bad&version=v1").status_code == 422        # 잘못된 트랙
