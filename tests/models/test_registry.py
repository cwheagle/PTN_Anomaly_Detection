"""
Model Registry 검증: 후보 저장 / 승격 / 롤백 / 경고 / 구버전 호환 / 동시 접근

실행 방법:
  pytest tests/models/test_registry.py -v
"""
import json
import os
import threading

import pytest

from src.models import registry as reg_mod
from src.models.registry import PromotionWarning, RegistryError

FT = "traffic"


def _entry(model_dir, version, threshold=0.3, val=0.02, files=True):
    """버전 엔트리 + (선택) 실제 파일 생성"""
    entry = {"version": version, "trained_at": "2026-10-01 10:00:00",
             "model_path": f"{FT}_ae_{version}.pth", "scaler_path": f"{FT}_scaler_{version}.joblib",
             "config_path": f"{FT}_ae_{version}.json", "threshold": threshold,
             "final_val_loss": val, "baseline_mse": 0.03, "samples_used": 1000}
    if files:
        for k in ("model_path", "scaler_path", "config_path"):
            (model_dir / entry[k]).write_text("x")
    return entry


def _add(model_dir, version, activate, **kw):
    with reg_mod.transaction(str(model_dir), FT) as reg:
        return reg_mod.add_version(reg, _entry(model_dir, version, **kw), activate)


@pytest.fixture
def d(tmp_path):
    return tmp_path


def test_load_missing_registry_is_empty_and_normalized(d):
    reg = reg_mod.load(str(d), FT)
    assert reg == {"active_version": None, "versions": [], "history": []}


def test_first_model_is_activated_even_when_candidate_requested(d):
    """활성 버전이 없는 최초 학습은 자동 활성화 (없으면 시스템이 모델 없이 시작됨)"""
    assert _add(d, "v1", activate=False) is True
    reg = reg_mod.load(str(d), FT)
    assert reg["active_version"] == "v1" and reg_mod.status_of(reg, "v1") == "active"


def test_second_model_stays_candidate_and_active_is_unchanged(d):
    _add(d, "v1", activate=True)
    assert _add(d, "v2", activate=False) is False
    reg = reg_mod.load(str(d), FT)
    assert reg["active_version"] == "v1"
    assert reg_mod.status_of(reg, "v2") == "candidate"
    assert [v["is_active"] for v in reg["versions"]] == [True, False]


def test_activate_true_switches_active_version(d):
    _add(d, "v1", activate=True)
    assert _add(d, "v2", activate=True) is True
    assert reg_mod.load(str(d), FT)["active_version"] == "v2"


def test_promote_candidate_records_history_and_previous(d):
    _add(d, "v1", True); _add(d, "v2", False)
    r = reg_mod.promote(str(d), FT, "v2")
    assert r == {"previous": "v1", "active": "v2", "warnings": []}
    reg = reg_mod.load(str(d), FT)
    assert reg["active_version"] == "v2"
    assert reg_mod.status_of(reg, "v1") == "retired"
    assert [h["version"] for h in reg["history"]] == ["v1", "v2"]


def test_promote_unknown_missing_files_and_already_active_are_rejected(d):
    _add(d, "v1", True)
    _add(d, "v2", False, files=False)                      # 파일 없는 후보
    with pytest.raises(RegistryError, match="없습니다"):
        reg_mod.promote(str(d), FT, "v9")
    with pytest.raises(RegistryError, match="파일이 없습니다"):
        reg_mod.promote(str(d), FT, "v2")
    with pytest.raises(RegistryError, match="이미 활성"):
        reg_mod.promote(str(d), FT, "v1")


def test_promote_warns_on_abnormal_threshold_and_force_overrides(d):
    """회귀: 2026-10-01 E2E 에서 임계치 9.0(기존 0.17)인 시험 모델이 검증 없이 활성화됨"""
    _add(d, "v1", True, threshold=0.17)
    _add(d, "v2", False, threshold=9.0)
    with pytest.raises(PromotionWarning) as e:
        reg_mod.promote(str(d), FT, "v2")
    assert any("임계치" in w for w in e.value.warnings)
    assert reg_mod.load(str(d), FT)["active_version"] == "v1"      # 거부되면 상태 불변
    r = reg_mod.promote(str(d), FT, "v2", force=True)
    assert r["active"] == "v2" and r["warnings"]


def test_promote_warns_on_validation_loss_blowup(d):
    _add(d, "v1", True, val=0.02)
    _add(d, "v2", False, val=0.5)
    with pytest.raises(PromotionWarning) as e:
        reg_mod.promote(str(d), FT, "v2")
    assert any("검증 손실" in w for w in e.value.warnings)


def test_normal_candidate_has_no_warnings(d):
    _add(d, "v1", True, threshold=0.27, val=3.0)
    _add(d, "v2", False, threshold=0.61, val=4.5)           # 2.3배, 1.5배: 정상 범위
    assert reg_mod.promote(str(d), FT, "v2")["warnings"] == []


def test_rollback_returns_to_previous_active_and_toggles(d):
    _add(d, "v1", True); _add(d, "v2", True)
    assert reg_mod.rollback(str(d), FT) == {"previous": "v2", "active": "v1", "warnings": []}
    assert reg_mod.load(str(d), FT)["active_version"] == "v1"
    assert reg_mod.rollback(str(d), FT)["active"] == "v2"    # 직전 활성으로 다시 토글


def test_rollback_without_previous_is_rejected(d):
    _add(d, "v1", True)
    with pytest.raises(RegistryError, match="이전 활성 버전이 없습니다"):
        reg_mod.rollback(str(d), FT)


def test_rollback_skips_versions_whose_files_were_removed(d):
    _add(d, "v1", True); _add(d, "v2", True); _add(d, "v3", True)
    os.remove(d / "traffic_ae_v2.pth")
    assert reg_mod.rollback(str(d), FT)["active"] == "v1"


def test_legacy_registry_without_history_is_supported(d):
    """Phase 11 시절 레지스트리(history 없음): 활성 버전을 이력의 시작으로 간주, 후보 저장/승격 가능"""
    legacy = {"active_version": "v1", "versions": [dict(_entry(d, "v1"), is_active=True)]}
    (d / f"{FT}_registry.json").write_text(json.dumps(legacy))
    reg = reg_mod.load(str(d), FT)
    assert [h["version"] for h in reg["history"]] == ["v1"]
    with pytest.raises(RegistryError):
        reg_mod.rollback(str(d), FT)                         # 되돌릴 이전 버전 없음
    _add(d, "v2", False)
    assert reg_mod.promote(str(d), FT, "v2")["previous"] == "v1"


def test_corrupt_registry_file_loads_as_empty_instead_of_crashing(d):
    (d / f"{FT}_registry.json").write_text("{not json")
    assert reg_mod.load(str(d), FT)["versions"] == []


def test_save_is_atomic_and_leaves_no_temp_file(d):
    _add(d, "v1", True)
    assert not (d / f"{FT}_registry.json.tmp").exists()
    assert json.loads((d / f"{FT}_registry.json").read_text())["active_version"] == "v1"


def test_version_ids_never_reuse_after_gaps(d):
    _add(d, "v1", True); _add(d, "v5", False)
    assert reg_mod.next_version_id(reg_mod.load(str(d), FT)) == "v6"


def test_list_versions_reports_status_for_ui(d):
    _add(d, "v1", True); _add(d, "v2", False)
    out = reg_mod.list_versions(str(d), FT)
    assert out["active_version"] == "v1"
    assert {v["version"]: v["status"] for v in out["versions"]} == {"v1": "active", "v2": "candidate"}


def test_concurrent_transactions_do_not_lose_versions(d):
    """학습 스레드와 승격 API 가 동시에 레지스트리를 수정해도 항목이 유실되지 않아야 함"""
    errors = []

    def worker():
        try:
            with reg_mod.transaction(str(d), FT) as reg:
                vid = reg_mod.next_version_id(reg)
                reg_mod.add_version(reg, _entry(d, vid), activate=False)
        except Exception as e:                                # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    [t.start() for t in threads]; [t.join() for t in threads]
    reg = reg_mod.load(str(d), FT)
    assert not errors
    assert len({v["version"] for v in reg["versions"]}) == 20
