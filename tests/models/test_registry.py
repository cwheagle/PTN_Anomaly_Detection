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


# ──────────────────────────────────────────────
# P1-1: 승격 게이트 결과 연동
# ──────────────────────────────────────────────
def _gate(status, checks=None, against="v1"):
    return {"status": status, "checks": checks or [], "evaluated_at": "2026-10-01 10:00:00", "data": {"against": against}}


def _chk(cid, status, msg="m"):
    return {"id": cid, "status": status, "message": msg}


def _setup_gated(d, gate, threshold=0.3):
    _add(d, "v1", activate=True)
    _add(d, "v2", activate=False, threshold=threshold)
    reg_mod.set_gate(str(d), FT, "v2", gate)


def test_gate_fail_blocks_promotion_until_force_and_carries_checks(d):
    """T-R1: gate FAIL -> PromotionWarning(검사 목록 포함), force 로 승격"""
    _setup_gated(d, _gate("FAIL", [_chk("G1", "PASS"), _chk("G3", "FAIL", "알람 과다")]))
    with pytest.raises(PromotionWarning) as e:
        reg_mod.promote(str(d), FT, "v2")
    assert e.value.warnings == ["[G3] 알람 과다"] and [c["id"] for c in e.value.checks] == ["G1", "G3"]
    assert reg_mod.load(str(d), FT)["active_version"] == "v1"
    out = reg_mod.promote(str(d), FT, "v2", force=True)
    assert out["active"] == "v2" and out["warnings"] == ["[G3] 알람 과다"]


def test_gate_g1_fail_cannot_be_forced(d):
    """T-R1: 깨진 모델(G1)은 force 로도 승격 불가"""
    _setup_gated(d, _gate("FAIL", [_chk("G1", "FAIL", "threshold 가 NaN")]))
    with pytest.raises(RegistryError) as e:
        reg_mod.promote(str(d), FT, "v2", force=True)
    assert not isinstance(e.value, PromotionWarning) and "G1" in str(e.value)
    assert reg_mod.load(str(d), FT)["active_version"] == "v1"


def test_gate_error_blocks_without_force(d):
    _setup_gated(d, _gate("ERROR", [_chk("ERROR", "ERROR", "게이트 실행 실패: db down")]))
    with pytest.raises(PromotionWarning):
        reg_mod.promote(str(d), FT, "v2")
    assert reg_mod.promote(str(d), FT, "v2", force=True)["active"] == "v2"


def test_gate_warn_promotes_and_returns_notes(d):
    """T-R2"""
    _setup_gated(d, _gate("WARN", [_chk("G1", "PASS"), _chk("G5", "WARN", "임계치 연속 상승")]))
    out = reg_mod.promote(str(d), FT, "v2")
    assert out["active"] == "v2" and out["warnings"] == ["[G5] 임계치 연속 상승"]


def test_gate_pass_promotes_cleanly_even_if_thresholds_differ_a_lot(d):
    """게이트가 있으면 그 결과가 판정 기준 (임계치 비교는 게이트의 G2 가 이미 수행)"""
    _setup_gated(d, _gate("PASS", [_chk("G2", "PASS")]), threshold=9.0)
    assert reg_mod.promote(str(d), FT, "v2")["warnings"] == []


def test_candidate_without_gate_keeps_threshold_comparison(d):
    """T-R3: 게이트 이전 후보 / 게이트 미기록 후보는 기존 compare_with_active 동작"""
    _add(d, "v1", activate=True)
    _add(d, "v2", activate=False, threshold=9.0)
    with pytest.raises(PromotionWarning) as e:
        reg_mod.promote(str(d), FT, "v2")
    assert "임계치" in e.value.warnings[0] and e.value.checks == []


def test_stale_gate_for_different_active_version_is_rechecked_against_current_active(d):
    """게이트가 v1 기준으로 계산됐는데 그 사이 활성 버전이 바뀌었으면, 임계치 비교는 현재 활성 기준으로 다시 하고 메모"""
    _add(d, "v1", activate=True)
    _add(d, "v2", activate=True, threshold=0.3)
    _add(d, "v3", activate=False, threshold=0.3)
    reg_mod.set_gate(str(d), FT, "v3", _gate("PASS", [_chk("G2", "PASS")], against="v1"))
    out = reg_mod.promote(str(d), FT, "v3")
    assert out["active"] == "v3" and any("게이트는 v1 기준" in w for w in out["warnings"])
    _add(d, "v4", activate=False, threshold=9.0)
    reg_mod.set_gate(str(d), FT, "v4", _gate("PASS", [_chk("G2", "PASS")], against="v1"))
    with pytest.raises(PromotionWarning):
        reg_mod.promote(str(d), FT, "v4")


def test_set_gate_unknown_version_and_list_versions_expose_gate(d):
    _add(d, "v1", activate=True)
    with pytest.raises(reg_mod.VersionNotFound):
        reg_mod.set_gate(str(d), FT, "v9", _gate("PASS"))
    _add(d, "v2", activate=False)
    reg_mod.set_gate(str(d), FT, "v2", _gate("WARN", [_chk("G5", "WARN")]))
    items = {v["version"]: v for v in reg_mod.list_versions(str(d), FT)["versions"]}
    assert items["v2"]["gate_status"] == "WARN" and items["v2"]["gate"]["checks"][0]["id"] == "G5"
    assert items["v1"]["gate_status"] is None and "trigger" in items["v1"]


def test_promote_reason_auto_gate_is_recorded_in_history(d):
    _setup_gated(d, _gate("PASS"))
    reg_mod.promote(str(d), FT, "v2", reason="auto-gate")
    assert reg_mod.load(str(d), FT)["history"][-1]["reason"] == "auto-gate"


# 발견 C: G2/G3 공통 배율 표기
@pytest.mark.parametrize("ratio,text", [(37.5, "×37.5"), (1.0, "×1.0"), (0.5, "1/2.0"), (0.4399 / 16.48, "1/37.5"),
                                         (None, "-"), (0, "-"), (float("inf"), "-"), (float("nan"), "-")])
def test_format_ratio(ratio, text):
    assert reg_mod.format_ratio(ratio) == text


def test_compare_with_active_messages_show_inverse_ratio_with_values_and_limits():
    reg = {"active_version": "v1", "versions": [{"version": "v1", "threshold": 16.48, "final_val_loss": 0.01}]}
    w = reg_mod.compare_with_active(reg, {"version": "v2", "threshold": 0.4399, "final_val_loss": 0.05})
    assert w[0] == "임계치 16.48 → 0.4399 (1/37.5, 기준 1/3 ~ ×3)"
    assert w[1] == "검증 손실 0.01 → 0.05 (×5.0, 기준 ≤ ×3)"
    assert not any("0.0배" in m for m in w)


# T-P3-R1: 정책 전용 버전 (P1-3, 재학습 없이 같은 가중치 + 다른 알람 정책)
def _with_meta_file(d, version="v1"):
    (d / f"{FT}_ae_{version}.json").write_text(json.dumps({"threshold": 0.3, "trained_at": "2026-10-01 10:00:00"}))


PRECISION = {"preset": "precision", "sigma_k": 2.0, "dyn_cap": 1.2, "threshold_scale": 3.0, "severity_decay": 0.5,
             "dampening_steps": {"1": 6, "2": 4, "3": 3}}


def test_policy_version_shares_weights_and_differs_only_in_policy(d):
    _add(d, "v1", True); _add(d, "v2", False)
    _with_meta_file(d, "v2")
    reg_mod.set_gate(str(d), FT, "v2", {"status": "PASS", "checks": []})
    e = reg_mod.create_policy_version(str(d), FT, "v2", PRECISION)
    assert e["version"] == "v3" and e["derived_from"] == "v2" and e["trigger"] == "policy"
    assert e["model_path"] == "traffic_ae_v2.pth" and e["scaler_path"] == "traffic_scaler_v2.joblib"   # 가중치·스케일러 공유
    assert e["config_path"] == "traffic_ae_v3.json" and e["alert_policy"] == PRECISION
    assert e["threshold"] == 0.3 and "gate" not in e                       # 원 버전의 게이트 결과는 가져오지 않음
    meta = json.load(open(d / "traffic_ae_v3.json"))
    assert meta["alert_policy"] == PRECISION and meta["derived_from"] == "v2" and meta["threshold"] == 0.3
    assert "alert_policy" not in json.load(open(d / "traffic_ae_v2.json"))  # 원 버전 메타는 불변


def test_policy_version_is_a_candidate_and_rollback_target_is_unchanged(d):
    _add(d, "v1", True); _add(d, "v2", False); _with_meta_file(d, "v2")
    reg_mod.create_policy_version(str(d), FT, "v2", PRECISION)
    reg = reg_mod.load(str(d), FT)
    assert reg["active_version"] == "v1" and reg_mod.status_of(reg, "v3") == "candidate"
    assert [h["version"] for h in reg["history"]] == ["v1"]
    reg_mod.promote(str(d), FT, "v3", force=True)
    assert reg_mod.rollback(str(d), FT)["active"] == "v1"                   # 롤백 대상은 직전 활성
    listing = {v["version"]: v for v in reg_mod.list_versions(str(d), FT)["versions"]}
    assert listing["v3"]["alert_policy"] == PRECISION and listing["v3"]["derived_from"] == "v2"
    assert listing["v1"]["alert_policy"] is None


def test_policy_version_errors(d):
    _add(d, "v1", True); _with_meta_file(d, "v1")
    with pytest.raises(reg_mod.VersionNotFound):
        reg_mod.create_policy_version(str(d), FT, "v9", PRECISION)
    (d / "traffic_scaler_v1.joblib").unlink()
    with pytest.raises(RegistryError):
        reg_mod.create_policy_version(str(d), FT, "v1", PRECISION)
    assert [v["version"] for v in reg_mod.load(str(d), FT)["versions"]] == ["v1"]   # 실패 시 등록 안 됨


# T-P3-G7 (U4'): G3 의 절대 상한 위반(WARN)은 수동 승격을 막지 않고, 상대 기준 위반(FAIL)은 force 가 필요
def test_g3_absolute_limit_warn_promotes_without_force_and_returns_the_message(d):
    msg = "홀드아웃 130포트 알람 0.100건/포트·일 > 절대 상한 0.050 (×2.0, 절대 상한 초과 — 장애가 많은 기간일 수 있음, 사람이 검토)"
    _setup_gated(d, _gate("WARN", [_chk("G1", "PASS"), _chk("G3", "WARN", msg)]))
    out = reg_mod.promote(str(d), FT, "v2")
    assert out["active"] == "v2" and out["warnings"] == [f"[G3] {msg}"]


def test_g3_relative_fail_needs_force(d):
    msg = "홀드아웃 130포트 알람 0.200건/포트·일 > 기준 0.150 (×1.3, 활성 모델 대비 과다)"
    _setup_gated(d, _gate("FAIL", [_chk("G1", "PASS"), _chk("G3", "FAIL", msg)]))
    with pytest.raises(PromotionWarning) as e:
        reg_mod.promote(str(d), FT, "v2")
    assert e.value.warnings == [f"[G3] {msg}"]
    out = reg_mod.promote(str(d), FT, "v2", force=True)
    assert out["active"] == "v2" and out["warnings"] == [f"[G3] {msg}"]
