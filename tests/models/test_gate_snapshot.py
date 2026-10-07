"""
T-P6-U0: 승격 게이트 기본 동작 스냅샷 (P1-6 변경 전 출력 고정)

P1-6 의 U0~U3 전 기간에 기본 설정의 `check_g2`·`check_g3`·`compare_with_active`·`registry._gate_findings`·
`registry.promote` 409 경고·`run_gate` 결과 키가 **바이트 단위로 같아야** 한다(설계서 7장, R8).
기존 테스트는 메시지 일부 문자열만 확인하므로 여기서는 상태·전체 메시지·value·limit·detail 을 리터럴로 고정한다.
기대값은 변경 전 코드를 그대로 실행해 얻은 값이다. 값을 바꾸는 변경은 이 파일을 함께 바꿔야 하며 그 자체가 기본 동작 변경이다.
"""
import pytest

from src.models import promotion_gate as pg
from src.models import registry as reg_mod
from src.models.registry import PromotionWarning
from src.pipeline.retrain_policy import RetrainPolicy
from tests.models.test_promotion_gate import SENSITIVE, _fetch, _gate_policy, _window

L = RetrainPolicy()
FT = "traffic"


def _meta(version="v2", threshold=0.3, val=0.02):
    return {"version": version, "threshold": threshold, "final_val_loss": val}


def _stats(rate, ports=100, port_days=None):
    d = {"ports": ports, "incidents": int(rate * ports * 3), "incidents_per_port_day": rate}
    if port_days:
        d["port_days"] = port_days
    return d


def _tup(c):
    return (c.id, c.status, c.value, c.limit, c.message, c.detail)


# ── check_g2 ──
G2_CASES = [
    ("pass", _meta("v2"), _meta("v1"),
     ("G2", "PASS", 1.0, 3.0, "임계치·검증 손실이 활성 모델과 비슷함", None)),
    ("threshold_high", _meta("v2", 0.91), _meta("v1"),
     ("G2", "FAIL", 0.91 / 0.3, 3.0, "임계치 0.3 → 0.91 (×3.0, 기준 1/3 ~ ×3)", None)),
    ("threshold_low", _meta("v2", 0.09), _meta("v1"),
     ("G2", "FAIL", 0.09 / 0.3, 3.0, "임계치 0.3 → 0.09 (1/3.3, 기준 1/3 ~ ×3)", None)),
    ("val_loss_worse", _meta("v2", 0.3, 0.07), _meta("v1"),
     ("G2", "FAIL", 1.0, 3.0, "검증 손실 0.02 → 0.07 (×3.5, 기준 ≤ ×3)", None)),
    ("both", _meta("v2", 0.91, 0.07), _meta("v1"),
     ("G2", "FAIL", 0.91 / 0.3, 3.0,
      "임계치 0.3 → 0.91 (×3.0, 기준 1/3 ~ ×3); 검증 손실 0.02 → 0.07 (×3.5, 기준 ≤ ×3)", None)),
    ("no_active", _meta("v2"), None, ("G2", "SKIP", None, None, "비교할 활성 모델 없음", None)),
    # 현행은 검증 손실이 훨씬 낮아져도(활성 3.01 대 후보 0.0002) PASS — P1-6 ② 의 동기, 기본 동작으로 고정
    ("val_loss_much_lower", _meta("v2", 0.3, 0.0002), _meta("v1", 0.3, 3.01),
     ("G2", "PASS", 1.0, 3.0, "임계치·검증 손실이 활성 모델과 비슷함", None)),
]


@pytest.mark.parametrize("name,cand,active,expected", G2_CASES, ids=[c[0] for c in G2_CASES])
def test_snapshot_check_g2(name, cand, active, expected):
    got = _tup(pg.check_g2(cand, active))
    assert got[:2] == expected[:2] and got[3:] == expected[3:]
    assert got[2] == (pytest.approx(expected[2]) if expected[2] is not None else None)


# ── check_g3 ──
D = {"cand": {"incidents": 3, "port_days": 300}, "active": {"incidents": 3, "port_days": 300}}
G3_CASES = [
    ("fail_relative", _stats(0.016), _stats(0.010), "FAIL", 0.016, 0.015,
     "홀드아웃 100포트 알람 0.016건/포트·일 > 기준 0.015 (×1.1, 활성 모델 대비 과다; 인시던트 4건/300포트·일)",
     {"cand": {"incidents": 4, "port_days": 300}, "active": {"incidents": 3, "port_days": 300}}),
    ("warn_absolute", _stats(0.051), _stats(0.2), "WARN", 0.051, 0.05,
     "홀드아웃 100포트 알람 0.051건/포트·일 > 절대 상한 0.050 (×1.0, 절대 상한 초과 — 장애가 많은 기간일 수 있음, "
     "사람이 검토; 인시던트 15건/300포트·일)",
     {"cand": {"incidents": 15, "port_days": 300}, "active": {"incidents": 60, "port_days": 300}}),
    ("pass_with_active", _stats(0.010), _stats(0.010), "PASS", 0.01, 0.015,
     "홀드아웃 100포트 알람 0.010건/포트·일 ≤ 0.015 (인시던트 3건/300포트·일)", D),
    ("pass_no_active", _stats(0.03), None, "PASS", 0.03, 0.05,
     "홀드아웃 100포트 알람 0.030건/포트·일 ≤ 0.050 (인시던트 9건/300포트·일)",
     {"cand": {"incidents": 9, "port_days": 300}, "active": None}),
    ("warn_no_active", _stats(0.06), None, "WARN", 0.06, 0.05,
     "홀드아웃 100포트 알람 0.060건/포트·일 > 절대 상한 0.050 (×1.2, 절대 상한 초과 — 장애가 많은 기간일 수 있음, "
     "사람이 검토; 인시던트 18건/300포트·일)",
     {"cand": {"incidents": 18, "port_days": 300}, "active": None}),
    ("skip_no_data", None, _stats(0.01), "SKIP", None, None, "게이트 데이터(홀드아웃 포트)가 없음", None),
    ("skip_small", _stats(0.01, ports=10, port_days=30), _stats(0.01), "SKIP", 30, 150,
     "포트·일 30<150, 판정 불가 (홀드아웃 10포트, 인시던트 0건) — 사람이 검토하세요",
     {"cand": {"incidents": 0, "port_days": 30}, "active": {"incidents": 3, "port_days": 300}}),
]


@pytest.mark.parametrize("name,cand,active,status,value,limit,msg,detail", G3_CASES, ids=[c[0] for c in G3_CASES])
def test_snapshot_check_g3(name, cand, active, status, value, limit, msg, detail):
    got = _tup(pg.check_g3(cand, active, L))
    assert got[:2] == ("G3", status) and got[4:] == (msg, detail)
    assert got[2] == (pytest.approx(value) if value is not None else None)
    assert got[3] == (pytest.approx(limit) if limit is not None else None)


# ── registry.compare_with_active ──
def _reg(active_th=0.3, active_val=0.02):
    return {"active_version": "v1", "versions": [_meta("v1", active_th, active_val)]}


CMP_CASES = [
    ("ok", _meta(), []),
    ("threshold_high", _meta("v2", 0.91), ["임계치 0.3 → 0.91 (×3.0, 기준 1/3 ~ ×3)"]),
    ("threshold_low", _meta("v2", 0.09), ["임계치 0.3 → 0.09 (1/3.3, 기준 1/3 ~ ×3)"]),
    ("val_worse", _meta("v2", 0.3, 0.07), ["검증 손실 0.02 → 0.07 (×3.5, 기준 ≤ ×3)"]),
    ("both", _meta("v2", 0.91, 0.07),
     ["임계치 0.3 → 0.91 (×3.0, 기준 1/3 ~ ×3)", "검증 손실 0.02 → 0.07 (×3.5, 기준 ≤ ×3)"]),
    ("val_much_lower", _meta("v2", 0.3, 0.0002), []),
    ("same_version", _meta("v1"), []),
]


@pytest.mark.parametrize("name,entry,expected", CMP_CASES, ids=[c[0] for c in CMP_CASES])
def test_snapshot_compare_with_active(name, entry, expected):
    assert reg_mod.compare_with_active(_reg(), entry) == expected


def test_snapshot_compare_with_active_no_active_or_no_val_loss():
    assert reg_mod.compare_with_active({"active_version": None, "versions": []}, _meta()) == []
    assert reg_mod.compare_with_active(_reg(0.3, None), _meta("v2", 0.3, 0.07)) == []


# ── registry._gate_findings ──
def _gate(status, checks, against="v1"):
    return {"status": status, "checks": checks, "data": {"against": against}}


def _chk(cid, status, msg="m"):
    return {"id": cid, "status": status, "message": msg}


def _with_gate(entry, gate):
    e = dict(entry)
    e["gate"] = gate
    return e


def test_snapshot_gate_findings_without_gate_uses_comparison_as_blocking():
    assert reg_mod._gate_findings(_reg(), _meta("v2", 0.91, 0.07)) == (
        ["임계치 0.3 → 0.91 (×3.0, 기준 1/3 ~ ×3)", "검증 손실 0.02 → 0.07 (×3.5, 기준 ≤ ×3)"], [], [])
    assert reg_mod._gate_findings(_reg(), _meta()) == ([], [], [])


def test_snapshot_gate_findings_with_gate():
    checks = [_chk("G2", "FAIL", "g2 msg"), _chk("G3", "WARN", "g3 warn"), _chk("G4", "SKIP")]
    e = _with_gate(_meta(), _gate("FAIL", checks))
    assert reg_mod._gate_findings(_reg(), e) == (
        ["[G2] g2 msg"], ["[G3] g3 warn"],
        [{"id": "G2", "status": "FAIL", "message": "g2 msg"}, {"id": "G3", "status": "WARN", "message": "g3 warn"},
         {"id": "G4", "status": "SKIP", "message": "m"}])
    e = _with_gate(_meta(), _gate("ERROR", [_chk("ERROR", "ERROR", "boom")]))
    assert reg_mod._gate_findings(_reg(), e) == (["[ERROR] boom"], [], [{"id": "ERROR", "status": "ERROR", "message": "boom"}])
    e = _with_gate(_meta(), {"status": "ERROR", "checks": [], "data": {"against": "v1"}})
    assert reg_mod._gate_findings(_reg(), e) == (["게이트 실행 오류"], [], [])


def test_snapshot_gate_findings_stale_gate_rechecks_g2_against_current_active():
    checks = [_chk("G2", "FAIL", "old g2"), _chk("G3", "FAIL", "g3 fail")]
    e = _with_gate(_meta("v2", 0.91, 0.07), _gate("FAIL", checks, against="v0"))
    assert reg_mod._gate_findings(_reg(), e) == (
        ["[G3] g3 fail", "임계치 0.3 → 0.91 (×3.0, 기준 1/3 ~ ×3)", "검증 손실 0.02 → 0.07 (×3.5, 기준 ≤ ×3)"],
        ["게이트는 v0 기준으로 계산됨 — 현재 활성 v1. 게이트 재실행을 권장합니다."],
        [{"id": "G2", "status": "FAIL", "message": "old g2"}, {"id": "G3", "status": "FAIL", "message": "g3 fail"}])
    e = _with_gate(_meta(), _gate("PASS", [_chk("G2", "PASS")], against=None))
    assert reg_mod._gate_findings(_reg(), e) == (
        [], ["게이트는 활성 모델 없음 기준으로 계산됨 — 현재 활성 v1. 게이트 재실행을 권장합니다."],
        [{"id": "G2", "status": "PASS", "message": "m"}])


# ── registry.promote 409 경고 ──
def _add(d, version, activate, **kw):
    entry = {"version": version, "trained_at": "2026-10-01 10:00:00", "model_path": f"{FT}_ae_{version}.pth",
             "scaler_path": f"{FT}_scaler_{version}.joblib", "config_path": f"{FT}_ae_{version}.json",
             "threshold": kw.get("threshold", 0.3), "final_val_loss": kw.get("val", 0.02),
             "baseline_mse": 0.03, "samples_used": 1000}
    for k in ("model_path", "scaler_path", "config_path"):
        (d / entry[k]).write_text("x")
    with reg_mod.transaction(str(d), FT) as reg:
        reg_mod.add_version(reg, entry, activate)


def test_snapshot_promote_warning_contents_without_gate(tmp_path):
    _add(tmp_path, "v1", True)
    _add(tmp_path, "v2", False, threshold=0.91, val=0.07)
    with pytest.raises(PromotionWarning) as e:
        reg_mod.promote(str(tmp_path), FT, "v2")
    assert e.value.warnings == ["임계치 0.3 → 0.91 (×3.0, 기준 1/3 ~ ×3)", "검증 손실 0.02 → 0.07 (×3.5, 기준 ≤ ×3)"]
    assert e.value.checks == []
    out = reg_mod.promote(str(tmp_path), FT, "v2", force=True, reason="snap")
    assert out == {"previous": "v1", "active": "v2",
                   "warnings": ["임계치 0.3 → 0.91 (×3.0, 기준 1/3 ~ ×3)", "검증 손실 0.02 → 0.07 (×3.5, 기준 ≤ ×3)"]}


def test_snapshot_promote_with_gate_warning_note_and_blocking(tmp_path):
    _add(tmp_path, "v1", True)
    _add(tmp_path, "v2", False)
    gate = _gate("FAIL", [_chk("G2", "FAIL", "g2 msg"), _chk("G3", "WARN", "g3 warn")])
    reg_mod.set_gate(str(tmp_path), FT, "v2", gate)
    with pytest.raises(PromotionWarning) as e:
        reg_mod.promote(str(tmp_path), FT, "v2")
    assert e.value.warnings == ["[G2] g2 msg"]
    assert [c["id"] for c in e.value.checks] == ["G2", "G3"]
    out = reg_mod.promote(str(tmp_path), FT, "v2", force=True)
    assert out["warnings"] == ["[G2] g2 msg", "[G3] g3 warn"]


# ── run_gate 결과 구조 (키 집합) ──
def test_snapshot_run_gate_result_keys(tiny_env, tiny_scenario):
    res = pg.run_gate(str(tiny_env), FT, "v2", _window(), _gate_policy(), _fetch(tiny_scenario), SENSITIVE)
    assert [c.id for c in res.checks] == ["G1", "G2", "G3", "G4", "G5"]
    assert set(res.to_dict()) == {"status", "checks", "evaluated_at", "data"}
    assert set(res.data) == {"version", "against", "gate_start", "gate_end", "track", "alert_policy", "ports",
                             "candidate_stats", "active_stats", "canary"}
    assert set(res.data["alert_policy"]) == {"candidate", "active"}
    assert set(res.data["candidate_stats"]) == {"ports", "incidents", "alarm_rows", "port_days",
                                                "incidents_per_port_day", "overlap_with_active"}
    assert set(res.data["active_stats"]) == {"ports", "incidents", "alarm_rows", "port_days", "incidents_per_port_day"}
    assert res.data["canary"] is None
    g2 = {c.id: c for c in res.checks}["G2"]
    assert set(vars(g2)) == {"id", "status", "value", "limit", "message", "detail"}
