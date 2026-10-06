"""
Lightweight Model Registry (버전 저장 / 후보 / 승격 / 롤백)

트랙(traffic/optical)마다 `<model_dir>/<ft>_registry.json` 하나를 관리한다.

  {
    "active_version": "v2",
    "versions": [ {version, trained_at, model_path, scaler_path, config_path, threshold,
                   final_val_loss, baseline_mse, samples_used, is_active}, ... ],
    "history":  [ {version, activated_at, reason}, ... ]      # 활성화 이력 (롤백 대상 계산용)
  }

버전의 상태는 저장하지 않고 파생한다:
  active    : active_version 과 같음
  retired   : 활성화된 적이 있으나 지금은 아님 (롤백 대상)
  candidate : 한 번도 활성화된 적 없음 (재학습 결과, 승격 대기)

설계 원칙
  - 재학습 결과는 기본적으로 '후보'로만 저장하고, 사람이 승격해야 Consumer 에 반영된다.
    (활성 버전이 아직 없는 최초 학습만 자동 활성화)
  - 레지스트리는 Consumer 등 다른 프로세스가 읽으므로 임시 파일에 쓴 뒤 교체(원자적)한다.
  - 승격 시 후보가 현재 모델과 크게 어긋나면(임계치/검증 손실) 경고하고, force 없이는 거부한다.
    승격 게이트(src/models/promotion_gate.py)의 결과가 엔트리의 `gate` 에 있으면 그것으로 판정하고,
    없으면(게이트 이전 후보 등) 임계치/검증 손실 비교만 한다.
"""
import json
import math
import os
import threading
from contextlib import contextmanager
from datetime import datetime

# 후보가 활성 모델 대비 이 배율을 넘게(또는 역수 미만으로) 어긋나면 경고
THRESHOLD_RATIO_LIMIT = 3.0
VAL_LOSS_RATIO_LIMIT = 3.0

_LOCK = threading.RLock()


class RegistryError(Exception):
    """레지스트리 조작 오류 (버전 없음, 파일 없음, 롤백 대상 없음 등)"""


class VersionNotFound(RegistryError):
    """요청한 버전이 레지스트리에 없음"""


class PromotionWarning(RegistryError):
    """후보가 현재 활성 모델과 크게 달라 force 없이는 승격할 수 없음 (checks: 게이트 검사 결과가 있으면 함께 전달)"""
    def __init__(self, warnings, checks=None):
        super().__init__("; ".join(warnings))
        self.warnings = warnings
        self.checks = checks or []


def registry_path(model_dir, ft):
    return os.path.join(model_dir, f"{ft}_registry.json")


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize(reg):
    """구버전/불완전한 레지스트리를 현재 스키마로 보정 (저장하지 않으면 파일은 바뀌지 않음)"""
    reg.setdefault("active_version", None)
    reg.setdefault("versions", [])
    reg.setdefault("history", [])
    active = reg["active_version"]
    # 이력 없는 구버전: 현재 활성 버전을 이력의 시작으로 간주 (롤백 대상 계산을 위해)
    if active and not any(h.get("version") == active for h in reg["history"]):
        entry = next((v for v in reg["versions"] if v.get("version") == active), {})
        reg["history"].append({"version": active, "activated_at": entry.get("trained_at"), "reason": "legacy"})
    for v in reg["versions"]:
        v["is_active"] = (v.get("version") == active)
    return reg


def load(model_dir, ft):
    path = registry_path(model_dir, ft)
    reg = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                reg = json.load(f)
        except (json.JSONDecodeError, OSError):
            reg = {}
    return _normalize(reg)


def save(model_dir, ft, reg):
    """임시 파일에 쓴 뒤 교체: 다른 프로세스(Consumer)가 쓰는 도중의 파일을 읽지 않도록"""
    os.makedirs(model_dir, exist_ok=True)
    path = registry_path(model_dir, ft)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_normalize(reg), f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)


@contextmanager
def transaction(model_dir, ft):
    """읽기-수정-쓰기를 하나의 락 안에서 수행 (학습 스레드와 승격 API 가 동시에 접근해도 안전)"""
    with _LOCK:
        reg = load(model_dir, ft)
        yield reg
        save(model_dir, ft, reg)


def next_version_id(reg):
    nums = [int(v["version"][1:]) for v in reg["versions"] if str(v.get("version", "")).startswith("v")
            and str(v["version"])[1:].isdigit()]
    return f"v{(max(nums) if nums else 0) + 1}"


def find(reg, version):
    return next((v for v in reg["versions"] if v.get("version") == version), None)


def _activate(reg, version, reason):
    reg["active_version"] = version
    reg["history"].append({"version": version, "activated_at": _now(), "reason": reason})
    for v in reg["versions"]:
        v["is_active"] = (v.get("version") == version)


def add_version(reg, entry, activate):
    """새 버전을 등록. 활성 버전이 없으면(최초 학습) activate 여부와 무관하게 활성화한다.
    Returns: 활성화되었으면 True, 후보로만 저장되었으면 False"""
    reg["versions"].append(entry)
    if activate or not reg.get("active_version"):
        _activate(reg, entry["version"], "trained" if activate else "initial")
        return True
    entry["is_active"] = False
    return False


def status_of(reg, version):
    if reg.get("active_version") == version:
        return "active"
    return "retired" if any(h.get("version") == version for h in reg["history"]) else "candidate"


def list_versions(model_dir, ft):
    reg = load(model_dir, ft)
    out = []
    for v in reg["versions"]:
        item = {k: v.get(k) for k in ("version", "trained_at", "threshold", "final_val_loss", "baseline_mse", "samples_used",
                                      "trigger", "suspect_stats")}
        item["status"] = status_of(reg, v["version"])
        item["gate"] = v.get("gate")                                  # 승격 게이트 결과 (없으면 None)
        item["gate_status"] = (v.get("gate") or {}).get("status")
        out.append(item)
    return {"active_version": reg["active_version"], "versions": out}


def format_ratio(ratio) -> str:
    """배율 표기 (G2·G3 공통): 1 이상은 `×37.5`, 1 미만은 역수로 `1/37.5`. 계산 불가면 `-`."""
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
        return "-"
    return f"×{ratio:.1f}" if ratio >= 1 else f"1/{1 / ratio:.1f}"


def compare_with_active(reg, entry):
    """후보를 현재 활성 모델과 비교해 명백히 비정상인 점을 경고 목록으로 반환"""
    active = find(reg, reg.get("active_version")) if reg.get("active_version") else None
    if not active or active.get("version") == entry.get("version"):
        return []
    warnings = []
    ta, tc = active.get("threshold"), entry.get("threshold")
    if ta and tc and ta > 0 and tc > 0:
        ratio = tc / ta
        if ratio > THRESHOLD_RATIO_LIMIT or ratio < 1 / THRESHOLD_RATIO_LIMIT:
            warnings.append(f"임계치 {ta:.4g} → {tc:.4g} ({format_ratio(ratio)}, "
                            f"기준 1/{THRESHOLD_RATIO_LIMIT:g} ~ ×{THRESHOLD_RATIO_LIMIT:g})")
    va, vc = active.get("final_val_loss"), entry.get("final_val_loss")
    if va and vc and va > 0 and vc / va > VAL_LOSS_RATIO_LIMIT:
        warnings.append(f"검증 손실 {va:.4g} → {vc:.4g} ({format_ratio(vc / va)}, 기준 ≤ ×{VAL_LOSS_RATIO_LIMIT:g})")
    return warnings


def set_gate(model_dir, ft, version, gate):
    """승격 게이트 결과(dict)를 버전 엔트리에 저장"""
    with transaction(model_dir, ft) as reg:
        entry = find(reg, version)
        if entry is None:
            raise VersionNotFound(f"버전 '{version}' 이(가) 없습니다.")
        entry["gate"] = gate


def _gate_findings(reg, entry):
    """게이트 결과를 승격 판정으로 해석: (차단 사유, 참고 메모, 검사 목록)

    - G1 FAIL(깨진 모델)은 force 로도 승격 불가 -> RegistryError
    - 그 외 FAIL / ERROR 는 차단 사유 (force 로 진행 가능), WARN 은 승격하되 메모로 반환
    - 게이트가 계산된 시점의 비교 대상(against)이 현재 활성 버전과 다르면 결과가 낡은 것 -> 임계치 비교(G2)는 현재 활성 기준으로 다시 하고 메모
    """
    gate = entry.get("gate")
    if not gate:
        return compare_with_active(reg, entry), [], []
    checks = gate.get("checks", [])
    for c in checks:
        if c.get("id") == "G1" and c.get("status") == "FAIL":
            raise RegistryError(f"{entry.get('version')} 은(는) 아티팩트 검증(G1)에 실패해 승격할 수 없습니다: {c.get('message')}")
    blocking = [f"[{c.get('id')}] {c.get('message')}" for c in checks if c.get("status") in ("FAIL", "ERROR")]
    notes = [f"[{c.get('id')}] {c.get('message')}" for c in checks if c.get("status") == "WARN"]
    against, active = (gate.get("data") or {}).get("against"), reg.get("active_version")
    if against != active:
        notes.append(f"게이트는 {against or '활성 모델 없음'} 기준으로 계산됨 — 현재 활성 {active}. 게이트 재실행을 권장합니다.")
        blocking = [b for b in blocking if not b.startswith("[G2]")] + compare_with_active(reg, entry)
    if gate.get("status") == "ERROR" and not blocking:
        blocking = ["게이트 실행 오류"]
    return blocking, notes, checks


def _check_files(model_dir, entry):
    for key in ("model_path", "scaler_path", "config_path"):
        p = entry.get(key)
        if not p or not os.path.exists(os.path.join(model_dir, p)):
            raise RegistryError(f"{entry.get('version')} 의 {key} 파일이 없습니다: {p}")


def promote(model_dir, ft, version, force=False, reason="manual"):
    """후보(또는 이전 버전)를 활성화. 경고가 있으면 force=True 가 아닌 한 PromotionWarning."""
    with transaction(model_dir, ft) as reg:
        entry = find(reg, version)
        if entry is None:
            raise VersionNotFound(f"버전 '{version}' 이(가) 없습니다.")
        if reg["active_version"] == version:
            raise RegistryError(f"'{version}' 은(는) 이미 활성 버전입니다.")
        _check_files(model_dir, entry)
        blocking, notes, checks = _gate_findings(reg, entry)
        if blocking and not force:
            raise PromotionWarning(blocking, checks)
        previous = reg["active_version"]
        _activate(reg, version, reason)
    return {"previous": previous, "active": version, "warnings": blocking + notes}


def rollback(model_dir, ft):
    """직전에 활성이었던 버전으로 되돌림 (활성화 이력 기준, 파일이 남아 있는 버전만)"""
    with transaction(model_dir, ft) as reg:
        current = reg["active_version"]
        target = None
        for h in reversed(reg["history"]):
            v = h.get("version")
            entry = find(reg, v)
            if v != current and entry is not None:
                try:
                    _check_files(model_dir, entry)
                except RegistryError:
                    continue
                target = v
                break
        if target is None:
            raise RegistryError("롤백할 이전 활성 버전이 없습니다.")
        _activate(reg, target, "rollback")
    return {"previous": current, "active": target, "warnings": []}
