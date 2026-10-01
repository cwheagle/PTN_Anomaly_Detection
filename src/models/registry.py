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
    (본격적인 승격 게이트는 별도 과제이며 여기서는 명백히 비정상인 경우만 거른다)
"""
import json
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
    """후보가 현재 활성 모델과 크게 달라 force 없이는 승격할 수 없음"""
    def __init__(self, warnings):
        super().__init__("; ".join(warnings))
        self.warnings = warnings


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
        item = {k: v.get(k) for k in ("version", "trained_at", "threshold", "final_val_loss", "baseline_mse", "samples_used")}
        item["status"] = status_of(reg, v["version"])
        out.append(item)
    return {"active_version": reg["active_version"], "versions": out}


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
            warnings.append(f"임계치가 현재 모델 대비 {ratio:.1f}배 ({ta:.4g} -> {tc:.4g})")
    va, vc = active.get("final_val_loss"), entry.get("final_val_loss")
    if va and vc and va > 0 and vc / va > VAL_LOSS_RATIO_LIMIT:
        warnings.append(f"검증 손실이 현재 모델 대비 {vc / va:.1f}배 ({va:.4g} -> {vc:.4g})")
    return warnings


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
        warnings = compare_with_active(reg, entry)
        if warnings and not force:
            raise PromotionWarning(warnings)
        previous = reg["active_version"]
        _activate(reg, version, reason)
    return {"previous": previous, "active": version, "warnings": warnings}


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
