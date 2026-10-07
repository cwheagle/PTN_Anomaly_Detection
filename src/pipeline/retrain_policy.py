"""
재학습 정책 (P1-1) — 설정 키와 로더

운영 알람 정책(`alerting.AlertPolicy`)과 같은 패턴: 모든 키는 코드 기본값을 가지며
`src/config.py` 의 `RETRAIN_POLICY`(선택)로 덮어쓴다. 환경변수 `DRIFT_RETRAIN_MODE` 가 mode 를 우선한다.

mode
  off       : 드리프트를 감지·알림만 하고 재학습하지 않음
  candidate : (기본) 재학습 결과를 후보로만 저장, 사람이 승격
  auto      : 승격 게이트 통과 시 자동 활성화 (게이트 신뢰가 쌓인 뒤에만 사용)
"""
import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

VALID_MODES = ("off", "candidate", "auto")
DEFAULT_MODE = "candidate"


@dataclass(frozen=True)
class RetrainPolicy:
    mode: str = DEFAULT_MODE

    # 학습 구간 / 데이터 분할 (알람 이력 보존 30일에 묶여 28일)
    train_days: int = 28
    val_port_fraction: float = 0.10
    split_salt: str = "ptn"
    gate_days: int = 3
    gate_max_ports: int = 2000

    # 장애 의심 구간 제외
    exclude_from_alarms: bool = True
    alarm_min_level: int = 1
    suspect_pre_steps: int = 16          # 장애 직전 램프 구간 (4h)
    suspect_post_steps: int = 4
    suspect_gap_steps: int = 4           # 이 간격 이하의 알람은 하나의 구간으로 묶음
    exclude_from_rules: bool = True
    rule_error_ge: float = 10.0          # 직전 24h 중앙값 대비 에러 증가량
    rule_rx_drop_db: float = 2.0         # 직전 24h 중앙값 대비 수신 광파워 하락
    rule_traffic_ratio: float = 0.3      # 직전 24h 중앙값 대비 트래픽 비율 하한
    rule_min_consecutive: int = 6        # 이 스텝 수 이상 지속될 때만 의심 (산발 에러/버스트는 남김)
    max_excluded_fraction: float = 0.20  # 초과 시 학습 중단 (대규모 장애 중 재학습 금지)
    port_drop_fraction: float = 0.50     # 포트 의심 비율이 이를 넘으면 그 포트 전체 제외

    # 드리프트 신호
    drift_window_hours: int = 24
    drift_min_port_rows: int = 48
    drift_median_factor: float = 1.5
    drift_port_fraction: float = 0.30
    localized_factor: float = 3.0
    min_hours_since_activation: int = 24

    # 재학습 결정
    drift_persist_checks: int = 3        # 연속 광역 드리프트 일수
    cooldown_hours: int = 72

    # 승격 게이트
    gate_max_alarm_ratio: float = 1.5
    gate_alarm_floor: float = 0.01
    gate_max_incidents_per_port_day: float = 0.05   # G3 절대 상한: 초과 시 WARN (P1-3 C4' — FAIL 이 아님. FAIL 은 활성 대비 gate_max_alarm_ratio 초과)
    gate_min_port_days: int = 150        # 홀드아웃 포트 x 일수가 이보다 작으면 G3 판정 불가(SKIP -> 종합 WARN)
    gate_canary_auprc_drop: float = 0.05
    gate_threshold_trend_versions: int = 3
    gate_threshold_trend_factor: float = 2.0

    def with_(self, **kw):
        return replace(self, **kw)


def _normalize_mode(value, source):
    mode = str(value).strip().lower()
    if mode in VALID_MODES:
        return mode
    logger.warning("잘못된 재학습 mode '%s' (%s) — '%s' 로 대체합니다. 허용: %s",
                   value, source, DEFAULT_MODE, "/".join(VALID_MODES))
    return DEFAULT_MODE


def load_retrain_policy():
    """코드 기본값 < config.RETRAIN_POLICY < 환경변수 DRIFT_RETRAIN_MODE(mode 만)"""
    try:
        from src import config
        override = getattr(config, "RETRAIN_POLICY", None) or {}
    except Exception:
        override = {}
    kw = {k: v for k, v in override.items() if k in RetrainPolicy.__dataclass_fields__}
    if "mode" in kw:
        kw["mode"] = _normalize_mode(kw["mode"], "config.RETRAIN_POLICY")
    policy = RetrainPolicy(**kw)
    env_mode = os.getenv("DRIFT_RETRAIN_MODE")
    if env_mode:
        policy = policy.with_(mode=_normalize_mode(env_mode, "DRIFT_RETRAIN_MODE"))
    return policy


# ─────────────────────────────────────────────
# 재학습 결정 (순수 함수) + 상태 영속화
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class Decision:
    action: str      # none | notify | train
    reason: str      # 사람이 읽는 사유 (UI/로그)
    track: str = ""


def _parse(value):
    try:
        return datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


# 중단·수집 오류 후 재시도까지의 대기 (일일 점검 주기와 같은 고정값 — 설정 키 아님)
SKIP_BACKOFF_HOURS = 24


def _empty_state():
    return {"consecutive_widespread": [], "last_trigger_at": None, "last_outcome": None, "last_version": None,
            "last_skipped_at": None}


def streak_days(state) -> int:
    return len(state.get("consecutive_widespread") or [])


def cooldown_remaining_hours(state, now, policy) -> float:
    last = _parse(state.get("last_trigger_at"))
    if not last:
        return 0.0
    return max(0.0, policy.cooldown_hours - (now - last).total_seconds() / 3600)


def skip_backoff_remaining_hours(state, now) -> float:
    last = _parse(state.get("last_skipped_at"))
    if not last:
        return 0.0
    return max(0.0, SKIP_BACKOFF_HOURS - (now - last).total_seconds() / 3600)


def record_check(state: dict, track_result: dict, now: datetime) -> dict:
    """드리프트 체크 결과를 지속성 카운트에 반영한 새 상태를 반환 (입력은 변경하지 않음).

    - 광역 드리프트: 오늘 날짜를 기록 (같은 날 여러 번 체크해도 1회). 어제까지 이어지지 않았으면 연속 기록을 새로 시작
    - 판정이 수행됐고 광역이 아니면 연속 기록 초기화
    - 판정 불가(error/warming_up/데이터 부족)이면 변경 없음
    """
    new = {**_empty_state(), **state}
    days = list(new.get("consecutive_widespread") or [])
    status = track_result.get("status")
    if status not in ("normal", "drifted", "localized"):
        return new
    today = now.strftime("%Y-%m-%d")
    if track_result.get("kind") == "widespread":
        if days and days[-1] == today:
            return new
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        new["consecutive_widespread"] = (days if days and days[-1] == yesterday else []) + [today]
    else:
        new["consecutive_widespread"] = []
    return new


def record_training(state: dict, now: datetime, version=None, outcome="started") -> dict:
    """학습 진행을 기록. outcome == 'started'(Trainer 시작 시점에만 사용)이면 쿨다운 기준 시각을 갱신하고
    지속성 카운트·중단 백오프를 초기화한다. 그 외 outcome 은 결과(last_outcome/last_version)만 갱신한다."""
    new = {**_empty_state(), **state}
    if outcome == "started":
        new["last_trigger_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
        new["consecutive_widespread"] = []
        new["last_skipped_at"] = None
    new["last_outcome"] = outcome
    if version:
        new["last_version"] = version
    return new


def record_skip(state: dict, now: datetime, outcome: str) -> dict:
    """Trainer 시작 전의 중단(의심 비율 초과)·수집 오류를 기록. 쿨다운은 걸지 않고 지속성도 유지하며,
    `last_skipped_at` 만 남겨 SKIP_BACKOFF_HOURS 동안 재시도를 보류한다."""
    new = {**_empty_state(), **state}
    new["last_outcome"] = outcome
    new["last_skipped_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
    return new


def _skip_reason(outcome, policy) -> str:
    """last_outcome 을 사람이 읽는 보류 사유로 (예: 'skipped: suspect_fraction=0.324')"""
    text = str(outcome or "")
    if text.startswith("skipped: suspect_fraction="):
        try:
            return f"의심 비율 {float(text.split('=', 1)[1]):.1%} > {policy.max_excluded_fraction:.0%} — 장애 의심"
        except ValueError:
            pass
    return text.replace("skipped: ", "").replace("failed: ", "") or "중단"


def decide(track_result: dict, state: dict, now: datetime, policy: RetrainPolicy,
           pending_candidate=None, is_training=False, track="") -> Decision:
    """드리프트 판정 -> 행동 (none / notify / train). 위에서부터 첫 번째로 해당하는 규칙을 적용.
    `record_check` 로 오늘 결과를 상태에 반영한 뒤 호출한다. pending_candidate: 대기 중인 드리프트 후보 버전(없으면 None)."""
    def d(action, reason):
        return Decision(action, reason, track)

    status = track_result.get("status")
    if status not in ("normal", "drifted", "localized"):
        return d("none", f"{status}: {track_result.get('message', '판정 불가')}")
    kind = track_result.get("kind", "none")
    if kind == "none":
        return d("none", "정상")
    if kind == "localized":
        return d("notify", f"국소 이상 {track_result.get('localized_ports', 0)}포트 — 장애 의심, 재학습 안 함")
    if track_result.get("baseline_legacy"):
        return d("notify", "구버전 기준값 — 감지만 (재학습하려면 수동 학습)")
    if policy.mode == "off":
        return d("notify", "광역 드리프트 감지 (mode=off — 재학습 안 함)")
    streak = streak_days(state)
    if streak < policy.drift_persist_checks:
        return d("notify", f"광역 드리프트 {streak}/{policy.drift_persist_checks}일")
    remaining = cooldown_remaining_hours(state, now, policy)
    if remaining > 0:
        return d("notify", f"쿨다운 {remaining:.0f}h 남음")
    backoff = skip_backoff_remaining_hours(state, now)
    if backoff > 0:
        return d("notify", f"재학습 보류: {_skip_reason(state.get('last_outcome'), policy)} ({backoff:.0f}h 후 재시도)")
    if pending_candidate:
        return d("notify", f"대기 중 후보 {pending_candidate} 검토 필요")
    if is_training:
        return d("notify", "이미 학습 중")
    return d("train", f"광역 드리프트 {streak}일 지속 — 후보 모델 학습 (mode={policy.mode})")


def pending_drift_candidate(reg) -> str:
    """드리프트로 만든 미승격 후보 중 활성 버전보다 새로운 것 (레지스트리에서 직접 파생 — 상태 파일 불일치 방지)"""
    def num(v):
        s = str(v.get("version", ""))
        return int(s[1:]) if s[1:].isdigit() else -1
    active = next((v for v in reg.get("versions", []) if v.get("version") == reg.get("active_version")), None)
    active_num = num(active) if active else -1
    activated = {h.get("version") for h in reg.get("history", [])}
    newest = None
    for v in reg.get("versions", []):
        if v.get("trigger") == "drift" and v.get("version") not in activated and num(v) > active_num:
            if newest is None or num(v) > num(newest):
                newest = v
    return newest["version"] if newest else None


def describe_state(state: dict, now: datetime, policy: RetrainPolicy) -> dict:
    """UI/API 표시용 요약"""
    return {**_empty_state(), **state, "streak_days": streak_days(state),
            "persist_required": policy.drift_persist_checks,
            "cooldown_remaining_hours": round(cooldown_remaining_hours(state, now, policy), 1),
            "skip_backoff_remaining_hours": round(skip_backoff_remaining_hours(state, now), 1), "mode": policy.mode}


def state_path(model_dir, ft):
    return os.path.join(model_dir, f"{ft}_retrain_state.json")


def load_state(model_dir, ft) -> dict:
    """상태 파일 로드. 없거나 손상되면 빈 상태 (참고용 상태이므로 판정이 막히지 않게)"""
    try:
        with open(state_path(model_dir, ft), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {**_empty_state(), **data}
    except (OSError, json.JSONDecodeError):
        pass
    return _empty_state()


def save_state(model_dir, ft, state) -> None:
    """임시 파일에 쓴 뒤 교체 (registry.save 와 같은 방식)"""
    os.makedirs(model_dir, exist_ok=True)
    path = state_path(model_dir, ft)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
