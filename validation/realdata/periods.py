"""
P1-2 개발/판정 기간 분할과 판정 구간 봉인 (설계서 9.7, 대안 1 확정 — Q12)

- 데이터 일 = 정상 ETH 행(4.3a 격리 후, 필터 통과)이 1행 이상 있는 날짜. 길이는 달력 일이 아니라 데이터 일로 센다.
- 개발 기간 [S, J0), J0 = 28번째 데이터일의 다음 날 KST 자정. 판정 기간 [J0, D), D = 마지막 데이터 날의 다음 날 자정.
- P4: 잠금 파일이 없으면 판정 구간과 겹치는 분석(alarms --summary / --count-only / join-check / replay / report / sample)을 거부한다.
  원본 추출 자체는 판정 구간을 포함해도 허용된다(봉인 대상은 분석이다).
이 파일의 값(28, 15)은 사전 고정이며 결과를 본 뒤 바꾸지 않는다.
"""
import json
import os
import re

import pandas as pd

DEV_DATA_DAYS = 28
MIN_JUDGE_DATA_DAYS = 15
ALTERNATIVE_ID = "fixed_split_v1"


class PeriodError(RuntimeError):
    """기간 규칙 위반 — 측정 전 거부."""


def data_days_from_eth(eth):
    """정상 ETH 스냅샷 프레임 → 데이터 일 목록(정렬된 Timestamp 자정). 일 단위는 snapshot 의 bin_end(없으면 occur_date) 날짜."""
    if eth is None or len(eth) == 0:
        return []
    col = "bin_end" if "bin_end" in eth.columns else "occur_date"
    days = pd.to_datetime(eth[col], errors="coerce").dropna().dt.normalize().unique()
    return sorted(pd.Timestamp(d) for d in days)


def compute_periods(days, dev_days=DEV_DATA_DAYS, min_judge_days=MIN_JUDGE_DATA_DAYS, start=None):
    """데이터 일 목록 → S, D, J0, 데이터 일 수.
    - 데이터일 < 28: 개발·판정을 나눌 수 없다 → explore_only (모든 분석은 purpose=explore, 장부에 남아 나중에 판정 구간이 될 수 없음, 9.7.4).
    - 28 ≤ 데이터일 < 43: ok=False(판정 < 15 데이터일)이지만 잠정 J0 = 28번째 데이터일 다음 날을 계산하고 P4 를 ok=True 와 똑같이 적용(잠금은 불가).
    - 데이터일 ≥ 43: ok=True."""
    days = sorted(pd.Timestamp(d).normalize() for d in days)
    out = {"alternative": ALTERNATIVE_ID, "data_days": len(days), "dev_data_days": dev_days, "ok": False, "reason": None,
           "provisional": False, "explore_only": False}
    if not days:
        out["reason"] = "데이터 일 0"
        out["explore_only"] = True
        return out
    out["S"] = pd.Timestamp(start) if start is not None else days[0]
    out["D"] = days[-1] + pd.Timedelta(days=1)
    if len(days) < dev_days:
        out["explore_only"] = True
        out["reason"] = f"데이터 일 {len(days)} < {dev_days} — 개발/판정을 나눌 수 없음, 모든 분석은 purpose=explore (장부에 남아 나중에 판정 구간이 될 수 없음)"
        return out
    out["J0"] = days[dev_days - 1] + pd.Timedelta(days=1)
    out["judge_data_days"] = len(days) - dev_days
    if len(days) < dev_days + min_judge_days:
        out["provisional"] = True
        out["reason"] = (f"데이터 일 {len(days)} < {dev_days + min_judge_days} (개발 {dev_days} + 판정 최소 {min_judge_days}) — 판정 실행 불가 (9.7.4), "
                         f"잠정 J0={out['J0']:%Y-%m-%d} 로 [J0, D) 분석은 봉인")
        return out
    out["ok"] = True
    return out


def range_bounds(periods, name):
    """'dev' → [S, J0), 'judge' → [J0, D). dev 는 잠정 J0 로도 가능, judge 는 ok 일 때만."""
    if periods.get("explore_only"):
        raise PeriodError(periods.get("reason") or "기간 계산 불가 — purpose=explore 의 START..END 범위만 허용")
    if name == "dev":
        return periods["S"], periods["J0"]
    if name == "judge":
        if not periods.get("ok"):
            raise PeriodError(periods.get("reason") or "판정 기간 계산 불가")
        return periods["J0"], periods["D"]
    raise PeriodError(f"알 수 없는 기간 이름: {name!r} (dev | judge)")


def lock_dir(runs_dir):
    return os.path.join(runs_dir, "locks")


def lock_path(runs_dir, periods):
    s, d = pd.Timestamp(periods["S"]).strftime("%Y%m%d"), pd.Timestamp(periods["D"]).strftime("%Y%m%d")
    return os.path.join(lock_dir(runs_dir), f"judge_{s}_{d}.json")


def overlaps_judge(start, end, periods):
    """[start, end) 가 판정 구간 [J0, D) 와 겹치는지."""
    if "J0" not in periods:
        return False
    j0, d = periods["J0"], periods["D"]
    return pd.Timestamp(start) < d and pd.Timestamp(end) > j0


def require_unsealed(start, end, periods, runs_dir, command):
    """잠금 파일이 없는데 [start, end) 가 판정 구간 [J0, D) 와 겹치면 PeriodError (측정 전 거부, P4).
    J0 는 데이터일 ≥ 28 이면 잠정값으로도 적용한다(잠금은 불가) — 데이터가 뒤로 늘어도 28번째 데이터일은 바뀌지 않는다."""
    if "J0" not in periods:
        return
    if overlaps_judge(start, end, periods) and not os.path.exists(lock_path(runs_dir, periods)):
        raise PeriodError(
            f"[{command}] 판정 구간 [{periods['J0']:%Y-%m-%d}, {periods['D']:%Y-%m-%d}) 와 겹치는 분석은 잠금(periods lock) 전에 "
            f"실행할 수 없습니다 (설계 9.7 P4). 개발 기간은 --range dev 로 지정하세요.")


def resolve_range(spec, periods, runs_dir, command):
    """'dev' | 'judge' | 'YYYY-MM-DD..YYYY-MM-DD' → (start, end). 판정 구간과 겹치면 잠금 확인."""
    if spec in ("dev", "judge"):
        start, end = range_bounds(periods, spec)
    elif re.match(r"^\d{4}-\d{2}-\d{2}(\s[\d:]+)?\.\.\d{4}-\d{2}-\d{2}(\s[\d:]+)?$", spec or ""):
        a, b = spec.split("..")
        start, end = pd.Timestamp(a), pd.Timestamp(b)
        if end <= start:
            raise PeriodError(f"기간 끝이 시작보다 앞입니다: {spec}")
    else:
        raise PeriodError(f"기간 형식 오류: {spec!r} (dev | judge | START..END)")
    require_unsealed(start, end, periods, runs_dir, command)
    return pd.Timestamp(start), pd.Timestamp(end)


def load_lock(runs_dir, periods):
    p = lock_path(runs_dir, periods)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---- 판정 모델 학습 구간 검사 (9.7.5, 측정 전 거부, fail closed) ---------------------------------------------
def _ts(v):
    try:
        t = pd.Timestamp(v)
        return None if pd.isna(t) else t
    except (TypeError, ValueError):
        return None


def check_training_window(meta, j0):
    """판정 모델 메타 검사. (a) trained_at ≤ J0 → 통과 (b) training_window 의 끝 시각이 있고 ≤ J0 → 통과
    (c) 그 밖(끝 시각 없음·파싱 불가·> J0)은 거부. 반환 (ok, 사유)."""
    j0 = pd.Timestamp(j0)
    trained = _ts((meta or {}).get("trained_at"))
    if trained is not None and trained <= j0:
        return True, f"trained_at {trained} ≤ J0 {j0} (a)"
    tw = (meta or {}).get("training_window")
    if isinstance(tw, dict):
        ends = [t for t in (_ts(tw.get(k)) for k in ("end", "train_end", "test_end")) if t is not None]
        if ends:
            e = max(ends)
            if e <= j0:
                return True, f"training_window 끝 {e} ≤ J0 {j0} (b)"
            return False, f"training_window 끝 {e} > J0 {j0} — 판정 구간과 겹칠 수 있음 (c)"
    return False, (f"trained_at {trained} > J0 {j0} 이고 training_window 의 끝 시각이 없거나 읽을 수 없음 — fail closed (c)"
                   if trained is not None else "trained_at 도 training_window 도 읽을 수 없음 — fail closed (c)")


def stable_hash(obj):
    """설정 해시: 키 정렬 JSON 의 sha256 (Timestamp 등은 문자열)."""
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def run_marker_path(runs_dir, periods):
    return lock_path(runs_dir, periods).replace(".json", ".run.json")


def create_lock(runs_dir, periods, payload):
    """잠금 파일을 배타적으로 만든다 (이미 있으면 PeriodError). 잠금 뒤 값은 바꾸지 않는다."""
    p = lock_path(runs_dir, periods)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        with open(p, "x", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    except FileExistsError:
        raise PeriodError(f"이미 잠긴 판정 기간입니다 (잠금 파일 존재, 다시 만들 수 없음): {p}")
    return p


# ---- 판정 실행 표식: phase 모델 (설계 9.7.6, 개정 4) ---------------------------------------------------------
# 원칙: 판정 구간 결과가 사람·로그에 보였을 수 있으면 소진, 아니면 되돌린다.
#   claimed   표식 직후(판정 데이터는 아직 모델·베이스라인에 들어가지 않음) → 예외 시 자동으로 aborted_<n> 으로 이름 바꾸고 재실행 허용
#   computing 판정 구간 데이터가 처음 모델·베이스라인에 들어가기 직전 → 예외 시 자동 되돌림 없음, `periods retry-judge` 1회만
#   completed 모든 출력 작성 후(출력 파일 sha256 기록) → 재실행 없음
def _marker_variant(runs_dir, periods, suffix):
    return lock_path(runs_dir, periods).replace(".json", f".{suffix}.json")


def _read_marker(runs_dir, periods):
    p = run_marker_path(runs_dir, periods)
    return (p, json.load(open(p, encoding="utf-8"))) if os.path.exists(p) else (p, None)


def _write_marker(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def count_variants(runs_dir, periods, prefix):
    d = os.path.dirname(lock_path(runs_dir, periods))
    base = os.path.basename(lock_path(runs_dir, periods))[:-5]
    return sorted(f for f in (os.listdir(d) if os.path.isdir(d) else []) if f.startswith(f"{base}.{prefix}_") and f.endswith(".json"))


def claim_judge_run(runs_dir, periods, info):
    """판정 실행 1회 표식을 배타적으로 만든다 (phase=claimed). 이미 있으면 phase 별 안내와 함께 거부."""
    p = run_marker_path(runs_dir, periods)
    attempt = len(count_variants(runs_dir, periods, "failed")) + 1
    payload = {**info, "phase": "claimed", "attempt": attempt, "claimed_at": str(pd.Timestamp.now())}
    try:
        with open(p, "x", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)
    except FileExistsError:
        phase = (_read_marker(runs_dir, periods)[1] or {}).get("phase")
        hint = {"completed": "완료된 판정은 다시 실행하지 않습니다",
                "computing": "computing 중 실패했다면 `periods retry-judge --reason ... --reviewed-by ...` 로 1회만 재시도할 수 있습니다",
                "claimed": "claimed 상태로 남아 있습니다 (프로세스 강제 종료 추정) — `periods retry-judge` 로 처리"}.get(phase, "")
        raise PeriodError(f"이 잠금에 대한 판정 실행 표식이 이미 있습니다 (phase={phase}; 판정 실행은 1회): {p}. {hint}")
    return p


def set_phase(runs_dir, periods, phase):
    p, m = _read_marker(runs_dir, periods)
    m["phase"] = phase
    m[f"{phase}_at"] = str(pd.Timestamp.now())
    _write_marker(p, m)


def abort_claimed(runs_dir, periods, error):
    """claimed 단계의 예외: 표식을 aborted_<n> 으로 이름을 바꾸고(삭제 아님, 오류 포함) 재실행을 허용한다. 반환: 새 파일 경로."""
    p, m = _read_marker(runs_dir, periods)
    n = len(count_variants(runs_dir, periods, "aborted")) + 1
    m.update(phase="aborted", error=str(error), aborted_at=str(pd.Timestamp.now()))
    dest = _marker_variant(runs_dir, periods, f"aborted_{n}")
    _write_marker(dest, m)
    os.remove(p)
    return dest


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def complete_judge_run(runs_dir, periods, files):
    """모든 출력 작성 후: phase=completed, 출력 파일 sha256 기록."""
    p, m = _read_marker(runs_dir, periods)
    m["outputs"] = {os.path.basename(f): _sha(f) for f in files if os.path.exists(f)}
    m["phase"] = "completed"
    m["completed_at"] = str(pd.Timestamp.now())
    _write_marker(p, m)


def retry_judge(runs_dir, periods, reason, reviewed_by, commit_now, partial_outputs=()):
    """computing(또는 강제 종료된 claimed)에서 실패한 판정을 사람 승인으로 1회만 다시 실행할 수 있게 한다.
    표식을 failed_1 로 이름을 바꾸고(부분 출력 목록·실패 전후 코드 커밋·사유·검토자 기록) 새 표식을 허용한다.
    잠금의 설정·모델·정책 해시 일치는 호출 측이 이 함수 전에 검사한다. 두 번째 실패는 소진(이 함수가 거부)."""
    if not (reason or "").strip() or not (reviewed_by or "").strip():
        raise PeriodError("retry-judge 는 --reason 과 --reviewed-by(수정 diff 가 판정 규칙·기준값·분류 규칙을 건드리지 않았음을 확인한 reviewer) 가 필요합니다")
    p, m = _read_marker(runs_dir, periods)
    if m is None:
        raise PeriodError("재시도할 판정 실행 표식이 없습니다")
    if m.get("phase") == "completed":
        raise PeriodError("완료된 판정은 재실행할 수 없습니다 (사후 정정은 별도 승인 절차, 9.7.6-5)")
    if count_variants(runs_dir, periods, "failed"):
        raise PeriodError("재시도는 1회만 허용됩니다 — 두 번째 실패로 이 잠금의 판정이 소진되었습니다. 이 잠금으로는 판정하지 않고 보고하세요 (새 판정 기간은 별도 결정)")
    m.update(phase="failed", reason=reason.strip(), reviewed_by=reviewed_by.strip(), commit_before=m.get("tool_commit"), commit_after=commit_now,
             partial_outputs=list(partial_outputs), retried_at=str(pd.Timestamp.now()))
    dest = _marker_variant(runs_dir, periods, "failed_1")
    _write_marker(dest, m)
    os.remove(p)
    return dest


# ---- 분석 장부 (설계 9.7.4, 추가 전용) ------------------------------------------------------------------------
LEDGER_NAME = "analysis_ledger.jsonl"


def ledger_path(runs_dir):
    return os.path.join(runs_dir, LEDGER_NAME)


def analysis_purpose(periods, start, end):
    """장부에 남길 purpose: 기간 미정(데이터일 < 28)이면 explore, 판정 구간과 겹치면 judge, 아니면 dev."""
    if periods.get("explore_only"):
        return "explore"
    return "judge" if overlaps_judge(start, end, periods) else "dev"


def append_ledger(runs_dir, command, purpose, start, end, periods, run_id=None):
    """분석한 시간 범위·purpose·스냅샷 S/D·시각을 덧붙인다 (파일을 지우거나 다른 runs 폴더를 쓰는 우회는 막지 못함: 운영자 규율)."""
    os.makedirs(runs_dir, exist_ok=True)
    rec = {"ts": str(pd.Timestamp.now()), "command": command, "purpose": purpose, "start": str(pd.Timestamp(start)), "end": str(pd.Timestamp(end)),
           "S": str(periods.get("S")), "D": str(periods.get("D")), "run": run_id}
    with open(ledger_path(runs_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def read_ledger(runs_dir):
    p = ledger_path(runs_dir)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def ledger_conflicts(runs_dir, periods):
    """[J0, D) 와 겹치는 장부 기록(이전 판정 실행·다른 S/D 의 스냅샷 것·explore 포함). 장부가 없으면 None."""
    recs = read_ledger(runs_dir)
    if recs is None:
        return None
    j0, d = pd.Timestamp(periods["J0"]), pd.Timestamp(periods["D"])
    return [r for r in recs if pd.Timestamp(r["start"]) < d and pd.Timestamp(r["end"]) > j0]
