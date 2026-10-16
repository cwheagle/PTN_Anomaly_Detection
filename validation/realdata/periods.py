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
    """데이터 일 목록 → S, J0, D, 데이터 일 수. 개발 데이터일 부족/판정 데이터일 부족이면 ok=False 와 사유."""
    days = sorted(pd.Timestamp(d).normalize() for d in days)
    out = {"alternative": ALTERNATIVE_ID, "data_days": len(days), "dev_data_days": dev_days, "ok": False, "reason": None}
    if not days:
        out["reason"] = "데이터 일 0"
        return out
    out["S"] = pd.Timestamp(start) if start is not None else days[0]
    out["D"] = days[-1] + pd.Timedelta(days=1)
    if len(days) < dev_days + min_judge_days:
        out["reason"] = (f"데이터 일 {len(days)} < {dev_days + min_judge_days} (개발 {dev_days} + 판정 최소 {min_judge_days}) — 판정 실행 불가 (9.7.4)")
        return out
    out["J0"] = days[dev_days - 1] + pd.Timedelta(days=1)
    out["judge_data_days"] = len(days) - dev_days
    out["ok"] = True
    return out


def range_bounds(periods, name):
    """'dev' → [S, J0), 'judge' → [J0, D)."""
    if not periods.get("ok"):
        raise PeriodError(periods.get("reason") or "기간 계산 불가")
    if name == "dev":
        return periods["S"], periods["J0"]
    if name == "judge":
        return periods["J0"], periods["D"]
    raise PeriodError(f"알 수 없는 기간 이름: {name!r} (dev | judge)")


def lock_dir(runs_dir):
    return os.path.join(runs_dir, "locks")


def lock_path(runs_dir, periods):
    s, d = pd.Timestamp(periods["S"]).strftime("%Y%m%d"), pd.Timestamp(periods["D"]).strftime("%Y%m%d")
    return os.path.join(lock_dir(runs_dir), f"judge_{s}_{d}.json")


def overlaps_judge(start, end, periods):
    """[start, end) 가 판정 구간 [J0, D) 와 겹치는지."""
    if not periods.get("ok"):
        return False
    j0, d = periods["J0"], periods["D"]
    return pd.Timestamp(start) < d and pd.Timestamp(end) > j0


def require_unsealed(start, end, periods, runs_dir, command):
    """잠금 파일이 없는데 [start, end) 가 판정 구간과 겹치면 PeriodError (측정 전 거부, P4)."""
    if not periods.get("ok"):
        # 기간을 정할 수 없으면(데이터일 부족) 판정 구간 자체가 없으므로 봉인 대상도 없다 — 단 판정 실행은 별도로 불가
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
