"""
P1-2 C5 라이브 결과 점검 (설계서 9.9) — 읽기 전용으로 `anomaly_detection` 을 조회한다 (권한 검사 4.1 동일, 문장 가드 통과).

- C5a 처리율 = 저장된 결과 행 / 처리 대상 행. 처리 대상 행 = 같은 스냅샷에 운영과 같은 DataProcessor 경로(AnomalyDetector.detect, 트래픽 트랙)를
  적용했을 때 추론 윈도우가 만들어지는 행 = 오프라인 재생 AI 프레임의 행.
- C5b 지연 = detect_time − occur_date 분포, 30분 이내 비율 (저장된 행의 99% 이상이 기준).
- C5c 근거 = 결과가 1시간 이상 비어 있는 구간 목록 (전역 시간축).
- 라이브와 오프라인 재생의 알람 일치율은 참고치 (필터 차이가 있으면 입력이 달라 일치하지 않을 수 있음, 2.2-2).
C5 는 라이브 가동 구간만 대상이다: 저장된 결과의 첫~마지막 occur_date 안에서 계산한다.
"""
import re

import pandas as pd

from validation.evaluation.metrics import KEY

SCHEMA_RE = re.compile(r"^[A-Za-z0-9_]+$")
C5A_MIN, C5B_MIN_RATIO, C5B_LIMIT_MIN = 0.999, 0.99, 30
GAP_HOURS = 1


def result_table(schema=None):
    if schema is None:
        return "`anomaly_detection`"
    if not SCHEMA_RE.match(schema):
        raise ValueError(f"결과 스키마 이름 형식 오류: {schema!r}")
    return f"`{schema}`.`anomaly_detection`"


def fetch_results(db, start, end, schema=None, chunk=50000):
    """저장된 결과 행 (id 키셋 청크). 열: id, occur_date, ip_addr, cid, lid, is_anomaly, detect_time."""
    tbl, last, rows = result_table(schema), 0, []
    while True:
        part = db.query(f"SELECT `id`, `occur_date`, `ip_addr`, `cid`, `lid`, `is_anomaly`, `detect_time` FROM {tbl} "
                        f"WHERE `occur_date` >= %s AND `occur_date` < %s AND `id` > %s ORDER BY `id` LIMIT %s",
                        [pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime(), last, chunk])
        rows += part
        if len(part) < chunk:
            break
        last = part[-1]["id"]
    df = pd.DataFrame(rows, columns=["id", "occur_date", "ip_addr", "cid", "lid", "is_anomaly", "detect_time"])
    df["occur_date"], df["detect_time"] = pd.to_datetime(df["occur_date"]), pd.to_datetime(df["detect_time"])
    df["ip_addr"] = df["ip_addr"].astype(str).str.strip()
    return df


def compute_c5(stored, target):
    """stored: fetch_results 결과, target: 재생 AI 프레임(KEY + occur_date [+ alarm]). 저장된 구간 안에서만 계산."""
    out = {"stored_rows": int(len(stored))}
    if stored.empty:
        out.update(hold=True, reason="저장된 결과 행이 없음 — 라이브 가동 구간 없음")
        return out
    lo, hi = stored["occur_date"].min(), stored["occur_date"].max()
    t = target[(target["occur_date"] >= lo) & (target["occur_date"] <= hi)]
    sk = stored.drop_duplicates(KEY + ["occur_date"])
    m = t.merge(sk, on=KEY + ["occur_date"], how="left", indicator=True)
    matched = int((m["_merge"] == "both").sum())
    out.update(live_interval=[lo, hi], target_rows=int(len(t)), matched_rows=matched,
               stored_not_in_target=int(len(sk) - matched),
               c5a=(matched / len(t)) if len(t) else float("nan"))
    out["c5a_pass"] = bool(out["c5a"] >= C5A_MIN) if len(t) else None
    lag = (stored["detect_time"] - stored["occur_date"]).dt.total_seconds() / 60
    lag = lag.dropna()
    out["c5b"] = {"within_30min": float((lag <= C5B_LIMIT_MIN).mean()) if len(lag) else float("nan"),
                  "p50_min": float(lag.median()) if len(lag) else float("nan"), "p99_min": float(lag.quantile(0.99)) if len(lag) else float("nan"),
                  "pass": bool((lag <= C5B_LIMIT_MIN).mean() >= C5B_MIN_RATIO) if len(lag) else None}
    ts = pd.Series(sorted(stored["occur_date"].unique()))
    gap = ts.diff()
    big = ts[gap > pd.Timedelta(hours=GAP_HOURS)]
    out["c5c_gaps"] = [{"from": ts[i - 1], "to": ts[i], "hours": float(gap[i] / pd.Timedelta(hours=1))} for i in big.index]
    both = m[m["_merge"] == "both"]
    if "alarm" in both.columns and len(both):
        out["alarm_agreement_ref"] = float((both["alarm"].astype(bool) == both["is_anomaly"].astype(bool)).mean())
    out["note"] = "C5c(서비스 중단 누적 ≤ 1시간·자동 복구)는 이 도구로 판정하지 않음 — 위 공백 구간을 운영 기록과 대조"
    return out
