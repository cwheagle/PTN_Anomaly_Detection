"""
재학습용 학습 구간 · 포트 분할 · 장애 의심 구간 산출 (P1-1, 순수 함수)

- 학습 구간은 트리거 시각까지 포함 [T - train_days, T]
- 검증은 시간이 아니라 **포트 단위 홀드아웃**: 최근 패턴을 학습하면서 처음 보는 포트로 검증
- 장애 의심 구간 = 자기 알람 이력(A) ∪ 모델과 무관한 규칙(B) ∪ 명시 CSV(C). 포트별로 병합.
  규칙(B)은 평가용 롤링 규칙 베이스라인(검증 도구 쪽)과 비슷한 아이디어지만 목적이 다르다
  (평가 베이스라인 vs 학습 데이터 정제). 솔루션(src)은 검증 도구를 import 할 수 없으므로 별도 구현하며 파라미터를 공유하지 않는다.

구간 DataFrame 컬럼: ip_addr, cid, lid, start_time, end_time, source  (양끝 포함)
"""
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)
PORT_KEYS = ['ip_addr', 'cid', 'lid']
INTERVAL_COLUMNS = PORT_KEYS + ['start_time', 'end_time', 'source']


@dataclass(frozen=True)
class TrainWindow:
    start: datetime                 # 학습·검증 공통 기간
    end: datetime
    gate_start: datetime            # 게이트 데이터 기간 (= [end - gate_days, end])
    gate_end: datetime


def plan_window(now: datetime, policy) -> TrainWindow:
    """기준시각 now(트리거 시각)까지 포함하는 학습 구간과 게이트 구간"""
    return TrainWindow(
        start=now - timedelta(days=policy.train_days), end=now,
        gate_start=now - timedelta(days=policy.gate_days), gate_end=now,
    )


# ─────────────────────────────────────────────
# 포트 분할
# ─────────────────────────────────────────────
def is_val_port(ip_addr, cid, lid, fraction: float, salt: str) -> bool:
    """안정적 해시로 검증 포트 여부 결정. 파이썬 hash() 는 프로세스마다 달라 쓰지 않는다."""
    digest = hashlib.md5(f"{salt}|{ip_addr}|{cid}|{lid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64 < fraction


def split_ports(df: pd.DataFrame, fraction: float, salt: str):
    """(학습 포트 행, 검증 포트 행) 로 분할. 같은 포트는 항상 같은 쪽."""
    if df is None or df.empty:
        return df, df
    ports = df[PORT_KEYS].drop_duplicates()
    val_keys = {(r.ip_addr, r.cid, r.lid) for r in ports.itertuples(index=False)
                if is_val_port(r.ip_addr, r.cid, r.lid, fraction, salt)}
    key_index = pd.MultiIndex.from_frame(df[PORT_KEYS])
    is_val = key_index.isin(list(val_keys)) if val_keys else np.zeros(len(df), dtype=bool)
    return df[~is_val].copy(), df[is_val].copy()


# ─────────────────────────────────────────────
# 구간 산출
# ─────────────────────────────────────────────
def _empty_intervals():
    return pd.DataFrame({c: pd.Series(dtype=object) for c in INTERVAL_COLUMNS}).astype(
        {'start_time': 'datetime64[ns]', 'end_time': 'datetime64[ns]'})


def _runs_to_intervals(runs: pd.DataFrame, pre_steps, post_steps, source):
    """runs: PORT_KEYS + first/last (Timestamp) -> 앞 pre·뒤 post 스텝 확장한 구간"""
    out = runs[PORT_KEYS].copy()
    out['start_time'] = runs['first'] - pre_steps * STEP
    out['end_time'] = runs['last'] + post_steps * STEP
    out['source'] = source
    return out.reset_index(drop=True)


def _group_runs(df, cond, max_gap, min_len):
    """df(포트·시간 정렬됨)에서 cond 가 참인 연속 구간(포트 경계/시간 공백 max_gap 초과 시 분리)을 first/last 로"""
    t = df['occur_date']
    same_port = (df[PORT_KEYS].shift() == df[PORT_KEYS]).all(axis=1)
    near = same_port & ((t - t.shift()) <= max_gap)
    prev_cond = cond.shift(fill_value=False)
    new_run = cond & ~(prev_cond & near)
    run_id = new_run.cumsum()
    sel = df[cond].assign(_run=run_id[cond])
    if sel.empty:
        return pd.DataFrame(columns=PORT_KEYS + ['first', 'last', 'n'])
    g = sel.groupby('_run').agg(ip_addr=('ip_addr', 'first'), cid=('cid', 'first'), lid=('lid', 'first'),
                                first=('occur_date', 'min'), last=('occur_date', 'max'), n=('occur_date', 'size'))
    return g[g['n'] >= min_len].reset_index(drop=True)


def intervals_from_alarms(alarm_rows: pd.DataFrame, pre_steps, post_steps, gap_steps=4) -> pd.DataFrame:
    """알람 행(occur_date, ip_addr, cid, lid) -> 포트별 연속 구간.
    gap_steps 이하 간격의 알람은 한 구간으로 묶고 [첫 알람 - pre, 마지막 알람 + post] 로 확장."""
    if alarm_rows is None or len(alarm_rows) == 0:
        return _empty_intervals()
    df = alarm_rows[PORT_KEYS + ['occur_date']].copy()
    df['occur_date'] = pd.to_datetime(df['occur_date'])
    df = df.sort_values(PORT_KEYS + ['occur_date']).reset_index(drop=True)
    cond = pd.Series(True, index=df.index)
    runs = _group_runs(df, cond, gap_steps * STEP, 1)
    return _runs_to_intervals(runs, pre_steps, post_steps, 'alarm')


def intervals_from_rules(raw: pd.DataFrame, policy) -> pd.DataFrame:
    """모델과 무관한 규칙으로 지속된 장애 구간을 찾는다 (직전 24h 포트별 중앙값 기준).

    - 에러: error_packet - 중앙값 >= rule_error_ge
    - 광 수신: 중앙값 - rx_avg_power >= rule_rx_drop_db
    - 트래픽: (tx+rx) < 중앙값 * rule_traffic_ratio (중앙값 > 0)
    조건이 rule_min_consecutive 스텝 이상 연속일 때만 의심 구간으로 본다. 산발 에러·버스트 같은
    정상 변동은 학습 데이터에 남겨야 하기 때문(lessons #24). 직전 중앙값이 4h(16스텝) 미만이면 판정하지 않음.
    """
    if raw is None or len(raw) == 0:
        return _empty_intervals()
    df = raw.copy()
    df['occur_date'] = pd.to_datetime(df['occur_date']).dt.round('15min')
    df = df.drop_duplicates(PORT_KEYS + ['occur_date'], keep='last')
    df = df.sort_values(PORT_KEYS + ['occur_date']).reset_index(drop=True)

    def prev_median(s):
        return s.shift(1).rolling(96, min_periods=16).median()

    cond = pd.Series(False, index=df.index)
    if 'error_packet' in df:
        err = pd.to_numeric(df['error_packet'], errors='coerce')
        med = err.groupby([df[k] for k in PORT_KEYS], sort=False).transform(prev_median)
        cond |= ((err - med) >= policy.rule_error_ge).fillna(False)
    if 'rx_avg_power' in df:
        rx = pd.to_numeric(df['rx_avg_power'], errors='coerce')
        med = rx.groupby([df[k] for k in PORT_KEYS], sort=False).transform(prev_median)
        cond |= ((med - rx) >= policy.rule_rx_drop_db).fillna(False)
    if 'tx_packet' in df and 'rx_packet' in df:
        tot = pd.to_numeric(df['tx_packet'], errors='coerce') + pd.to_numeric(df['rx_packet'], errors='coerce')
        med = tot.groupby([df[k] for k in PORT_KEYS], sort=False).transform(prev_median)
        cond |= ((med > 0) & (tot < med * policy.rule_traffic_ratio)).fillna(False)

    runs = _group_runs(df, cond, 1.5 * STEP, policy.rule_min_consecutive)
    return _runs_to_intervals(runs, policy.suspect_pre_steps, policy.suspect_post_steps, 'rule')


def merge_intervals(*frames, touch=STEP) -> pd.DataFrame:
    """여러 구간 DataFrame 을 포트별로 합친다. 겹치거나 touch 이내로 맞닿은 구간은 하나로 병합.
    source 는 병합된 출처를 '+' 로 연결 (예: 'alarm+rule'). 구간 컬럼은 start_time/end_time
    (구형 CSV 의 failure_time 은 end_time 으로 간주)."""
    parts = []
    for f in frames:
        if f is None or len(f) == 0:
            continue
        f = f.copy()
        if 'end_time' not in f.columns and 'failure_time' in f.columns:
            f = f.rename(columns={'failure_time': 'end_time'})
        if 'source' not in f.columns:
            f['source'] = 'explicit'
        parts.append(f[INTERVAL_COLUMNS])
    if not parts:
        return _empty_intervals()
    df = pd.concat(parts, ignore_index=True)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df = df.sort_values(PORT_KEYS + ['start_time']).reset_index(drop=True)
    keys = [df[k] for k in PORT_KEYS]
    prev_end = df['end_time'].groupby(keys, sort=False).transform(lambda s: s.cummax().shift())
    new = prev_end.isna() | (df['start_time'] > prev_end + touch)
    df['_g'] = new.cumsum()
    out = df.groupby('_g').agg(ip_addr=('ip_addr', 'first'), cid=('cid', 'first'), lid=('lid', 'first'),
                               start_time=('start_time', 'min'), end_time=('end_time', 'max'),
                               source=('source', lambda s: '+'.join(sorted({x for v in s for x in str(v).split('+')}))))
    return out.reset_index(drop=True)[INTERVAL_COLUMNS]


# ─────────────────────────────────────────────
# 제외 적용
# ─────────────────────────────────────────────
def exclusion_mask(df: pd.DataFrame, intervals: pd.DataFrame) -> np.ndarray:
    """df 의 각 행이 (같은 포트, start_time <= occur_date <= end_time) 구간에 속하는지.
    구간 수에 관계없이 포트별 searchsorted 로 계산 (구간 수 × 전체 행 순회 방지)."""
    mask = np.zeros(len(df), dtype=bool)
    if len(df) == 0 or intervals is None or len(intervals) == 0:
        return mask
    merged = merge_intervals(intervals, touch=pd.Timedelta(0))
    times = pd.to_datetime(df['occur_date']).values.astype('datetime64[ns]')
    ports = df.groupby(PORT_KEYS, sort=False).indices
    for key, g in merged.groupby(PORT_KEYS, sort=False):
        rows = ports.get(key if isinstance(key, tuple) else (key,))
        if rows is None:
            continue
        starts = g['start_time'].values.astype('datetime64[ns]')   # merge 결과: 정렬·비중첩
        ends = g['end_time'].values.astype('datetime64[ns]')
        t = times[rows]
        idx = np.searchsorted(starts, t, side='right') - 1
        ok = idx >= 0
        ok[ok] = t[ok] <= ends[idx[ok]]
        mask[rows[ok]] = True
    return mask


def suspect_stats(df: pd.DataFrame, intervals: pd.DataFrame, port_drop_fraction: float = 0.5) -> dict:
    """{rows, excluded_rows, fraction, ports, dropped_ports, by_source}
    dropped_ports: 의심 비율이 port_drop_fraction 을 넘는 포트(만성 불량 포트) 목록."""
    rows = len(df) if df is not None else 0
    stats = {'rows': rows, 'excluded_rows': 0, 'fraction': 0.0, 'ports': 0, 'dropped_ports': [], 'by_source': {}}
    if rows == 0:
        return stats
    stats['ports'] = int(df[PORT_KEYS].drop_duplicates().shape[0])
    if intervals is None or len(intervals) == 0:
        return stats
    mask = exclusion_mask(df, intervals)
    stats['excluded_rows'] = int(mask.sum())
    stats['fraction'] = stats['excluded_rows'] / rows
    per_port = pd.Series(mask, index=df.index).groupby([df[k] for k in PORT_KEYS], sort=False).mean()
    stats['dropped_ports'] = [[str(k[0]), int(k[1]), int(k[2])] for k in per_port[per_port > port_drop_fraction].index]   # JSON 직렬화 가능
    tokens = sorted({t for s in intervals['source'].astype(str) for t in s.split('+')}) if 'source' in intervals else []
    for tok in tokens:
        sub = intervals[intervals['source'].astype(str).apply(lambda s, tok=tok: tok in s.split('+'))]
        stats['by_source'][tok] = int(exclusion_mask(df, sub).sum())
    return stats


def exceeds_suspect_limit(stats: dict, policy) -> bool:
    """의심 비율이 한도를 넘으면 학습을 중단해야 함 (대규모 장애 중 재학습 금지)"""
    return stats['fraction'] > policy.max_excluded_fraction


def apply_suspect_filter(df: pd.DataFrame, intervals: pd.DataFrame, port_drop_fraction: float = 0.5, stats=None):
    """의심 구간 행과 만성 불량 포트 전체를 제거. Returns: (필터링된 df, suspect_stats)
    stats 는 이미 계산한 suspect_stats 를 재사용할 때 전달."""
    stats = stats or suspect_stats(df, intervals, port_drop_fraction)
    if df is None or df.empty:
        return df, stats
    drop = exclusion_mask(df, intervals)
    if stats['dropped_ports']:
        drop |= pd.MultiIndex.from_frame(df[PORT_KEYS]).isin([tuple(p) for p in stats['dropped_ports']])
    return df[~drop].copy(), stats
