"""
train_window 검증 (P1-1): 학습 구간 · 포트 분할 · 장애 의심 구간 산출
"""
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data import train_window as tw
from src.pipeline.retrain_policy import RetrainPolicy

STEP = pd.Timedelta(minutes=15)
T0 = pd.Timestamp('2026-07-01 00:00')


def _alarm_rows(steps, lid=1):
    return pd.DataFrame({'occur_date': [T0 + i * STEP for i in steps],
                         'ip_addr': '1.1.1.1', 'cid': 0, 'lid': lid, 'alarm_level': 1})


def _traffic(n=300, lid=1, **overrides):
    """정상 트래픽 n 스텝 (tx=rx=1000, error=0). overrides: {col: {step: value}}"""
    df = pd.DataFrame({'occur_date': [T0 + i * STEP for i in range(n)], 'ip_addr': '1.1.1.1', 'cid': 0, 'lid': lid,
                       'tx_packet': 1000.0, 'rx_packet': 1000.0, 'error_packet': 0.0})
    for col, mapping in overrides.items():
        for step, val in mapping.items():
            df.loc[step, col] = val
    return df


# T-W1
def test_plan_window_includes_trigger_time_and_gate_is_last_days():
    now = datetime(2026, 10, 1, 3, 0)
    w = tw.plan_window(now, RetrainPolicy(train_days=28, gate_days=3))
    assert w.end == now and w.start == now - timedelta(days=28)
    assert w.gate_end == now and w.gate_start == now - timedelta(days=3)
    assert w.start < w.gate_start < w.end


# T-W2
def test_is_val_port_deterministic_across_processes_and_fraction():
    ports = [('10.0.%d.%d' % (i // 250, i % 250), i % 7, i % 13) for i in range(10000)]
    flags = [tw.is_val_port(ip, c, l, 0.10, 'ptn') for ip, c, l in ports]
    assert abs(np.mean(flags) - 0.10) < 0.01
    # 다른 프로세스(PYTHONHASHSEED 상이)에서도 같은 결과여야 함 — 내장 hash() 사용 금지
    code = ("from src.data.train_window import is_val_port;"
            "print(''.join('1' if is_val_port('10.0.%d.%d' % (i // 250, i % 250), i % 7, i % 13, 0.10, 'ptn') else '0'"
            " for i in range(300)))")
    outs = {subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
                           env={'PYTHONHASHSEED': seed, 'PYTHONPATH': '.', 'PATH': ''}).stdout.strip()
            for seed in ('1', '2')}
    assert len(outs) == 1
    assert outs.pop() == ''.join('1' if f else '0' for f in flags[:300])
    assert tw.is_val_port('a', 1, 1, 0.1, 'x') == tw.is_val_port('a', 1, 1, 0.1, 'x')


def test_split_ports_keeps_each_port_on_one_side():
    df = pd.concat([_traffic(10, lid=l) for l in range(200)], ignore_index=True)
    train, val = tw.split_ports(df, 0.10, 'ptn')
    assert len(train) + len(val) == len(df)
    assert not set(train.lid) & set(val.lid)
    assert 0 < val.lid.nunique() < 50


# T-W3
def test_intervals_from_alarms_merges_small_gaps_and_expands():
    # 스텝 100~102, 간격 4 이하(106) 는 병합, 간격 5(112 -> 6 차이) 는 분리
    rows = _alarm_rows([100, 101, 102, 106, 112])
    iv = tw.intervals_from_alarms(rows, pre_steps=16, post_steps=4, gap_steps=4)
    assert len(iv) == 2
    first, second = iv.iloc[0], iv.iloc[1]
    assert first.start_time == T0 + (100 - 16) * STEP and first.end_time == T0 + (106 + 4) * STEP
    assert second.start_time == T0 + (112 - 16) * STEP and second.end_time == T0 + (112 + 4) * STEP
    assert set(iv.source) == {'alarm'}


def test_intervals_from_alarms_empty_input():
    assert len(tw.intervals_from_alarms(None, 16, 4)) == 0
    assert list(tw.intervals_from_alarms(_alarm_rows([]), 16, 4).columns) == tw.INTERVAL_COLUMNS


def test_intervals_from_alarms_separates_ports():
    rows = pd.concat([_alarm_rows([100, 101], lid=1), _alarm_rows([100, 101], lid=2)], ignore_index=True)
    assert len(tw.intervals_from_alarms(rows, 16, 4)) == 2


# T-W4
def test_rule_intervals_ignore_sporadic_errors_but_flag_sustained():
    pol = RetrainPolicy()      # 기본 rule_min_consecutive=6, rule_error_ge=10
    n = pol.rule_min_consecutive
    sporadic = _traffic(error_packet={150 + i: 80 for i in range(n - 1)})      # n-1 스텝: 정상 변동으로 남김
    sustained = _traffic(error_packet={150 + i: 80 for i in range(n)})         # n 스텝 연속: 의심
    assert len(tw.intervals_from_rules(sporadic, pol)) == 0
    iv = tw.intervals_from_rules(sustained, pol)
    assert len(iv) == 1
    assert iv.iloc[0].start_time == T0 + (150 - pol.suspect_pre_steps) * STEP
    assert iv.iloc[0].end_time == T0 + (150 + n - 1 + pol.suspect_post_steps) * STEP
    assert iv.iloc[0].source == 'rule'


def test_rule_intervals_non_consecutive_errors_not_merged_into_run():
    pol = RetrainPolicy()
    df = _traffic(error_packet={150: 80, 151: 80, 153: 80, 154: 80})        # 152 가 정상 -> 2+2
    assert len(tw.intervals_from_rules(df, pol)) == 0


def test_rule_intervals_flag_traffic_drop_and_rx_power_drop():
    pol = RetrainPolicy()
    drop = _traffic(tx_packet={i: 10 for i in range(150, 156)}, rx_packet={i: 10 for i in range(150, 156)})
    assert len(tw.intervals_from_rules(drop, pol)) == 1
    optical = pd.DataFrame({'occur_date': [T0 + i * STEP for i in range(300)], 'ip_addr': '1.1.1.1', 'cid': 0,
                            'lid': 1, 'tx_avg_power': -3.0, 'rx_avg_power': -10.0})
    optical.loc[150:155, 'rx_avg_power'] = -16.0
    assert len(tw.intervals_from_rules(optical, pol)) == 1


def test_rule_intervals_need_history_before_judging():
    pol = RetrainPolicy()
    df = _traffic(40, error_packet={i: 80 for i in range(3, 10)})            # 직전 이력 < 16 스텝
    assert len(tw.intervals_from_rules(df, pol)) == 0


def test_merge_intervals_merges_overlap_and_adjacent_and_joins_sources():
    a = tw.intervals_from_alarms(_alarm_rows([100]), 2, 2)                     # [98, 102]
    b = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 1, 'start_time': T0 + 101 * STEP,
                       'end_time': T0 + 110 * STEP, 'source': 'rule'}])        # 겹침
    c = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 1, 'start_time': T0 + 111 * STEP,
                       'end_time': T0 + 115 * STEP, 'source': 'explicit'}])    # 맞닿음(+1스텝)
    far = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 1, 'start_time': T0 + 200 * STEP,
                         'end_time': T0 + 201 * STEP, 'source': 'rule'}])
    m = tw.merge_intervals(a, b, c, far)
    assert len(m) == 2
    assert m.iloc[0].start_time == T0 + 98 * STEP and m.iloc[0].end_time == T0 + 115 * STEP
    assert m.iloc[0].source == 'alarm+explicit+rule'


def test_merge_intervals_accepts_legacy_failure_time_column():
    legacy = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 1, 'start_time': T0, 'failure_time': T0 + STEP}])
    m = tw.merge_intervals(legacy)
    assert m.iloc[0].end_time == T0 + STEP and m.iloc[0].source == 'explicit'


# T-W5
def _sequence_windows_around_gap(gap, gap_start=150, n=300):
    """n 스텝 포트에서 [gap_start, gap_start+gap) 을 제외하고 전처리 -> 만들어진 각 윈도우의 (공백 이전 행 존재, 이후 행 존재, 공백 행 수)"""
    from src.data.data_processor import DataProcessor
    rng = np.random.default_rng(1)
    df = pd.DataFrame({'occur_date': [T0 + i * STEP for i in range(n)], 'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 1,
                       'tx_packet': rng.integers(900, 1100, n), 'rx_packet': rng.integers(900, 1100, n),
                       'error_packet': 0})
    iv = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 1, 'start_time': T0 + gap_start * STEP,
                        'end_time': T0 + (gap_start + gap - 1) * STEP}])
    kept = df[~tw.exclusion_mask(df, iv)]
    assert len(kept) == n - gap

    proc = DataProcessor('traffic')
    clean = proc.preprocess(kept, is_train=False)
    proc.create_sequences(clean, is_train=True)             # 스케일러 fit
    _, last_idx = proc.create_sequences(clean, is_train=False)[('1.1.1.1', 0, 1)]
    g0, g1 = T0 + gap_start * STEP, T0 + (gap_start + gap) * STEP
    out = []
    for li in last_idx:
        pos = clean.index.get_loc(li)
        w = clean['occur_date'].iloc[pos - proc.window_size + 1: pos + 1]
        out.append(((w < g0).any(), (w >= g1).any(), int(((w >= g0) & (w < g1)).sum())))
    assert out
    return out


@pytest.mark.parametrize("gap", [3, 4, 8, 40])
def test_gap_left_by_exclusion_is_never_crossed_by_a_sequence(gap):
    """제외로 생긴 시간축 공백(3스텝 이상)을 가로지르는 시퀀스가 만들어지면 안 된다.
    전처리가 공백을 NaN 으로 남겨 윈도우를 버리며, 보간은 공백 양끝에서 최대 1행씩만 메운다."""
    for before, after, gap_rows in _sequence_windows_around_gap(gap):
        assert not (before and after)
        assert gap_rows <= 1


def test_short_gap_of_two_steps_or_less_is_still_interpolated():
    """한계 명시: 전처리 보간(limit=1, both)은 2스텝 이하 공백을 메우므로 이 길이의 제외 구간은 시퀀스가 가로지른다.
    의심 구간은 앞 16·뒤 4스텝 확장으로 항상 20스텝 이상이라 영향이 없지만, 명시 CSV 의 짧은 구간에는 해당."""
    assert any(b and a for b, a, _ in _sequence_windows_around_gap(2))


# T-W6
def test_suspect_stats_and_abort_decision():
    pol = RetrainPolicy(max_excluded_fraction=0.20)
    df = _traffic(100)
    ok_iv = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 1, 'start_time': T0, 'end_time': T0 + 14 * STEP,
                           'source': 'alarm'}])         # 15 / 100 = 15%
    bad_iv = ok_iv.assign(end_time=T0 + 29 * STEP)       # 30%
    s_ok, s_bad = tw.suspect_stats(df, ok_iv), tw.suspect_stats(df, bad_iv)
    assert s_ok['excluded_rows'] == 15 and s_ok['fraction'] == pytest.approx(0.15)
    assert not tw.exceeds_suspect_limit(s_ok, pol) and tw.exceeds_suspect_limit(s_bad, pol)
    assert s_ok['by_source'] == {'alarm': 15}


def test_suspect_filter_drops_chronically_bad_ports_entirely():
    df = pd.concat([_traffic(100, lid=1), _traffic(100, lid=2)], ignore_index=True)
    iv = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 1, 'start_time': T0, 'end_time': T0 + 59 * STEP,
                        'source': 'rule'}])               # 포트1 의 60% > 50%
    kept, stats = tw.apply_suspect_filter(df, iv, port_drop_fraction=0.5)
    assert stats['dropped_ports'] == [['1.1.1.1', 0, 1]]
    assert set(kept.lid) == {2} and len(kept) == 100


def test_suspect_filter_keeps_port_below_drop_fraction():
    df = _traffic(100)
    iv = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 1, 'start_time': T0, 'end_time': T0 + 19 * STEP,
                        'source': 'rule'}])
    kept, stats = tw.apply_suspect_filter(df, iv, port_drop_fraction=0.5)
    assert stats['dropped_ports'] == [] and len(kept) == 80


def test_exclusion_mask_matches_naive_loop_on_random_intervals():
    rng = np.random.default_rng(7)
    df = pd.concat([_traffic(500, lid=l) for l in range(1, 6)], ignore_index=True)
    rows = []
    for _ in range(60):
        s = int(rng.integers(0, 480))
        rows.append({'ip_addr': '1.1.1.1', 'cid': 0, 'lid': int(rng.integers(1, 7)),
                     'start_time': T0 + s * STEP, 'end_time': T0 + (s + int(rng.integers(0, 20))) * STEP})
    iv = pd.DataFrame(rows)
    naive = np.zeros(len(df), dtype=bool)
    for r in iv.itertuples():
        naive |= ((df.lid == r.lid) & (df.occur_date >= r.start_time) & (df.occur_date <= r.end_time)).values
    assert (tw.exclusion_mask(df, iv) == naive).all()
