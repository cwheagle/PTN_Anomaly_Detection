import pytest
import pandas as pd
import numpy as np
from src.data.data_processor import DataProcessor

N = 40  # 포트 필터의 최소 요건(window_size * 2 = 24행)을 넘기는 길이


def _port_df(lid=1, n=N, tx=None, error=0):
    """현 표준 스키마(ip_addr/cid/lid)를 따르는 단일 포트 Mock 데이터"""
    return pd.DataFrame({
        'occur_date': pd.date_range(start='2026-04-27', periods=n, freq='15min'),
        'ip_addr': '192.168.1.1',
        'cid': 1,
        'lid': lid,
        'tx_packet': np.linspace(1000, 2000, n) if tx is None else tx,
        'rx_packet': np.linspace(900, 1900, n),
        'error_packet': error,
        'tx_avg_power': -5.0,
        'rx_avg_power': -5.2,
    })


@pytest.fixture
def mock_raw_df():
    return _port_df()


def test_processor_logic(mock_raw_df):
    """결측 보간(1개 한정), 파생 변수 생성, 길이 유지 검증"""
    processor = DataProcessor('traffic')
    df_test = mock_raw_df.copy()
    df_test.loc[5, 'tx_packet'] = np.nan

    df = processor.preprocess(df_test, is_train=True)

    assert df is not None
    # 1. 단건 결측은 선형 보간으로 채워져야 함
    assert not df['tx_packet'].isnull().any()
    # 2. 시간축 재구성 후에도 길이 유지
    assert len(df) == len(mock_raw_df)
    # 3. 파생 변수(ma/var/lag)가 모두 생성되어야 함
    for col in processor.extended_feature_cols:
        assert col in df.columns, f"파생 변수 누락: {col}"
    # 4. traffic은 log1p 변환되므로 원본(>=1000)보다 훨씬 작아야 함
    assert df['tx_packet'].max() < 20


def test_processor_consecutive_missing_not_interpolated(mock_raw_df):
    """2개 이상 연속 결측은 보간하지 않음 (limit=1) -> NaN 유지"""
    processor = DataProcessor('traffic')
    df_test = mock_raw_df.copy()
    df_test.loc[[10, 11, 12], 'tx_packet'] = np.nan

    df = processor.preprocess(df_test, is_train=True)

    assert df['tx_packet'].isnull().any()


def test_processor_sequence_creation_skips_gap(mock_raw_df):
    """결측으로 NaN이 남은 구간을 포함하는 윈도우는 버려져야 함"""
    processor = DataProcessor('traffic')
    full = processor.preprocess(mock_raw_df, is_train=True)
    full_seqs = processor.create_sequences(full, is_train=True)
    assert len(full_seqs) == N - processor.window_size + 1

    gap = mock_raw_df.copy()
    gap.loc[[20, 21, 22], 'tx_packet'] = np.nan
    gap_df = processor.preprocess(gap, is_train=True)
    gap_seqs = processor.create_sequences(gap_df, is_train=True)

    assert 0 < len(gap_seqs) < len(full_seqs)


def test_processor_scaling_shape_and_clip(mock_raw_df):
    """시퀀스 차원은 (N, window, 파생 변수 포함 컬럼 수), 값은 [-10, 10]으로 클리핑"""
    processor = DataProcessor('traffic')
    df = processor.preprocess(mock_raw_df, is_train=True)
    seqs = processor.create_sequences(df, is_train=True)

    assert seqs.shape[1:] == (processor.window_size, len(processor.extended_feature_cols))
    assert seqs.shape[2] == 15
    assert seqs.min() >= -10.0 and seqs.max() <= 10.0


def test_optical_has_ten_features():
    assert len(DataProcessor('optical').extended_feature_cols) == 10


# --- 회귀 테스트: 포트 품질 필터 (.any() > 0.5 버그) ---

def test_quality_filter_keeps_port_with_single_spike():
    """1e9 초과가 단 1건뿐인 포트는 폐기되면 안 됨 (수정 전: .any()가 True라 폐기됨)"""
    spike = np.full(N, 1000.0)
    spike[3] = 2e9
    df = pd.concat([_port_df(lid=1), _port_df(lid=2, tx=spike)], ignore_index=True)

    kept = DataProcessor('traffic')._filter_low_quality(df)

    assert sorted(kept['lid'].unique()) == [1, 2]


def test_quality_filter_drops_port_mostly_over_limit():
    """1e9 초과가 절반 넘게 지속되는 오염 포트는 폐기"""
    df = pd.concat([_port_df(lid=1), _port_df(lid=3, tx=np.full(N, 2e9))], ignore_index=True)

    kept = DataProcessor('traffic')._filter_low_quality(df)

    assert sorted(kept['lid'].unique()) == [1]


def test_quality_filter_drops_port_with_persistent_errors():
    df = pd.concat([_port_df(lid=1), _port_df(lid=4, error=5000)], ignore_index=True)

    kept = DataProcessor('traffic')._filter_low_quality(df)

    assert sorted(kept['lid'].unique()) == [1]
