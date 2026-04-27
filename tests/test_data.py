import pytest
import pandas as pd
import numpy as np
from src.data.data_processor import DataProcessor

@pytest.fixture
def mock_raw_df():
    """현 표준 스키마를 따르는 테스트용 Mock 데이터 생성"""
    data = {
        'occur_date': pd.date_range(start='2026-04-27', periods=20, freq='15min'),
        'ip_addr': ['192.168.1.1']*20,
        'cid': [1]*20,
        'lid': [1]*20,
        'tx_packet': np.linspace(1000, 2000, 20),
        'rx_packet': np.linspace(900, 1900, 20),
        'error_packet': [0]*19 + [10], # 마지막에 에러 주입
        'tx_avg_power': [-5.0]*20,
        'rx_avg_power': [-5.2]*20
    }
    return pd.DataFrame(data)

def test_processor_logic(mock_raw_df):
    """DataProcessor의 전처리 로직 검증 (결측치, 클리핑)"""
    processor = DataProcessor()
    
    # 의도적으로 결측치와 비정상 범위 데이터 주입
    df_test = mock_raw_df.copy()
    df_test.loc[5, 'tx_packet'] = np.nan
    df_test.loc[10, 'rx_avg_power'] = -150 # Clipping 범위 밖
    
    df_processed = processor.preprocess(df_test)
    
    # 1. 결측치 보간 확인
    assert not df_processed['tx_packet'].isnull().any()
    # 2. 클리핑 확인 (-100 ~ 20)
    assert df_processed.loc[10, 'rx_avg_power'] >= -100
    # 3. 데이터 길이 유지 확인
    assert len(df_processed) == len(mock_raw_df)

def test_processor_sequence_creation(mock_raw_df):
    """시퀀스 생성 및 시간 단절 대응 로직 검증"""
    processor = DataProcessor()
    
    # 10번 인덱스와 11번 인덱스 사이에 1시간 공백 주입
    df_gap = mock_raw_df.copy()
    df_gap.loc[11:, 'occur_date'] = df_gap.loc[11:, 'occur_date'] + pd.Timedelta(hours=1)
    
    grouped_res, _ = processor.prepare_inference_data(df_gap)
    
    # 공백으로 인해 연속된 시퀀스가 줄어들어야 함
    # 20개 데이터, window=12일 때 공백이 없으면 9개 시퀀스 생성
    # 1시간 공백이 있으면 중간의 윈도우들이 스킵됨
    for key, sequences in grouped_res.items():
        assert len(sequences) < (20 - processor.window_size + 1)
        print(f"\n[Test] Created {len(sequences)} sequences with time gap.")

def test_processor_scaling(mock_raw_df):
    """스케일링 차원 및 범위 검증"""
    processor = DataProcessor()
    df_processed = processor.preprocess(mock_raw_df)
    scaled = processor.scale_data(df_processed, is_train=True)
    
    # 피처 개수 5개 확인
    assert scaled.shape[1] == 5
    # MinMaxScaler 범위 확인
    assert np.min(scaled) >= 0 and np.max(scaled) <= 1
