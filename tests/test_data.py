import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock, patch
from src.data.data_processor import DataProcessor
from src.data.db_connector import DBConnector

@pytest.fixture
def real_sample_df():
    """이미 생성된 train_data.csv를 로드합니다. 없으면 테스트를 스킵합니다."""
    path = "data/train_data.csv"
    if not os.path.exists(path):
        # 더미 데이터 생성 (테스트용)
        data = {
            'occur_date': pd.date_range(start='2026-04-27', periods=20, freq='15min'),
            'ip_addr': ['1.1.1.1']*20,
            'cid': [1]*20,
            'lid': [1]*20,
            'tx_packet': np.random.rand(20)*1000,
            'rx_packet': np.random.rand(20)*1000,
            'error_packet': np.random.rand(20)*10,
            'tx_avg_power': np.random.rand(20)-10,
            'rx_avg_power': np.random.rand(20)-10
        }
        return pd.DataFrame(data)
    return pd.read_csv(path).iloc[:100]

def test_data_processor_preprocess(real_sample_df):
    processor = DataProcessor()
    # 결측치 주입 테스트
    df_test = real_sample_df.copy()
    # error_packet에 결측치 주입
    df_test.loc[0, 'error_packet'] = np.nan
    
    df_processed = processor.preprocess(df_test)
    
    assert df_processed is not None
    assert df_processed['error_packet'].isnull().sum() == 0
    assert len(df_processed) == len(real_sample_df)

def test_data_processor_scaling(real_sample_df):
    processor = DataProcessor()
    df_processed = processor.preprocess(real_sample_df)
    scaled_data = processor.scale_data(df_processed, is_train=True)
    
    # feature_cols는 5개 (tx_packet, rx_packet, error_packet, tx_avg_power, rx_avg_power)
    assert scaled_data.shape == (len(df_processed), 5)
    # 스케일링 범위 확인 (MinMaxScaler)
    assert np.all(scaled_data >= -1e-7) and np.all(scaled_data <= 1 + 1e-7)

@patch('mysql.connector.pooling.MySQLConnectionPool')
def test_db_connector_pool(mock_pool):
    connector = DBConnector()
    assert connector is not None
    assert mock_pool.called
