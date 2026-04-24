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
        pytest.skip("train_data.csv가 없습니다. test_simul을 먼저 실행하세요.")
    return pd.read_csv(path).iloc[:100] # 테스트 속도를 위해 일부만 사용

def test_data_processor_preprocess(real_sample_df):
    processor = DataProcessor()
    # 결측치 주입 테스트
    df_test = real_sample_df.copy()
    df_test.iloc[0, df_test.columns.get_loc('tx_error')] = np.nan
    
    df_processed = processor.preprocess(df_test)
    
    assert df_processed is not None
    assert df_processed['tx_error'].isnull().sum() == 0
    assert len(df_processed) == len(real_sample_df)

def test_data_processor_scaling(real_sample_df):
    processor = DataProcessor()
    df_processed = processor.preprocess(real_sample_df)
    scaled_data = processor.scale_data(df_processed, is_train=True)
    
    assert scaled_data.shape == (len(df_processed), 8)
    assert np.all(scaled_data >= -1e-7) and np.all(scaled_data <= 1 + 1e-7)

@patch('mysql.connector.connect')
def test_db_connector_fetch(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    connector = DBConnector()
    with patch('pandas.read_sql') as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame({'test': [1, 2, 3]})
        df = connector.fetch_performance_data(minutes=15)
        assert df is not None
        assert mock_read_sql.called
