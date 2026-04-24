import pytest
import pandas as pd
import os
from src.simul.data_generator import generate_ptn_data, run_generation

def test_generate_ptn_data_shape():
    """데이터 생성 결과의 컬럼과 형태가 올바른지 확인"""
    days = 1
    df = generate_ptn_data(days=days, add_anomaly=False)
    
    expected_columns = [
        'collect_time', 'equipment_id', 'port_id', 
        'tx_bps', 'rx_bps', 'tx_pps', 'rx_pps', 
        'tx_error', 'rx_error', 'tx_power', 'rx_power'
    ]
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in expected_columns:
        assert col in df.columns

def test_run_generation_files_creation():
    """run_generation 함수가 실제 파일을 생성하는지 테스트"""
    # 기존 파일이 있다면 삭제 (테스트 신뢰성)
    for f in ["data/train_data.csv", "data/test_data.csv"]:
        if os.path.exists(f):
            os.remove(f)
            
    # 실행
    run_generation()
    
    # 파일 존재 확인
    assert os.path.exists("data/train_data.csv")
    assert os.path.exists("data/test_data.csv")
    
    # 데이터 내용이 비어있지 않은지 확인
    df_train = pd.read_csv("data/train_data.csv")
    df_test = pd.read_csv("data/test_data.csv")
    assert len(df_train) > 0
    assert len(df_test) > 0
