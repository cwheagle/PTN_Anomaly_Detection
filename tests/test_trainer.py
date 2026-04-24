import pytest
import os
import pandas as pd
from src.models.trainer import Trainer
from src.config import PATHS

def test_trainer_with_real_data(monkeypatch):
    """실제 생성된 train_data.csv를 사용하여 학습 프로세스 검증"""
    train_data_path = "data/train_data.csv"
    if not os.path.exists(train_data_path):
        pytest.skip("train_data.csv가 없습니다. test_simul을 먼저 실행하세요.")
        
    trainer = Trainer()
    
    # 설정 최적화: 테스트 속도를 위해 최소한의 실행
    trainer.config['epochs'] = 1
    
    # 모델 및 스케일러 저장 경로를 테스트용으로 임시 변경
    test_model_path = "models/test_model.pth"
    test_scaler_path = "models/test_scaler.joblib"
    monkeypatch.setitem(PATHS, 'model_save_path', test_model_path)
    monkeypatch.setitem(PATHS, 'scaler_save_path', test_scaler_path)
    
    try:
        # 학습 실행 (data/train_data.csv 사용)
        trainer.train(train_data_path)
        
        # 파일 생성 확인
        assert os.path.exists(test_model_path)
        assert os.path.exists(test_scaler_path)
        
    finally:
        # 테스트 완료 후 임시 파일 삭제
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        if os.path.exists(test_scaler_path):
            os.remove(test_scaler_path)
