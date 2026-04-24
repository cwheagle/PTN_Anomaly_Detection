import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.pipeline.scheduler import PTNAnomalyScheduler

def test_run_inference_job_success():
    """스케줄러의 추론 작업 실행 성공 테스트"""
    with patch('src.pipeline.scheduler.DBConnector') as MockDB, \
         patch('src.pipeline.scheduler.AnomalyDetector') as MockDetector:
        
        # Mock 객체 설정
        mock_db = MockDB.return_value
        mock_detector = MockDetector.return_value
        
        # 1. 데이터 추출 결과 Mock
        mock_df = pd.DataFrame({'test': range(20)})
        mock_db.fetch_performance_data.return_value = mock_df
        
        # 2. 추론 결과 Mock
        mock_results = pd.DataFrame({
            'collect_time': ['2026-04-22'],
            'equipment_id': ['EQ001'],
            'port_id': ['P1'],
            'anomaly_score': [0.5],
            'is_anomaly': [False]
        })
        mock_detector.detect.return_value = mock_results
        
        scheduler = PTNAnomalyScheduler()
        scheduler.run_inference_job()
        
        # 핵심 메서드들이 호출되었는지 확인
        assert mock_db.fetch_performance_data.called
        assert mock_detector.detect.called
        assert mock_db.save_anomaly_results.called
        assert mock_db.disconnect.called

def test_run_inference_job_no_data():
    """데이터가 없을 경우 스케줄러가 정상적으로 건너뛰는지 테스트"""
    with patch('src.pipeline.scheduler.DBConnector') as MockDB, \
         patch('src.pipeline.scheduler.AnomalyDetector') as MockDetector:
        
        mock_db = MockDB.return_value
        mock_detector = MockDetector.return_value
        
        # 데이터가 비어있는 경우
        mock_db.fetch_performance_data.return_value = pd.DataFrame()
        
        scheduler = PTNAnomalyScheduler()
        scheduler.run_inference_job()
        
        # 데이터가 없으므로 추론과 저장은 호출되지 않아야 함
        assert mock_db.fetch_performance_data.called
        assert not mock_detector.detect.called
        assert not mock_db.save_anomaly_results.called
        assert mock_db.disconnect.called
