import pandas as pd
from src.models.trainer import Trainer
from src.pipeline.inference import AnomalyDetector
from src.config import MODEL_CONFIG
import os

def test_manual_run_cycle():
    """실제 학습 및 추론 사이클을 수동으로 실행하고 결과를 출력합니다."""
    print("\n" + "="*50)
    print("STARTING MANUAL TRAINING & INFERENCE CYCLE")
    print("="*50)

    # 1. 학습
    trainer = Trainer()
    trainer.config['epochs'] = 100  # 제대로 학습
    trainer.train("data/train_data.csv")
    
    # 2. 추론
    detector = AnomalyDetector()
    test_df = pd.read_csv("data/test_data.csv")
    results = detector.detect(test_df)
    
    # 3. 결과 출력
    if results is not None:
        anomalies = results[results['is_anomaly'] == True]
        print(f"\n[Summary]")
        print(f"- Total Records: {len(results)}")
        print(f"- Anomalies Detected: {len(anomalies)}")
        
        if not anomalies.empty:
            print("\n[Detected Anomalies (Top 10)]")
            cols = ['collect_time', 'equipment_id', 'port_id', 'anomaly_score', 'anomaly_reason']
            print(anomalies[cols].sort_values('anomaly_score', ascending=False).head(10))
    
    print("\n" + "="*50)
    print("CYCLE COMPLETE")
    print("="*50)
