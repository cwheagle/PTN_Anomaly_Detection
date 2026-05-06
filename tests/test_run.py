import pandas as pd
from src.models.trainer import Trainer
from src.pipeline.inference import AnomalyDetector
import os

def test_ensemble_cycle():
    """정제된 코드로 전체 학습 및 앙상블 추론 사이클 검증"""
    print("\n" + "="*70)
    print("FINAL REFACTORED ENSEMBLE CYCLE TEST")
    print("="*70)

    # 1. 독립 트랙 모델 학습
    for ft in ['traffic', 'optical']:
        Trainer(ft).train()
    
    # 2. 통합 추론
    print("\n[*] Initializing Refactored Detector...")
    detector = AnomalyDetector()
    
    # 데이터 로드
    df_t = pd.read_csv("data/traffic_test.csv") if os.path.exists("data/traffic_test.csv") else None
    df_o = pd.read_csv("data/optical_test.csv") if os.path.exists("data/optical_test.csv") else None
    
    print("[*] Running Ensemble Analysis...")
    # 개별 트랙 데이터프레임을 직접 전달하도록 변경
    results = detector.detect(df_traffic=df_t, df_optical=df_o)
    
    if results is not None:
        anomalies = results[results['is_anomaly']]
        print(f"\n[Final Results Summary]")
        print(f"- Processed: {len(results)} items")
        print(f"- Anomalies: {len(anomalies)} items")
        
        if not anomalies.empty:
            print("\n[Detected Anomalies (Top 10)]")
            cols = ['occur_date', 'ip_addr', 'cid', 'lid', 'anomaly_score', 'anomaly_reason']
            print(anomalies[cols].sort_values('anomaly_score', ascending=False).head(10))
    else:
        print("\n[SKIP] No data available for analysis.")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_ensemble_cycle()
