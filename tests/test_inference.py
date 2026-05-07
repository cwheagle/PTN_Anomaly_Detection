import sys
import os
import pandas as pd
sys.path.append(os.getcwd())
from src.pipeline.inference import AnomalyDetector

def run_inference_test():
    print("="*70)
    print("ENSEMBLE INFERENCE & REASON VERIFICATION")
    print("="*70)
    
    detector = AnomalyDetector()
    
    df_t = pd.read_csv("data/traffic_test.csv") if os.path.exists("data/traffic_test.csv") else None
    df_o = pd.read_csv("data/optical_test.csv") if os.path.exists("data/optical_test.csv") else None
    
    results = detector.detect(df_traffic=df_t, df_optical=df_o)
    
    if results is not None:
        anomalies = results[results['is_anomaly']]
        print(f"[*] Processed: {len(results)} items, Anomalies: {len(anomalies)}")

        if not anomalies.empty:
            print("\n--- Detected Anomalies (Top 10) ---")
            print(anomalies.sort_values('anomaly_score', ascending=False).head(10)[['occur_date', 'ip_addr', 'cid', 'lid', 'anomaly_score', 'anomaly_reason']])
    else:
        print("[SKIP] No data for analysis.")
    print("="*70)

if __name__ == "__main__":
    run_inference_test()
