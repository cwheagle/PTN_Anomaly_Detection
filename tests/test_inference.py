import sys
import os
import pandas as pd
sys.path.append(os.getcwd())
from src.pipeline.inference import AnomalyDetector

def diagnose_reason(port_res, ft):
    """특정 트랙(Traffic/Optical)의 데이터 변화를 분석하여 진단 메시지 생성"""
    if port_res is None or len(port_res) == 0:
        return "No data for diagnosis"
        
    if ft == 'optical':
        curr_rx = port_res['rx_avg_power'].iloc[-1]
        curr_tx = port_res['tx_avg_power'].iloc[-1]
        
        if curr_rx <= -39.9:
            return f"Signal Lost (RX: -40dBm)"
        elif curr_rx < -20:
            return f"Weak Signal (RX: {curr_rx:.2f}dBm)"
        else:
            return f"Power Fluctuation (TX: {curr_tx:.2f}, RX: {curr_rx:.2f})"
            
    elif ft == 'traffic':
        curr_err = port_res['error_packet'].iloc[-1]
        curr_tx = port_res['tx_packet'].iloc[-1]
        
        if curr_err > 100:
            return f"High Error Packets ({int(curr_err)} pkts)"
        else:
            return f"Traffic Anomaly (TX: {int(curr_tx)} pkts)"
    return "Unknown Pattern"

def run_inference_test():
    print("\n" + "="*90)
    print("INTEGRATED ANOMALY DIAGNOSIS REPORT")
    print("="*90)
    
    detector = AnomalyDetector()
    
    # 1. 데이터 로드
    df_t = pd.read_csv("data/traffic_test.csv") if os.path.exists("data/traffic_test.csv") else None
    df_o = pd.read_csv("data/optical_test.csv") if os.path.exists("data/optical_test.csv") else None
    
    # 2. 추론 실행 (테스트 시에는 전체 이력 분석을 위해 latest_only=False 설정)
    results = detector.detect(df_traffic=df_t, df_optical=df_o, latest_only=False)
    
    if results is not None:
        anomalies = results[results['is_anomaly']].copy()
        print(f"[*] Processed: {len(results)} items, Anomalies found: {len(anomalies)}")

        # 3. 결과 CSV 저장
        save_path = "data/detected_anomalies_test.csv"
        results.to_csv(save_path, index=False)
        print(f"[*] Full results saved to: {save_path}")

        if not anomalies.empty:
            # 4. 포트별 요약 보고서 출력
            print("\n--- Port-wise Anomaly Summary ---")
            print(f"{'IP ADDRESS':<15} | {'CID':<3} | {'LID':<3} | {'COUNT':<5} | {'MAX SCORE':<10} | {'PRIMARY REASON'}")
            print("-" * 90)
            
            summary = anomalies.groupby(['ip_addr', 'cid', 'lid']).agg({
                'is_anomaly': 'count',
                'anomaly_score': 'max',
                'anomaly_reason': lambda x: x.mode()[0]
            }).reset_index().sort_values('anomaly_score', ascending=False)
            
            for _, row in summary.iterrows():
                print(f"{row['ip_addr']:<15} | {int(row['cid']):<3} | {int(row['lid']):<3} | {row['is_anomaly']:<5} | {row['anomaly_score']:<10.4f} | {row['anomaly_reason']}")
    else:
        print("[SKIP] No data for analysis.")
    print("\n" + "="*90)

if __name__ == "__main__":
    run_inference_test()
