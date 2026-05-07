import os
import sys
import pandas as pd
import numpy as np

# 경로 추가
sys.path.append(os.getcwd())

from src.pipeline.inference import AnomalyDetector

def diagnose_reason(port_res, ft):
    """특정 트랙(Traffic/Optical)의 데이터 변화를 분석하여 진단 메시지 생성"""
    if ft == 'optical':
        curr_tx = port_res['tx_avg_power'].iloc[-1]
        curr_rx = port_res['rx_avg_power'].iloc[-1]
        
        if curr_rx <= -39.9:
            return f"Signal Lost (RX is at -40dBm)"
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

def export_all_anomalies():
    detector = AnomalyDetector()
    
    # 최신 데이터 로드
    df_t = pd.read_csv("data/traffic_test.csv")
    df_o = pd.read_csv("data/optical_test.csv")
    
    # 추론 실행
    res = detector.detect(df_traffic=df_t, df_optical=df_o)
    
    if res is None:
        print("[!] No results to analyze.")
        return

    # 이상치만 필터링
    anomalies = res[res['is_anomaly']].copy()
    
    # 1. 상세 CSV 저장
    output_path = "data/detected_anomalies_all.csv"
    anomalies.to_csv(output_path, index=False)
    print(f"\n[OK] Full anomaly list saved to: {output_path}")

    # 2. 포트별 요약 보고서 출력
    print("\n" + "="*90)
    print(f"{'IP ADDRESS':<15} | {'CID':<3} | {'LID':<3} | {'COUNT':<5} | {'MAIN REASON':<15} | {'MAX SCORE':<10}")
    print("-" * 90)
    
    summary = anomalies.groupby(['ip_addr', 'cid', 'lid']).agg({
        'is_anomaly': 'count',
        'anomaly_reason': lambda x: x.mode()[0],
        'anomaly_score': 'max'
    }).reset_index()
    
    summary = summary.sort_values('anomaly_score', ascending=False)
    
    for _, row in summary.iterrows():
        print(f"{row['ip_addr']:<15} | {int(row['cid']):<3} | {int(row['lid']):<3} | {row['is_anomaly']:<5} | {row['anomaly_reason']:<15} | {row['anomaly_score']:<10.4f}")
    
    print("-" * 90)

    # 3. 상위 장애 포트 정밀 진단 (Top 3)
    print("\n[TOP ANOMALY DIAGNOSIS]")
    for _, row in summary.head(3).iterrows():
        ip, cid, lid = row['ip_addr'], row['cid'], row['lid']
        
        # 해당 포트의 가장 높은 점수 시점 데이터 추출
        port_anom = anomalies[(anomalies['ip_addr']==ip) & (anomalies['cid']==cid) & (anomalies['lid']==lid)]
        worst_case = port_anom.sort_values('anomaly_score', ascending=False).iloc[0]
        
        # 진단을 위해 원본 데이터에서 해당 시점의 수치 가져오기
        main_ft = 'optical' if 'OPTICAL' in worst_case['anomaly_reason'] else 'traffic'
        raw_df = df_o if main_ft == 'optical' else df_t
        
        # 시간 타입 통일 후 매칭
        raw_df['occur_date'] = pd.to_datetime(raw_df['occur_date']).dt.round('15min')
        target_time = pd.to_datetime(worst_case['occur_date'])
        
        raw_row = raw_df[(raw_df['ip_addr']==ip) & (raw_df['cid']==cid) & (raw_df['lid']==lid) & (raw_df['occur_date']==target_time)]
        
        if not raw_row.empty:
            diagnosis = diagnose_reason(raw_row, main_ft)
            print(f"[*] Port {ip}-{int(cid)}-{int(lid)}: {diagnosis}")
            print(f"    - Occurrence: {worst_case['occur_date']}")
            print(f"    - Max Anomaly Score: {worst_case['anomaly_score']:.4f}")
        else:
            print(f"[*] Port {ip}-{int(cid)}-{int(lid)}: Pattern analysis failed (Raw data not found)")
    
    print("="*90)

if __name__ == "__main__":
    export_all_anomalies()
