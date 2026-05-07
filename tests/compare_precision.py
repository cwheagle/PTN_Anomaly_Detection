import pandas as pd
import numpy as np
import sys
import os

# 경로 추가
sys.path.append(os.getcwd())

def compare_raw_vs_detection():
    # 1. 데이터 로드
    df_raw = pd.read_csv("data/optical_test.csv")
    df_res = pd.read_csv("data/detected_anomalies_all.csv")
    
    if df_res.empty:
        print("No anomalies detected to compare.")
        return

    # 시간 정규화 (비교를 위해)
    df_raw['occur_date'] = pd.to_datetime(df_raw['occur_date']).dt.round('15min')
    df_res['occur_date'] = pd.to_datetime(df_res['occur_date'])

    # 상위 이상치 포트 하나 선택
    sample_port = df_res.iloc[0] # 192.168.143.76-5-5
    ip, cid, lid = sample_port['ip_addr'], sample_port['cid'], sample_port['lid']
    
    print(f"\n=== Precision Comparison: {ip} (Slot {cid}, Port {lid}) ===")
    
    # 해당 포트의 전체 원본 데이터
    port_raw = df_raw[(df_raw['ip_addr']==ip) & (df_raw['cid']==cid) & (df_raw['lid']==lid)].sort_values('occur_date')
    # 해당 포트의 탐지 결과 (전체 추론 결과가 필요하므로 다시 계산하거나 병합)
    
    # 윈도우 시각화 (이상 발생 시점 전후)
    target_time = pd.to_datetime(sample_port['occur_date'])
    context = port_raw[(port_raw['occur_date'] <= target_time)].tail(15)
    
    print(f"\n[Raw Data Flow leading to Anomaly at {target_time}]")
    print(f"{'TIME':<20} | {'TX':<8} | {'RX':<8} | {'IS_ANOMALY'}")
    print("-" * 55)
    
    for _, row in context.iterrows():
        is_anom = "★ DETECTED" if row['occur_date'] == target_time else ""
        print(f"{str(row['occur_date']):<20} | {row['tx_avg_power']:<8} | {row['rx_avg_power']:<8} | {is_anom}")

    # 2. 다른 포트 (192.168.99.227-1-3) 대조
    print(f"\n\n=== Case 2: 192.168.99.227 (Slot 1, Port 3) ===")
    sample_port2 = df_res[df_res['ip_addr'] == '192.168.99.227'].iloc[0]
    target_time2 = pd.to_datetime(sample_port2['occur_date'])
    port_raw2 = df_raw[(df_raw['ip_addr']=='192.168.99.227') & (df_raw['cid']==1) & (df_raw['lid']==3)].sort_values('occur_date')
    context2 = port_raw2[(port_raw2['occur_date'] <= target_time2)].tail(15)
    
    print(f"\n[Raw Data Flow leading to Anomaly at {target_time2}]")
    print(f"{'TIME':<20} | {'TX':<8} | {'RX':<8} | {'IS_ANOMALY'}")
    print("-" * 55)
    for _, row in context2.iterrows():
        is_anom = "★ DETECTED" if row['occur_date'] == target_time2 else ""
        print(f"{str(row['occur_date']):<20} | {row['tx_avg_power']:<8} | {row['rx_avg_power']:<8} | {is_anom}")

if __name__ == "__main__":
    compare_raw_vs_detection()
