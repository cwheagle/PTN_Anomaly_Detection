import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_ptn_data(days=30, equipment_count=1, ports_per_eq=2, add_anomaly=False):
    """
    PTN 장비의 성능 데이터를 시뮬레이션 생성합니다.
    """
    start_time = datetime.now() - timedelta(days=days)
    periods = days * 24 * 4  # 15분 단위
    
    data_list = []
    
    for eq_id in range(1, equipment_count + 1):
        for port_id in range(1, ports_per_eq + 1):
            eq_name = f"EQ_{eq_id:03d}"
            port_name = f"P_{port_id}"
            
            # 시간축 생성
            times = [start_time + timedelta(minutes=15 * i) for i in range(periods)]
            
            # 1. 트래픽 패턴 (BPS/PPS) - 24시간 주기 사인 곡선 + 노이즈
            time_indices = np.array([(t.hour * 60 + t.minute) / (24 * 60) for t in times])
            base_traffic = (np.sin(2 * np.pi * (time_indices - 0.25)) + 1) / 2
            
            tx_bps = (base_traffic * 800 + 100 + np.random.normal(0, 20, periods)).clip(min=0)
            rx_bps = (base_traffic * 750 + 80 + np.random.normal(0, 15, periods)).clip(min=0)
            tx_pps = tx_bps / 1.5 + np.random.normal(0, 5, periods).clip(min=0)
            rx_pps = rx_bps / 1.5 + np.random.normal(0, 5, periods).clip(min=0)
            
            # 2. 광 파워 (Tx/Rx Power)
            tx_power = np.full(periods, -5.0) + np.random.normal(0, 0.1, periods)
            rx_power = np.full(periods, -5.5) + np.random.normal(0, 0.1, periods)
            
            # 3. 에러 (Error)
            tx_error = np.zeros(periods)
            rx_error = np.zeros(periods)
            
            # 이상 징후 삽입 (필요 시)
            if add_anomaly:
                # 약 1% 확률로 이상 발생 시뮬레이션
                anomaly_indices = np.random.choice(range(periods), max(1, periods // 100), replace=False)
                for idx in anomaly_indices:
                    anomaly_type = np.random.choice(['error_spike', 'power_drop', 'traffic_surge'])
                    duration = np.random.randint(4, 12) # 1~3시간 지속
                    
                    if anomaly_type == 'error_spike':
                        tx_error[idx : idx + duration] = np.random.randint(100, 500, size=min(duration, periods-idx))
                    elif anomaly_type == 'power_drop':
                        rx_power[idx : idx + duration] -= np.random.uniform(5.0, 10.0)
                    elif anomaly_type == 'traffic_surge':
                        tx_bps[idx : idx + duration] *= np.random.uniform(2.0, 5.0)
            
            for i in range(periods):
                data_list.append({
                    'collect_time': times[i],
                    'equipment_id': eq_name,
                    'port_id': port_name,
                    'tx_bps': tx_bps[i],
                    'rx_bps': rx_bps[i],
                    'tx_pps': tx_pps[i],
                    'rx_pps': rx_pps[i],
                    'tx_error': tx_error[i],
                    'rx_error': rx_error[i],
                    'tx_power': tx_power[i],
                    'rx_power': rx_power[i]
                })
                
    return pd.DataFrame(data_list)

def run_generation():
    """전체 30일 데이터를 생성하여 80/20 비율(24일/6일)로 학습 및 테스트셋을 분리 저장합니다."""
    os.makedirs("data", exist_ok=True)
    
    total_days = 30
    train_ratio = 0.8
    train_days = int(total_days * train_ratio)
    test_days = total_days - train_days

    # 1. 학습용 데이터 생성 (전체 30일 중 앞 24일치, 정상 데이터만)
    print(f"Generating training data ({train_days} days, normal only)...")
    df_full_normal = generate_ptn_data(days=total_days, add_anomaly=False)
    # 뒤에서부터 test_days만큼 제외한 나머지를 학습용으로 사용
    split_idx = int(len(df_full_normal) * train_ratio)
    df_train = df_full_normal.iloc[:split_idx].copy()
    df_train.to_csv("data/train_data.csv", index=False)

    # 2. 테스트용 데이터 생성 (전체 30일 중 뒤 6일치, 이상 징후 포함)
    print(f"Generating test data ({test_days} days, with anomalies)...")
    df_full_anomaly = generate_ptn_data(days=total_days, add_anomaly=True)
    # 뒤의 20% 영역을 테스트용으로 사용
    df_test = df_full_anomaly.iloc[split_idx:].copy()
    df_test.to_csv("data/test_data.csv", index=False)

    print(f"Data generation complete.")
    print(f"- Train data: data/train_data.csv ({len(df_train)} records, {train_days} days)")
    print(f"- Test data: data/test_data.csv ({len(df_test)} records, {test_days} days)")

