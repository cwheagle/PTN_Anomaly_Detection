import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
sys.path.append(os.getcwd())

def visualize_top_anomalies(result_path="data/detected_anomalies_test.csv", top_n=10):
    if not os.path.exists(result_path):
        print(f"[!] Result file not found: {result_path}")
        return

    # 1. 데이터 로드 및 병합
    df_res = pd.read_csv(result_path)
    df_res['occur_date'] = pd.to_datetime(df_res['occur_date'])
    
    # 원본 데이터와 결합 (원본 수치를 가져오기 위함)
    if os.path.exists("data/traffic_test.csv"):
        df_raw = pd.read_csv("data/traffic_test.csv")
        df_raw['occur_date'] = pd.to_datetime(df_raw['occur_date']).dt.round('15min')
        df = pd.merge(df_res, df_raw, on=['ip_addr', 'cid', 'lid', 'occur_date'], how='left')
    else:
        df = df_res

    # 이상치가 있는 포트들만 추출
    anomalous_ports = df[df['is_anomaly']].groupby(['ip_addr', 'cid', 'lid'])['anomaly_score'].max().sort_values(ascending=False).head(top_n).index
    
    if len(anomalous_ports) == 0:
        print("[*] No anomalies found to visualize. Plotting top score ports instead.")
        anomalous_ports = df.groupby(['ip_addr', 'cid', 'lid'])['anomaly_score'].max().sort_values(ascending=False).head(top_n).index

    # 2. 저장 폴더 생성
    output_dir = "logs/plots"
    os.makedirs(output_dir, exist_ok=True)

    # 3. 포트별 시각화
    for i, (ip, cid, lid) in enumerate(anomalous_ports):
        port_data = df[(df['ip_addr'] == ip) & (df['cid'] == cid) & (df['lid'] == lid)].sort_values('occur_date')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # --- 상단: 원본 트래픽 (TX/RX Packet 출력) ---
        ax1.plot(port_data['occur_date'], port_data['tx_packet'], label='TX Packet', color='blue', alpha=0.6)
        ax1.plot(port_data['occur_date'], port_data['rx_packet'], label='RX Packet', color='green', alpha=0.6, linestyle='--')
        
        # 이상 지점 강조 (TX/RX 양쪽에 점을 찍어 어디서 튀었는지 명확히 함)
        anom_points = port_data[port_data['is_anomaly']]
        ax1.scatter(anom_points['occur_date'], anom_points['tx_packet'], color='red', s=30, label='Anomaly (TX)', zorder=5)
        ax1.scatter(anom_points['occur_date'], anom_points['rx_packet'], color='darkviolet', s=30, label='Anomaly (RX)', zorder=5)

        ax1.set_title(f"Traffic Pattern: {ip} (Port {int(cid)}-{int(lid)})")
        ax1.set_ylabel("Packets (Raw)")
        ax1.legend(loc='upper right', fontsize='small', ncol=2)
        ax1.grid(True, alpha=0.3)


        # --- 하단: 이상 점수 (Anomaly Score) ---
        ax2.plot(port_data['occur_date'], port_data['anomaly_score'], label='Anomaly Score', color='orange')
        
        # 임계치 추정선 (score가 threshold보다 크면 이상이므로, is_anomaly가 바뀌는 지점의 score 참고)
        # 실제 threshold는 모델마다 다를 수 있으므로 여기서는 시각적 참고용으로만 표시
        ax2.axhline(y=port_data['anomaly_score'].mean() * 3, color='gray', linestyle='--', label='Reference Line', alpha=0.5)
        
        ax2.set_ylabel("Anomaly Score")
        ax2.set_xlabel("Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"{output_dir}/anomaly_{ip}_{int(cid)}_{int(lid)}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"[*] Visualization saved: {save_path}")

if __name__ == "__main__":
    visualize_top_anomalies()
