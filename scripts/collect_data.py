import os
import sys
from datetime import datetime, timedelta

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.db_connector import DBConnector

def run_collection(days=37):
    """트래픽과 광파워 데이터를 각각 독립적으로 수집하여 CSV 저장"""
    print("="*60)
    print("PTN DATA COLLECTION SYSTEM")
    print("="*60)
    
    db = DBConnector()
    now_dt = datetime.now()
    start_dt = now_dt - timedelta(days=days)
    split_dt = now_dt - timedelta(days=7)
    
    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')
    split_str = split_dt.strftime('%Y-%m-%d')
    
    os.makedirs("data", exist_ok=True)
    
    # 1. 트래픽 수집
    print(f"[*] Collecting Traffic data...")
    df_t = db.fetch_traffic(start_str, end_str)
    if df_t is not None:
        # 가독성을 위한 정렬 추가
        df_t = df_t.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
        
        train = df_t[df_t['occur_date'] < split_str]
        test = df_t[df_t['occur_date'] >= split_str]
        train.to_csv("data/traffic_train.csv", index=False)
        test.to_csv("data/traffic_test.csv", index=False)
        print(f"    [OK] Traffic: {len(train)} train / {len(test)} test")

    # 2. 광파워 수집
    print(f"\n[*] Collecting Optical data...")
    df_o = db.fetch_optical(start_str, end_str)
    if df_o is not None:
        # 가독성을 위한 정렬 추가
        df_o = df_o.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
        
        train = df_o[df_o['occur_date'] < split_str]
        test = df_o[df_o['occur_date'] >= split_str]
        train.to_csv("data/optical_train.csv", index=False)
        test.to_csv("data/optical_test.csv", index=False)
        print(f"    [OK] Optical: {len(train)} train / {len(test)} test")
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_collection()
