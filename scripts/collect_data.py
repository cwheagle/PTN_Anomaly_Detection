import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.db_connector import DBConnector

def collect_real_data(train_days=30, test_days=7):
    """
    운영 DB에서 실제 데이터를 수집하여 학습/테스트용 CSV를 생성합니다.
    """
    print("="*50)
    print("PTN REAL DATA COLLECTION UTILITY")
    print("="*50)
    
    connector = DBConnector()
    now_dt = datetime.now()
    
    # 기간 설정
    test_start_dt = now_dt - timedelta(days=test_days)
    train_start_dt = test_start_dt - timedelta(days=train_days)
    
    os.makedirs("data", exist_ok=True)
    
    # 1. 학습 데이터 수집
    print(f"[*] Step 1: Fetching training data ({train_days} days)...")
    print(f"    Range: {train_start_dt.strftime('%Y-%m-%d')} ~ {test_start_dt.strftime('%Y-%m-%d')}")
    
    df_train = connector.fetch_real_data(
        train_start_dt.strftime('%Y-%m-%d %H:%M:%S'),
        test_start_dt.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    if df_train is not None and not df_train.empty:
        train_path = "data/train_data.csv"
        df_train.to_csv(train_path, index=False)
        print(f"    [SUCCESS] Saved to {train_path} ({len(df_train)} rows)")
    else:
        print("    [FAILURE] No training data found.")

    # 2. 테스트 데이터 수집
    print(f"\n[*] Step 2: Fetching test data ({test_days} days)...")
    print(f"    Range: {test_start_dt.strftime('%Y-%m-%d')} ~ {now_dt.strftime('%Y-%m-%d')}")
    
    df_test = connector.fetch_real_data(
        test_start_dt.strftime('%Y-%m-%d %H:%M:%S'),
        now_dt.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    if df_test is not None and not df_test.empty:
        test_path = "data/test_data.csv"
        df_test.to_csv(test_path, index=False)
        print(f"    [SUCCESS] Saved to {test_path} ({len(df_test)} rows)")
    else:
        print("    [FAILURE] No test data found.")
    
    print("\n" + "="*50)
    print("COLLECTION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    # 인자가 있으면 기간을 조절할 수 있게 확장 가능
    collect_real_data()
