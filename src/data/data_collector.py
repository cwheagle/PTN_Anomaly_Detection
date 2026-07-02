import os
from src.data.db_connector import DBConnector

class DataCollector:
    """운영 환경에서의 배치 데이터 수집 엔진"""
    def __init__(self):
        self.db = DBConnector()

    def collect_and_save(self, train_start, train_end, test_start, test_end, feature_type=None, output_dir="data", stop_checker=None):
        """데이터를 수집하여 학습/테스트용 CSV로 저장 (Trainer 호출용)
        
        Args:
            train_start/end, test_start/end: 명시적 날짜 지정 방식 (YYYY-MM-DD)
            feature_type: 'traffic' 또는 'optical'. None이면 둘 다 수집.
            stop_checker: 중지 요청 여부를 확인할 콜백 함수
        """
        # 1. 날짜 범위 산출 및 수집
        fetch_start = min(train_start, test_start)
        fetch_end = max(train_end, test_end)
        
        t_start = f"{train_start} 00:00:00"
        t_end = f"{train_end} 23:59:59"
        v_start = f"{test_start} 00:00:00"
        v_end = f"{test_end} 23:59:59"
        
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # 정답지 로드하여 오염 데이터 필터링 준비
        gt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tools', 'simulator', 'data', 'eval_dataset.csv')
        gt_df = None
        if os.path.exists(gt_path):
            import pandas as pd
            gt_df = pd.read_csv(gt_path)
            gt_df['start_time'] = pd.to_datetime(gt_df['start_time'])
            gt_df['failure_time'] = pd.to_datetime(gt_df['failure_time'])
            print(f"[*] DataCollector: Found {len(gt_df)} anomaly scenarios. Will filter them out from training data.")

        def filter_anomalies(df):
            if df is None or df.empty or gt_df is None: return df
            df['occur_date'] = pd.to_datetime(df['occur_date'])
            drop_mask = pd.Series(False, index=df.index)
            print("[*] DataCollector: Filtering anomalies (this might take a few seconds)...")
            for _, gt in gt_df.iterrows():
                mask = (df['ip_addr'] == gt['ip_addr']) & \
                       (df['cid'] == gt['cid']) & \
                       (df['lid'] == gt['lid']) & \
                       (df['occur_date'] >= gt['start_time']) & \
                       (df['occur_date'] <= gt['failure_time'])
                drop_mask = drop_mask | mask
            dropped_count = drop_mask.sum()
            print(f"[*] DataCollector: Dropped {dropped_count} contaminated records.")
            return df[~drop_mask].copy()

        # 1. 트래픽 수집 및 독립 필터링
        if feature_type is None or feature_type == 'traffic':
            df_t = self.db.fetch_traffic(fetch_start, fetch_end, stop_checker=stop_checker)
            if stop_checker and stop_checker(): return results # 중지 시 조기 리턴

            if df_t is not None and not df_t.empty:
                df_t = filter_anomalies(df_t)
                df_t = df_t.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
                train = df_t[(df_t['occur_date'] >= t_start) & (df_t['occur_date'] <= t_end)]
                test = df_t[(df_t['occur_date'] >= v_start) & (df_t['occur_date'] <= v_end)]
                
                train.to_csv(os.path.join(output_dir, "traffic_train.csv"), index=False)
                test.to_csv(os.path.join(output_dir, "traffic_test.csv"), index=False)
                results['traffic'] = {'train': len(train), 'test': len(test)}
                print(f"[*] Traffic collected: Train({len(train)}), Test({len(test)})")

        # 2. 광파워 수집 및 독립 필터링
        if feature_type is None or feature_type == 'optical':
            df_o = self.db.fetch_optical(fetch_start, fetch_end, stop_checker=stop_checker)
            if stop_checker and stop_checker(): return results # 중지 시 조기 리턴

            if df_o is not None and not df_o.empty:
                df_o = filter_anomalies(df_o)
                df_o = df_o.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
                train = df_o[(df_o['occur_date'] >= t_start) & (df_o['occur_date'] <= t_end)]
                test = df_o[(df_o['occur_date'] >= v_start) & (df_o['occur_date'] <= v_end)]
                
                train.to_csv(os.path.join(output_dir, "optical_train.csv"), index=False)
                test.to_csv(os.path.join(output_dir, "optical_test.csv"), index=False)
                results['optical'] = {'train': len(train), 'test': len(test)}
                print(f"[*] Optical collected: Train({len(train)}), Test({len(test)})")
            
        return results
