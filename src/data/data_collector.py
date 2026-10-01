import os
import pandas as pd
from src.data.db_connector import DBConnector

class DataCollector:
    """운영 환경에서의 배치 데이터 수집 엔진"""
    def __init__(self):
        self.db = DBConnector()

    @staticmethod
    def load_exclusions(path):
        """학습에서 제외할 구간 목록(CSV)을 로드. path가 없으면 None (제외 없음).

        CSV 컬럼: ip_addr, cid, lid, start_time, failure_time
        운영 환경에는 정답지가 없으므로 기본은 제외 없음이며, 시뮬레이션 등 명시적으로
        제외 구간을 알고 있을 때만 주입한다. (평가용 정답지를 학습에 암묵적으로 쓰지 않기 위함)
        """
        if not path:
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"Exclusion file not found: {path}")
        gt_df = pd.read_csv(path)
        gt_df['start_time'] = pd.to_datetime(gt_df['start_time'])
        gt_df['failure_time'] = pd.to_datetime(gt_df['failure_time'])
        return gt_df

    @staticmethod
    def filter_excluded(df, exclusions):
        """exclusions의 (포트, start_time ~ failure_time) 구간에 해당하는 행을 제거"""
        if df is None or df.empty or exclusions is None or exclusions.empty:
            return df
        df = df.copy()
        df['occur_date'] = pd.to_datetime(df['occur_date'])
        drop_mask = pd.Series(False, index=df.index)
        for _, ex in exclusions.iterrows():
            drop_mask |= ((df['ip_addr'] == ex['ip_addr']) &
                          (df['cid'] == ex['cid']) &
                          (df['lid'] == ex['lid']) &
                          (df['occur_date'] >= ex['start_time']) &
                          (df['occur_date'] <= ex['failure_time']))
        print(f"[*] DataCollector: Dropped {int(drop_mask.sum())} excluded records.")
        return df[~drop_mask].copy()

    def collect_and_save(self, train_start, train_end, test_start, test_end, feature_type=None, output_dir="data", stop_checker=None, exclude_path=None):
        """데이터를 수집하여 학습/테스트용 CSV로 저장 (Trainer 호출용)
        
        Args:
            train_start/end, test_start/end: 명시적 날짜 지정 방식 (YYYY-MM-DD)
            feature_type: 'traffic' 또는 'optical'. None이면 둘 다 수집.
            stop_checker: 중지 요청 여부를 확인할 콜백 함수
            exclude_path: 학습 제외 구간 CSV 경로. None이면 환경변수 TRAIN_EXCLUDE_CSV, 둘 다 없으면 제외 없음.
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

        # 학습 제외 구간 (명시적으로 지정된 경우에만 적용)
        exclusions = self.load_exclusions(exclude_path or os.getenv("TRAIN_EXCLUDE_CSV"))
        if exclusions is not None:
            print(f"[*] DataCollector: {len(exclusions)} exclusion intervals will be removed from training data.")

        def filter_anomalies(df):
            return self.filter_excluded(df, exclusions)

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
