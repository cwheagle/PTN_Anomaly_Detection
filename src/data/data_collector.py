import os
from src.data.db_connector import DBConnector

class DataCollector:
    """운영 환경에서의 배치 데이터 수집 엔진"""
    def __init__(self):
        self.db = DBConnector()

    def collect_and_save(self, train_start, train_end, test_start, test_end, output_dir="data"):
        """데이터를 수집하여 학습/테스트용 CSV로 저장 (Trainer 호출용)
        
        Args:
            train_start/end, test_start/end: 명시적 날짜 지정 방식 (YYYY-MM-DD)
        """
        # 1. 날짜 범위 산출 및 수집
        # 두 구간을 모두 포함하는 전체 범위를 가져온 후 개별 필터링
        fetch_start = min(train_start, test_start)
        fetch_end = max(train_end, test_end)
        
        # 검색 조건 (문자열 비교를 위해 시분초 포함 처리)
        t_start = f"{train_start} 00:00:00"
        t_end = f"{train_end} 23:59:59"
        v_start = f"{test_start} 00:00:00"
        v_end = f"{test_end} 23:59:59"
        
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # 1. 트래픽 수집 및 독립 필터링
        df_t = self.db.fetch_traffic(fetch_start, fetch_end)
        if df_t is not None and not df_t.empty:
            df_t = df_t.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
            # 문자열 비교 또는 Timestamp 비교를 통한 독립 구간 추출
            train = df_t[(df_t['occur_date'] >= t_start) & (df_t['occur_date'] <= t_end)]
            test = df_t[(df_t['occur_date'] >= v_start) & (df_t['occur_date'] <= v_end)]
            
            train.to_csv(os.path.join(output_dir, "traffic_train.csv"), index=False)
            test.to_csv(os.path.join(output_dir, "traffic_test.csv"), index=False)
            results['traffic'] = {'train': len(train), 'test': len(test)}
            print(f"[*] Traffic collected: Train({len(train)}), Test({len(test)})")

        # 2. 광파워 수집 및 독립 필터링
        df_o = self.db.fetch_optical(fetch_start, fetch_end)
        if df_o is not None and not df_o.empty:
            df_o = df_o.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
            train = df_o[(df_o['occur_date'] >= t_start) & (df_o['occur_date'] <= t_end)]
            test = df_o[(df_o['occur_date'] >= v_start) & (df_o['occur_date'] <= v_end)]
            
            train.to_csv(os.path.join(output_dir, "optical_train.csv"), index=False)
            test.to_csv(os.path.join(output_dir, "optical_test.csv"), index=False)
            results['optical'] = {'train': len(train), 'test': len(test)}
            print(f"[*] Optical collected: Train({len(train)}), Test({len(test)})")
            
        return results
