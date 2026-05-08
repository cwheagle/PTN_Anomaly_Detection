import os
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
from src.data.db_connector import DBConnector
from src.pipeline.inference import AnomalyDetector
from src.config import INTERVAL_MINUTES, RETENTION_DAYS

class PTNAnomalyScheduler:
    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.detector = AnomalyDetector()
        self.db = DBConnector()

    def run_job(self):
        """15분 주기 독립 트랙 기반 통합 탐지 작업"""
        now_dt = datetime.now()
        now_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')
        start_str = (now_dt - timedelta(minutes=180)).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n[{now_str}] Starting anomaly detection cycle...")
        
        try:
            # 1. 독립 트랙별 데이터 수집
            df_t = self.db.fetch_traffic(start_str, now_str)
            df_o = self.db.fetch_optical(start_str, now_str)
            
            # 2. 앙상블 탐지 실행
            results = self.detector.detect(df_traffic=df_t, df_optical=df_o)
            
            if results is not None:
                anomalies = results[results['is_anomaly']]
                print(f"    - Processed items: {len(results)}")
                print(f"    - Anomalies found: {len(anomalies)}")
                
                # 3. DB 저장
                self.db.save_results(results)
                self.db.delete_old_data(RETENTION_DAYS)

                # 4. 하이브리드 로그 (CSV 파일 누적)
                csv_path = "data/live_anomalies.csv"
                file_exists = os.path.exists(csv_path)
                # 전체 결과가 아닌 이상치만 CSV에 누적 (운영자 확인용)
                if not anomalies.empty:
                    anomalies.to_csv(csv_path, mode='a', index=False, header=not file_exists)
                    print(f"    - Anomalies appended to: {csv_path}")
            else:
                print("    - No valid data to analyze in this cycle.")
                
        except Exception as e:
            print(f"    [ERROR] Job failed: {e}")

    def start(self):
        """스케줄러 가동"""
        print(f"[*] PTN Scheduler initialized. Interval: {INTERVAL_MINUTES}m")
        # 즉시 실행 후 예약
        self.run_job()
        self.scheduler.add_job(self.run_job, 'interval', minutes=INTERVAL_MINUTES)
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            print("[*] Scheduler stopped.")
