from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import pandas as pd
from src.data.db_connector import DBConnector
from src.pipeline.inference import AnomalyDetector
from src.config import INTERVAL_MINUTES

class PTNAnomalyScheduler:
    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.detector = AnomalyDetector()
        self.db_connector = DBConnector()

    def run_inference_job(self):
        """15분마다 실행될 핵심 추론 작업입니다."""
        from datetime import datetime, timedelta
        
        now_dt = datetime.now()
        now_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[{now_str}] Starting anomaly detection job...")
        
        try:
            # 1. 데이터 가져오기 (충분한 시퀀스 확보를 위해 최근 3시간 데이터를 가져옴)
            start_time = (now_dt - timedelta(minutes=180)).strftime('%Y-%m-%d %H:%M:%S')
            end_time = now_str
            df = self.db_connector.fetch_real_data(start_time, end_time) 
            
            if df is None or df.empty:
                print("No data fetched from DB. Skipping this cycle.")
                return

            # 2. 이상 탐지 수행
            results = self.detector.detect(df)
            
            if results is not None:
                anomalies = results[results['is_anomaly'] == True]
                print(f"Processed {len(results)} records. Detected {len(anomalies)} anomalies.")
                
                # 3. 결과 DB 저장
                self.db_connector.save_anomaly_results(results)
                
                if not anomalies.empty:
                    print("--- ANOMALY DETECTED ---")
                    cols = ['occur_date', 'ip_addr', 'cid', 'lid', 'anomaly_score']
                    print(anomalies[cols].head())
            else:
                print("Not enough data to perform inference (check window size).")
                
        except Exception as e:
            print(f"Error during scheduled job: {e}")

    def start(self):
        """스케줄러를 시작합니다."""
        # 시작하자마자 한 번 실행
        self.run_inference_job()
        
        # 주기적 실행 등록
        self.scheduler.add_job(
            self.run_inference_job, 
            'interval', 
            minutes=INTERVAL_MINUTES
        )
        
        print(f"Scheduler started. Running every {INTERVAL_MINUTES} minutes.")
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            print("Scheduler stopped.")
