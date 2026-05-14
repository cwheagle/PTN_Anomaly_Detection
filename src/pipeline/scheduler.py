import os
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timedelta
from src.data.db_connector import DBConnector
from src.pipeline.inference import AnomalyDetector
from src.config import INTERVAL_MINUTES, RETENTION_DAYS, MODEL_CONFIG

class PTNAnomalyScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.detector = AnomalyDetector()
        self.db = DBConnector()
        self.callback = None

    def set_callback(self, callback_func):
        """추론 결과 처리를 위한 콜백 함수 등록"""
        self.callback = callback_func

    async def run_job(self):
        """15분 주기 독립 트랙 기반 통합 탐지 작업"""
        now_dt = datetime.now()
        now_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 추세 분석을 위해 (window_size + 4개 시점) 만큼 데이터 조회
        # 12(window) + 4(trend) = 16시점 * 15분 = 240분
        fetch_minutes = (MODEL_CONFIG['window_size'] + 4) * INTERVAL_MINUTES
        start_str = (now_dt - timedelta(minutes=fetch_minutes)).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n[{now_str}] Starting anomaly detection cycle (Fetch: {fetch_minutes}m)...")
        
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

                # 4. 콜백 실행 (SSE 알림 등 외부 연동) - 비동기 대응
                if self.callback:
                    if asyncio.iscoroutinefunction(self.callback):
                        await self.callback(results)
                    else:
                        self.callback(results)

                # 5. 하이브리드 로그 (전체 결과 CSV 누적 - 일자별 분리)
                date_str = now_dt.strftime('%Y%m%d')
                csv_path = f"data/history_{date_str}.csv"
                file_exists = os.path.exists(csv_path)
                
                # 전체 결과를 CSV에 누적
                results.to_csv(csv_path, mode='a', index=False, header=not file_exists)
                print(f"    - Full results appended to: {csv_path}")
            else:
                print("    - No valid data to analyze in this cycle.")
                
        except Exception as e:
            print(f"    [ERROR] Job failed: {e}")

    def start(self):
        """스케줄러 가동"""
        print(f"[*] PTN Scheduler initialized. Interval: {INTERVAL_MINUTES}m")
        # 비동기 루프 내에서 작업 추가
        self.scheduler.add_job(self.run_job, 'interval', minutes=INTERVAL_MINUTES, next_run_time=datetime.now())
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            print("[*] Scheduler stopped.")

    def restart(self):
        """스케줄러 재가동"""
        self.scheduler.remove_all_jobs()
        self.scheduler.add_job(self.run_job, 'interval', minutes=INTERVAL_MINUTES, next_run_time=datetime.now())
