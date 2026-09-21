import os
import sys

# 프로젝트 루트 디렉토리를 path에 추가
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

import json
import time
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
from confluent_kafka import Producer
from apscheduler.schedulers.background import BlockingScheduler

from src.data.db_connector import DBConnector
from src.config import KAFKA_CONFIG, INTERVAL_MINUTES

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class PTNKafkaProducer:
    """
    DB에서 최신 15분 데이터를 폴링하여 Kafka로 전송하는 스트리밍 시뮬레이터 / 수집기
    """
    def __init__(self):
        self.db = DBConnector()
        # Kafka Producer 초기화
        conf = {
            'bootstrap.servers': KAFKA_CONFIG['bootstrap_servers'],
            'client.id': 'ptn_producer'
        }
        self.producer = Producer(conf)
        self.topic = KAFKA_CONFIG['topic_metrics']
        self.scheduler = BlockingScheduler()

    def delivery_report(self, err, msg):
        """ Kafka 메시지 전송 콜백 """
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            pass # 성공 로그는 너무 많아질 수 있으므로 생략

    def fetch_and_produce(self, start_time: str, end_time: str):
        """ DB에서 데이터를 조회하여 Kafka로 전송 """
        logger.info(f"Fetching data from {start_time} to {end_time}")
        
        df_t = self.db.fetch_traffic(start_time, end_time)
        df_o = self.db.fetch_optical(start_time, end_time)
        
        # Traffic과 Optical 병합 (Inference 모듈과 동일한 로직)
        if df_t is None and df_o is None:
            logger.warning("No data found for the given time range.")
            return

        df = pd.DataFrame()
        if df_t is not None and df_o is not None:
            df = pd.merge(df_t, df_o, on=['occur_date', 'ip_addr', 'cid', 'lid'], how='outer')
        elif df_t is not None:
            df = df_t
        elif df_o is not None:
            df = df_o

        if df.empty:
            logger.warning("Merged dataframe is empty.")
            return
            
        # Nan 처리
        df.fillna(0, inplace=True)

        count = 0
        for _, row in df.iterrows():
            # 날짜형은 문자열로 변환
            record = row.to_dict()
            if 'occur_date' in record:
                record['occur_date'] = str(record['occur_date'])
            
            # Key 생성 (예: 192.168.1.1:1:2)
            ip = record.get('ip_addr', 'unknown')
            cid = record.get('cid', 0)
            lid = record.get('lid', 0)
            key = f"{ip}:{cid}:{lid}"
            
            # Kafka Produce
            self.producer.produce(
                self.topic,
                key=key.encode('utf-8'),
                value=json.dumps(record).encode('utf-8'),
                callback=self.delivery_report
            )
            count += 1
            
        self.producer.flush()
        logger.info(f"Produced {count} messages to topic '{self.topic}'.")

    def run_job(self):
        """ 스케줄러에 의해 주기적으로 실행될 작업 """
        now_dt = datetime.now()
        # 정확히 최근 15분치(Interval) 데이터만 가져옴
        start_str = (now_dt - timedelta(minutes=INTERVAL_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
        end_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')
        self.fetch_and_produce(start_str, end_str)

    def start_scheduler(self):
        """ 실시간 모드 (15분 간격 스케줄링) """
        logger.info(f"Starting Producer Scheduler (Interval: {INTERVAL_MINUTES}m)")
        self.scheduler.add_job(self.run_job, 'interval', minutes=INTERVAL_MINUTES, next_run_time=datetime.now())
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Producer Scheduler stopped.")

    def run_simulation(self, start_time: str, end_time: str, step_minutes: int = 15):
        """ 과거 데이터를 순차적으로 스트리밍하는 시뮬레이션 모드 (E2E 테스트용) """
        logger.info(f"Starting Historical Simulation from {start_time} to {end_time}")
        current = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        
        while current < end:
            next_time = current + timedelta(minutes=step_minutes)
            self.fetch_and_produce(
                current.strftime('%Y-%m-%d %H:%M:%S'),
                (next_time - timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
            )
            current = next_time
            time.sleep(1) # 시뮬레이션 간격 조정 (너무 빠르면 Consumer가 밀림)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTN Kafka Producer")
    parser.add_argument('--mode', type=str, default='live', choices=['live', 'sim'], help="Run mode: live or sim")
    parser.add_argument('--start', type=str, help="Simulation start time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument('--end', type=str, help="Simulation end time (YYYY-MM-DD HH:MM:SS)")
    
    args = parser.parse_args()
    producer = PTNKafkaProducer()
    
    if args.mode == 'sim':
        if not args.start or not args.end:
            logger.error("--start and --end are required for simulation mode")
        else:
            producer.run_simulation(args.start, args.end)
    else:
        producer.start_scheduler()
