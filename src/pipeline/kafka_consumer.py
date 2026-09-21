import os
import sys

# 프로젝트 루트 디렉토리를 path에 추가
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

import json
import logging
import pandas as pd
import requests
import redis
import numpy as np
from confluent_kafka import Consumer, KafkaError
from src.pipeline.window_state import WindowStateManager
from src.pipeline.inference import AnomalyDetector
from src.data.db_connector import DBConnector
from src.config import KAFKA_CONFIG, MODEL_CONFIG, RETENTION_DAYS, REDIS_CONFIG, API_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class PTNKafkaConsumer:
    """
    Kafka 토픽을 구독하여 실시간 스트림 데이터를 처리하는 추론 엔진 워커 (Consumer).
    분산 환경에서 여러 Pod가 실행되면 Kafka의 Consumer Group 기능으로 자동 부하 분산(Load Balancing)됨.
    """
    def __init__(self):
        self.db = DBConnector()
        self.detector = AnomalyDetector()
        self.window_manager = WindowStateManager()
        
        # 모델의 window_size(기본 12) + 추세 분석용 과거 4시점 = 총 16시점 필요
        self.required_size = MODEL_CONFIG.get('window_size', 12) + 4

        conf = {
            'bootstrap.servers': KAFKA_CONFIG['bootstrap_servers'],
            'group.id': KAFKA_CONFIG['consumer_group_id'],
            'auto.offset.reset': KAFKA_CONFIG['auto_offset_reset'],
            # K8s 등 컨테이너 환경을 위해 session timeout 등을 튜닝할 수 있음
        }
        self.consumer = Consumer(conf)
        self.topic = KAFKA_CONFIG['topic_metrics']
        self.consumer.subscribe([self.topic])

        # Redis Pub/Sub for Model Hot-Reload
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(**{'ptn_control': self.handle_control_message})
        self.pubsub_thread = self.pubsub.run_in_thread(sleep_time=1.0)

    def handle_control_message(self, message):
        """ Redis Pub/Sub 제어 채널 수신 (Hot-Reload) """
        if message['type'] == 'message':
            try:
                data = json.loads(message['data'])
                if data.get('action') == 'reload':
                    track = data.get('track')
                    logger.info(f"[Control] Received reload request for {track} model")
                    self.detector.reload_model(track)
            except Exception as e:
                logger.error(f"[Control] Failed to parse control message: {e}")

    def process_message(self, key_str: str, record: dict):
        """
        단일 메시지가 들어왔을 때 Redis에 저장하고, 조건이 충족되면 추론을 수행합니다.
        """
        # 1. Redis Window 버퍼에 데이터 추가 (최대 16개 유지)
        self.window_manager.add_event(key_str, record, max_size=self.required_size)
        
        # 2. 현재 버퍼 크기 확인
        current_size = self.window_manager.get_window_size(key_str)
        if current_size < self.required_size:
            # 너무 많은 로그가 찍히지 않도록 일부 포트에 대해서만 진행 상황 표시
            if current_size == 1 and key_str.endswith("0:1"):
                logger.info(f"[{key_str}] Buffering data... ({current_size}/{self.required_size})")
            return

        # 3. 버퍼에서 전체 윈도우 추출 (과거 -> 현재 정렬됨)
        window_data = self.window_manager.get_window(key_str)
        
        # 4. DataFrame으로 변환
        df = pd.DataFrame(window_data)
        if 'occur_date' in df.columns:
            df['occur_date'] = pd.to_datetime(df['occur_date'])

        # 5. AnomalyDetector 실행 (최신 시점 결과 1건만 반환)
        # Producer에서 Traffic/Optical을 병합해서 보냈으므로 동일한 df를 양쪽에 넣음
        results = self.detector.detect(df_traffic=df, df_optical=df, latest_only=True)
        
        # 6. 추론 결과 DB 저장
        if results is not None and not results.empty:
            is_anomaly = results.iloc[0].get('is_anomaly', False)
            alarm = results.iloc[0].get('alarm_label', 'NORMAL')
            
            logger.info(f"[{key_str}] Processed! Anomaly: {is_anomaly}, Alarm: {alarm}")
            
            if is_anomaly or alarm != "NORMAL":
                logger.info(f"[{key_str}] Anomaly Detected! Alarm: {alarm}")
                # 웹훅을 통해 API 서버에 SSE 발송 요청
                try:
                    payload = results.replace({np.nan: None}).to_dict(orient="records")
                    payload_json = json.dumps(payload, default=str)
                    requests.post(
                        f"{API_URL}/api/internal/alarm", 
                        data=payload_json, 
                        headers={'Content-Type': 'application/json'}, 
                        timeout=2
                    )
                except Exception as e:
                    logger.error(f"Failed to send SSE webhook: {e}")
            
            try:
                self.db.save_results(results)
            except Exception as e:
                logger.error(f"DB Save Error: {e}")
        else:
            if current_size == 16 and key_str.endswith("0:1"):
                logger.info(f"[{key_str}] Debug: DataFrame shape before detect={df.shape}, head=\n{df[['occur_date', 'tx_packet', 'rx_avg_power']].head(3)}")
            logger.info(f"[{key_str}] Results empty or None")

    def start(self):
        """ 무한 루프로 Kafka 토픽 폴링 """
        logger.info(f"Starting PTN Consumer for topic: {self.topic}")
        logger.info(f"Required Window Size: {self.required_size}")
        
        try:
            # DB 오래된 데이터 정리 (일 1회 실행하면 좋으나, 여기선 스트림 시작 시 1회 정리)
            self.db.delete_old_data(RETENTION_DAYS)

            while True:
                # 1초 타임아웃으로 메시지 폴링
                msg = self.consumer.poll(1.0)
                
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # 끝까지 읽음 (일반적으로 에러 아님)
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        break

                # 메시지 처리
                try:
                    key = msg.key().decode('utf-8') if msg.key() else "unknown"
                    val_str = msg.value().decode('utf-8')
                    record = json.loads(val_str)
                    
                    self.process_message(key, record)
                    
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON from message")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user.")
        finally:
            self.consumer.close()
            if self.pubsub_thread:
                self.pubsub_thread.stop()
            logger.info("Consumer closed.")

if __name__ == "__main__":
    consumer = PTNKafkaConsumer()
    consumer.start()
