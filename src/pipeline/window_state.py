import json
import redis
import logging
from typing import List, Dict, Optional
from src.config import REDIS_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

class WindowStateManager:
    """
    Kafka 스트림 처리 중 장비(Port)별 과거 데이터를 보관하는 Redis 기반 상태 관리자.
    분산 환경에서 여러 Consumer Pod가 떠 있더라도 상태를 외부에 유지하여 무결성을 보장합니다.
    """
    def __init__(self, host=None, port=None, db=None):
        self.host = host or REDIS_CONFIG.get("host", "localhost")
        self.port = port or REDIS_CONFIG.get("port", 6379)
        self.db = db or REDIS_CONFIG.get("db", 0)
        self.decode_responses = REDIS_CONFIG.get("decode_responses", True)
        
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=self.decode_responses
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise e

    def add_event(self, key: str, event_data: dict, max_size: int = None) -> None:
        """
        새로운 데이터를 Redis 리스트의 맨 앞(Left)에 밀어넣고, max_size만큼만 자릅니다 (Sliding Window).
        """
        if max_size is None:
            max_size = MODEL_CONFIG.get("window_size", 12)
            
        try:
            json_data = json.dumps(event_data)
            pipeline = self.redis_client.pipeline()
            # LPUSH: 리스트의 가장 앞(index 0)에 추가
            pipeline.lpush(key, json_data)
            # LTRIM: 0부터 max_size - 1까지만 남기고 삭제
            pipeline.ltrim(key, 0, max_size - 1)
            pipeline.execute()
        except Exception as e:
            logger.error(f"Error adding event to Redis for key {key}: {e}")
            raise e

    def get_window(self, key: str) -> List[Dict]:
        """
        해당 키의 전체 윈도우(과거~현재 데이터)를 시간순(과거->최신)으로 정렬하여 반환합니다.
        """
        try:
            # LRANGE: index 0(가장 최신)부터 -1(마지막)까지 조회
            raw_data = self.redis_client.lrange(key, 0, -1)
            if not raw_data:
                return []
            
            # 파싱 후, 시간순(과거 데이터가 먼저 오도록)으로 역순 정렬
            parsed_data = [json.loads(item) for item in raw_data]
            parsed_data.reverse()
            return parsed_data
        except Exception as e:
            logger.error(f"Error getting window from Redis for key {key}: {e}")
            return []

    def get_window_size(self, key: str) -> int:
        """
        현재 버퍼에 쌓인 데이터의 개수를 반환합니다.
        """
        try:
            return self.redis_client.llen(key)
        except Exception as e:
            logger.error(f"Error getting length for key {key}: {e}")
            return 0

    def clear_window(self, key: str) -> None:
        """
        해당 키의 데이터를 삭제합니다.
        """
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
