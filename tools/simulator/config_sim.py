import os

# 시뮬레이터 구동용 DB 설정 (기본적으로 localhost에 cowptn_test 데이터베이스 사용 권장)
DB_CONFIG_SIM = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "root",
    "database": "cowptn_test",  # 테스트용 독립 스키마
    "port": 3306
}

# 시뮬레이터 시나리오 설정
SIM_CONFIG = {
    "nodes": 10,          # 가상 PTN 장비 개수
    "ports_per_node": 10, # 장비당 포트(CID, LID) 개수 -> 총 100개 링크
    "interval_minutes": 15,
    "history_days": 30,   # 과거 데이터 생성 일수
    
    # 이상치 발생 확률 (15분 단위 1건 생성 시 해당 확률로 이상치 부여)
    "anomaly_probabilities": {
        "spike": 0.05,       # 5% 개별 트랙 급격한 스파이크 이상
        "spike_both": 0.02,  # 2% 광파워/트래픽 동시 심각한 장애 (물리적 단선 등)
        "trend_start": 0.01  # 1% 확률로 점진적 열화(Trend) 시작
    },
    
    # 트래픽 기본 값 (평균)
    "traffic_base": {
        "tx_mean": 100000,
        "rx_mean": 100000,
        "noise_std": 5000,
        "error_mean": 0      # 정상 시 에러 패킷은 0
    },
    
    # 광파워 기본 값 (dBm)
    "optical_base": {
        "tx_mean": -4.0,
        "rx_mean": -5.0,
        "noise_std": 0.1
    }
}

SIGNAL_TYPES = {
    "ETH": 6,
    "OPTIC": 44
}
