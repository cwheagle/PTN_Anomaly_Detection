import os

# Database Configuration
DB_CONFIG = {
    "host": "192.168.99.253",
    "user": "root",
    "password": "root",
    "database": "cowptn",
    "port": 3306
}

# Feature Groups
FEATURE_GROUPS = {
    "traffic": ['tx_packet', 'rx_packet', 'error_packet'],
    "optical": ['tx_avg_power', 'rx_avg_power']
}

# Model Configuration (공통 아키텍처 설정)
MODEL_CONFIG = {
    "hidden_dim": 32,
    "latent_dim": 16,
    "num_layers": 2,
    "window_size": 12,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "threshold_percentile": 99.9  # 극도로 보수적으로 상향 (상위 0.1%)
}

# Severity Configuration (Phase 6: 다단계 경보 체계)
SEVERITY_CONFIG = {
    "NORMAL": {"min": 0, "max": 50, "level": 0, "label": "NORMAL"},
    "MINOR": {"min": 50, "max": 70, "level": 1, "label": "MINOR"},
    "MAJOR": {"min": 70, "max": 90, "level": 2, "label": "MAJOR"},
    "CRITICAL": {"min": 90, "max": 100, "level": 3, "label": "CRITICAL"},
    "slope_threshold": 1.0,  # 추세 분석(RISING/FALLING) 판단 기준 수치 (15분당 1.0점 상승)
    "rul_target": 90.0       # RUL 예측 목표 심각도 (Critical 임계치)
}

# Signal Types
SIGNAL_TYPES = {
    "ETH": 6,
    "OPTIC": 44
}

# Path Configuration (앙상블 모델용 분리)
PATHS = {
    "traffic": {
        "model": "models/traffic_ae.pth",
        "scaler": "models/traffic_scaler.joblib",
        "threshold": "models/traffic_threshold.json"
    },
    "optical": {
        "model": "models/optical_ae.pth",
        "scaler": "models/optical_scaler.joblib",
        "threshold": "models/optical_threshold.json"
    },
    "log_path": "logs/anomaly_detection.log"
}

# Scheduler Configuration
INTERVAL_MINUTES = 15
RETENTION_DAYS = 30
