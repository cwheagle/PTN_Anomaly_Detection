import os

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
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
    "threshold_percentile": 99.5  # 조금 더 보수적으로 상향 (상위 0.5%)
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
