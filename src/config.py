import os

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "your_username",
    "password": "your_password",
    "database": "ptn_ems_db",
    "port": 3306
}

# Model Configuration
MODEL_CONFIG = {
    "input_dim": 8,       # BPS(tx/rx), PPS(tx/rx), Error(tx/rx), Power(tx/rx) 등
    "hidden_dim": 32,
    "latent_dim": 16,
    "num_layers": 2,
    "window_size": 12,    # 15분 단위 기준, 3시간 분량의 시퀀스 (12 * 15분)
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "threshold_percentile": 99  # 이상 탐지 임계치 (상위 1% 오차)
}

# Path Configuration
PATHS = {
    "model_save_path": "models/lstm_ae_v1.pth",
    "scaler_save_path": "models/scaler.joblib",
    "threshold_path": "models/threshold.json",
    "log_path": "logs/anomaly_detection.log"
}

# Scheduler Configuration
INTERVAL_MINUTES = 15
