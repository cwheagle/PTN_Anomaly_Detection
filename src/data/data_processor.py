import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from src.config import MODEL_CONFIG, PATHS

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.window_size = MODEL_CONFIG['window_size']
        self.feature_cols = [
            'tx_bps', 'rx_bps', 'tx_pps', 'rx_pps', 
            'tx_error', 'rx_error', 'tx_power', 'rx_power'
        ]

    def save_scaler(self, path=None):
        save_path = path or PATHS['scaler_save_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.scaler, save_path)
        print(f"Scaler saved to {save_path}")

    def load_scaler(self, path=None):
        load_path = path or PATHS['scaler_save_path']
        if os.path.exists(load_path):
            self.scaler = joblib.load(load_path)
            print(f"Scaler loaded from {load_path}")
            return True
        return False

    def preprocess(self, df):
        """데이터 정제 및 파생 변수 생성을 수행합니다."""
        if df is None or df.empty:
            return None

        # 1. 결측치 처리 (선형 보간 후 남은 결측치는 0으로 채움)
        df = df.sort_values(['equipment_id', 'port_id', 'collect_time'])
        df[self.feature_cols] = df.groupby(['equipment_id', 'port_id'])[self.feature_cols].transform(
            lambda x: x.interpolate().fillna(0)
        )

        # 2. 누적 데이터(Packet)의 증감량 계산 (필요시)
        # 이 예제에서는 BPS/PPS가 이미 있으므로 생략하거나 보조 지표로 활용 가능합니다.
        
        return df

    def scale_data(self, df, is_train=True):
        """데이터 스케일링을 수행합니다."""
        data = df[self.feature_cols].values
        
        if is_train:
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)
            
        return scaled_data

    def create_sequences(self, data):
        """LSTM 입력을 위한 시퀀스 데이터를 생성합니다. (Samples, Window_Size, Features)"""
        sequences = []
        for i in range(len(data) - self.window_size + 1):
            sequences.append(data[i : i + self.window_size])
        
        return np.array(sequences)

    def prepare_inference_data(self, df):
        """
        추론을 위해 데이터를 정제, 스케일링 및 그룹별 시퀀스화합니다.
        Returns:
            grouped_sequences: { (equip_id, port_id): sequence_tensor }
            df_cleaned: 정제된 전체 데이터프레임
        """
        df_clean = self.preprocess(df)
        if df_clean is None or df_clean.empty:
            return None, None
        
        # 스케일링 (전체 데이터에 대해 한꺼번에 수행)
        scaled_data = self.scale_data(df_clean, is_train=False)
        df_scaled = df_clean.copy()
        df_scaled[self.feature_cols] = scaled_data
        
        grouped_results = {}
        
        # 장비별, 포트별로 그룹화하여 시퀀스 생성
        for (equip_id, port_id), group in df_scaled.groupby(['equipment_id', 'port_id']):
            if len(group) < self.window_size:
                continue
            
            group_data = group[self.feature_cols].values
            sequences = self.create_sequences(group_data)
            grouped_results[(equip_id, port_id)] = sequences
            
        return grouped_results, df_clean
