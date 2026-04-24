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

        # 1. 필수 칼럼 확인
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in input data: {missing_cols}")
            for col in missing_cols:
                df[col] = 0

        # 2. 시간 데이터 변환 및 정렬
        df['collect_time'] = pd.to_datetime(df['collect_time'])
        df = df.sort_values(['equipment_id', 'port_id', 'collect_time'])

        # 3. 비정상적인 값 처리 (Clipping)
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # 4. 결측치 처리 (선형 보간 후 남은 결측치는 0으로 채움)
        df[self.feature_cols] = df.groupby(['equipment_id', 'port_id'])[self.feature_cols].transform(
            lambda x: x.interpolate().fillna(0)
        )

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
        시간 간격이 너무 크면(30분 이상) 시퀀스를 분리합니다.
        """
        df_clean = self.preprocess(df)
        if df_clean is None or df_clean.empty:
            return None, None
        
        # 스케일링
        scaled_data = self.scale_data(df_clean, is_train=False)
        df_scaled = df_clean.copy()
        df_scaled[self.feature_cols] = scaled_data
        
        grouped_results = {}
        
        # 장비별, 포트별로 그룹화
        for (equip_id, port_id), group in df_scaled.groupby(['equipment_id', 'port_id']):
            if len(group) < self.window_size:
                continue
            
            # 시간 간격 체크: 이전 행과의 시간 차이가 30분 이상이면 데이터가 끊긴 것으로 간주
            time_diffs = group['collect_time'].diff().dt.total_seconds() / 60
            group_data = group[self.feature_cols].values
            
            sequences = []
            for i in range(len(group_data) - self.window_size + 1):
                # 해당 윈도우 내에 30분 이상 시간 단절이 있는지 확인
                window_diffs = time_diffs.iloc[i+1 : i + self.window_size]
                if (window_diffs > 30).any(): 
                    continue
                sequences.append(group_data[i : i + self.window_size])
            
            if sequences:
                grouped_results[(equip_id, port_id)] = np.array(sequences)
            
        return grouped_results, df_clean
