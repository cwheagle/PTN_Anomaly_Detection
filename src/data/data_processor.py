import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from src.config import MODEL_CONFIG, FEATURE_GROUPS

class DataProcessor:
    def __init__(self, feature_type='traffic'):
        self.window_size = MODEL_CONFIG['window_size']
        self.feature_type = feature_type
        self.feature_cols = FEATURE_GROUPS[feature_type]
        self.scaler = StandardScaler()

    def save_scaler(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        if os.path.exists(path):
            self.scaler = joblib.load(path)
            return True
        return False

    def _filter_low_quality(self, df, threshold=0.3):
        """결측치가 많은 그룹 제거"""
        if df is None or df.empty: return df
        valid_indices = []
        for _, group in df.groupby(['ip_addr', 'cid', 'lid']):
            if group[self.feature_cols].isna().mean().max() < threshold:
                valid_indices.extend(group.index)
        return df.loc[valid_indices].copy()

    def preprocess(self, df, is_train=True):
        """데이터 정제, 보간 및 필터링"""
        if df is None or df.empty: return None
        
        df = df.copy()
        df['occur_date'] = pd.to_datetime(df['occur_date'])
        df = df.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])

        # 숫자형 변환 (필수)
        for col in self.feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 학습 시에만 엄격한 품질 필터링 적용
        if is_train:
            df = self._filter_low_quality(df)
            if df.empty: return None

        # 클리핑 및 채우기
        if self.feature_type == 'traffic':
            df[self.feature_cols] = df[self.feature_cols].clip(lower=0).fillna(0)
        else:
            df[self.feature_cols] = df[self.feature_cols].clip(lower=-100, upper=20).fillna(-40)

        # 선형 보간 (그룹별)
        for _, group in df.groupby(['ip_addr', 'cid', 'lid']):
            if len(group) > 1:
                df.loc[group.index, self.feature_cols] = group[self.feature_cols].interpolate()
        
        return df

    def create_sequences(self, df, is_train=True):
        """시간 연속성이 보장된 윈도우 시퀀스 생성"""
        # 스케일링 적용
        data_vals = df[self.feature_cols].values
        scaled_vals = self.scaler.fit_transform(data_vals) if is_train else self.scaler.transform(data_vals)
        
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = scaled_vals
        
        sequences = []
        grouped_data = {} if not is_train else None

        for (ip, cid, lid), group in df_scaled.groupby(['ip_addr', 'cid', 'lid']):
            if len(group) < self.window_size: continue
            
            group_data = group[self.feature_cols].values
            time_diffs = group['occur_date'].diff().dt.total_seconds() / 60
            
            group_seqs = []
            for i in range(len(group_data) - self.window_size + 1):
                # 20분 이상 단절 시 해당 윈도우 폐기
                if not (time_diffs.iloc[i+1 : i+self.window_size] > 20).any():
                    group_seqs.append(group_data[i : i+self.window_size])
            
            if group_seqs:
                if is_train:
                    sequences.extend(group_seqs)
                else:
                    grouped_data[(ip, cid, lid)] = np.array(group_seqs)
        
        return np.array(sequences) if is_train else grouped_data
