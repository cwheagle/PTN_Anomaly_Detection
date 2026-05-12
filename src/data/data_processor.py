import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
import os
from src.config import MODEL_CONFIG, FEATURE_GROUPS

class DataProcessor:
    def __init__(self, feature_type='traffic', config=None):
        # 1. 설정값 우선순위: config 파라미터 > MODEL_CONFIG 기본값
        active_config = config if config else MODEL_CONFIG
        self.window_size = active_config.get('window_size', MODEL_CONFIG['window_size'])
        
        self.feature_type = feature_type
        self.feature_cols = FEATURE_GROUPS[feature_type]
        # 이상치에 강한 RobustScaler 사용
        self.scaler = RobustScaler()

    def save_scaler(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        """저장된 스케일러 로드"""
        if os.path.exists(path):
            try:
                self.scaler = joblib.load(path)
                return True
            except Exception as e:
                print(f" [!] Error loading scaler: {e}")
                return False
        return False

    def _clean_and_transform(self, df):
        """특성별 맞춤형 전처리 (Traffic: Log, Optical: Linear)"""
        if df is None or df.empty: return df
        df = df.copy()
        
        if self.feature_type == 'traffic':
            for col in self.feature_cols:
                # 음수 방지 및 로그 변환 (10^19 등 거대 수치는 학습 시 필터링됨)
                df[col] = np.log1p(df[col].clip(lower=0))
        # Optical은 이미 dBm이므로 별도 변환 없이 그대로 반환
        return df

    def _filter_low_quality(self, df):
        """결측치 및 무의미하거나 오염된 포트 데이터 필터링 (원본 수치 기준)"""
        if df is None or df.empty: return df
        valid_indices = []
        for _, group in df.groupby(['ip_addr', 'cid', 'lid']):
            # 1. 최소 데이터 개수 체크
            if len(group.dropna(subset=self.feature_cols)) < self.window_size * 2:
                continue

            # 2. 광파워 죽은 채널 체크 (-40 고정)
            if self.feature_type == 'optical':
                if (group[self.feature_cols] <= -39.9).all().all():
                    continue

            # 3. 트래픽 오염 데이터 체크 (원본 수치 기준)
            if self.feature_type == 'traffic':
                # 에러 패킷 1,000개 초과 또는 트래픽 10억(1e9) 초과 시 포트 폐기
                if (group['error_packet'] > 1000).any() or \
                   (group[['tx_packet', 'rx_packet']] > 1e9).any().any():
                    continue

            valid_indices.extend(group.index)
        return df.loc[valid_indices].copy()

    def preprocess(self, df, is_train=True):
        """데이터 정제, 보간 및 필터링"""
        if df is None or df.empty: return None

        df = df.copy()
        df['occur_date'] = pd.to_datetime(df['occur_date'])
        df['occur_date'] = df['occur_date'].dt.round('15min')

        # 중복 제거 및 정렬
        df = df.drop_duplicates(subset=['ip_addr', 'cid', 'lid', 'occur_date'], keep='last')
        df = df.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])

        # 숫자형 변환 (필터링 및 변환 전 필수)
        for col in self.feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 1. 학습 시 품질 필터링 (원본 수치 기준)
        if is_train:
            df = self._filter_low_quality(df)
            if df is None or df.empty: return None

        # 2. 특성별 맞춤형 변환 적용 (클리핑 및 로그)
        # 필터링을 통과한 데이터에 대해 변환 수행
        df = self._clean_and_transform(df)

        # 3. 시간축 재구성 (누락된 행 강제 생성)
        reindexed_dfs = []
        for (ip, cid, lid), group in df.groupby(['ip_addr', 'cid', 'lid']):
            orig_max = group['occur_date'].max()
            group = group.set_index('occur_date')
            full_range = pd.date_range(start=group.index.min(), end=orig_max, freq='15min')
            group = group.reindex(full_range)
            group['ip_addr'], group['cid'], group['lid'] = ip, cid, lid
            reindexed_dfs.append(group.reset_index().rename(columns={'index': 'occur_date'}))

        df = pd.concat(reindexed_dfs, ignore_index=True) if reindexed_dfs else df
        df = df.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])

        # 4. 결측치 보간 (최대 1개 연속 누락까지만)
        for _, group in df.groupby(['ip_addr', 'cid', 'lid']):
            if len(group) > 1:
                df.loc[group.index, self.feature_cols] = group[self.feature_cols].interpolate(
                    method='linear', limit=1, limit_direction='both'
                )

        return df

    def create_sequences(self, df, is_train=True):
        """NaN(보간되지 않은 결측치)이 포함된 윈도우는 버리고 유효한 시퀀스만 생성"""
        df = df.copy()
        
        # 1. 스케일러 학습 (학습 시에만, NaN을 제외한 순수 데이터 분포만 학습)
        if is_train:
            valid_data = df[self.feature_cols].dropna()
            if not valid_data.empty:
                self.scaler.fit(valid_data)
        
        # 2. 스케일링 수행 (숫자 변환을 위해 임시로 fillna 적용)
        temp_fill = 0 if self.feature_type == 'traffic' else -40
        data_vals = df[self.feature_cols].fillna(temp_fill)
        scaled_vals = self.scaler.transform(data_vals)
        
        # 3. 원래 어디가 NaN(2개 이상 연속 누락)이었는지 마스크 생성
        is_nan_mask = df[self.feature_cols].isna().any(axis=1).values
        
        sequences = []
        grouped_data = {} if not is_train else None

        current_idx = 0
        for (ip, cid, lid), group in df.groupby(['ip_addr', 'cid', 'lid']):
            n_rows = len(group)
            group_scaled = scaled_vals[current_idx : current_idx + n_rows]
            group_nan = is_nan_mask[current_idx : current_idx + n_rows]
            current_idx += n_rows
            
            if n_rows < self.window_size: continue
            
            group_seqs = []
            group_indices = []
            for i in range(n_rows - self.window_size + 1):
                # 윈도우 내에 NaN(보간되지 않은 결측치)이 하나라도 있으면 해당 윈도우 통째로 버림
                if not group_nan[i : i + self.window_size].any():
                    group_seqs.append(group_scaled[i : i + self.window_size])
                    # 윈도우의 마지막 시점 인덱스 저장
                    group_indices.append(group.index[i + self.window_size - 1])
            
            if group_seqs:
                if is_train:
                    sequences.extend(group_seqs)
                else:
                    grouped_data[(ip, cid, lid)] = (np.array(group_seqs), group_indices)
        
        return np.array(sequences) if is_train else grouped_data
