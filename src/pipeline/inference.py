import torch
import numpy as np
import pandas as pd
import os
import json
from src.models.model import LSTMAutoencoder
from src.data.data_processor import DataProcessor
from src.config import MODEL_CONFIG, PATHS

class AnomalyDetector:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = MODEL_CONFIG
        self.model = LSTMAutoencoder(self.config).to(self.device)
        self.global_threshold = None
        
        # 모델 로드
        path = model_path or PATHS['model_save_path']
        try:
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.eval()
                print(f"Model loaded from {path}")
            else:
                print(f"Warning: Model file not found at {path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            
        self.processor = DataProcessor()
        self.processor.load_scaler()
        self.load_threshold()

    def load_threshold(self):
        """저장된 임계치 정보를 로드합니다."""
        path = PATHS.get('threshold_path', 'models/threshold.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.global_threshold = data['threshold']
                print(f"Global threshold loaded: {self.global_threshold:.6f}")
        else:
            print("Global threshold not found. Will use dynamic thresholding.")

    def detect(self, df):
        """
        입력 데이터프레임에 대해 장비/포트별로 이상 탐지를 수행합니다.
        """
        # 1. 데이터 준비 (그룹별 시퀀스 생성)
        grouped_sequences, df_clean = self.processor.prepare_inference_data(df)
        
        if not grouped_sequences or df_clean is None:
            return None
            
        all_results = []
        feature_names = self.processor.feature_cols
        
        # 2. 그룹별(IP/CID/LID) 추론 수행
        for (ip, cid, lid), sequences in grouped_sequences.items():
            input_tensor = torch.from_numpy(sequences).float().to(self.device)
            
            with torch.no_grad():
                reconstructed = self.model(input_tensor)
                # 시퀀스의 마지막 시점(t)에 대한 복원 오차 계산
                diff = (input_tensor[:, -1, :] - reconstructed[:, -1, :]) ** 2
                feature_mse = diff.cpu().numpy()
                mse = np.mean(feature_mse, axis=1)
            
            # 해당 그룹 데이터 필터링
            group_df = df_clean[
                (df_clean['ip_addr'] == ip) & 
                (df_clean['cid'] == cid) & 
                (df_clean['lid'] == lid)
            ].copy()
            
            # 시간 단절 대응 인덱스 매핑
            valid_indices = []
            time_diffs = group_df['occur_date'].diff().dt.total_seconds() / 60
            for i in range(len(group_df) - self.config['window_size'] + 1):
                window_diffs = time_diffs.iloc[i+1 : i + self.config['window_size']]
                if not (window_diffs > 30).any():
                    valid_indices.append(group_df.index[i + self.config['window_size'] - 1])
            
            if not valid_indices or len(valid_indices) != len(mse):
                continue
                
            df_res = group_df.loc[valid_indices].copy()
            
            # 임계치 적용
            threshold = self.global_threshold if self.global_threshold is not None else np.percentile(mse, 99.5)
            
            df_res['anomaly_score'] = mse
            df_res['is_anomaly'] = mse > threshold
            df_res['threshold'] = threshold
            
            # 원인 분석 (RCA)
            df_res['anomaly_reason'] = self.analyze_root_cause(feature_mse, feature_names)
            all_results.append(df_res)
            
        if not all_results:
            return None
            
        return pd.concat(all_results).sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])

    def analyze_root_cause(self, feature_mse, feature_names):
        """복원 오차 기여도 기반으로 이상 원인을 분석합니다."""
        total_error = np.sum(feature_mse, axis=1, keepdims=True)
        total_error = np.where(total_error == 0, 1e-9, total_error)
        contribution = (feature_mse / total_error) * 100
        
        reasons = []
        for i in range(len(feature_mse)):
            main_causes = []
            # 기여도가 높은 순으로 정렬
            sorted_idx = np.argsort(contribution[i])[::-1]
            
            for idx in sorted_idx:
                if contribution[i, idx] >= 15.0: # 15% 이상 기여 시 원인으로 포함
                    main_causes.append(f"{feature_names[idx]}({contribution[i, idx]:.1f}%)")
                if len(main_causes) >= 2: # 최대 2개까지 표시
                    break
            
            if not main_causes:
                top_idx = sorted_idx[0]
                main_causes.append(f"{feature_names[top_idx]}({contribution[i, top_idx]:.1f}%)")
                
            reasons.append(", ".join(main_causes))
        return reasons
