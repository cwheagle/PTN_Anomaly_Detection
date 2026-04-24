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
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from {path}")
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
        Returns:
            df_total_res: 모든 장비/포트의 이상 점수와 판정 결과, 원인 분석이 포함된 데이터프레임
        """
        # 1. 데이터 준비 (그룹별 시퀀스 생성)
        grouped_sequences, df_clean = self.processor.prepare_inference_data(df)
        
        if not grouped_sequences:
            return None
            
        all_results = []
        feature_names = self.processor.feature_cols
        
        # 2. 그룹별(장비/포트) 추론 수행
        for (equip_id, port_id), sequences in grouped_sequences.items():
            input_tensor = torch.from_numpy(sequences).float().to(self.device)
            
            with torch.no_grad():
                reconstructed = self.model(input_tensor)
                # 시퀀스의 마지막 시점(t)에 대한 복원 오차를 피처별로 계산
                diff = (input_tensor[:, -1, :] - reconstructed[:, -1, :]) ** 2
                feature_mse = diff.cpu().numpy() # (batch, features)
                mse = np.mean(feature_mse, axis=1) # (batch,)
            
            # 해당 그룹의 데이터만 추출하여 결과 매칭
            group_df = df_clean[(df_clean['equipment_id'] == equip_id) & (df_clean['port_id'] == port_id)].copy()
            result_indices = group_df.index[self.config['window_size'] - 1:]
            df_res = group_df.loc[result_indices].copy()
            
            # 임계치 결정 (저장된 값이 있으면 우선 사용)
            threshold = self.global_threshold if self.global_threshold is not None else self.get_threshold(mse)
            
            df_res['anomaly_score'] = mse
            df_res['is_anomaly'] = mse > threshold
            df_res['threshold'] = threshold
            
            # 원인 분석 (RCA): 기여도 기반으로 개선
            total_error = np.sum(feature_mse, axis=1, keepdims=True)
            total_error = np.where(total_error == 0, 1e-9, total_error) # 0 나누기 방지
            contribution = (feature_mse / total_error) * 100
            
            reasons = []
            for i in range(len(mse)):
                # 기여도가 15% 이상인 피처만 추출하여 중요도 순으로 정렬
                main_causes = []
                sorted_idx = np.argsort(contribution[i])[::-1]
                
                for idx in sorted_idx:
                    if contribution[i, idx] >= 15.0: # 15% 이상 기여한 것만 인정
                        main_causes.append(f"{feature_names[idx]}({contribution[i, idx]:.1f}%)")
                    if len(main_causes) >= 2: # 최대 2개까지만 표시
                        break
                
                if not main_causes: # 만약 15% 넘는게 없으면 가장 높은 것 하나라도 표시
                    top_idx = sorted_idx[0]
                    main_causes.append(f"{feature_names[top_idx]}({contribution[i, top_idx]:.1f}%)")
                    
                reasons.append(", ".join(main_causes))
            
            df_res['anomaly_reason'] = reasons
            
            all_results.append(df_res)
            
        # 3. 전체 결과 통합
        if not all_results:
            return None
            
        df_total_res = pd.concat(all_results).sort_values(['equipment_id', 'port_id', 'collect_time'])
        return df_total_res

    def get_threshold(self, scores):
        """이상 판정을 위한 임계치를 결정합니다. (Global Threshold가 없을 때 사용)"""
        percentile = self.config.get('threshold_percentile', 99.5)
        return np.percentile(scores, percentile)
