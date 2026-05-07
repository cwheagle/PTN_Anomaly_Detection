import torch
import numpy as np
import pandas as pd
import os
import json
from src.models.model import LSTMAutoencoder
from src.data.data_processor import DataProcessor
from src.config import MODEL_CONFIG, PATHS, FEATURE_GROUPS

class AnomalyDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracks = {} # {f_type: {'model': m, 'proc': p, 'th': t}}

        for ft in ['traffic', 'optical']:
            cfg = MODEL_CONFIG.copy()
            cfg['input_dim'] = len(FEATURE_GROUPS[ft])
            model = LSTMAutoencoder(cfg).to(self.device)
            
            p = PATHS[ft]
            if os.path.exists(p['model']):
                model.load_state_dict(torch.load(p['model'], map_location=self.device))
                model.eval()
                
                proc = DataProcessor(ft)
                proc.load_scaler(p['scaler'])
                
                with open(p['threshold'], 'r') as f:
                    th = json.load(f)['threshold']
                
                self.tracks[ft] = {'model': model, 'proc': proc, 'th': th}

    def _analyze_track(self, df, ft):
        """특정 트랙(Traffic/Optical)의 데이터를 분석하여 이상 점수 산출"""
        if ft not in self.tracks: return None
        
        track = self.tracks[ft]
        df_clean = track['proc'].preprocess(df, is_train=False)
        if df_clean is None: return None
        
        grouped_data = track['proc'].create_sequences(df_clean, is_train=False)
        if not grouped_data: return None
        
        all_res = []
        for (ip, cid, lid), (seqs, indices) in grouped_data.items():
            inputs = torch.from_numpy(seqs).float().to(self.device)
            with torch.no_grad():
                outputs = track['model'](inputs)
                # 마지막 시점의 오차만 계산 (추론 시점의 점수)
                diff = (inputs[:, -1, :] - outputs[:, -1, :]) ** 2
                mse = np.mean(diff.cpu().numpy(), axis=1)
            
            # 매핑된 인덱스를 사용하여 결과 데이터프레임 생성
            res = df_clean.loc[indices].copy()
            res[f'{ft}_score'] = mse
            res[f'is_{ft}_anomaly'] = mse > track['th']
            all_res.append(res[['occur_date', 'ip_addr', 'cid', 'lid', f'{ft}_score', f'is_{ft}_anomaly']])
        
        return pd.concat(all_res) if all_res else None

    def detect(self, df_traffic=None, df_optical=None):
        """앙상블 분석 통합 인터페이스"""
        res_t = self._analyze_track(df_traffic, 'traffic') if df_traffic is not None else None
        res_o = self._analyze_track(df_optical, 'optical') if df_optical is not None else None
        
        if res_t is None and res_o is None: return None
        
        # 병합
        if res_t is not None and res_o is not None:
            final = pd.merge(res_t, res_o, on=['occur_date', 'ip_addr', 'cid', 'lid'], how='outer')
        else:
            final = res_t if res_t is not None else res_o
            
        # NaN 값 처리 (데이터가 없는 트랙은 정상으로 간주)
        for col in ['is_traffic_anomaly', 'is_optical_anomaly']:
            if col in final.columns:
                final[col] = final[col].fillna(False)
        for col in ['traffic_score', 'optical_score']:
            if col in final.columns:
                final[col] = final[col].fillna(0.0)

        # 통합 이상 판정
        final['is_anomaly'] = (final.get('is_traffic_anomaly', False) == True) | \
                             (final.get('is_optical_anomaly', False) == True)
        
        def get_reason(row):
            reasons = []
            if row.get('is_traffic_anomaly') is True: reasons.append("TRAFFIC")
            if row.get('is_optical_anomaly') is True: reasons.append("OPTICAL")
            return " + ".join(reasons) if reasons else "NORMAL"
            
        final['anomaly_reason'] = final.apply(get_reason, axis=1)
        final['anomaly_score'] = final[['traffic_score', 'optical_score']].max(axis=1)
        
        return final.sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
