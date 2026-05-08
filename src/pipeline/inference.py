import torch
import numpy as np
import pandas as pd
import os
import json
from src.models.model import LSTMAutoencoder
from src.data.data_processor import DataProcessor
from src.config import MODEL_CONFIG, PATHS, FEATURE_GROUPS, SEVERITY_CONFIG

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

    def _get_alarm_info(self, severity):
        """심각도 점수에 따른 경보 등급 및 라벨 반환"""
        for tier in ["CRITICAL", "MAJOR", "MINOR"]:
            if severity >= SEVERITY_CONFIG[tier]["min"]:
                return SEVERITY_CONFIG[tier]["level"], SEVERITY_CONFIG[tier]["label"]
        return SEVERITY_CONFIG["NORMAL"]["level"], SEVERITY_CONFIG["NORMAL"]["label"]

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
                # 마지막 시점의 오차만 계산
                diff = (inputs[:, -1, :] - outputs[:, -1, :]) ** 2
                mse = np.mean(diff.cpu().numpy(), axis=1)
            
            res = df_clean.loc[indices].copy()
            res[f'{ft}_score'] = mse
            res[f'is_{ft}_anomaly'] = mse > track['th']
            res[f'{ft}_threshold'] = track['th']
            
            # 심각도 점수 산출 로직 (0~100 정규화)
            def calculate_severity(mse, threshold):
                if threshold <= 0: return 0.0
                ratio = mse / threshold
                if ratio <= 1.0:
                    return ratio * 50.0 # 정상 구간 (0~50)
                else:
                    # 이상 구간 (50~100 수렴): 50 + 50 * (1 - exp(-0.5 * (ratio-1)))
                    return 50.0 + 50.0 * (1 - np.exp(-0.5 * (ratio - 1.0)))

            res[f'{ft}_severity'] = res[f'{ft}_score'].apply(lambda x: calculate_severity(x, track['th']))
            
            # [Phase 6] 추세 분석 (Slope): 포트별 점수 변화율 산출
            # 여러 시점의 결과가 있을 경우 선형 회귀 기울기 계산
            if len(res) >= 2:
                y = res[f'{ft}_severity'].values
                x = np.arange(len(y))
                slope, _ = np.polyfit(x, y, 1)
                res[f'{ft}_slope'] = slope
            else:
                res[f'{ft}_slope'] = 0.0

            # 상세 사유 진단 로직 (원본 수치 복원)
            def get_detailed_reason(row):
                if not row[f'is_{ft}_anomaly']: return "NORMAL"
                if ft == 'traffic':
                    # 로그 역변환하여 원본 수치 근사값 산출
                    err = int(np.expm1(row.get('error_packet', 0)))
                    tx = int(np.expm1(row.get('tx_packet', 0)))
                    rx = int(np.expm1(row.get('rx_packet', 0)))
                    reason = f"Traffic (TX:{tx}, RX:{rx}"
                    if err > 0: reason += f", Err:{err}"
                    reason += ")"
                    return reason
                elif ft == 'optical':
                    rx = row.get('rx_avg_power', 0)
                    tx = row.get('tx_avg_power', 0)
                    return f"Optical (RX:{rx:.2f}, TX:{tx:.2f})"
                return "Anomaly"
            
            res[f'{ft}_reason'] = res.apply(get_detailed_reason, axis=1)
            all_res.append(res[['occur_date', 'ip_addr', 'cid', 'lid', 
                               f'{ft}_score', f'{ft}_severity', f'{ft}_slope', f'is_{ft}_anomaly', f'{ft}_threshold', f'{ft}_reason']])
        
        return pd.concat(all_res) if all_res else None

    def detect(self, df_traffic=None, df_optical=None, latest_only=True):
        """앙상블 분석 통합 인터페이스"""
        res_t = self._analyze_track(df_traffic, 'traffic') if df_traffic is not None else None
        res_o = self._analyze_track(df_optical, 'optical') if df_optical is not None else None
        
        if res_t is None and res_o is None: return None
        
        if res_t is not None and res_o is not None:
            final = pd.merge(res_t, res_o, on=['occur_date', 'ip_addr', 'cid', 'lid'], how='outer')
        else:
            final = res_t if res_t is not None else res_o
            
        # 결측값 및 통합 필드 처리
        for ft in ['traffic', 'optical']:
            if f'is_{ft}_anomaly' in final.columns:
                final[f'is_{ft}_anomaly'] = final[f'is_{ft}_anomaly'].fillna(False)
                final[f'{ft}_score'] = final[f'{ft}_score'].fillna(0.0)
                final[f'{ft}_severity'] = final[f'{ft}_severity'].fillna(0.0)
                final[f'{ft}_reason'] = final[f'{ft}_reason'].fillna("NORMAL")
                if f'{ft}_threshold' not in final.columns: final[f'{ft}_threshold'] = 0.0

        final['is_anomaly'] = (final.get('is_traffic_anomaly', False) == True) | \
                             (final.get('is_optical_anomaly', False) == True)
        
        def merge_reasons(row):
            reasons = []
            if row.get('is_traffic_anomaly'): reasons.append(f"T:{row['traffic_reason']}")
            if row.get('is_optical_anomaly'): reasons.append(f"O:{row['optical_reason']}")
            return " + ".join(reasons) if reasons else "NORMAL"
            
        final['anomaly_reason'] = final.apply(merge_reasons, axis=1)
        
        # 통합 점수
        score_cols = [c for c in ['traffic_score', 'optical_score'] if c in final.columns]
        final['anomaly_score'] = final[score_cols].max(axis=1) if score_cols else 0.0

        # 통합 심각도
        sev_cols = [c for c in ['traffic_severity', 'optical_severity'] if c in final.columns]
        final['severity'] = final[sev_cols].max(axis=1) if sev_cols else 0.0

        # 통합 기울기
        slope_cols = [c for c in ['traffic_slope', 'optical_slope'] if c in final.columns]
        final['slope'] = final[slope_cols].max(axis=1) if slope_cols else 0.0

        # 통합 임계치
        th_cols = [c for c in ['traffic_threshold', 'optical_threshold'] if c in final.columns]
        final['threshold'] = final[th_cols].max(axis=1) if th_cols else 0.0

        # 추세 라벨
        def get_slope_label(slope):
            th = SEVERITY_CONFIG.get('slope_threshold', 3.0)
            if slope > th: return "RISING"
            elif slope < -th: return "FALLING"
            return "STABLE"
        final['slope_label'] = final['slope'].apply(get_slope_label)

        # 경보 등급 및 라벨 추가
        alarm_data = final['severity'].apply(self._get_alarm_info)
        final['alarm_level'] = alarm_data.apply(lambda x: x[0])
        final['alarm_label'] = alarm_data.apply(lambda x: x[1])

        # [표준화] DB 확장형 스키마 및 CSV 저장 형식을 23개 컬럼으로 확장
        standard_cols = [
            'occur_date', 'ip_addr', 'cid', 'lid', 
            'traffic_score', 'traffic_severity', 'traffic_slope', 'traffic_threshold', 'is_traffic_anomaly',
            'optical_score', 'optical_severity', 'optical_slope', 'optical_threshold', 'is_optical_anomaly',
            'anomaly_score', 'severity', 'slope', 'slope_label', 'threshold', 'is_anomaly', 
            'alarm_level', 'alarm_label', 'anomaly_reason'
        ]


        
        final_cols = [c for c in standard_cols if c in final.columns]
        results = final[final_cols].sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
        
        # [최신화] 스케줄러 동작 시(latest_only=True) 최신 1건만 반환, 백테스트 시(False) 전체 반환
        if latest_only:
            return results.groupby(['ip_addr', 'cid', 'lid']).tail(1).copy()
        return results
