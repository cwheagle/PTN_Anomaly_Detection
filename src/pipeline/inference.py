import os
import json
import math
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from src.models.model import LSTMAutoencoder
from src.data.data_processor import DataProcessor
from src.config import MODEL_CONFIG, PATHS, FEATURE_GROUPS, SEVERITY_CONFIG

class AnomalyDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracks = {} # {f_type: {'model': m, 'proc': p, 'th': t, 'config': c}}

        for ft in ['traffic', 'optical']:
            p = PATHS[ft]
            # 모델 파일명과 동일한 설정 파일 (예: traffic_ae.json)
            meta_path = p['model'].replace('.pth', '.json')
            
            # 1. 설정 파일이 있으면 거기서 모델 구성을 로드, 없으면 기본 config.py 사용
            cfg = MODEL_CONFIG.copy()
            th = None
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        cfg.update(meta.get('config', {}))
                        th = meta.get('threshold')
                        print(f"[*] Loaded metadata for {ft} model (Trained at: {meta.get('trained_at')})")
                except Exception as e:
                    print(f"[!] Error reading meta for {ft}: {e}")

            cfg['input_dim'] = len(FEATURE_GROUPS[ft])
            model = LSTMAutoencoder(cfg).to(self.device)
            
            if os.path.exists(p['model']):
                try:
                    model.load_state_dict(torch.load(p['model'], map_location=self.device, weights_only=True))
                    model.eval()
                    
                    # 2. 로드된 cfg를 DataProcessor에도 전달 (window_size 정합성 확보)
                    proc = DataProcessor(ft, config=cfg)
                    if proc.load_scaler(p['scaler']):
                        if th is not None:
                            self.tracks[ft] = {'model': model, 'proc': proc, 'th': th, 'config': cfg}
                            print(f"[*] Loaded {ft} model track successfully.")
                        else:
                            print(f"[!] No threshold found for {ft}")
                    else:
                        print(f" [!] Failed to load scaler for {ft}")
                except Exception as e:
                    print(f"[!] Error loading {ft} model weight: {e}")
            else:
                print(f"[!] {ft.capitalize()} model file not found: {p['model']}")

    def reload_config(self, ft):
        """저장된 메타데이터 파일(.json)에서 설정을 다시 읽어 메모리에 반영"""
        if ft not in self.tracks: return False
        
        p = PATHS[ft]
        meta_path = p['model'].replace('.pth', '.json')
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    # 기존 config 유지하면서 meta의 config로 업데이트
                    self.tracks[ft]['config'].update(meta.get('config', {}))
                    self.tracks[ft]['th'] = meta.get('threshold', self.tracks[ft]['th'])
                    print(f"[*] Reloaded config for {ft} track (Threshold: {self.tracks[ft]['th']})")
                    return True
            except Exception as e:
                print(f"[!] Error reloading config for {ft}: {e}")
        return False

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

            # 1. 데이터 복구 (Inverse Transform) - 사유 진단 및 DB 저장용
            if ft == 'traffic':
                res['tx_packet'] = np.expm1(res['tx_packet']).astype(int)
                res['rx_packet'] = np.expm1(res['rx_packet']).astype(int)
                res['error_packet'] = np.expm1(res['error_packet']).astype(int)

            # 2. 상세 사유 진단 로직 (복구된 수치 사용)
            def get_detailed_reason(row):
                if not row[f'is_{ft}_anomaly']: return "NORMAL"
                if ft == 'traffic':
                    err = int(row.get('error_packet', 0))
                    tx = int(row.get('tx_packet', 0))
                    rx = int(row.get('rx_packet', 0))
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
            
            # 3. 최종 컬럼 정리 (DB 저장용)
            cols_to_keep = ['occur_date', 'ip_addr', 'cid', 'lid', 
                           f'{ft}_score', f'{ft}_severity', f'{ft}_slope', f'is_{ft}_anomaly', f'{ft}_threshold', f'{ft}_reason']
            
            if ft == 'traffic':
                cols_to_keep.extend(['tx_packet', 'rx_packet', 'error_packet'])
            elif ft == 'optical':
                cols_to_keep.extend(['tx_avg_power', 'rx_avg_power'])
                
            all_res.append(res[cols_to_keep])
        
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
        
        # 원본 수치 결측값 처리
        for col in ['tx_packet', 'rx_packet', 'error_packet', 'tx_avg_power', 'rx_avg_power']:
            if col in final.columns:
                final[col] = final[col].fillna(0)

        final['is_anomaly'] = (final.get('is_traffic_anomaly', False) == True) | \
                             (final.get('is_optical_anomaly', False) == True)
        
        def merge_reasons(row):
            reasons = []
            if row.get('is_traffic_anomaly'): reasons.append(row['traffic_reason'])
            if row.get('is_optical_anomaly'): reasons.append(row['optical_reason'])
            return " + ".join(reasons) if reasons else "NORMAL"
            
        final['anomaly_reason'] = final.apply(merge_reasons, axis=1)
        
        # 통합 지표 결정: 가장 심각도가 높은(dominant) 트랙의 값을 선택
        # 점수, 임계치, 기울기, RUL(장애 예측)까지 해당 트랙의 전문 설정으로 일괄 계산
        def get_dominant_metrics(row):
            t_sev = row.get('traffic_severity', 0.0)
            o_sev = row.get('optical_severity', 0.0)
            
            # 우세 트랙 결정
            ft = 'traffic' if t_sev >= o_sev else 'optical'
            
            # 기본 지표 추출
            res = {
                'anomaly_score': row.get(f'{ft}_score', 0.0),
                'threshold': row.get(f'{ft}_threshold', 0.0),
                'slope': row.get(f'{ft}_slope', 0.0),
                'severity': row.get(f'{ft}_severity', 0.0),
                'slope_label': "STABLE",
                'ttf_minutes': None,
                'expected_fatal_time': None
            }
            
            # 해당 트랙의 전문 설정(config) 참조하여 지능형 지표 산출
            if ft in self.tracks:
                cfg = self.tracks[ft]['config']
                
                # 1. 추세 라벨 판정
                slope_th = cfg.get('slope_threshold', 1.0)
                if res['slope'] > slope_th: res['slope_label'] = "RISING"
                elif res['slope'] < -slope_th: res['slope_label'] = "FALLING"
                
                # 2. 잔여 수명 예측 (RUL)
                target_sev = cfg.get('rul_target', 90.0)
                if res['slope_label'] == "RISING" and res['severity'] < target_sev:
                    # 원본 TTF 계산 (분 단위)
                    raw_ttf = ((target_sev - res['severity']) / res['slope']) * 15
                    # 15분 단위 정규화 (올림)
                    res['ttf_minutes'] = math.ceil(raw_ttf / 15) * 15
                    res['expected_fatal_time'] = row['occur_date'] + timedelta(minutes=res['ttf_minutes'])
            
            return pd.Series([
                res['anomaly_score'], res['threshold'], res['slope'], 
                res['severity'], res['slope_label'], res['ttf_minutes'], res['expected_fatal_time']
            ])

        final[['anomaly_score', 'threshold', 'slope', 'severity', 'slope_label', 'ttf_minutes', 'expected_fatal_time']] = final.apply(get_dominant_metrics, axis=1)

        # 경보 등급 및 라벨 추가
        alarm_data = final['severity'].apply(self._get_alarm_info)
        final['alarm_level'] = alarm_data.apply(lambda x: x[0])
        final['alarm_label'] = alarm_data.apply(lambda x: x[1])

        # [표준화] DB 확장형 스키마 및 CSV 저장 형식을 확장
        standard_cols = [
            'occur_date', 'ip_addr', 'cid', 'lid', 
            'tx_packet', 'rx_packet', 'error_packet', 'tx_avg_power', 'rx_avg_power',
            'traffic_score', 'traffic_severity', 'traffic_slope', 'traffic_threshold', 'is_traffic_anomaly',
            'optical_score', 'optical_severity', 'optical_slope', 'optical_threshold', 'is_optical_anomaly',
            'anomaly_score', 'severity', 'slope', 'slope_label', 'threshold', 'is_anomaly', 
            'alarm_level', 'alarm_label', 'ttf_minutes', 'expected_fatal_time', 'anomaly_reason'
        ]
        
        final_cols = [c for c in standard_cols if c in final.columns]
        results = final[final_cols].sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
        
        # [최신화] 스케줄러 동작 시(latest_only=True) 배치 내 가장 최신 시점(max)의 데이터만 반환
        # tail(1)을 쓰면 과거에 데이터가 끊긴 포트의 마지막 데이터가 섞여 들어오는 현상 방지
        if latest_only and not results.empty:
            latest_time = results['occur_date'].max()
            return results[results['occur_date'] == latest_time].copy()
        return results
