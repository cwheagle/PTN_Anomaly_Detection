import os
import json
import traceback
import math
import torch
import numpy as np
import pandas as pd
from datetime import timedelta
from src.models.model import LSTMAutoencoder
from src.data.data_processor import DataProcessor
from src.config import MODEL_CONFIG, PATHS, FEATURE_GROUPS, SEVERITY_CONFIG
from src.rca.feature_contribution import FeatureContributionAnalyzer
from src.rca.rule_engine import RCAEngine

class AnomalyDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tracks = {} # {f_type: {'model': m, 'proc': p, 'th': t, 'config': c}}

        # [Phase 8] RCA 엔진 초기화 (Feature Contribution + 도메인 룰 기반 진단)
        self.rca_engine = RCAEngine()

        # 초기 구동 시 저장된 모델이 있으면 로드
        for ft in ['traffic', 'optical']:
            self.reload_model(ft)

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

    def reload_model(self, ft):
        """학습 완료 후 또는 초기화 시 파일로부터 모델 가중치, 스케일러, 설정을 모두 로드"""
        p = PATHS[ft]
        if not os.path.exists(p['model']):
            print(f"[*] {ft.capitalize()} model file not found, skipping load.")
            return False

        print(f"[*] (Re)loading {ft} model from disk...")
        try:
            # 설정 파일(.json) 로드
            meta_path = p['model'].replace('.pth', '.json')
            cfg = MODEL_CONFIG.copy()
            th = None
            
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    cfg.update(meta.get('config', {}))
                    th = meta.get('threshold', cfg.get('threshold'))
                    print(f"[*] Loaded metadata for {ft} model (Trained at: {meta.get('trained_at')})")

            # 모델 인스턴스 생성 및 가중치 로드
            cfg['input_dim'] = len(FEATURE_GROUPS[ft])
            model = LSTMAutoencoder(cfg).to(self.device)
            state_dict = torch.load(p['model'], map_location=self.device, weights_only=True)
            new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            
            # [Blackwell 최적화] PyTorch 2.0+ 및 CUDA 환경에서 컴파일 적용 (이식성 유지)
            # BUG FIX: RuntimeError: Detected that you are using FX to symbolically trace a dynamo-optimized function...
            # torch.compile과 AMP(Autocast)가 충돌하는 현상이 있으므로 주석 처리 (이미 6초대로 충분히 빠름)
            # if hasattr(torch, "compile") and self.device.type == "cuda":
            #     try:
            #         print(f"[*] Compiling {ft} model for inference optimization...")
            #         model = torch.compile(model)
            #     except: pass
            
            model.eval()
            
            # 프로세서 및 스케일러 리로드
            proc = DataProcessor(ft, config=cfg)
            if not proc.load_scaler(p['scaler']): 
                print(f"[!] Failed to load scaler for {ft}")
                return False
            
            # 메모리 교체
            self.tracks[ft] = {'model': model, 'proc': proc, 'th': th if th is not None else 0.1, 'config': cfg}
            print(f"[SUCCESS] {ft.capitalize()} model track (re)loaded. (Threshold: {self.tracks[ft]['th']:.6f})")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to reload {ft} model: {e}")
            print(traceback.format_exc())
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
        
        # [Blackwell 최적화] 라이브러리 임포트 및 존재 여부 확인
        try:
            from transformer_engine.pytorch import fp8_autocast
        except (ImportError, ModuleNotFoundError):
            fp8_autocast = None

        all_res = []
        for (ip, cid, lid), (seqs, indices) in grouped_data.items():
            inputs = torch.from_numpy(seqs).float().to(self.device)
            
            # [Blackwell 최적화] FP8/AMP 컨텍스트를 포트(배치)마다 새로 생성
            if fp8_autocast and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
                fp8_ctx = fp8_autocast(enabled=True)
            else:
                fp8_ctx = torch.amp.autocast('cuda', enabled=True if torch.cuda.is_available() else False)

            with torch.no_grad():
                with fp8_ctx:
                    outputs = track['model'](inputs)
                # 마지막 시점의 통합 오차 계산 (기존 로직 유지)
                diff = (inputs[:, -1, :] - outputs[:, -1, :]) ** 2
                mse = np.mean(diff.cpu().numpy(), axis=1)

                # [Phase 9] 동적 임계치 (Dynamic Threshold) 3-Sigma 산출
                diff_all = (inputs - outputs) ** 2
                mse_all = np.mean(diff_all.cpu().numpy(), axis=2) # (batch, seq_len)
                
                # 과거 시점(마지막 제외) 평균/표준편차
                past_mse = mse_all[:, :-1]
                past_mean = np.mean(past_mse, axis=1)
                past_std = np.std(past_mse, axis=1)
                
                # 동적 임계치 밴드 (mu + 3 sigma)
                dyn_th = past_mean + 3 * past_std
                
                # 노이즈 방지용 글로벌 하한선 (학습된 전역 임계치의 100% 보장)
                # 과거 변동성이 너무 없어서 dyn_th가 0에 수렴하더라도 최소한 글로벌 임계치는 넘겨야 이상으로 판별
                min_th = track['th'] * 1.0
                final_threshold = np.maximum(dyn_th, min_th)

                # [Phase 8] Feature별 기여도 계산용 NumPy 변환
                inputs_np = inputs.cpu().numpy()
                outputs_np = outputs.cpu().numpy()

            res = df_clean.loc[indices].copy()
            res[f'{ft}_score'] = mse
            res[f'is_{ft}_anomaly'] = mse > final_threshold
            res[f'{ft}_threshold'] = final_threshold

            # [Phase 8] Feature Contribution 계산 (포트 단위 배치 전체에 대해 평균)
            feature_names = FEATURE_GROUPS[ft]
            fc_analyzer = FeatureContributionAnalyzer(feature_names)
            # 포트 배치(seqs) 전체 평균 기여도 산출
            contributions = fc_analyzer.compute(inputs_np, outputs_np)
            # JSON 직렬화 가능한 문자열로 저장
            contributions_json = json.dumps(contributions, ensure_ascii=False)
            
            # 심각도 점수 산출 로직 (0~100 정규화) - Vectorized 적용 (Phase 9)
            def calculate_severity(mse_arr, threshold_arr):
                # 0 나누기 방지
                th_safe = np.where(threshold_arr <= 0, 1e-9, threshold_arr)
                ratio = mse_arr / th_safe
                sev = np.where(ratio <= 1.0,
                               ratio * 50.0, # 정상 구간
                               50.0 + 50.0 * (1 - np.exp(-0.5 * (ratio - 1.0)))) # 이상 구간
                # threshold가 0 이하인 예외 케이스 처리
                return np.where(threshold_arr <= 0, 0.0, sev)

            res[f'{ft}_severity'] = calculate_severity(res[f'{ft}_score'].values, res[f'{ft}_threshold'].values)
            
            # [Phase 6] 추세 분석 (Slope): 포트별 점수 변화율 산출
            # 최근 4시점을 이용해 선형 회귀 기울기 계산 (Rolling)
            def rolling_slope(s):
                s_clean = s.dropna()
                if len(s_clean) < 2: return 0.0
                x = np.arange(len(s_clean))
                return np.polyfit(x, s_clean.values, 1)[0]
            
            res[f'{ft}_slope'] = res[f'{ft}_severity'].rolling(4, min_periods=2).apply(rolling_slope, raw=False)
            res[f'{ft}_slope'] = res[f'{ft}_slope'].fillna(0.0)

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

            # [Phase 9] RCA 진단명 생성은 통합 단계(detect)로 연기하고 기여도만 저장
            res['feature_contribution'] = contributions_json

            # 3. 최종 컬럼 정리 (DB 저장용)
            cols_to_keep = ['occur_date', 'ip_addr', 'cid', 'lid', 
                           f'{ft}_score', f'{ft}_severity', f'{ft}_slope', f'is_{ft}_anomaly', f'{ft}_threshold', f'{ft}_reason',
                           'feature_contribution']

            
            if ft == 'traffic':
                cols_to_keep.extend(['tx_packet', 'rx_packet', 'error_packet'])
            elif ft == 'optical':
                cols_to_keep.extend(['tx_avg_power', 'rx_avg_power'])
                
            # [Phase 9] 추가된 RCA 컨텍스트 지표(Ratio, Slope) 유지
            extra_cols = [c for c in res.columns if c.endswith('_ratio') or c.endswith('_trend_slope')]
            cols_to_keep.extend(extra_cols)
                
            all_res.append(res[cols_to_keep])
        
        return pd.concat(all_res) if all_res else None

    def detect(self, df_traffic=None, df_optical=None, latest_only=True):
        """앙상블 분석 통합 인터페이스"""
        
        # [Phase 9] 실시간 인메모리 RCA 컨텍스트 지표(Ratio, Trend Slope) 산출
        def append_rca_metrics(df_in, ft):
            if df_in is None or df_in.empty: return df_in
            df_out = df_in.copy().sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
            
            def slope_calc(s):
                s_clean = s.dropna()
                if len(s_clean) < 2: return 0.0
                x = np.arange(len(s_clean))
                return np.polyfit(x, s_clean.values, 1)[0]
                
            grouped = df_out.groupby(['ip_addr', 'cid', 'lid'])
            
            if ft == 'traffic':
                for col in ['tx_packet', 'rx_packet', 'error_packet']:
                    if col in df_out.columns:
                        # 4시점 이동평균 (자신 미포함 과거 4시점)
                        past_mean = grouped[col].transform(lambda x: x.rolling(4, min_periods=1).mean().shift(1))
                        # 0으로 나누기 방지
                        df_out[f'{col}_ratio'] = df_out[col] / (past_mean.replace(0, 1e-9))
                        df_out[f'{col}_ratio'] = df_out[f'{col}_ratio'].fillna(1.0)
                        df_out[f'{col}_trend_slope'] = grouped[col].transform(lambda x: x.rolling(4, min_periods=2).apply(slope_calc, raw=False))
            elif ft == 'optical':
                for col in ['tx_avg_power', 'rx_avg_power']:
                    if col in df_out.columns:
                        df_out[f'{col}_trend_slope'] = grouped[col].transform(lambda x: x.rolling(4, min_periods=2).apply(slope_calc, raw=False))
            return df_out

        df_traffic = append_rca_metrics(df_traffic, 'traffic')
        df_optical = append_rca_metrics(df_optical, 'optical')

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

        # [Phase 9] 통합 RCA 진단 (Cross-Track) 및 원시 데이터 연동
        def evaluate_rca(row):
            is_t = row.get('is_traffic_anomaly', False)
            is_o = row.get('is_optical_anomaly', False)
            is_any_anomaly = row.get('is_anomaly', False)
            
            # 1. 원시 데이터(Raw Data) 추출 및 RCA 컨텍스트 스냅샷
            raw_data = {}
            for col in ['tx_packet', 'rx_packet', 'error_packet', 'tx_avg_power', 'rx_avg_power',
                        'traffic_severity', 'optical_severity']:
                if col in row.index:
                    raw_data[col] = row[col]
                    
            for col in row.index:
                if isinstance(col, str) and (col.endswith('_ratio') or col.endswith('_trend_slope')):
                    # Float64 등 JSON 변환 에러 방지를 위한 처리
                    val = row[col]
                    if pd.isna(val): raw_data[col] = 0.0
                    else: raw_data[col] = float(val)

            # 2. 기여도(Contribution) 파싱 및 병합
            fc_t = {}
            fc_o = {}
            fc_t_str = row.get('feature_contribution_x') if 'feature_contribution_x' in row.index else row.get('feature_contribution')
            if fc_t_str and isinstance(fc_t_str, str):
                try: fc_t = json.loads(fc_t_str)
                except: pass
                
            fc_o_str = row.get('feature_contribution_y') if 'feature_contribution_y' in row.index else row.get('feature_contribution')
            if fc_o_str and isinstance(fc_o_str, str):
                try: fc_o = json.loads(fc_o_str)
                except: pass

            # 점수 기반 가중치 정규화 (총합 100%)
            score_t = row.get('traffic_score', 0)
            score_o = row.get('optical_score', 0)
            total_score = score_t + score_o
            
            w_t = score_t / total_score if total_score > 0 else 0.5
            w_o = score_o / total_score if total_score > 0 else 0.5
            
            normalized_fc = {}
            for k, v in fc_t.items(): normalized_fc[k] = round(v * w_t, 2)
            for k, v in fc_o.items(): normalized_fc[k] = round(v * w_o, 2)

            if not is_any_anomaly:
                return pd.Series(["NORMAL", None, None])

            # 3. 진단 트랙 결정 및 수행 (DB 저장용 rca_context 스냅샷 포함)
            if is_t and is_o:
                diagnosis, action = self.rca_engine.diagnose("integrated", normalized_fc, True, raw_data)
                normalized_fc['rca_context'] = raw_data
                return pd.Series([diagnosis, action, json.dumps(normalized_fc)])
            elif is_t:
                diagnosis, action = self.rca_engine.diagnose("traffic", fc_t, True, raw_data)
                fc_t['rca_context'] = raw_data
                return pd.Series([diagnosis, action, json.dumps(fc_t)])
            elif is_o:
                diagnosis, action = self.rca_engine.diagnose("optical", fc_o, True, raw_data)
                fc_o['rca_context'] = raw_data
                return pd.Series([diagnosis, action, json.dumps(fc_o)])
            
            return pd.Series(["NORMAL", None, None])

        final[['rca_diagnosis', 'rca_action', 'feature_contribution']] = final.apply(evaluate_rca, axis=1)

        
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
                slope_th = cfg.get('slope_threshold', 3.0)
                if res['slope'] > slope_th: res['slope_label'] = "RISING"
                elif res['slope'] < -slope_th: res['slope_label'] = "FALLING"
                
                # 2. 잔여 수명 예측 (RUL)
                target_sev = cfg.get('rul_target', 90.0)
                # [수정] 현재 심각도가 최소 MINOR(50) 이상인 '진짜 이상 상태'일 때만 잔여 수명을 예측합니다.
                # 이 조건이 없으면 정상(Severity 10)인데 노이즈로 잠깐 올랐다고 수명을 예측해버려 오탐이 폭발합니다.
                if res['slope_label'] == "RISING" and res['severity'] >= 50.0 and res['severity'] < target_sev:
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
            'alarm_level', 'alarm_label', 'ttf_minutes', 'expected_fatal_time',
            'anomaly_reason', 'rca_diagnosis', 'rca_action', 'feature_contribution'  # [Phase 8, 9.1]
        ]
        
        final_cols = [c for c in standard_cols if c in final.columns]
        results = final[final_cols].sort_values(['ip_addr', 'cid', 'lid', 'occur_date'])
        
        # [최신화] 스케줄러 동작 시(latest_only=True) 배치 내 가장 최신 시점(max)의 데이터만 반환
        # tail(1)을 쓰면 과거에 데이터가 끊긴 포트의 마지막 데이터가 섞여 들어오는 현상 방지
        if latest_only and not results.empty:
            latest_time = results['occur_date'].max()
            return results[results['occur_date'] == latest_time].copy()
        return results
