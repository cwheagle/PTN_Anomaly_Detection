import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from . import registry
from .model import LSTMAutoencoder
from src.data.data_processor import DataProcessor
from src.config import MODEL_CONFIG, PATHS

class Trainer:
    def __init__(self, feature_type='traffic', config_override=None, progress_callback=None, activate=True,
                 trigger='manual', window_info=None, suspect_stats=None, alert_policy=None):
        self.feature_type = feature_type
        self.progress_callback = progress_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stop_requested = False
        self.early_stopped = False # 조기 종료 여부 플래그 추가
        # activate=False 이면 학습 결과를 '후보'로만 저장 (활성 버전이 없는 최초 학습은 예외적으로 활성화).
        # API(UI Train 버튼, 드리프트 재학습)는 False 를 사용하고, 사람이 승격해야 Consumer 에 반영된다.
        self.activate = activate
        self.activated = None       # 학습 후: 이 버전이 활성화되었는지
        self.version = None         # 학습 후: 저장된 버전 ID
        # 재학습 출처 메타데이터 (기록 전용): trigger = manual | drift, window_info = 학습 구간·분할, suspect_stats = 제외 통계
        self.trigger = trigger
        self.window_info = window_info
        self.suspect_stats = suspect_stats
        # 모델과 짝으로 배포할 알람 정책 (alerting.policy_to_meta 형식 dict). None 이면 메타에 기록하지 않음(= 기본 정책)
        self.alert_policy = alert_policy

        # 1. 설정값 병합 (기본값 + 외부 주입값)
        self.config = MODEL_CONFIG.copy()
        if config_override:
            self.config.update(config_override)
        
        # 2. 주입된 설정값으로 데이터 프로세서 우선 초기화 (파생 변수 차원 확인 목적)
        self.processor = DataProcessor(feature_type, config=self.config)
        self.config['input_dim'] = len(self.processor.extended_feature_cols)
        
        # 3. 늘어난 차원(input_dim)을 반영하여 모델 초기화
        self.model = LSTMAutoencoder(self.config).to(self.device)
        # 전역 PATHS를 오염시키지 않도록 복사본 사용 (학습 시 버전별 경로로 덮어씀)
        self.paths = dict(PATHS[feature_type])
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

    def stop(self):
        """학습 중지 요청"""
        self.stop_requested = True
        print(f"[*] Stop requested for {self.feature_type} trainer.")

    def _prepare_loader(self, data_path, fit_scaler=True):
        if not os.path.exists(data_path):
            print(f"[ERROR] Data file not found: {data_path}")
            return None
            
        print(f"[*] Loading CSV: {data_path}")
        df = pd.read_csv(data_path)
        if self.stop_requested: return None

        print(f"[*] Preprocessing data...")
        df_clean = self.processor.preprocess(df, is_train=True)
        if df_clean is None or self.stop_requested: 
            print(f"[!] Preprocessing returned None (Insufficient data or stop requested)")
            return None
        
        print(f"[*] Creating sequences...")
        sequences = self.processor.create_sequences(df_clean, is_train=True, fit_scaler=fit_scaler)
        if len(sequences) == 0 or self.stop_requested: 
            print(f"[!] No sequences created")
            return None
        
        print(f"[*] Converting to Tensor and creating DataLoader (Size: {len(sequences)})")
        return DataLoader(
            TensorDataset(torch.from_numpy(sequences).float()), 
            batch_size=self.config['batch_size'], 
            shuffle=True
        ), sequences

    def train(self, train_path=None, val_path=None):
        t_path = train_path or f"data/{self.feature_type}_train.csv"
        v_path = val_path or f"data/{self.feature_type}_test.csv"
        print(f"[*] Training [{self.feature_type}] specialist model...")
        
        # 1. 데이터 로더 준비
        print(f"[*] Preparing training loader...")
        t_res = self._prepare_loader(t_path)
        if self.stop_requested: return False
        
        print(f"[*] Preparing validation loader...")
        v_res = self._prepare_loader(v_path, fit_scaler=False) # 검증 데이터: 학습 데이터로 fit 한 스케일러를 그대로 사용
        if self.stop_requested: return False
        
        if not t_res:
            print(f"[SKIP] Insufficient training data for {self.feature_type}")
            return False
            
        train_loader, train_sequences = t_res
        val_loader = v_res[0] if v_res else None
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = self.config['patience']
        no_improve_count = 0
        
        print(f"[*] Starting epoch loop (Total: {self.config['epochs']})")
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # --- 훈련 단계 ---
            self.model.train()
            train_loss_sum = 0
            
            # [Blackwell 최적화] 라이브러리 임포트 및 존재 여부 확인
            try:
                from transformer_engine.pytorch import fp8_autocast
            except (ImportError, ModuleNotFoundError):
                fp8_autocast = None

            for batch_idx, batch in enumerate(train_loader):
                if self.stop_requested: return False # 배치 단위 중지 체크
                
                inputs = batch[0].to(self.device)
                
                # [Blackwell 최적화] FP8/AMP 컨텍스트를 배치마다 새로 생성
                if fp8_autocast and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
                    fp8_ctx = fp8_autocast(enabled=True)
                else:
                    fp8_ctx = torch.amp.autocast('cuda', enabled=True if torch.cuda.is_available() else False)

                try:
                    with fp8_ctx:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, inputs)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss_sum += loss.item()
                except Exception as e:
                    print(f"[!] Error during training at epoch {epoch+1}, batch {batch_idx+1}: {e}")
                    raise e
            
            avg_train_loss = train_loss_sum / len(train_loader)
            
            # --- 검증 단계 ---
            avg_val_loss = None
            if val_loader:
                self.model.eval()
                val_loss_sum = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if self.stop_requested: return False # 배치 단위 중지 체크
                        
                        inputs = batch[0].to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, inputs)
                        val_loss_sum += loss.item()
                avg_val_loss = val_loss_sum / len(val_loader)
                
                # 최적 모델 체크 및 조기 종료 로직
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve_count = 0
                    print(f"[SAVE] Best model updated at epoch {epoch+1} (Val Loss: {best_val_loss:.6f})")
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"[EARLY STOP] No improvement for {patience} epochs. Stopping at epoch {epoch+1}")
                        self.early_stopped = True # 플래그 설정
                        break

            # 로그 출력
            epoch_duration = time.time() - epoch_start
            log_msg = f"    Epoch [{epoch+1}/{self.config['epochs']}] Train Loss: {avg_train_loss:.6f}"
            if avg_val_loss is not None:
                log_msg += f", Val Loss: {avg_val_loss:.6f}"
                if no_improve_count > 0:
                    log_msg += f" (No improvement for {no_improve_count} epochs)"
            log_msg += f" ({epoch_duration:.1f}s)"
            print(log_msg)

            # 진행 상황 콜백 호출
            if self.progress_callback:
                self.progress_callback(epoch + 1, self.config['epochs'], avg_train_loss, avg_val_loss)

            # 중지 요청 확인 (Early Exit)
            if self.stop_requested:
                print(f"[STOP] Training interrupted by user at epoch {epoch+1}")
                return False
        
        # 3. 모델 및 스케일러 저장
        model_dir = os.path.dirname(self.paths['model'])
        os.makedirs(model_dir, exist_ok=True)
        
        # [Phase 11] Lightweight Model Registry (버전 관리, src/models/registry.py)
        new_version_id = registry.next_version_id(registry.load(model_dir, self.feature_type))
        self.version = new_version_id
        
        # 파일 경로 버저닝
        base_model = os.path.basename(self.paths['model']).replace('.pth', '')
        v_model_path = os.path.join(model_dir, f"{base_model}_{new_version_id}.pth")
        
        base_scaler = os.path.basename(self.paths['scaler']).replace('.joblib', '')
        v_scaler_path = os.path.join(model_dir, f"{base_scaler}_{new_version_id}.joblib")
        
        # 실제 저장 경로 갱신
        self.paths['model'] = v_model_path
        self.paths['scaler'] = v_scaler_path
        
        # 최적의 모델 상태가 있다면 그것을 로드한 뒤 저장, 없으면 현재 상태 저장
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"[*] Deploying best model {new_version_id} (Val Loss: {best_val_loss:.6f})")
        
        # [Blackwell 최적화] 저장 시 torch.compile 접두어('_orig_mod.') 제거하여 이식성 확보
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in self.model.state_dict().items()}
        torch.save(state_dict, self.paths['model'])
        self.processor.save_scaler(self.paths['scaler'])
        
        # 4. 임계치 산출 및 통합 메타데이터 저장
        # 훈련 데이터 기반으로 임계치 결정
        # 드리프트 기준값: 검증(홀드아웃 포트) 데이터의 '포트별 평균 score' 분포 (없으면 학습 데이터로, 출처 표시)
        baseline_src, baseline_path = ("val", v_path) if v_res else ("train", t_path)
        port_baseline = self._port_baseline(baseline_path)
        port_baseline["baseline_source"] = baseline_src
        meta = self._save_metadata(train_sequences, best_val_loss if val_loader else None, extra=port_baseline)
        
        # 레지스트리 갱신 (후보로 저장하거나, 활성 버전이 없으면 활성화)
        registry_entry = {
            "version": new_version_id,
            "trained_at": meta["trained_at"],
            "model_path": os.path.basename(v_model_path),
            "scaler_path": os.path.basename(v_scaler_path),
            "config_path": os.path.basename(v_model_path.replace('.pth', '.json')),
            "threshold": meta["threshold"],
            "final_val_loss": meta["final_val_loss"],
            "baseline_mse": meta.get("baseline_mse"),
            "samples_used": meta.get("samples_used"),
            "trigger": self.trigger,
            "training_window": self.window_info,
            "suspect_stats": self.suspect_stats,
            **{k: meta.get(k) for k in ("baseline_port_median", "baseline_port_p90", "baseline_ports", "baseline_source")},
        }
        if self.alert_policy:
            registry_entry["alert_policy"] = self.alert_policy
        with registry.transaction(model_dir, self.feature_type) as reg:
            self.activated = registry.add_version(reg, registry_entry, self.activate)
            
        if self.activated:
            print(f"[SUCCESS] {self.feature_type.capitalize()} model deployed. (Version: {new_version_id})")
        else:
            print(f"[SUCCESS] {self.feature_type.capitalize()} model saved as CANDIDATE. (Version: {new_version_id}, not active — promote to deploy)")
        return True

    def _port_baseline(self, data_path):
        """추론과 같은 지표(포트별 윈도우 마지막 시점 MSE 의 평균)의 포트 분포 -> median / p90.
        드리프트 감지가 같은 지표끼리 비교하도록 학습 시 저장한다 (lessons #19). 계산 불가 시 빈 dict."""
        try:
            df = pd.read_csv(data_path)
            df_clean = self.processor.preprocess(df, is_train=False)
            grouped = self.processor.create_sequences(df_clean, is_train=False) if df_clean is not None else {}
        except Exception as e:
            print(f"[!] Port baseline skipped: {e}")
            return {}
        self.model.eval()
        port_means = []
        with torch.no_grad():
            for seqs, _ in grouped.values():
                x = torch.from_numpy(seqs).float().to(self.device)
                out = self.model(x)
                mse = torch.mean((x[:, -1, :] - out[:, -1, :]) ** 2, dim=1)
                port_means.append(float(mse.mean().cpu()))
        if not port_means:
            return {}
        return {
            "baseline_port_median": float(np.median(port_means)),
            "baseline_port_p90": float(np.percentile(port_means, 90)),
            "baseline_ports": len(port_means),
        }

    def _save_metadata(self, sequences, val_loss=None, extra=None):
        """임계치와 학습 당시의 설정을 하나의 JSON으로 저장 (Inference 로드용)"""
        self.model.eval()
        mses = []
        loader = DataLoader(
            TensorDataset(torch.from_numpy(sequences).float()), 
            batch_size=self.config['batch_size']
            )
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                # 마지막 타임스텝의 MSE 기준
                diff = (inputs[:, -1, :] - outputs[:, -1, :]) ** 2
                mses.extend(torch.mean(diff, dim=1).cpu().numpy())
        
        threshold = float(np.percentile(mses, self.config['threshold_percentile']))
        # 드리프트 감지용 기준값: 추론 시 모니터링하는 score(마지막 시점 MSE)와 동일 지표의 평균
        baseline_mse = float(np.mean(mses))

        # 훈련된 임계치를 config 내부에도 업데이트 (통합 관리)
        self.config['threshold'] = threshold
        
        # 메타데이터 구성
        meta = {
            "model_type": "LSTM-Autoencoder",
            "feature_type": self.feature_type,
            "trained_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "config": self.config,
            "threshold": threshold,
            "baseline_mse": baseline_mse,
            "samples_used": len(sequences),
            "final_val_loss": float(val_loss) if val_loss is not None else None,
            "trigger": self.trigger,
            "training_window": self.window_info,
            "suspect_stats": self.suspect_stats,
        }
        if self.alert_policy:
            meta["alert_policy"] = self.alert_policy
        meta.update(extra or {})
        
        # 모델명과 동일하게 .json 확장자로 저장 (예: traffic_ae.pth -> traffic_ae.json)
        config_path = self.paths['model'].replace('.pth', '.json')
        with open(config_path, 'w') as f:
            json.dump(meta, f, indent=4)
            
        return meta
