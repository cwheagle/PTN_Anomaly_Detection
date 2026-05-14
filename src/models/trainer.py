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
from .model import LSTMAutoencoder
from src.data.data_processor import DataProcessor
from src.config import MODEL_CONFIG, PATHS, FEATURE_GROUPS

class Trainer:
    def __init__(self, feature_type='traffic', config_override=None, progress_callback=None):
        self.feature_type = feature_type
        self.progress_callback = progress_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stop_requested = False
        self.early_stopped = False # 조기 종료 여부 플래그 추가

        # 1. 설정값 병합 (기본값 + 외부 주입값)
        self.config = MODEL_CONFIG.copy()
        if config_override:
            self.config.update(config_override)
        
        self.config['input_dim'] = len(FEATURE_GROUPS[feature_type])
        
        # 2. 주입된 설정값으로 모델 및 프로세서 초기화
        self.model = LSTMAutoencoder(self.config).to(self.device)
        
        # [Blackwell 최적화] PyTorch 2.0+ 및 CUDA 환경에서 컴파일 적용 (이식성 유지)
        if hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                # 첫 실행 시 컴파일 시간이 걸리지만 이후 속도가 비약적으로 향상됨
                print(f"[*] Optimizing model with torch.compile (mode: max-autotune)...")
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception as e:
                print(f"[!] torch.compile failed, falling back to eager mode: {e}")

        self.processor = DataProcessor(feature_type, config=self.config)
        self.paths = PATHS[feature_type]
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

    def stop(self):
        """학습 중지 요청"""
        self.stop_requested = True
        print(f"[*] Stop requested for {self.feature_type} trainer.")

    def _prepare_loader(self, data_path):
        if not os.path.exists(data_path):
            print(f"    [ERROR] Data file not found: {data_path}")
            return None
            
        df = pd.read_csv(data_path)
        if self.stop_requested: return None

        df_clean = self.processor.preprocess(df, is_train=True)
        if df_clean is None or self.stop_requested: return None
        
        sequences = self.processor.create_sequences(df_clean, is_train=True)
        if len(sequences) == 0 or self.stop_requested: return None
        
        dataset = TensorDataset(torch.from_numpy(sequences).float())
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True), sequences

    def train(self, train_path=None, val_path=None):
        t_path = train_path or f"data/{self.feature_type}_train.csv"
        v_path = val_path or f"data/{self.feature_type}_test.csv"
        print(f"[*] Training [{self.feature_type}] specialist model...")
        
        # 1. 데이터 로더 준비
        t_res = self._prepare_loader(t_path)
        if self.stop_requested: return False
        
        v_res = self._prepare_loader(v_path) # 검증 데이터
        if self.stop_requested: return False
        
        if not t_res:
            print(f"    [SKIP] Insufficient training data for {self.feature_type}")
            return False
            
        train_loader, train_sequences = t_res
        val_loader = v_res[0] if v_res else None
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = self.config['patience']
        no_improve_count = 0
        
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

            # [Blackwell 최적화] FP8 Autocast 컨텍스트 준비
            if fp8_autocast and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
                fp8_ctx = fp8_autocast(enabled=True)
            else:
                fp8_ctx = torch.amp.autocast('cuda', enabled=False)

            for batch in train_loader:
                if self.stop_requested: return False # 배치 단위 중지 체크
                
                inputs = batch[0].to(self.device)
                
                with fp8_ctx:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, inputs)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss_sum += loss.item()
            
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
                    print(f"    [SAVE] Best model updated at epoch {epoch+1} (Val Loss: {best_val_loss:.6f})")
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"    [EARLY STOP] No improvement for {patience} epochs. Stopping at epoch {epoch+1}")
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
                print(f"    [STOP] Training interrupted by user at epoch {epoch+1}")
                return False
        
        # 3. 모델 및 스케일러 저장
        os.makedirs(os.path.dirname(self.paths['model']), exist_ok=True)
        
        # 최적의 모델 상태가 있다면 그것을 로드한 뒤 저장, 없으면 현재 상태 저장
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"[*] Deploying best model (Val Loss: {best_val_loss:.6f})")
        
        torch.save(self.model.state_dict(), self.paths['model'])
        self.processor.save_scaler(self.paths['scaler'])
        
        # 4. 임계치 산출 및 통합 메타데이터 저장
        # 훈련 데이터 기반으로 임계치 결정
        self._save_metadata(train_sequences, best_val_loss if val_loader else None)
        print(f"    [SUCCESS] {self.feature_type.capitalize()} model deployed.")
        return True

    def _save_metadata(self, sequences, val_loss=None):
        """임계치와 학습 당시의 설정을 하나의 JSON으로 저장 (Inference 로드용)"""
        self.model.eval()
        mses = []
        loader = DataLoader(TensorDataset(torch.from_numpy(sequences).float()), batch_size=32)
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                # 마지막 타임스텝의 MSE 기준
                diff = (inputs[:, -1, :] - outputs[:, -1, :]) ** 2
                mses.extend(torch.mean(diff, dim=1).cpu().numpy())
        
        threshold = float(np.percentile(mses, self.config['threshold_percentile']))
        
        # 훈련된 임계치를 config 내부에도 업데이트 (통합 관리)
        self.config['threshold'] = threshold
        
        # 메타데이터 구성
        meta = {
            "model_type": "LSTM-Autoencoder",
            "feature_type": self.feature_type,
            "trained_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "config": self.config,
            "threshold": threshold,
            "samples_used": len(sequences),
            "final_val_loss": float(val_loss) if val_loss is not None else None
        }
        
        # 모델명과 동일하게 .json 확장자로 저장 (예: traffic_ae.pth -> traffic_ae.json)
        config_path = self.paths['model'].replace('.pth', '.json')
        with open(config_path, 'w') as f:
            json.dump(meta, f, indent=4)
