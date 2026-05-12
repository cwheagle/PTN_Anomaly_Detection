import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from .model import LSTMAutoencoder
from src.data.data_processor import DataProcessor
from src.config import MODEL_CONFIG, PATHS, FEATURE_GROUPS

class Trainer:
    def __init__(self, feature_type='traffic', config_override=None):
        self.feature_type = feature_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 설정값 병합 (기본값 + 외부 주입값)
        self.config = MODEL_CONFIG.copy()
        if config_override:
            self.config.update(config_override)
        
        self.config['input_dim'] = len(FEATURE_GROUPS[feature_type])
        
        # 2. 주입된 설정값으로 모델 및 프로세서 초기화
        self.model = LSTMAutoencoder(self.config).to(self.device)
        self.processor = DataProcessor(feature_type, config=self.config)
        self.paths = PATHS[feature_type]
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

    def _prepare_loader(self, data_path):
        if not os.path.exists(data_path):
            print(f"    [ERROR] Data file not found: {data_path}")
            return None
            
        df = pd.read_csv(data_path)
        df_clean = self.processor.preprocess(df, is_train=True)
        if df_clean is None: return None
        
        sequences = self.processor.create_sequences(df_clean, is_train=True)
        if len(sequences) == 0: return None
        
        dataset = TensorDataset(torch.from_numpy(sequences).float())
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True), sequences

    def train(self, train_path=None, val_path=None):
        t_path = train_path or f"data/{self.feature_type}_train.csv"
        v_path = val_path or f"data/{self.feature_type}_test.csv"
        print(f"[*] Training [{self.feature_type}] specialist model...")
        
        # 1. 데이터 로더 준비
        t_res = self._prepare_loader(t_path)
        v_res = self._prepare_loader(v_path) # 검증 데이터
        
        if not t_res:
            print(f"    [SKIP] Insufficient training data for {self.feature_type}")
            return False
            
        train_loader, train_sequences = t_res
        val_loader = v_res[0] if v_res else None
        
        start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # --- 훈련 단계 ---
            self.model.train()
            train_loss_sum = 0
            for batch in train_loader:
                inputs = batch[0].to(self.device)
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
                        inputs = batch[0].to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, inputs)
                        val_loss_sum += loss.item()
                avg_val_loss = val_loss_sum / len(val_loader)
                
                # 최적 모델 체크 (필요시 저장 로직 확장 가능)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

            # 로그 출력
            epoch_duration = time.time() - epoch_start
            log_msg = f"    Epoch [{epoch+1}/{self.config['epochs']}] Train Loss: {avg_train_loss:.6f}"
            if avg_val_loss is not None:
                log_msg += f", Val Loss: {avg_val_loss:.6f}"
            log_msg += f" ({epoch_duration:.1f}s)"
            print(log_msg)
        
        # 3. 모델 및 스케일러 저장
        os.makedirs(os.path.dirname(self.paths['model']), exist_ok=True)
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

if __name__ == "__main__":
    # 직접 실행 시 기본값으로 학습
    for ft in ['traffic', 'optical']:
        Trainer(ft).train()
