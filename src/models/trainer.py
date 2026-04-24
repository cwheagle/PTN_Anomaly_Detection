import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json
from .model import LSTMAutoencoder
from src.data.data_processor import DataProcessor
from src.config import MODEL_CONFIG, PATHS

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMAutoencoder(MODEL_CONFIG).to(self.device)
        self.processor = DataProcessor()
        self.config = MODEL_CONFIG
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

    def load_and_prepare_data(self, file_path):
        """데이터를 로드하고 학습용 DataLoader를 생성합니다."""
        df = pd.read_csv(file_path)
        
        # 1. 전처리 및 스케일링
        df_clean = self.processor.preprocess(df)
        scaled_data = self.processor.scale_data(df_clean, is_train=True)
        
        # 2. 시퀀스 생성
        sequences = self.processor.create_sequences(scaled_data)
        
        # 3. PyTorch 텐서 변환
        tensor_x = torch.from_numpy(sequences).float()
        
        # 4. DataLoader 생성
        dataset = TensorDataset(tensor_x)
        # 임계치 계산 시 셔플링되지 않은 전체 데이터가 필요할 수 있으므로 저장
        self.full_train_loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def train(self, data_path):
        """모델 학습을 수행합니다."""
        print(f"Starting training on {self.device}...")
        train_loader = self.load_and_prepare_data(data_path)
        
        self.model.train()
        for epoch in range(self.config['epochs']):
            epoch_loss = 0
            for batch in train_loader:
                inputs = batch[0].to(self.device)
                
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}], Loss: {epoch_loss/len(train_loader):.6f}")
        
        self.save_model()
        self.processor.save_scaler()
        self.calculate_and_save_threshold()

    def calculate_and_save_threshold(self):
        """학습 데이터에 대한 오차 분포를 기반으로 임계치를 계산하고 저장합니다."""
        self.model.eval()
        mses = []
        with torch.no_grad():
            for batch in self.full_train_loader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                
                # 각 샘플의 마지막 시점 오차 계산 (추론 시와 동일한 방식)
                diff = (inputs[:, -1, :] - outputs[:, -1, :]) ** 2
                batch_mse = torch.mean(diff, dim=1).cpu().numpy()
                mses.extend(batch_mse)
        
        threshold = np.percentile(mses, self.config['threshold_percentile'])
        
        threshold_data = {
            "threshold": float(threshold),
            "percentile": self.config['threshold_percentile'],
            "mean_mse": float(np.mean(mses)),
            "std_mse": float(np.std(mses))
        }
        
        os.makedirs(os.path.dirname(PATHS['threshold_path']), exist_ok=True)
        with open(PATHS['threshold_path'], 'w') as f:
            json.dump(threshold_data, f, indent=4)
        
        print(f"Threshold saved to {PATHS['threshold_path']}: {threshold:.6f}")

    def save_model(self):
        """학습된 모델 가중치를 저장합니다."""
        os.makedirs(os.path.dirname(PATHS['model_save_path']), exist_ok=True)
        torch.save(self.model.state_dict(), PATHS['model_save_path'])
        print(f"Model saved to {PATHS['model_save_path']}")
