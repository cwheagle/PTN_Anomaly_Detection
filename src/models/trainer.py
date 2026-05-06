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
from src.config import MODEL_CONFIG, PATHS, FEATURE_GROUPS

class Trainer:
    def __init__(self, feature_type='traffic'):
        self.feature_type = feature_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config = MODEL_CONFIG.copy()
        config['input_dim'] = len(FEATURE_GROUPS[feature_type])
        
        self.model = LSTMAutoencoder(config).to(self.device)
        self.processor = DataProcessor(feature_type)
        self.paths = PATHS[feature_type]
        self.config = config
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])

    def _prepare_loader(self, data_path):
        df = pd.read_csv(data_path)
        df_clean = self.processor.preprocess(df, is_train=True)
        if df_clean is None: return None
        
        sequences = self.processor.create_sequences(df_clean, is_train=True)
        if len(sequences) == 0: return None
        
        dataset = TensorDataset(torch.from_numpy(sequences).float())
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True), sequences

    def train(self, data_path=None):
        path = data_path or f"data/{self.feature_type}_train.csv"
        print(f"[*] Training [{self.feature_type}] specialist model...")
        
        result = self._prepare_loader(path)
        if not result:
            print(f"    [SKIP] Insufficient data for {self.feature_type}")
            return
            
        loader, sequences = result
        self.model.train()
        for epoch in range(self.config['epochs']):
            loss_sum = 0
            for batch in loader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch [{epoch+1}/{self.config['epochs']}], Loss: {loss_sum/len(loader):.6f}")
        
        # 저장
        torch.save(self.model.state_dict(), self.paths['model'])
        self.processor.save_scaler(self.paths['scaler'])
        self._save_threshold(sequences)
        print(f"    [SUCCESS] {self.feature_type.capitalize()} model deployed.")

    def _save_threshold(self, sequences):
        self.model.eval()
        mses = []
        loader = DataLoader(TensorDataset(torch.from_numpy(sequences).float()), batch_size=32)
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                diff = (inputs[:, -1, :] - outputs[:, -1, :]) ** 2
                mses.extend(torch.mean(diff, dim=1).cpu().numpy())
        
        threshold = np.percentile(mses, self.config['threshold_percentile'])
        with open(self.paths['threshold'], 'w') as f:
            json.dump({"threshold": float(threshold), "percentile": self.config['threshold_percentile']}, f)

if __name__ == "__main__":
    for ft in ['traffic', 'optical']:
        Trainer(ft).train()
