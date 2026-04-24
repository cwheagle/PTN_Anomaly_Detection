import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        self.latent_linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        # 마지막 레이어의 hidden state 사용 (batch_size, hidden_dim)
        last_hidden = hidden[-1]
        latent = self.latent_linear(last_hidden)
        return latent

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers, seq_len):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: (batch_size, latent_dim)
        # Latent vector를 hidden dimension으로 확장
        h = self.latent_to_hidden(z)
        # Sequence length만큼 복제 (RepeatVector와 유사한 역할)
        # (batch_size, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        x = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        x, _ = self.lstm(x)
        out = self.output_linear(x)
        return out

class LSTMAutoencoder(nn.Module):
    """
    PTN 이상탐지를 위한 고도화된 LSTM-Autoencoder 모델
    """
    def __init__(self, config):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        self.num_layers = config['num_layers']
        self.window_size = config['window_size']

        self.encoder = Encoder(
            self.input_dim, self.hidden_dim, self.latent_dim, self.num_layers
        )
        self.decoder = Decoder(
            self.latent_dim, self.hidden_dim, self.input_dim, self.num_layers, self.window_size
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_reconstruction_loss(self, x):
        """추론 시 오차 계산을 위한 편의 메서드"""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # (batch_size, seq_len, input_dim) -> 각 샘플별 평균 오차 반환
            loss = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return loss
