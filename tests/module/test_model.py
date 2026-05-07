import pytest
import torch
from src.models.model import LSTMAutoencoder
from src.config import MODEL_CONFIG

@pytest.fixture
def model():
    return LSTMAutoencoder(MODEL_CONFIG)

def test_model_forward_shape(model):
    batch_size = 32
    seq_len = MODEL_CONFIG['window_size']
    input_dim = MODEL_CONFIG['input_dim']
    
    # 더미 입력 생성 (Batch, Seq, Feature)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x)
    
    # 출력 차원이 입력과 동일한지 확인
    assert output.shape == (batch_size, seq_len, input_dim)

def test_model_latent_extraction(model):
    batch_size = 8
    seq_len = MODEL_CONFIG['window_size']
    input_dim = MODEL_CONFIG['input_dim']
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 인코더만 테스트하여 Latent 공간 확인
    latent = model.encoder(x)
    
    assert latent.shape == (batch_size, MODEL_CONFIG['latent_dim'])

def test_reconstruction_loss_method(model):
    batch_size = 5
    seq_len = MODEL_CONFIG['window_size']
    input_dim = MODEL_CONFIG['input_dim']
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 오차 계산 메서드 테스트
    loss = model.get_reconstruction_loss(x)
    
    # 각 샘플별로 하나의 손실값이 나와야 함
    assert loss.shape == (batch_size,)
    assert torch.all(loss >= 0)
