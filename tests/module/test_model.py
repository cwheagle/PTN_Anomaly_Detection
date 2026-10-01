import pytest
import torch
from src.models.model import LSTMAutoencoder
from src.config import MODEL_CONFIG
from src.data.data_processor import DataProcessor

# input_dim은 config에 고정되지 않고 파생 변수 포함 컬럼 수로 동적 결정된다 (Phase 11)
INPUT_DIM = len(DataProcessor('traffic').extended_feature_cols)
CONFIG = {**MODEL_CONFIG, 'input_dim': INPUT_DIM}

@pytest.fixture
def model():
    return LSTMAutoencoder(CONFIG)

def test_input_dim_matches_derived_features():
    """traffic 3개 + optical 2개 원본 변수 x (원본, ma_4, ma_16, var_4, lag_1) = 15 / 10"""
    assert len(DataProcessor('traffic').extended_feature_cols) == 15
    assert len(DataProcessor('optical').extended_feature_cols) == 10

def test_model_forward_shape(model):
    batch_size = 32
    seq_len = MODEL_CONFIG['window_size']
    input_dim = INPUT_DIM
    
    # 더미 입력 생성 (Batch, Seq, Feature)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x)
    
    # 출력 차원이 입력과 동일한지 확인
    assert output.shape == (batch_size, seq_len, input_dim)

def test_model_latent_extraction(model):
    batch_size = 8
    seq_len = MODEL_CONFIG['window_size']
    input_dim = INPUT_DIM
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 인코더만 테스트하여 Latent 공간 확인
    latent = model.encoder(x)
    
    assert latent.shape == (batch_size, MODEL_CONFIG['latent_dim'])

def test_reconstruction_loss_method(model):
    batch_size = 5
    seq_len = MODEL_CONFIG['window_size']
    input_dim = INPUT_DIM
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 오차 계산 메서드 테스트
    loss = model.get_reconstruction_loss(x)
    
    # 각 샘플별로 하나의 손실값이 나와야 함
    assert loss.shape == (batch_size,)
    assert torch.all(loss >= 0)
