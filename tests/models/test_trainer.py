import json
import os

import numpy as np
import pandas as pd
import pytest

from src.config import PATHS
from src.models.trainer import Trainer


@pytest.fixture
def tiny_train_csv(tmp_path):
    """학습 가능한 최소 규모의 합성 traffic 데이터 (2개 포트 x 80시점)"""
    n = 80
    rng = np.random.default_rng(0)
    frames = []
    for lid in (1, 2):
        frames.append(pd.DataFrame({
            'occur_date': pd.date_range('2026-04-01', periods=n, freq='15min'),
            'ip_addr': '10.0.0.1', 'cid': 1, 'lid': lid,
            'tx_packet': rng.integers(900, 1100, n),
            'rx_packet': rng.integers(900, 1100, n),
            'error_packet': 0,
        }))
    path = tmp_path / "traffic_train.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """모델 산출물이 프로젝트의 models/ 를 건드리지 않도록 PATHS를 임시 경로로 교체"""
    paths = {
        'model': str(tmp_path / 'traffic_ae.pth'),
        'scaler': str(tmp_path / 'traffic_scaler.joblib'),
    }
    monkeypatch.setitem(PATHS, 'traffic', paths)
    return paths


def _train(csv_path):
    trainer = Trainer('traffic', config_override={'epochs': 1, 'batch_size': 16})
    assert trainer.train(train_path=csv_path, val_path=csv_path) is True
    return trainer


def test_trainer_does_not_mutate_global_paths(tiny_train_csv, isolated_paths):
    """회귀: 학습이 전역 PATHS를 버전 경로로 덮어쓰면 다음 학습 파일명이 v2_v3처럼 누적됨"""
    before = dict(isolated_paths)

    trainer = _train(tiny_train_csv)

    assert PATHS['traffic'] == before
    assert os.path.basename(trainer.paths['model']) == 'traffic_ae_v1.pth'


def test_registry_versions_increment_cleanly(tiny_train_csv, isolated_paths, tmp_path):
    """연속 학습 시 traffic_ae_v1 -> traffic_ae_v2 로 증가 (traffic_ae_v1_v2 금지)"""
    _train(tiny_train_csv)
    _train(tiny_train_csv)

    registry = json.load(open(tmp_path / 'traffic_registry.json'))
    assert registry['active_version'] == 'v2'
    assert [v['model_path'] for v in registry['versions']] == ['traffic_ae_v1.pth', 'traffic_ae_v2.pth']
    assert os.path.exists(tmp_path / 'traffic_ae_v2.pth')
    assert os.path.exists(tmp_path / 'traffic_ae_v2.json')


def test_metadata_contains_threshold_and_baseline(tiny_train_csv, isolated_paths, tmp_path):
    """드리프트 감지용 baseline_mse가 메타데이터에 저장되고 임계치 이하의 양수여야 함"""
    _train(tiny_train_csv)

    meta = json.load(open(tmp_path / 'traffic_ae_v1.json'))
    assert meta['config']['input_dim'] == 15
    assert 0 < meta['baseline_mse'] <= meta['threshold']


def test_saved_scaler_is_fit_on_train_data_not_validation(tiny_train_csv, isolated_paths, tmp_path):
    """회귀: 검증 데이터 로더가 스케일러를 다시 fit 하면, 학습 시퀀스와 저장된 스케일러의 스케일링이 달라진다
    (추론 시 학습과 다른 스케일링이 적용되고, 임계치도 어긋남)"""
    import joblib
    from src.data.data_processor import DataProcessor

    # 검증 데이터: 분포가 크게 다른(50배 큰 트래픽) 파일
    val = pd.read_csv(tiny_train_csv)
    val[['tx_packet', 'rx_packet']] = val[['tx_packet', 'rx_packet']] * 50
    val_path = tmp_path / "traffic_val.csv"
    val.to_csv(val_path, index=False)

    trainer = Trainer('traffic', config_override={'epochs': 1, 'batch_size': 16})
    assert trainer.train(train_path=tiny_train_csv, val_path=str(val_path)) is True

    # 기대값: 학습 데이터만으로 fit 한 스케일러
    ref = DataProcessor('traffic')
    ref.create_sequences(ref.preprocess(pd.read_csv(tiny_train_csv), is_train=True), is_train=True)

    saved = joblib.load(trainer.paths['scaler'])
    assert saved.center_ == pytest.approx(ref.scaler.center_)
    assert saved.scale_ == pytest.approx(ref.scaler.scale_)


def test_trainer_activate_false_saves_candidate_and_keeps_active_model(tiny_train_csv, isolated_paths, tmp_path):
    """API 경로(activate=False): 첫 학습은 자동 활성화, 이후 학습은 후보로만 저장되어 활성 모델이 바뀌지 않아야 함"""
    from src.models import registry

    first = Trainer('traffic', config_override={'epochs': 1, 'batch_size': 16}, activate=False)
    assert first.train(train_path=tiny_train_csv, val_path=tiny_train_csv) is True
    assert first.activated is True and first.version == 'v1'              # 활성 모델이 없으니 최초 학습은 활성화

    second = Trainer('traffic', config_override={'epochs': 1, 'batch_size': 16}, activate=False)
    assert second.train(train_path=tiny_train_csv, val_path=tiny_train_csv) is True
    assert second.activated is False and second.version == 'v2'

    reg = registry.load(str(tmp_path), 'traffic')
    assert reg['active_version'] == 'v1'                                  # 후보 학습이 활성 모델을 바꾸지 않음
    assert registry.status_of(reg, 'v2') == 'candidate'
    assert os.path.exists(tmp_path / 'traffic_ae_v2.pth')                 # 파일은 저장되어 승격 가능

    registry.promote(str(tmp_path), 'traffic', 'v2', force=True)
    assert registry.load(str(tmp_path), 'traffic')['active_version'] == 'v2'


def test_registry_entry_contains_baseline_and_samples(tiny_train_csv, isolated_paths, tmp_path):
    """승격 시 비교/표시에 쓰이도록 레지스트리 항목에 baseline_mse 와 samples_used 가 저장되어야 함"""
    from src.models import registry
    _train(tiny_train_csv)
    entry = registry.load(str(tmp_path), 'traffic')['versions'][0]
    assert entry['baseline_mse'] > 0 and entry['samples_used'] > 0
