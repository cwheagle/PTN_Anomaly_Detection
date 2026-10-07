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


def test_metadata_and_registry_record_port_baseline_trigger_and_window(tiny_train_csv, isolated_paths, tmp_path):
    """T-T1: 드리프트 기준값(포트별 평균 score 분포)과 출처 메타데이터가 메타 JSON 과 레지스트리에 기록"""
    from src.models import registry
    window = {'start': '2026-04-01 00:00:00', 'end': '2026-04-01 20:00:00', 'val_port_fraction': 0.1, 'salt': 'ptn'}
    stats = {'rows': 160, 'excluded_rows': 8, 'fraction': 0.05, 'dropped_ports': [], 'by_source': {'alarm': 8}}
    trainer = Trainer('traffic', config_override={'epochs': 1, 'batch_size': 16},
                      trigger='drift', window_info=window, suspect_stats=stats)
    assert trainer.train(train_path=tiny_train_csv, val_path=tiny_train_csv) is True

    meta = json.load(open(tmp_path / 'traffic_ae_v1.json'))
    assert meta['trigger'] == 'drift' and meta['training_window'] == window and meta['suspect_stats'] == stats
    assert meta['baseline_ports'] == 2 and meta['baseline_source'] == 'val'
    assert 0 < meta['baseline_port_median'] <= meta['baseline_port_p90']
    assert 'baseline_mse' in meta                                           # 하위 호환 유지

    entry = registry.load(str(tmp_path), 'traffic')['versions'][0]
    for key in ('trigger', 'training_window', 'suspect_stats', 'baseline_port_median', 'baseline_port_p90',
                'baseline_ports', 'baseline_source'):
        assert entry[key] == meta[key]


def test_baseline_falls_back_to_train_data_when_no_validation_file(tiny_train_csv, isolated_paths, tmp_path):
    trainer = Trainer('traffic', config_override={'epochs': 1, 'batch_size': 16})
    assert trainer.train(train_path=tiny_train_csv, val_path=str(tmp_path / 'missing.csv')) is True
    meta = json.load(open(tmp_path / 'traffic_ae_v1.json'))
    assert meta['baseline_source'] == 'train' and meta['baseline_port_median'] > 0
    assert meta['trigger'] == 'manual'


def test_port_baseline_matches_per_port_mean_of_last_step_mse(tiny_train_csv, isolated_paths):
    """기준값의 지표가 추론 score 와 같은지(포트별 윈도우 마지막 시점 MSE 평균) 직접 재계산해 확인"""
    import torch
    trainer = _train(tiny_train_csv)
    base = trainer._port_baseline(tiny_train_csv)
    clean = trainer.processor.preprocess(pd.read_csv(tiny_train_csv), is_train=False)
    grouped = trainer.processor.create_sequences(clean, is_train=False)
    means = []
    with torch.no_grad():
        for seqs, _ in grouped.values():
            x = torch.from_numpy(seqs).float().to(trainer.device)
            means.append(float(((x[:, -1] - trainer.model(x)[:, -1]) ** 2).mean(dim=1).mean()))
    assert base['baseline_port_median'] == pytest.approx(float(np.median(means)), rel=1e-5)
    assert base['baseline_port_p90'] == pytest.approx(float(np.percentile(means, 90)), rel=1e-5)


def test_alert_policy_is_recorded_in_meta_and_registry_only_when_given(tiny_train_csv, isolated_paths, tmp_path):
    """T-P3-T1: alert_policy 지정 시 메타·레지스트리 엔트리에 기록, 미지정이면 어디에도 기록하지 않음(= 기본 정책)"""
    policy = {"preset": "precision", "sigma_k": 2.0, "dyn_cap": 1.2, "threshold_scale": 3.0, "severity_decay": 0.5,
              "dampening_steps": {"1": 6, "2": 4, "3": 3}}
    plain = Trainer('traffic', config_override={'epochs': 1, 'batch_size': 16}, activate=False)
    assert plain.train(train_path=tiny_train_csv, val_path=tiny_train_csv) is True
    with_policy = Trainer('traffic', config_override={'epochs': 1, 'batch_size': 16}, activate=False, alert_policy=policy)
    assert with_policy.train(train_path=tiny_train_csv, val_path=tiny_train_csv) is True

    assert "alert_policy" not in json.load(open(tmp_path / 'traffic_ae_v1.json'))
    assert json.load(open(tmp_path / 'traffic_ae_v2.json'))["alert_policy"] == policy
    versions = {v["version"]: v for v in json.load(open(tmp_path / 'traffic_registry.json'))["versions"]}
    assert "alert_policy" not in versions["v1"] and versions["v2"]["alert_policy"] == policy
