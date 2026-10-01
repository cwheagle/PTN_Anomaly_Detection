"""
test_phase12_fixes.py — Phase 12 정합성 점검에서 수정한 항목의 회귀 테스트

검증 대상:
  1. DB 접속 실패 시 조용히 넘어가지 않고 예외 발생 (lessons #14)
  2. 드리프트 baseline이 baseline_mse를 우선 사용하고, 구버전 메타는 val_loss로 fallback
  3. Producer가 결측을 0이 아닌 null로 보내고, 해당 포트가 Consumer 경로에서 오탐 없이 처리됨
  4. Consumer 제어 채널: RCA 룰 리로드 메시지 처리

실행 방법:
  python -m pytest tests/test_phase12_fixes.py -v
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ──────────────────────────────────────────────
# 1. DB silent failure 방지
# ──────────────────────────────────────────────

def test_db_pool_failure_raises(monkeypatch):
    from src.config import DB_CONFIG
    from src.data.db_connector import DBConnector

    monkeypatch.setitem(DB_CONFIG, 'host', '127.0.0.1')
    monkeypatch.setitem(DB_CONFIG, 'port', 1)  # 아무도 listen하지 않는 포트

    with pytest.raises(RuntimeError, match="Connection pool init failed"):
        DBConnector()


# ──────────────────────────────────────────────
# 2. Drift baseline
# ──────────────────────────────────────────────

def _monitor_with_meta(tmp_path, meta):
    from src.pipeline.drift_monitor import DriftMonitor

    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta))
    monitor = DriftMonitor.__new__(DriftMonitor)  # __init__의 DB 접속 우회
    monitor._get_active_meta_path = lambda ft: str(meta_path)
    return monitor


def test_drift_baseline_prefers_baseline_mse(tmp_path):
    monitor = _monitor_with_meta(tmp_path, {"baseline_mse": 0.05, "final_val_loss": 3.0, "threshold": 0.27})
    assert monitor._get_single_baseline("traffic") == pytest.approx(0.05)


def test_drift_baseline_falls_back_to_val_loss_for_legacy_model(tmp_path):
    monitor = _monitor_with_meta(tmp_path, {"final_val_loss": 3.0, "threshold": 0.27})
    assert monitor._get_single_baseline("traffic") == pytest.approx(3.0)


# ──────────────────────────────────────────────
# 3. 결측 처리 (Producer -> Kafka JSON -> Consumer -> 추론)
# ──────────────────────────────────────────────

def _traffic_only_port(n=16):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'occur_date': pd.date_range('2026-09-30', periods=n, freq='15min'),
        'ip_addr': '10.0.0.1', 'cid': 1, 'lid': 1,
        'tx_packet': rng.integers(900, 1100, n),
        'rx_packet': rng.integers(900, 1100, n),
        'error_packet': 0,
        'tx_avg_power': np.nan, 'rx_avg_power': np.nan,   # optical 결측 (outer merge 결과)
    })


def _kafka_roundtrip(df):
    """Producer 변환(NaN -> None) 후 JSON 직렬화/역직렬화"""
    df = df.astype(object).where(df.notna(), None)
    recs = [{**r, 'occur_date': str(r['occur_date'])} for r in df.to_dict('records')]
    return [json.loads(json.dumps(r)) for r in recs]


def test_missing_optical_is_sent_as_null_not_zero():
    recs = _kafka_roundtrip(_traffic_only_port())
    assert recs[0]['tx_avg_power'] is None
    assert recs[0]['rx_avg_power'] is None
    assert recs[0]['tx_packet'] is not None


@pytest.mark.skipif(not os.path.exists("models/traffic_registry.json"),
                    reason="학습된 모델(models/)이 없어 추론 경로 검증을 건너뜀 (gitignore 대상)")
def test_traffic_only_port_processed_without_error():
    """optical이 전부 null인 포트도 Consumer 변환 후 추론이 에러 없이 끝나야 하고, optical 오탐이 없어야 함"""
    from src.pipeline.inference import AnomalyDetector
    from src.pipeline.kafka_consumer import METRIC_COLS

    df = pd.DataFrame(_kafka_roundtrip(_traffic_only_port()))
    df['occur_date'] = pd.to_datetime(df['occur_date'])
    assert df['tx_avg_power'].dtype == object          # 변환 전: 전부 None이면 object -> 변환이 필요한 이유
    for col in METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    result = AnomalyDetector().detect(df_traffic=df, df_optical=df, latest_only=True)

    assert result is not None and len(result) == 1
    assert not bool(result.get('is_optical_anomaly', pd.Series([False])).iloc[0])


# ──────────────────────────────────────────────
# 4. Consumer 제어 채널
# ──────────────────────────────────────────────

def test_consumer_control_channel_reloads_rules():
    from src.pipeline.kafka_consumer import PTNKafkaConsumer

    calls = []

    class _Engine:
        def reload_rules(self): calls.append('rules')

    class _Detector:
        rca_engine = _Engine()
        def reload_model(self, track): calls.append(f'model:{track}')

    consumer = PTNKafkaConsumer.__new__(PTNKafkaConsumer)  # Kafka/Redis 접속 우회
    consumer.detector = _Detector()

    consumer.handle_control_message({'type': 'message', 'data': json.dumps({'action': 'reload_rules'})})
    consumer.handle_control_message({'type': 'message', 'data': json.dumps({'action': 'reload', 'track': 'traffic'})})

    assert calls == ['rules', 'model:traffic']
