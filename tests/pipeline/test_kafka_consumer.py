"""
Kafka Consumer 경로 검증: 결측(null) 처리, 제어 채널(reload), 웹훅 전송 조건
"""
import json
import os

import numpy as np
import pandas as pd
import pytest



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


# ──────────────────────────────────────────────
# 2. Consumer 웹훅 전송 조건
# ──────────────────────────────────────────────

def test_consumer_notifies_on_critical_recovery():
    from src.pipeline.kafka_consumer import PTNKafkaConsumer
    notify = PTNKafkaConsumer.should_notify

    assert notify(False, "NORMAL", 0) is False        # 평시: 전송 안 함
    assert notify(False, "NORMAL", 3) is True         # CRITICAL -> 정상 복구: CLEAR 전달
    assert notify(False, "NORMAL", 2) is False        # MAJOR -> 정상: 해제할 활성 알람 없음
    assert notify(True, "CRITICAL", 0) is True        # 이상 발생
    assert notify(False, "NORMAL (DAMPENED)", 0) is False  # 회귀: 댐프닝으로 억제된 행은 알람이 아님 (로그/웹훅 대상 아님)
    assert notify(False, "NORMAL (DAMPENED)", 3) is True   # 단, 직전이 CRITICAL 이었다면 해제(CLEAR) 전달 필요
    assert notify(False, "MAJOR", 0) is True
