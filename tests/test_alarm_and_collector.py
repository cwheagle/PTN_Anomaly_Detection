"""
test_alarm_and_collector.py — 2차 정합성 점검 수정 항목의 회귀 테스트

검증 대상:
  1. 알람 발생/해제: 포트 단위로 판단 (다른 포트의 웹훅이 이 포트의 알람을 해제하면 안 됨)
  2. Consumer 웹훅 전송 조건: CRITICAL 복구 시 CLEAR가 전달되어야 함
  3. DataCollector: 학습 제외 구간은 명시적으로 주입될 때만 적용 (정답지 암묵 의존 제거)
  4. DBConnector: SQL 파라미터 바인딩 및 비정상 시각 입력 차단

실행 방법:
  python -m pytest tests/test_alarm_and_collector.py -v
"""
import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.alarm_tracker import plan_alarm_events


# ──────────────────────────────────────────────
# 1. 알람 발생/해제 (포트 단위)
# ──────────────────────────────────────────────

def _row(ip, label, occur='2026-10-01 10:00:00', cid=1, lid=1):
    return {'occur_date': occur, 'ip_addr': ip, 'cid': cid, 'lid': lid,
            'alarm_label': label, 'anomaly_reason': 'reason'}


def _apply(events, active):
    """호출자(alarm_callback)와 동일하게 상태 반영"""
    for ev, key, occur in events:
        if ev['type'] == 'ALARM':
            active[key] = occur
        else:
            active.pop(key, None)


def test_other_port_webhook_does_not_clear_active_alarm():
    """회귀: 포트 B의 웹훅(1건 payload)이 포트 A의 알람을 CLEAR 하던 버그"""
    active = {}
    _apply(plan_alarm_events([_row('A', 'CRITICAL')], active), active)
    events = plan_alarm_events([_row('B', 'CRITICAL')], active)
    _apply(events, active)

    assert [e['type'] for e, _, _ in events] == ['ALARM']          # CLEAR 없음
    assert set(k[0] for k in active) == {'A', 'B'}


def test_port_recovery_emits_clear_only_for_that_port():
    active = {('A', 1, 1): 't0', ('B', 1, 1): 't0'}

    events = plan_alarm_events([_row('A', 'NORMAL', occur='t1')], active)

    assert [(e['type'], e['ip_addr']) for e, _, _ in events] == [('CLEAR', 'A')]
    _apply(events, active)
    assert set(k[0] for k in active) == {'B'}


def test_duplicate_alarm_same_occur_date_is_suppressed():
    active = {('A', 1, 1): '2026-10-01 10:00:00'}
    assert plan_alarm_events([_row('A', 'CRITICAL')], active) == []


def test_new_occur_date_re_alarms():
    active = {('A', 1, 1): '2026-10-01 10:00:00'}
    events = plan_alarm_events([_row('A', 'CRITICAL', occur='2026-10-01 10:15:00')], active)
    assert [e['type'] for e, _, _ in events] == ['ALARM']


def test_non_critical_without_active_alarm_emits_nothing():
    assert plan_alarm_events([_row('A', 'MAJOR'), _row('B', 'NORMAL')], {}) == []


def test_state_not_mutated_by_planning():
    """발송 성공 시에만 상태를 반영하므로, 계획 단계에서는 active를 바꾸지 않아야 함"""
    active = {('A', 1, 1): 't0'}
    plan_alarm_events([_row('A', 'NORMAL'), _row('B', 'CRITICAL')], active)
    assert active == {('A', 1, 1): 't0'}


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
    assert notify(False, "NORMAL (DAMPENED)", 0) is True   # 기존 동작 유지


# ──────────────────────────────────────────────
# 3. DataCollector: 학습 제외 구간
# ──────────────────────────────────────────────

def _port_rows(lid, n=10):
    return pd.DataFrame({
        'occur_date': pd.date_range('2026-07-01 00:00', periods=n, freq='15min'),
        'ip_addr': '1.1.1.1', 'cid': 0, 'lid': lid, 'tx_packet': 1,
    })


def _exclusions(lid, start, end):
    return pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': lid,
                          'start_time': pd.Timestamp(start), 'failure_time': pd.Timestamp(end)}])


def test_load_exclusions_none_means_no_filtering():
    from src.data.data_collector import DataCollector
    assert DataCollector.load_exclusions(None) is None
    assert DataCollector.load_exclusions("") is None


def test_load_exclusions_missing_file_is_explicit_error(tmp_path):
    from src.data.data_collector import DataCollector
    with pytest.raises(FileNotFoundError):
        DataCollector.load_exclusions(str(tmp_path / "nope.csv"))


def test_filter_excluded_drops_only_matching_port_and_window():
    from src.data.data_collector import DataCollector
    df = pd.concat([_port_rows(1), _port_rows(2)], ignore_index=True)
    ex = _exclusions(1, '2026-07-01 00:30', '2026-07-01 01:00')   # 포트1의 3개 시점 (00:30, 00:45, 01:00 — 양끝 포함)

    out = DataCollector.filter_excluded(df, ex)

    assert len(out) == len(df) - 3
    assert len(out[out.lid == 2]) == 10                            # 다른 포트는 보존
    p1 = out[out.lid == 1]['occur_date']
    assert not ((p1 >= '2026-07-01 00:30') & (p1 <= '2026-07-01 01:00')).any()


def test_filter_excluded_without_exclusions_returns_input_unchanged():
    from src.data.data_collector import DataCollector
    df = _port_rows(1)
    assert DataCollector.filter_excluded(df, None) is df


def test_collector_does_not_read_simulator_ground_truth_implicitly():
    """회귀: 학습 코드가 tools/simulator/data/eval_dataset.csv 를 암묵적으로 읽지 않아야 함"""
    import inspect
    from src.data import data_collector
    assert 'eval_dataset' not in inspect.getsource(data_collector)
    assert 'tools' not in inspect.getsource(data_collector)


# ──────────────────────────────────────────────
# 4. DBConnector: SQL 파라미터 바인딩
# ──────────────────────────────────────────────

class _FakeCursor:
    def execute(self, *_a, **_k): pass
    def fetchone(self): return (1,)          # SHOW TABLES: 테이블 존재
    def close(self): pass


class _FakeConn:
    def cursor(self): return _FakeCursor()
    def close(self): pass


@pytest.fixture
def db_with_captured_queries(monkeypatch):
    from src.data import db_connector
    db = db_connector.DBConnector.__new__(db_connector.DBConnector)  # 실제 DB 접속 우회
    db.get_connection = lambda: _FakeConn()
    calls = []

    def fake_read_sql(query, conn, params=None):
        calls.append((query, params))
        return pd.DataFrame({'occur_date': [pd.Timestamp('2026-07-01')], 'ip_addr': ['x'], 'cid': [0], 'lid': [1]})

    monkeypatch.setattr(db_connector.pd, 'read_sql', fake_read_sql)
    return db, calls


def test_fetch_uses_bound_parameters_not_string_interpolation(db_with_captured_queries):
    db, calls = db_with_captured_queries

    db.fetch_traffic('2026-07-01 10:00:00', '2026-07-01 10:29:59')
    db.fetch_optical('2026-07-01 10:00:00', '2026-07-01 10:29:59')

    (tq, tp), (oq, op) = calls
    assert '2026-07-01' not in tq and '2026-07-01' not in oq          # 값이 SQL 문자열에 박히지 않음
    assert tq.count('%s') == 3 and tp[1:] == ('2026-07-01 10:00:00', '2026-07-01 10:29:59')
    assert oq.count('%s') == 2 and op == ('2026-07-01 10:00:00', '2026-07-01 10:29:59')


def test_fetch_rejects_non_datetime_input_before_any_query(db_with_captured_queries):
    db, calls = db_with_captured_queries

    with pytest.raises(Exception):
        db.fetch_traffic("2026-07-01' OR '1'='1", '2026-07-02')

    assert calls == []
