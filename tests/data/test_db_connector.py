"""
DBConnector 검증: 접속 실패 시 명시적 예외(lessons #14), SQL 파라미터 바인딩
"""
import pandas as pd
import pytest



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
