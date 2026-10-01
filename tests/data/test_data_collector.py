"""
DataCollector 검증: 학습 제외 구간은 명시적으로 주입될 때만 적용 (평가 정답지 암묵 의존 금지)
"""
import pandas as pd
import pytest



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
    src_text = inspect.getsource(data_collector)
    assert 'eval_dataset' not in src_text
    assert 'validation' not in src_text and 'tools' not in src_text
