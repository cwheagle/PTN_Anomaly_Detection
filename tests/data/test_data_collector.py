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


# ──────────────────────────────────────────────
# P1-1: 포트 분할 모드 / 구간 조인 / 의심 구간 자동 제외
# ──────────────────────────────────────────────
class _FakeDB:
    """DataCollector 가 쓰는 DB 인터페이스만 흉내 (DB 접속 없이 검증)"""
    def __init__(self, traffic=None, optical=None, alarms=None):
        self.traffic, self.optical = traffic, optical
        self.alarms = alarms if alarms is not None else pd.DataFrame(
            columns=['occur_date', 'ip_addr', 'cid', 'lid', 'alarm_level'])
        self.alarm_calls = []

    def fetch_traffic(self, start, end, stop_checker=None):
        return self.traffic

    def fetch_optical(self, start, end, stop_checker=None):
        return self.optical

    def fetch_alarm_rows(self, start, end, min_level=1):
        self.alarm_calls.append((start, end, min_level))
        return self.alarms


def _collector(db):
    from src.data.data_collector import DataCollector
    c = DataCollector.__new__(DataCollector)     # __init__ 의 DB 접속 우회
    c.db = db
    return c


def _traffic_ports(n_ports=60, n=200):
    frames = []
    for lid in range(n_ports):
        frames.append(pd.DataFrame({
            'occur_date': pd.date_range('2026-07-01', periods=n, freq='15min'),
            'ip_addr': '1.1.1.1', 'cid': 0, 'lid': lid,
            'tx_packet': 1000, 'rx_packet': 1000, 'error_packet': 0}))
    return pd.concat(frames, ignore_index=True)


def test_port_split_mode_train_and_test_ports_disjoint_with_same_period(tmp_path):
    """T-C1"""
    db = _FakeDB(traffic=_traffic_ports())
    out = _collector(db).collect_and_save('2026-07-01', '2026-07-03', feature_type='traffic',
                                          output_dir=str(tmp_path), val_port_fraction=0.2)
    train, test = pd.read_csv(tmp_path / 'traffic_train.csv'), pd.read_csv(tmp_path / 'traffic_test.csv')
    assert out['traffic'] == {'train': len(train), 'test': len(test)}
    assert len(test) > 0 and len(train) > len(test)
    assert not set(train.lid) & set(test.lid)
    assert train.occur_date.min() == test.occur_date.min() and train.occur_date.max() == test.occur_date.max()


def test_port_split_mode_accepts_datetime_bounds(tmp_path):
    db = _FakeDB(traffic=_traffic_ports(20, 100))
    out = _collector(db).collect_and_save(pd.Timestamp('2026-07-01 00:00'), pd.Timestamp('2026-07-01 12:00'),
                                          feature_type='traffic', output_dir=str(tmp_path), val_port_fraction=0.3)
    assert out['traffic']['train'] > 0


def test_filter_excluded_matches_naive_implementation_on_random_intervals():
    """T-C2: 구간 조인 방식이 기존 순회 구현과 같은 결과"""
    import numpy as np
    from src.data.data_collector import DataCollector
    rng = np.random.default_rng(3)
    df = pd.concat([_port_rows(l, 400) for l in range(1, 8)], ignore_index=True)
    ex = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': int(rng.integers(1, 9)),
                        'start_time': pd.Timestamp('2026-07-01') + pd.Timedelta(minutes=15 * int(s)),
                        'failure_time': pd.Timestamp('2026-07-01') + pd.Timedelta(minutes=15 * (int(s) + int(rng.integers(0, 30))))}
                       for s in rng.integers(0, 390, 80)])
    drop = pd.Series(False, index=df.index)           # 기존 구현 그대로
    for _, e in ex.iterrows():
        drop |= ((df['ip_addr'] == e['ip_addr']) & (df['cid'] == e['cid']) & (df['lid'] == e['lid']) &
                 (df['occur_date'] >= e['start_time']) & (df['occur_date'] <= e['failure_time']))
    out = DataCollector.filter_excluded(df, ex)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df[~drop].reset_index(drop=True))


def test_legacy_failure_time_csv_and_end_time_csv_both_load(tmp_path):
    """T-C3: 구형(failure_time) / 신형(end_time) CSV 모두 제외 구간으로 사용 가능"""
    from src.data.data_collector import DataCollector
    old = tmp_path / 'old.csv'
    old.write_text("ip_addr,cid,lid,start_time,failure_time\n1.1.1.1,0,1,2026-07-01 00:30:00,2026-07-01 01:00:00\n")
    new = tmp_path / 'new.csv'
    new.write_text("ip_addr,cid,lid,start_time,end_time\n1.1.1.1,0,1,2026-07-01 00:30:00,2026-07-01 01:00:00\n")
    df = _port_rows(1)
    for path in (old, new):
        out = DataCollector.filter_excluded(df, DataCollector.load_exclusions(str(path)))
        assert len(out) == len(df) - 3


def test_date_split_path_unchanged_without_new_options(tmp_path):
    """T-C3: 기존 날짜 분할 경로(validation/ 스크립트 호환)는 새 인자 없이 그대로 동작"""
    df = pd.concat([_traffic_ports(3, 96 * 4)], ignore_index=True)     # 4일치
    out = _collector(_FakeDB(traffic=df)).collect_and_save(
        '2026-07-01', '2026-07-02', '2026-07-03', '2026-07-04', feature_type='traffic', output_dir=str(tmp_path))
    train, test = pd.read_csv(tmp_path / 'traffic_train.csv'), pd.read_csv(tmp_path / 'traffic_test.csv')
    assert out['traffic'] == {'train': 3 * 96 * 2, 'test': 3 * 96 * 2}
    assert pd.to_datetime(train.occur_date).max() < pd.Timestamp('2026-07-03')
    assert pd.to_datetime(test.occur_date).min() >= pd.Timestamp('2026-07-03')


def test_collector_unions_injected_exclusions_with_exclude_path(tmp_path):
    csv = tmp_path / 'ex.csv'
    csv.write_text("ip_addr,cid,lid,start_time,end_time\n1.1.1.1,0,0,2026-07-01 00:00:00,2026-07-01 00:45:00\n")   # 4행
    inj = pd.DataFrame([{'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 0, 'start_time': pd.Timestamp('2026-07-01 02:00'),
                         'end_time': pd.Timestamp('2026-07-01 02:15')}])                                         # 2행
    db = _FakeDB(traffic=_traffic_ports(1, 96))
    out = _collector(db).collect_and_save('2026-07-01', '2026-07-01', '2026-07-01', '2026-07-01',
                                          feature_type='traffic', output_dir=str(tmp_path),
                                          exclude_path=str(csv), exclusions=inj)
    assert out['traffic']['train'] == 96 - 6


def test_suspect_policy_excludes_alarm_history_and_reports_stats(tmp_path):
    from src.pipeline.retrain_policy import RetrainPolicy
    alarms = pd.DataFrame({'occur_date': pd.date_range('2026-07-01 12:00', periods=3, freq='15min'),
                           'ip_addr': '1.1.1.1', 'cid': 0, 'lid': 5, 'alarm_level': 2})
    db = _FakeDB(traffic=_traffic_ports(40, 200), alarms=alarms)
    pol = RetrainPolicy(exclude_from_rules=False)
    out = _collector(db).collect_and_save('2026-07-01', '2026-07-03', feature_type='traffic',
                                          output_dir=str(tmp_path), val_port_fraction=0.0, suspect_policy=pol)
    stats = out['traffic']['suspect_stats']
    # 알람 3행 + 앞 16 + 뒤 4 스텝 = 최대 23행 제외 (포트 5 한 곳)
    assert stats['excluded_rows'] == 23 and stats['by_source'] == {'alarm': 23}
    assert db.alarm_calls and db.alarm_calls[0][2] == pol.alarm_min_level
    train = pd.read_csv(tmp_path / 'traffic_train.csv')
    assert len(train) == 40 * 200 - 23


def test_suspect_fraction_over_limit_skips_track_without_writing_csv(tmp_path):
    from src.pipeline.retrain_policy import RetrainPolicy
    n = 200
    raw = _traffic_ports(10, n)
    raw.loc[raw.occur_date >= '2026-07-01 12:00', 'error_packet'] = 500        # 모든 포트가 12시 이후 지속 오류
    out = _collector(_FakeDB(traffic=raw)).collect_and_save(
        '2026-07-01', '2026-07-03', feature_type='traffic', output_dir=str(tmp_path),
        val_port_fraction=0.1, suspect_policy=RetrainPolicy(exclude_from_alarms=False, max_excluded_fraction=0.05))
    assert out['traffic']['skipped'].startswith('suspect_fraction=')
    assert out['traffic']['suspect_stats']['fraction'] > 0.05
    assert not (tmp_path / 'traffic_train.csv').exists()


def test_alarm_history_fetch_failure_propagates(tmp_path):
    """lessons #14: 알람 이력을 못 읽으면 조용히 넘기지 않고 중단 (이력 없이 학습하면 장애가 정상으로 학습됨)"""
    from src.pipeline.retrain_policy import RetrainPolicy

    class Broken(_FakeDB):
        def fetch_alarm_rows(self, *a, **k):
            raise RuntimeError("db down")

    with pytest.raises(RuntimeError):
        _collector(Broken(traffic=_traffic_ports(3, 50))).collect_and_save(
            '2026-07-01', '2026-07-02', feature_type='traffic', output_dir=str(tmp_path),
            val_port_fraction=0.1, suspect_policy=RetrainPolicy())
