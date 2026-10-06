"""
DriftMonitor 검증: baseline 지표 선택 (baseline_mse 우선, 구버전 메타는 val_loss fallback)
"""
import json

import pytest



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
# P1-1: 포트별 신호 / 광역·국소 구분 / 활성화 직후 / legacy 기준값
# ──────────────────────────────────────────────
from datetime import datetime, timedelta

import pandas as pd

from src.pipeline.drift_monitor import DriftMonitor, evaluate_track
from src.pipeline.retrain_policy import RetrainPolicy

BASELINE = {"port_median": 0.05, "port_p90": 0.08, "baseline_mse": 0.05, "legacy": False}


def _ports(scores, n=96):
    return pd.DataFrame({"ip_addr": [f"10.0.0.{i}" for i in range(len(scores))], "cid": 1, "lid": 1,
                         "mean_score": scores, "n": n})


def _normal_scores(k=40):
    return [0.04 + 0.04 * i / (k - 1) for i in range(k)]            # 0.04~0.08: p90 이하, 중앙값 0.06 (비율 1.2)


def test_evaluate_normal_distribution_is_none():
    r = evaluate_track(_ports(_normal_scores()), BASELINE, RetrainPolicy())
    assert r["kind"] == "none" and r["status"] == "normal" and r["ports"] == 40
    assert r["median_ratio"] == pytest.approx(1.2) and r["drifted_port_fraction"] == 0.0


def test_few_ports_spiking_is_localized_not_drift():
    """T-D2: 소수 포트 급등 = 장애 의심. drift 로 보지 않음"""
    scores = _normal_scores() + [0.9, 1.2]
    r = evaluate_track(_ports(scores), BASELINE, RetrainPolicy())
    assert r["kind"] == "localized" and r["status"] == "localized" and r["localized_ports"] == 2
    assert [p["mean_score"] for p in r["top_ports"][:2]] == [1.2, 0.9]


def test_most_ports_shifting_is_widespread():
    """T-D3: 다수 포트의 이동 = 정상 패턴 변화"""
    r = evaluate_track(_ports([0.12] * 40), BASELINE, RetrainPolicy())
    assert r["kind"] == "widespread" and r["status"] == "drifted" and r["median_ratio"] == pytest.approx(2.4)
    assert r["drift_ratio"] == r["median_ratio"]                    # UI 호환 키
    assert r["baseline_mse"] == 0.05 and r["mean_mse"] == pytest.approx(0.12)


def test_widespread_by_port_fraction_even_if_median_is_normal():
    scores = [0.05] * 26 + [0.1] * 14                               # 중앙값 비율 1.0 이지만 35% 포트가 p90 초과
    r = evaluate_track(_ports(scores), BASELINE, RetrainPolicy())
    assert r["median_ratio"] == pytest.approx(1.0) and r["drifted_port_fraction"] == pytest.approx(0.35)
    assert r["kind"] == "widespread"


def test_scores_of_zero_and_sparse_ports_are_excluded():
    """T-D1(판정부): 0 점수(트랙 없음)와 표본이 부족한 포트는 판정에 쓰지 않는다"""
    stats = pd.concat([_ports([0.0] * 30), _ports([0.06] * 10), _ports([5.0] * 3, n=5)], ignore_index=True)
    r = evaluate_track(stats, BASELINE, RetrainPolicy())
    assert r["ports"] == 10 and r["kind"] == "none"


def test_insufficient_and_missing_baseline_statuses():
    assert evaluate_track(_ports([0.0] * 5), BASELINE, RetrainPolicy())["status"] == "insufficient_data"
    assert evaluate_track(_ports([0.1] * 5), {"legacy": True}, RetrainPolicy())["status"] == "no_baseline"


def test_legacy_baseline_is_flagged_and_approximated():
    """T-D5"""
    legacy = {"baseline_mse": 0.05, "legacy": True}
    r = evaluate_track(_ports([0.12] * 40), legacy, RetrainPolicy())
    assert r["baseline_legacy"] is True and r["kind"] == "widespread" and r["median_ratio"] == pytest.approx(2.4)


def test_get_baseline_distinguishes_port_distribution_from_legacy(tmp_path):
    m = _monitor_with_meta(tmp_path, {"baseline_mse": 0.05, "baseline_port_median": 0.04, "baseline_port_p90": 0.09})
    b = m._get_baseline("traffic")
    assert b["legacy"] is False and b["port_median"] == 0.04 and b["port_p90"] == 0.09
    m = _monitor_with_meta(tmp_path, {"baseline_mse": 0.05, "threshold": 0.3})
    assert m._get_baseline("traffic") == {"baseline_mse": 0.05, "legacy": True}


class _FakeDB:
    def __init__(self, stats):
        self.stats, self.calls = stats, []

    def fetch_port_score_stats(self, ft, since):
        self.calls.append((ft, since))
        return self.stats


def _monitor(stats, activated_at, policy=None):
    m = DriftMonitor.__new__(DriftMonitor)
    m.db, m._policy = _FakeDB(stats), policy
    m._get_baseline = lambda ft: dict(BASELINE)
    m._active_activated_at = lambda ft: activated_at
    return m


NOW = datetime(2026, 10, 4, 3, 0)


def test_check_drift_right_after_activation_is_warming_up_and_does_not_query():
    """T-D4: 활성화 후 24h 미만이면 판정하지 않음 (이전 모델의 score 가 섞임)"""
    m = _monitor(_ports([0.12] * 40), NOW - timedelta(hours=6))
    r = m.check_drift(NOW)
    assert r["traffic"]["status"] == "warming_up" and "6h" in r["traffic"]["message"]
    assert r["drift_detected"] is False and m.db.calls == []


def test_check_drift_query_window_starts_after_activation():
    pol = RetrainPolicy(min_hours_since_activation=1, drift_window_hours=24)
    activated = NOW - timedelta(hours=3)
    m = _monitor(_ports([0.12] * 40), activated, pol)
    m.check_drift(NOW)
    assert all(since == activated for _, since in m.db.calls) and len(m.db.calls) == 2      # 두 트랙 모두


def test_check_drift_normal_window_and_result_shape():
    m = _monitor(_ports([0.12] * 40), NOW - timedelta(days=5))
    r = m.check_drift(NOW)
    assert m.db.calls[0][1] == NOW - timedelta(hours=24)
    assert r["drift_detected"] is True and r["drifted_tracks"] == ["traffic", "optical"]
    for key in ("status", "mean_mse", "baseline_mse", "drift_ratio", "kind", "median_ratio",
                "drifted_port_fraction", "top_ports"):
        assert key in r["traffic"]


def test_check_drift_localized_does_not_flag_drift_detected():
    m = _monitor(_ports(_normal_scores() + [1.5]), NOW - timedelta(days=5))
    r = m.check_drift(NOW)
    assert r["drift_detected"] is False and r["drifted_tracks"] == [] and r["traffic"]["kind"] == "localized"


def test_check_drift_no_data_sets_message_for_ui():
    m = _monitor(_ports([]), NOW - timedelta(days=5))
    assert "No recent data" in m.check_drift(NOW)["message"]


def test_check_drift_db_error_returns_error_status():
    m = _monitor(None, None)
    m.db.fetch_port_score_stats = lambda ft, since: (_ for _ in ()).throw(RuntimeError("db down"))
    r = m.check_drift(NOW)
    assert r["status"] == "error" and "db down" in r["message"]


def test_port_score_query_excludes_zero_scores_and_binds_parameters(monkeypatch):
    """T-D1(쿼리): score 0.0 은 '트랙 없음' 이므로 평균에서 제외, 값은 바인딩"""
    from unittest import mock
    from src.data import db_connector
    seen = {}
    monkeypatch.setattr(db_connector.pd, "read_sql", lambda q, conn, params=None: seen.update(q=q, p=params) or pd.DataFrame())
    db = db_connector.DBConnector.__new__(db_connector.DBConnector)
    db.get_connection = lambda: mock.MagicMock()
    db.fetch_port_score_stats("optical", datetime(2026, 10, 3, 3, 0))
    assert "optical_score > 0" in seen["q"] and "GROUP BY ip_addr, cid, lid" in seen["q"]
    assert seen["p"] == ("2026-10-03 03:00:00",) and "2026-10-03" not in seen["q"]
    with pytest.raises(ValueError):
        db.fetch_port_score_stats("x; DROP TABLE y", datetime.now())


def test_active_activated_at_reads_registry_history(tmp_path, monkeypatch):
    from src.models import registry
    from src.pipeline import drift_monitor
    entry = {"version": "v1", "model_path": "m", "scaler_path": "s", "config_path": "c", "trained_at": "2026-09-01 00:00:00"}
    with registry.transaction(str(tmp_path), "traffic") as reg:
        registry.add_version(reg, entry, True)
        reg["history"][-1]["activated_at"] = "2026-10-03 21:00:00"
    monkeypatch.setitem(drift_monitor.PATHS, "traffic", {"model": str(tmp_path / "traffic_ae.pth")})
    m = DriftMonitor.__new__(DriftMonitor)
    assert m._active_activated_at("traffic") == datetime(2026, 10, 3, 21, 0)
