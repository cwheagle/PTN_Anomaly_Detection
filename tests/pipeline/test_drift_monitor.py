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
