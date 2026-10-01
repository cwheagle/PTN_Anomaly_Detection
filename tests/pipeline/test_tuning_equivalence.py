"""
튜닝 도구와 운영 추론의 동등성: 저장된 점수 + 정책으로 재계산한 알람이 detect() 와 한 건도 다르지 않아야 함

이게 깨지면 튜닝에서 찾은 값이 운영에서 같은 결과를 내지 않으므로, 튜닝 결과를 신뢰할 수 없다.
학습된 모델(models/)이 없으면 건너뛴다.

실행 방법:
  pytest tests/pipeline/test_tuning_equivalence.py -v
"""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("models/traffic_registry.json"),
                                reason="학습된 모델(models/)이 없어 건너뜀")

KEY = ["ip_addr", "cid", "lid", "occur_date"]


@pytest.fixture(scope="module")
def data():
    from validation.simulator.scenario_generator import ScenarioConfig, generate
    return generate(ScenarioConfig(seed=3, nodes=1, ports_per_node=8, days=4, mean_gap_days=1.0, warmup_days=0.5,
                                   start="2026-03-01 00:00:00"))


def _compare(data, policy):
    from src.pipeline.inference import AnomalyDetector
    from validation.evaluation.tuning import compute_track_scores, simulate_alarms

    det = AnomalyDetector(policy=policy)
    res = det.detect(df_traffic=data["traffic"], df_optical=data["optical"], latest_only=False)
    sim = simulate_alarms(compute_track_scores(det, data["traffic"], data["optical"]), policy)

    m = res[KEY + ["is_anomaly", "alarm_level"]].merge(sim, on=KEY, how="outer", indicator=True, suffixes=("_det", "_sim"))
    assert (m["_merge"] == "both").all(), "detect() 와 튜닝 도구가 다루는 (포트, 시각)이 다름"
    return m


def test_default_policy_alarms_match_detect_exactly(data):
    from src.pipeline.alerting import AlertPolicy
    m = _compare(data, AlertPolicy())
    assert (m["is_anomaly"].astype(bool) == m["alarm"]).all()
    assert (m["alarm_level"] == m["level"].where(m["alarm"], 0)).all()      # 억제된 행의 등급은 0


@pytest.mark.parametrize("policy_kwargs", [
    dict(threshold_scale=1.5),
    dict(sigma_k=2.0, threshold_scale=0.8),
    dict(dampening_steps={1: 1, 2: 1, 3: 1}),
    dict(dampening_steps={1: 6, 2: 4, 3: 3}, severity_decay=0.3),
])
def test_non_default_policies_also_match_detect(data, policy_kwargs):
    from src.pipeline.alerting import AlertPolicy
    m = _compare(data, AlertPolicy(**policy_kwargs))
    assert (m["is_anomaly"].astype(bool) == m["alarm"]).all(), policy_kwargs
