"""
회귀(골든) 테스트: 알람 로직을 alerting.py 로 분리하기 전의 detect() 출력과 한 컬럼도 달라지지 않아야 함

골든 값은 분리 이전 코드의 detect() 결과에서 뽑은 요약이다 (시드 고정 시나리오, 활성 모델 v1 기준).
학습된 모델(models/)이 없으면 건너뛴다 (gitignore 대상).

실행 방법:
  pytest tests/pipeline/test_inference_golden.py -v
"""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("models/traffic_registry.json"),
                                reason="학습된 모델(models/)이 없어 건너뜀")

# 분리 이전 코드의 결과 (2026-10-01): 8포트 x 4일, seed=3
GOLDEN = {"rows": 2984, "alarms": 491,
          "labels": {"NORMAL": 2366, "CRITICAL": 428, "NORMAL (DAMPENED)": 127, "MAJOR": 41, "MINOR": 22}}


def test_detect_output_summary_unchanged_by_alerting_refactor():
    from src.pipeline.inference import AnomalyDetector
    from validation.simulator.scenario_generator import ScenarioConfig, generate

    d = generate(ScenarioConfig(seed=3, nodes=1, ports_per_node=8, days=4, mean_gap_days=1.0, warmup_days=0.5,
                                start="2026-03-01 00:00:00"))
    res = AnomalyDetector().detect(df_traffic=d["traffic"], df_optical=d["optical"], latest_only=False)

    assert len(res) == GOLDEN["rows"]
    assert int(res.is_anomaly.sum()) == GOLDEN["alarms"]
    assert res.alarm_label.value_counts().to_dict() == GOLDEN["labels"]


def test_policy_override_changes_alarm_count():
    """정책이 실제로 추론에 반영되는지: 댐프닝을 모두 즉시(1)로 풀면 알람이 같거나 늘어남"""
    from src.pipeline.alerting import AlertPolicy
    from src.pipeline.inference import AnomalyDetector
    from validation.simulator.scenario_generator import ScenarioConfig, generate

    d = generate(ScenarioConfig(seed=3, nodes=1, ports_per_node=8, days=4, mean_gap_days=1.0, warmup_days=0.5,
                                start="2026-03-01 00:00:00"))
    relaxed = AnomalyDetector(policy=AlertPolicy(dampening_steps={1: 1, 2: 1, 3: 1}))
    res = relaxed.detect(df_traffic=d["traffic"], df_optical=d["optical"], latest_only=False)
    assert int(res.is_anomaly.sum()) > GOLDEN["alarms"]               # 억제됐던 127건이 알람으로 풀림
    assert (res.alarm_label == "NORMAL (DAMPENED)").sum() == 0
