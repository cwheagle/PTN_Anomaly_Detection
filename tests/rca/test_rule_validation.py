"""
룰 정의 검증: 추론이 제공하지 않는 키를 쓴 룰은 조용히 영원히 매칭되지 않는다 (과거 6/16 룰이 이 상태였음)

실행 방법:
  pytest tests/rca/test_rule_validation.py -v
"""
import json
import os

from src.rca.rule_engine import DEFAULT_RULES_PATH, RAW_FIELDS, RCAEngine, RuleTable, validate_rule


def _default_rules():
    with open(DEFAULT_RULES_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_all_default_rules_use_fields_the_inference_provides():
    problems = {r["id"]: validate_rule(r) for r in _default_rules() if validate_rule(r)}
    assert not problems, f"매칭될 수 없는 조건을 가진 룰: {problems}"


def test_raw_fields_match_what_inference_actually_produces():
    """RAW_FIELDS 가 실제 detect() 의 raw_data 키와 어긋나지 않는지 (inference.py 가 만드는 이름 규칙 확인)"""
    import inspect
    from src.pipeline import inference
    src = inspect.getsource(inference.AnomalyDetector.detect)
    # append_rca_metrics 가 만드는 접미어
    assert "_ratio" in src and "_trend_slope" in src
    assert "rx_avg_power_trend_slope" in RAW_FIELDS and "rx_power_trend_slope" not in RAW_FIELDS


def test_validate_rule_flags_unknown_keys():
    bad = {"id": "X", "track": "optical", "priority": 1, "diagnosis": "d", "action": "a",
           "contributions": {"error_packet": 50},                # optical 트랙에 없는 feature
           "raw_conditions": {"max_rx_power_trend_slope": -1.0,   # 옛 오타 (추론은 rx_avg_power_trend_slope)
                              "rx_avg_power": -20}}               # min_/max_ 접두어 없음
    issues = validate_rule(bad)
    assert len(issues) == 3


def test_add_rule_rejects_invalid_and_accepts_valid(tmp_path):
    table = RuleTable(rules_path=str(tmp_path / "rules.json"))
    base = {"id": "T-1", "track": "optical", "priority": 1, "diagnosis": "d", "action": "a"}

    assert table.add_rule({**base, "raw_conditions": {"max_rx_power_trend_slope": -1.0}}) is False
    assert table.add_rule({**base, "raw_conditions": {"max_rx_avg_power_trend_slope": -1.0}}) is True


def test_op003_actually_matches_with_inference_style_raw_data():
    """회귀: 광 수신 급락 룰(OP-003)이 추론이 만드는 키 이름으로 실제 매칭되어야 함"""
    engine = RCAEngine()
    rule = next(r for r in engine.get_rules() if r["id"] == "OP-003")
    contrib = {"rx_avg_power": 70.0, "tx_avg_power": 30.0}
    raw = {"rx_avg_power": -12.0, "tx_avg_power": -4.0,
           "rx_avg_power_trend_slope": -2.0, "tx_avg_power_trend_slope": -0.1}

    diagnosis, action = engine.diagnose("optical", contrib, True, raw)

    assert diagnosis == rule["diagnosis"] and action == rule["action"]
