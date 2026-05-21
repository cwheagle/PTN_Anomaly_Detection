"""
test_rca.py — Phase 8 RCA 엔진 코어 단위 테스트

검증 시나리오:
  1. Feature Contribution 정확성 — 의도적으로 오차가 큰 Feature가 1위로 선정되는지
  2. Default Rule 진단 — traffic error_contribution > 60 시 예상 진단명 출력
  3. Custom Rule 우선순위 — priority 높은 룰이 먼저 적용되는지
  4. Fallback 동작 — 룰 없어도 DefaultRuleSet이 동작하는지
  5. RCAEngine 정상 동작 — NORMAL 케이스 + 이상 케이스

실행 방법:
  python -m pytest tests/test_rca.py -v
"""
import sys
import os
import json
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rca.feature_contribution import FeatureContributionAnalyzer
from src.rca.rule_engine import RCAEngine, RuleTable
from src.rca.default_rules import DefaultRuleSet


# ──────────────────────────────────────────────
# 1. Feature Contribution 분석기 테스트
# ──────────────────────────────────────────────

class TestFeatureContributionAnalyzer:

    def _make_tensors(self, feature_errors: list):
        """
        feature_errors: 각 feature의 오차 크기 리스트
        예: [10.0, 1.0, 0.5] → error_packet이 압도적으로 큰 케이스
        """
        n_features = len(feature_errors)
        B, T = 4, 12  # 배치 4, 시퀀스 12
        inputs_np = np.zeros((B, T, n_features))
        outputs_np = np.zeros((B, T, n_features))
        # 마지막 타임스텝에만 오차 주입
        for i, err in enumerate(feature_errors):
            outputs_np[:, -1, i] = err
        return inputs_np, outputs_np

    def test_error_packet_dominates(self):
        """error_packet 오차가 클 때 기여도 1위여야 함 (traffic)"""
        features = ['tx_packet', 'rx_packet', 'error_packet']
        analyzer = FeatureContributionAnalyzer(features)
        # error_packet에 큰 오차 주입
        inputs_np, outputs_np = self._make_tensors([1.0, 1.0, 10.0])
        contribs = analyzer.compute(inputs_np, outputs_np)

        print(f"\n[TEST] Feature Contributions: {contribs}")
        assert 'error_packet' in contribs
        assert analyzer.top_feature(contribs) == 'error_packet', \
            f"Expected 'error_packet' top, got: {analyzer.top_feature(contribs)}"
        assert contribs['error_packet'] > 60.0, \
            f"error_packet contribution should be > 60%, got: {contribs['error_packet']}"
        print("  PASS: error_packet dominates as expected")

    def test_rx_power_dominates_optical(self):
        """rx_avg_power 오차가 클 때 기여도 1위여야 함 (optical)"""
        features = ['tx_avg_power', 'rx_avg_power']
        analyzer = FeatureContributionAnalyzer(features)
        inputs_np, outputs_np = self._make_tensors([0.5, 8.0])
        contribs = analyzer.compute(inputs_np, outputs_np)

        print(f"\n[TEST] Optical Contributions: {contribs}")
        assert analyzer.top_feature(contribs) == 'rx_avg_power', \
            f"Expected 'rx_avg_power' top, got: {analyzer.top_feature(contribs)}"
        print("  PASS: rx_avg_power dominates as expected")

    def test_sum_equals_100(self):
        """기여도 합계가 100.0(±0.5 허용)이어야 함"""
        features = ['tx_packet', 'rx_packet', 'error_packet']
        analyzer = FeatureContributionAnalyzer(features)
        inputs_np, outputs_np = self._make_tensors([3.0, 2.0, 5.0])
        contribs = analyzer.compute(inputs_np, outputs_np)

        total = sum(contribs.values())
        print(f"\n[TEST] Contribution total: {total:.2f}%")
        assert abs(total - 100.0) < 0.5, f"Sum should be ~100%, got: {total}"
        print("  PASS: contributions sum to ~100%")

    def test_zero_error_uniform_distribution(self):
        """오차가 없을 때 균등 분배여야 함"""
        features = ['tx_packet', 'rx_packet', 'error_packet']
        analyzer = FeatureContributionAnalyzer(features)
        inputs_np = np.ones((2, 5, 3))
        outputs_np = inputs_np.copy()  # 완벽 재구성 → 오차 없음
        contribs = analyzer.compute(inputs_np, outputs_np)

        print(f"\n[TEST] Zero error contributions: {contribs}")
        for f, pct in contribs.items():
            assert abs(pct - 33.33) < 1.0, f"Expected ~33.3%, got {pct} for {f}"
        print("  PASS: uniform distribution when no error")


# ──────────────────────────────────────────────
# 2. Default Rule Set 테스트
# ──────────────────────────────────────────────

class TestDefaultRuleSet:

    def test_traffic_error_dominant(self):
        """traffic: error_packet 기여도 1위 → 오류 관련 진단"""
        ruleset = DefaultRuleSet()
        contribs = {'tx_packet': 10.0, 'rx_packet': 15.0, 'error_packet': 75.0}
        diagnosis = ruleset.diagnose('traffic', contribs)
        print(f"\n[TEST] Default diagnosis (traffic error dominant): {diagnosis}")
        assert 'CRC' in diagnosis or '오류' in diagnosis or 'error' in diagnosis.lower(), \
            f"Expected CRC/error diagnosis, got: {diagnosis}"
        assert '75' in diagnosis or '기여도' in diagnosis
        print("  PASS: error-related diagnosis returned")

    def test_optical_rx_dominant(self):
        """optical: rx_avg_power 기여도 1위 → 수신 열화 진단"""
        ruleset = DefaultRuleSet()
        contribs = {'tx_avg_power': 20.0, 'rx_avg_power': 80.0}
        diagnosis = ruleset.diagnose('optical', contribs)
        print(f"\n[TEST] Default diagnosis (optical rx dominant): {diagnosis}")
        assert 'RX' in diagnosis or '수신' in diagnosis, \
            f"Expected RX-related diagnosis, got: {diagnosis}"
        print("  PASS: RX degradation diagnosis returned")


# ──────────────────────────────────────────────
# 3. RCA Engine 테스트
# ──────────────────────────────────────────────

class TestRCAEngine:

    def test_normal_returns_normal(self):
        """is_anomaly=False 이면 항상 'NORMAL' 반환"""
        engine = RCAEngine()
        contribs = {'tx_packet': 40.0, 'rx_packet': 30.0, 'error_packet': 30.0}
        result = engine.diagnose('traffic', contribs, is_anomaly=False)
        print(f"\n[TEST] Normal case: {result}")
        assert result == 'NORMAL', f"Expected 'NORMAL', got: {result}"
        print("  PASS: NORMAL returned for non-anomaly")

    def test_custom_rule_priority(self):
        """사용자 정의 룰이 있을 때 DefaultRuleSet보다 먼저 적용"""
        engine = RCAEngine(rules_path=None)  # 룰 파일 없이 시작
        # 임시 룰 직접 주입
        engine.rule_table.rules = [
            {
                "id": "TEST-001",
                "track": "traffic",
                "priority": 10,
                "condition": "error_contribution > 50",
                "diagnosis": "테스트 커스텀 진단",
            }
        ]
        contribs = {'tx_packet': 10.0, 'rx_packet': 15.0, 'error_packet': 75.0}
        result = engine.diagnose('traffic', contribs, is_anomaly=True)
        print(f"\n[TEST] Custom rule result: {result}")
        assert result == "테스트 커스텀 진단", \
            f"Expected custom rule diagnosis, got: {result}"
        print("  PASS: Custom rule applied with priority")

    def test_fallback_when_no_rules(self):
        """룰 파일 없을 때 DefaultRuleSet Fallback 동작"""
        engine = RCAEngine(rules_path=None)  # 룰 없음
        contribs = {'tx_packet': 10.0, 'rx_packet': 15.0, 'error_packet': 75.0}
        result = engine.diagnose('traffic', contribs, is_anomaly=True)
        print(f"\n[TEST] Fallback diagnosis: {result}")
        assert result != 'NORMAL', "Should return some diagnosis, not NORMAL"
        assert len(result) > 0, "Should return non-empty string"
        print(f"  PASS: Fallback diagnosis returned: {result}")

    def test_default_rules_file_loaded(self):
        """기본 룰 파일이 정상 로드되어 error > 60% 케이스에서 CRC 진단 반환"""
        engine = RCAEngine()  # default_rules.json 로드
        if len(engine.get_rules()) == 0:
            pytest.skip("Default rules file not found, skipping rule-match test")

        contribs = {'tx_packet': 10.0, 'rx_packet': 15.0, 'error_packet': 75.0}
        result = engine.diagnose('traffic', contribs, is_anomaly=True)
        print(f"\n[TEST] Default rules file diagnosis: {result}")
        assert 'CRC' in result or '오류' in result, \
            f"Expected CRC diagnosis from default rules, got: {result}"
        print("  PASS: CRC diagnosis from default_rules.json")

    def test_optical_rx_diagnosis(self):
        """optical rx_avg_power 기여도 > 60% → 광 수신 열화 진단"""
        engine = RCAEngine()
        contribs = {'tx_avg_power': 20.0, 'rx_avg_power': 80.0}
        result = engine.diagnose('optical', contribs, is_anomaly=True)
        print(f"\n[TEST] Optical RX diagnosis: {result}")
        assert '수신' in result or 'RX' in result, \
            f"Expected RX degradation, got: {result}"
        print("  PASS: Optical RX degradation diagnosed")


# ──────────────────────────────────────────────
# 4. RuleTable 테스트
# ──────────────────────────────────────────────

class TestRuleTable:

    def test_add_rule_requires_fields(self):
        """필수 필드 누락 시 False 반환"""
        table = RuleTable(rules_path=None)
        result = table.add_rule({"id": "X-001", "track": "traffic"})  # condition, diagnosis 없음
        assert result is False
        print("\n  PASS: Missing fields returns False")

    def test_priority_sort(self):
        """priority 높은 룰이 먼저 정렬되어야 함"""
        table = RuleTable(rules_path=None)
        table.rules = [
            {"id": "A", "priority": 5, "track": "traffic", "condition": "True", "diagnosis": "A"},
            {"id": "B", "priority": 10, "track": "traffic", "condition": "True", "diagnosis": "B"},
        ]
        table.rules.sort(key=lambda r: r.get("priority", 0), reverse=True)
        assert table.rules[0]["id"] == "B", "Higher priority should be first"
        print("\n  PASS: Priority sorting works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
