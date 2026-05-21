"""
RCA Rule Engine

JSON 파일 기반의 도메인 룰 테이블을 로드하고, Feature Contribution 결과를 바탕으로
가장 적합한 장애 진단명(rca_diagnosis)을 출력한다.

설계 원칙:
  - Phase 9에서 DB 기반으로 확장 가능한 구조 (RuleTable 추상화)
  - 룰 파일이 없거나 매칭 실패 시 DefaultRuleSet으로 Fallback
  - 룰 조건은 Fully Structured JSON(contributions, raw_conditions) 기반
"""
import os
import json
from typing import Optional
from src.rca.default_rules import DefaultRuleSet


# 기본 룰 파일 경로
DEFAULT_RULES_PATH = os.path.join(
    os.path.dirname(__file__), "rules", "default_rules.json"
)


class RuleTable:
    """
    JSON 파일 기반 RCA 룰 테이블.
    
    룰 구조:
      {
        "id": "TR-001",
        "track": "traffic",          # "traffic" | "optical"
        "priority": 10,              # 높을수록 먼저 평가
        "condition": "error_contribution > 60",  # 평가 표현식
        "diagnosis": "CRC/비트 오류 의심 ...",
        "action": "광 커넥터 청소 ..."
      }
    """

    def __init__(self, rules_path: str = DEFAULT_RULES_PATH):
        self.rules_path = rules_path
        self.rules = self._load(rules_path)

    def _load(self, path: str) -> list:
        """JSON 파일에서 룰 로드. 파일 없으면 빈 리스트 반환."""
        if not path or not os.path.exists(path):
            print(f"[RCA] Rules file not found: {path}. Using DefaultRuleSet only.")
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                rules = json.load(f)
            # priority 내림차순 정렬 (높은 우선순위 먼저 평가)
            rules.sort(key=lambda r: r.get("priority", 0), reverse=True)
            print(f"[RCA] Loaded {len(rules)} rules from {path}")
            return rules
        except Exception as e:
            print(f"[RCA] Failed to load rules from {path}: {e}")
            return []

    def reload(self):
        """룰 파일 핫 리로드 (운영 중 룰 추가 시 사용)"""
        self.rules = self._load(self.rules_path)

    def add_rule(self, rule_dict: dict) -> bool:
        """
        새로운 도메인 룰을 메모리 및 파일에 추가 (Fully Structured JSON 지원)
        """
        required = {"id", "track", "priority", "diagnosis", "action"}
        if not required.issubset(rule_dict.keys()):
            logger.error(f"[RCA Engine] Missing required fields. Rule: {rule_dict}")
            return False
            
        if rule_dict["track"] not in ("traffic", "optical", "integrated"):
            logger.error(f"[RCA Engine] Invalid track: {rule_dict.get('track')}")
            return False

        # 중복 id 체크
        existing_ids = {r["id"] for r in self.rules}
        if rule_dict["id"] in existing_ids:
            print(f"[RCA] Rule ID '{rule_dict['id']}' already exists. Use a different ID.")
            return False

        self.rules.append(rule_dict)
        self.rules.sort(key=lambda r: r.get("priority", 0), reverse=True)

        try:
            os.makedirs(os.path.dirname(self.rules_path), exist_ok=True)
            with open(self.rules_path, "w", encoding="utf-8") as f:
                json.dump(self.rules, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[RCA] Failed to save rule: {e}")
            return False

    def get_rules(self) -> list:
        """현재 룰 테이블 반환"""
        return self.rules


class RCAEngine:
    """
    Feature Contribution 결과를 받아 장애명(rca_diagnosis)을 출력하는 엔진.
    
    평가 순서:
      1. RuleTable의 룰을 priority 순으로 평가 → 첫 번째 매칭 룰의 diagnosis 반환
      2. 매칭 없으면 DefaultRuleSet의 기여도 1위 Feature 기반 기본 진단 반환
      3. 이상이 아닌 경우 → "NORMAL" 반환
    """

    def __init__(self, rules_path: str = DEFAULT_RULES_PATH):
        self.rule_table = RuleTable(rules_path)
        self.default_ruleset = DefaultRuleSet()

    def reload_rules(self):
        """운영 중 룰 파일 변경 시 핫 리로드"""
        self.rule_table.reload()

    def add_rule(self, rule: dict) -> bool:
        """새 룰 추가 (API 엔드포인트에서 호출)"""
        return self.rule_table.add_rule(rule)

    def get_rules(self) -> list:
        """현재 룰 목록 반환"""
        return self.rule_table.get_rules()

    def diagnose(self, ft: str, contributions: dict, is_anomaly: bool, raw_data: dict = None) -> tuple:
        """
        Feature Contribution 결과로 장애명 진단.

        Args:
            ft (str): 트랙 유형 ("traffic" | "optical" | "integrated")
            contributions (dict): {"error_packet": 72.3, ...}
            is_anomaly (bool): 이상 탐지 여부
            raw_data (dict): 원시 Feature 값 (예: {"error_packet": 5000, "rx_avg_power": -30.5})

        Returns:
            tuple: (진단명 텍스트 또는 "NORMAL", 추천 조치 텍스트 또는 None)
        """
        if not is_anomaly:
            return "NORMAL", None

        # 1단계: 사용자 정의 룰 평가 (priority 순)
        for rule in self.rule_table.get_rules():
            if rule.get("track") != ft:
                continue
                
            match = True
            
            # Contribution 평가
            rule_contribs = rule.get("contributions", {})
            for feature, threshold in rule_contribs.items():
                if contributions.get(feature, 0) < threshold:
                    match = False
                    break
                    
            if not match:
                continue
                
            # Raw Condition 평가 (Min / Max)
            rule_raw = rule.get("raw_conditions", {})
            if raw_data:
                for key, val in rule_raw.items():
                    if key.startswith("min_"):
                        feature = key[4:]
                        if raw_data.get(feature, 0) < val:
                            match = False
                            break
                    elif key.startswith("max_"):
                        feature = key[4:]
                        if raw_data.get(feature, 0) > val:
                            match = False
                            break
                            
            if match:
                return rule.get("diagnosis", "Unknown"), rule.get("action", None)

        # 2단계: DefaultRuleSet Fallback
        diagnosis = self.default_ruleset.diagnose(ft, contributions, raw_data)
        return diagnosis, None
