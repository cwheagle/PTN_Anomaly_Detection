"""
Default Rule Set

도메인 룰 파일이 없거나, 어떤 사용자 정의 룰도 매칭되지 않을 때
Feature Contribution 기여도 1위 기반으로 동작하는 Fallback 진단 세트.

이 모듈은 코드 내에 하드코딩된 안전망 역할을 한다.
Phase 9에서 룰 파일이 충분히 정교해지면 이 모듈은 거의 호출되지 않는다.
"""


class DefaultRuleSet:
    """
    기여도 1위 Feature 기반의 간단한 Fallback 진단.
    
    - 룰 조건 평가 없이 순수하게 contributions.max()만으로 동작
    - 항상 어떤 진단이든 반환 (빈 문자열 없음)
    """

    # Traffic 트랙: feature 이름 → 기본 진단 메시지
    TRAFFIC_DIAGNOSIS = {
        "error_packet": "패킷 오류 급증 (CRC/비트 오류 의심)",
        "tx_packet": "송신 트래픽 패턴 이상 (TX 급변)",
        "rx_packet": "수신 트래픽 패턴 이상 (RX 패킷 손실 의심)",
    }

    # Optical 트랙: feature 이름 → 기본 진단 메시지
    OPTICAL_DIAGNOSIS = {
        "rx_avg_power": "광 수신 전력 이상 (RX 열화 의심)",
        "tx_avg_power": "광 송신 전력 이상 (TX 열화 의심)",
    }

    def diagnose(self, ft: str, contributions: dict, raw_data: dict = None) -> str:
        """
        기여도 1위 Feature 기반으로 기본 진단명 반환.

        Args:
            ft (str): "traffic" | "optical" | "integrated"
            contributions (dict): {"error_packet": 72.3, ...}
            raw_data (dict): 원시 데이터

        Returns:
            str: 기본 진단명
        """
        if not contributions:
            return "이상 패턴 감지 (원인 미상)"

        # 기여도 1위 Feature 선정
        top_feature = max(contributions, key=contributions.get)
        top_pct = contributions[top_feature]

        if ft == "traffic":
            base_msg = self.TRAFFIC_DIAGNOSIS.get(
                top_feature, f"트래픽 이상 ({top_feature} 기여도 {top_pct:.1f}%)"
            )
        elif ft == "optical":
            base_msg = self.OPTICAL_DIAGNOSIS.get(
                top_feature, f"광성능 이상 ({top_feature} 기여도 {top_pct:.1f}%)"
            )
        elif ft == "integrated":
            # 트래픽과 광성능 기여도 1위를 각각 찾거나 단순히 복합 장애로 표시
            base_msg = f"복합 장애 감지 (광/트래픽 동시 발생, 주요 원인: {top_feature})"
        else:
            base_msg = f"이상 패턴 감지 ({top_feature} 기여도 {top_pct:.1f}%)"

        return f"{base_msg} [기여도 {top_pct:.1f}%]"
