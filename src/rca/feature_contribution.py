"""
Feature Contribution Analyzer

LSTM-Autoencoder의 Reconstruction Error(MSE)를 Feature 단위로 분해하여,
각 Feature가 이상 점수 상승에 기여한 비율(%)을 산출한다.

계산 기준:
  - 마지막 타임스텝(t=-1)의 (input - output)^2 를 Feature 별로 계산
  - 전체 합 대비 각 Feature의 비율(%) 반환
"""
import numpy as np


class FeatureContributionAnalyzer:
    """
    Feature별 MSE 기여도를 계산하는 분석기.

    Args:
        feature_names (list[str]): Feature 이름 목록 (FEATURE_GROUPS[ft] 순서와 동일해야 함)
    """

    def __init__(self, feature_names: list):
        self.feature_names = feature_names

    def compute(self, inputs_np: np.ndarray, outputs_np: np.ndarray) -> dict:
        """
        Feature별 MSE 기여도(%) 계산.

        Args:
            inputs_np  (np.ndarray): shape [B, T, F] — 입력 시퀀스 (NumPy)
            outputs_np (np.ndarray): shape [B, T, F] — 재구성 출력 (NumPy)

        Returns:
            dict: {"error_packet": 72.3, "tx_packet": 18.1, ...}
                  총합이 항상 100.0이 되도록 정규화됨.
                  B > 1 인 경우 배치 평균으로 산출.
        """
        # 마지막 타임스텝 기준으로 feature별 제곱 오차 계산
        # shape: [B, F]
        diff_sq = (inputs_np[:, -1, :] - outputs_np[:, -1, :]) ** 2

        # 배치 평균 → [F]
        per_feature_mse = diff_sq.mean(axis=0)

        total = per_feature_mse.sum()

        if total < 1e-12:
            # 오차가 거의 없는 경우 → 균등 분배
            n = len(self.feature_names)
            return {f: round(100.0 / n, 2) for f in self.feature_names}

        contributions = {
            f: round(float(v / total * 100), 2)
            for f, v in zip(self.feature_names, per_feature_mse)
        }
        return contributions

    def top_feature(self, contributions: dict) -> str:
        """기여도가 가장 높은 Feature 이름 반환"""
        return max(contributions, key=contributions.get)

    def rank_features(self, contributions: dict) -> list:
        """기여도 높은 순서로 (feature, %) 튜플 리스트 반환"""
        return sorted(contributions.items(), key=lambda x: x[1], reverse=True)
