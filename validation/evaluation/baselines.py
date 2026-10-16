"""
자명한 베이스라인 (AI 모델이 실제로 더해주는 가치를 재기 위한 비교 대상)

모든 베이스라인은 AI 모델과 '같은 스텝'에서 같은 채점(metrics.summarize)으로 평가한다.
raw: 트래픽+광 병합 프레임 (occur_date, ip_addr, cid, lid, tx_packet, rx_packet, error_packet,
                            tx_avg_power, rx_avg_power)  — 포트별 시간순 정렬 필요 없음(내부 정렬)
각 함수는 raw 와 같은 인덱스를 가진 DataFrame(alarm, score) 을 반환한다.
"""
import numpy as np
import pandas as pd

KEY = ["ip_addr", "cid", "lid"]


def always_alarm(raw: pd.DataFrame) -> pd.DataFrame:
    """모든 스텝에서 알람 (정밀도 = 유병률, 재현율 = 1). 이 전략을 못 이기면 모델은 가치가 없다."""
    return pd.DataFrame({"alarm": True, "score": 1.0}, index=raw.index)


def fixed_threshold(raw: pd.DataFrame, error_ge: int = 100, rx_lt: float = -15.0, use_optical: bool = True) -> pd.DataFrame:
    """고정 임계치: 에러 패킷 >= error_ge 또는 광 수신 < rx_lt dBm (포트별 기준선 없음).
    use_optical=False 면 광 조건을 끄고 에러 조건만 쓴다 (광 열이 없어도 동작). 기본값은 기존 동작과 같다."""
    if not use_optical:
        return pd.DataFrame({"alarm": raw["error_packet"] >= error_ge, "score": raw["error_packet"] / error_ge}, index=raw.index)
    alarm = (raw["error_packet"] >= error_ge) | (raw["rx_avg_power"] < rx_lt)
    score = np.maximum(raw["error_packet"] / error_ge, (-raw["rx_avg_power"]) / (-rx_lt))
    return pd.DataFrame({"alarm": alarm, "score": score}, index=raw.index)


def rolling_rule(raw: pd.DataFrame, window: int = 96, min_periods: int = 24,
                 error_ge: int = 20, rx_drop_db: float = 3.0, traffic_ratio: float = 0.15,
                 use_optical: bool = True) -> pd.DataFrame:
    """
    포트별 적응형 규칙 (최근 24시간 중앙값 대비 편차) — 학습이 필요 없는 가장 단순한 '똑똑한' 기준
      - 에러 패킷 >= error_ge
      - 광 수신 파워가 직전 24h 중앙값보다 rx_drop_db(dB) 이상 하락   (use_optical=False 면 사용 안 함, 광 열이 없어도 동작)
      - 트래픽 합계가 직전 24h 중앙값의 traffic_ratio 배 미만
    score = 세 조건의 정규화 값 중 최대 (>=1 이면 알람)
    """
    df = raw.sort_values(KEY + ["occur_date"]).copy()
    g = df.groupby(KEY)
    tot = df["tx_packet"] + df["rx_packet"]
    df["_tot"] = tot
    med_tot = df.groupby(KEY)["_tot"].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).median())

    e = df["error_packet"] / error_ge
    t = (1.0 - tot / med_tot.replace(0, np.nan)) / (1.0 - traffic_ratio)
    parts = [e, t]
    if use_optical:
        med_rx = g["rx_avg_power"].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).median())
        parts = [e, (med_rx - df["rx_avg_power"]) / rx_drop_db, t]
    score = pd.concat(parts, axis=1).max(axis=1).fillna(0.0)
    out = pd.DataFrame({"alarm": score >= 1.0, "score": score}, index=df.index)
    return out.reindex(raw.index)


BASELINES = {
    "always_alarm": always_alarm,
    "fixed_threshold": fixed_threshold,
    "rolling_rule": rolling_rule,
}
