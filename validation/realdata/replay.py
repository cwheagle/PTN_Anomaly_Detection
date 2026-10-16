"""
P1-2 오프라인 재생 (설계서 9.1~9.3): 스냅샷의 트래픽 행을 AI(트래픽 트랙만)·고정 임계치·롤링 규칙에 같은 입력으로 통과시킨다.

- 입력은 4.3a 격리 후 정상 ETH 행(스냅샷)이고 운영과 같은 DataProcessor 경로(AnomalyDetector.detect)를 쓴다. 공백을 메우는 보간은 하지 않는다.
- 재생 구간은 평가 구간보다 2일 앞에서 시작하고 앞 2일 결과는 버린다(워밍업, 9.2).
- 베이스라인은 광 조건을 끈다(use_optical=False). 임계값은 바꾸지 않는다.
- 방법마다 '판정할 수 있었던 슬롯'이 다르므로(AI: 연속 required_rows 행 필요) 방법별 재생 가능 슬롯 비율을 함께 낸다.
"""
import numpy as np
import pandas as pd

from validation.evaluation.baselines import fixed_threshold, rolling_rule
from validation.evaluation.metrics import KEY, make_incidents

WARMUP_DAYS = 2
METHODS = ("ai", "fixed", "rolling")
ROLLING_MIN_PERIODS = 24


def traffic_frame(eth, start, end, warmup_days=WARMUP_DAYS):
    """스냅샷 ETH → 재생 입력 [start − warmup, end). 열: occur_date(=구간 끝), ip_addr, cid, lid, tx/rx/error_packet.
    같은 (포트, 구간)의 중복은 운영과 같이 마지막 행을 유지한다."""
    df = eth.copy()
    df["occur_date"] = pd.to_datetime(df["bin_end"] if "bin_end" in df.columns else df["occur_date"])
    lo, hi = pd.Timestamp(start) - pd.Timedelta(days=warmup_days), pd.Timestamp(end)
    df = df[(df["occur_date"] >= lo) & (df["occur_date"] < hi)]
    for c in ("tx_packet", "rx_packet", "error_packet"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ip_addr"] = df["ip_addr"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["ip_addr", "cid", "lid", "occur_date"], keep="last")
    return df[["occur_date", "ip_addr", "cid", "lid", "tx_packet", "rx_packet", "error_packet"]].sort_values(KEY + ["occur_date"]).reset_index(drop=True)


def run_ai(df_traffic, versions=None, policy=None, detector=None):
    """운영과 같은 AnomalyDetector.detect (트래픽 트랙만). 반환: 점수가 나온 행의 (occur_date, 키, alarm, severity, score)."""
    if detector is None:
        from src.pipeline.inference import AnomalyDetector
        detector = AnomalyDetector(policy=policy, versions=versions, tracks=("traffic",))
    res = detector.detect(df_traffic=df_traffic, df_optical=None, latest_only=False)
    if res is None or len(res) == 0:
        return pd.DataFrame(columns=KEY + ["occur_date", "alarm", "severity", "score"])
    out = res.copy()
    out["occur_date"] = pd.to_datetime(out["occur_date"])
    out["alarm"] = out["is_anomaly"].astype(bool)
    score = out["traffic_score"] if "traffic_score" in out.columns else out.get("anomaly_score", pd.Series(np.nan, index=out.index))
    out["score"] = pd.to_numeric(score, errors="coerce")
    cols = KEY + ["occur_date", "alarm", "severity", "score"] + (["alarm_level"] if "alarm_level" in out.columns else [])
    return out[cols].reset_index(drop=True)


def run_baselines(df_traffic):
    """고정 임계치·롤링 규칙(광 조건 끔). 반환: {이름: DataFrame(키, occur_date, alarm, score, playable)}."""
    out = {}
    base = df_traffic.reset_index(drop=True)
    fx = fixed_threshold(base, use_optical=False)
    out["fixed"] = base[KEY + ["occur_date"]].assign(alarm=fx["alarm"].to_numpy(), score=fx["score"].to_numpy(), playable=True)
    rr = rolling_rule(base, use_optical=False)
    prior = base.sort_values(KEY + ["occur_date"]).groupby(KEY).cumcount()
    play = (prior.reindex(base.index) >= ROLLING_MIN_PERIODS)
    # 롤링 규칙은 에러 조건이 이력 없이도 동작하므로 playable 은 '이력(24행)이 있는 슬롯' 기준(트래픽 급감 조건 판단 가능)
    out["rolling"] = base[KEY + ["occur_date"]].assign(alarm=rr["alarm"].to_numpy(), score=rr["score"].to_numpy(), playable=play.to_numpy())
    return out


def build_detector(versions=None, policy=None, models_dir=None):
    """운영과 같은 AnomalyDetector (트래픽 트랙만). 모델·스케일러·정책 로드가 여기서 일어난다. models_dir 가 있으면 그 폴더의 모델(자체 레지스트리)을 로드한다
    (PATHS 는 로드하는 동안만 바꾸고 복구)."""
    from src.pipeline.inference import AnomalyDetector
    if models_dir:
        from validation.realdata.retrain import use_models_dir
        with use_models_dir(models_dir):
            return AnomalyDetector(policy=policy, versions=versions, tracks=("traffic",))
    return AnomalyDetector(policy=policy, versions=versions, tracks=("traffic",))


def replay(eth, start, end, versions=None, policy=None, detector=None, warmup_days=WARMUP_DAYS, inp=None):
    """세 방법을 한 번에 재생. 결과는 평가 구간 [start, end) 로 자른 프레임들과 방법별 재생 가능 슬롯 비율.
    반환 dict: frames{방법: DataFrame(키, occur_date, alarm, [severity, score])}, slots{방법: {...}}, input_rows, normal_rows"""
    if inp is None:
        inp = traffic_frame(eth, start, end, warmup_days)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    normal = inp[(inp["occur_date"] >= s) & (inp["occur_date"] < e)]
    ai = run_ai(inp, versions, policy, detector)
    ai = ai[(ai["occur_date"] >= s) & (ai["occur_date"] < e)]
    bl = run_baselines(inp)
    frames = {"ai": ai}
    for k, v in bl.items():
        frames[k] = v[(v["occur_date"] >= s) & (v["occur_date"] < e)].reset_index(drop=True)
    n_norm = len(normal)
    slots = {"ai": {"playable": int(len(ai)), "normal": n_norm},
             "fixed": {"playable": int(frames["fixed"]["playable"].sum()) if n_norm else 0, "normal": n_norm},
             "rolling": {"playable": int(frames["rolling"]["playable"].sum()) if n_norm else 0, "normal": n_norm}}
    for v in slots.values():
        v["ratio"] = (v["playable"] / v["normal"]) if v["normal"] else float("nan")
    return {"frames": frames, "slots": slots, "input_rows": int(len(inp)), "normal_rows": n_norm, "range": (s, e)}


def incidents_of(frame, gap_steps=4):
    """make_incidents(gap_steps=4, nuisance=False). 반환 (인시던트 DataFrame, 알람 행 DataFrame). 인시던트에 최고 심각도(AI)·시작/끝."""
    ev = frame[KEY + ["occur_date", "alarm"]].copy()
    ev["nuisance"] = False
    inc, alarms = make_incidents(ev, gap_steps)
    if len(inc) and "severity" in frame.columns:
        idx = pd.MultiIndex.from_frame(alarms[KEY + ["occur_date"]])
        a = alarms.copy()
        f = frame.set_index(KEY + ["occur_date"])
        a["severity"] = f["severity"].reindex(idx).to_numpy()
        inc = inc.merge(a.groupby("incident")["severity"].max().rename("severity_max"), left_on="incident", right_index=True, how="left")
        if "alarm_level" in frame.columns:
            a["alarm_level"] = f["alarm_level"].reindex(idx).to_numpy()
            inc = inc.merge(a.groupby("incident")["alarm_level"].max().rename("level_max"), left_on="incident", right_index=True, how="left")
    return inc, alarms


def port_days(frame):
    """재생 결과가 있는 포트·일 수 (C1 분모)."""
    if frame.empty:
        return 0
    return int(frame.assign(_d=frame["occur_date"].dt.normalize()).drop_duplicates(KEY + ["_d"]).shape[0])
