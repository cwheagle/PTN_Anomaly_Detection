"""
이벤트 중심 평가 지표 (예지 정비용)

기존 평가(evaluate_model.py)의 문제를 보완한다:
  - 알람 단위 정밀도와 시나리오 단위 재현율을 섞은 F1 -> 모든 지표를 '이벤트(인시던트/에피소드)' 단위로 통일
  - start ~ failure+168h 를 모두 정답으로 인정 -> 에피소드가 실제로 활성인 구간(+작은 허용 오차)만 정답
  - TTF 없는 알람을 채점에서 제외 -> 모든 알람을 채점
  - 사후 알람도 TP -> 'failure 이전 조기 탐지'와 '사후 탐지'를 분리
  - 베이스라인 없음 -> 같은 채점으로 자명한 규칙과 항상 비교

입력 약속 (ev: 평가 프레임, 스코어링된 스텝만 포함)
  occur_date, ip_addr, cid, lid       : 키
  alarm (bool), score (float)         : 방법이 낸 알람/연속 점수
  state (0/1/2), episode_id, scenario, nuisance : 정답 (labels.csv 와 조인)
episodes: episode_id, ip_addr, cid, lid, scenario, t_start, t_fail, t_end
"""
import numpy as np
import pandas as pd

KEY = ["ip_addr", "cid", "lid"]
STEP = pd.Timedelta(minutes=15)


def step_metrics(ev: pd.DataFrame) -> dict:
    """스텝 단위: 장애(state>0) 여부를 알람이 맞췄는가"""
    pos = ev["state"] > 0
    a = ev["alarm"].astype(bool)
    tp, fp, fn = int((a & pos).sum()), int((a & ~pos).sum()), int((~a & pos).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "steps": int(len(ev)),
        "prevalence": float(pos.mean()),
        "precision": p, "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "false_positive_rate": fp / max(int((~pos).sum()), 1),
        "recall_ramp": float(a[ev["state"] == 1].mean()) if (ev["state"] == 1).any() else None,
        "recall_plateau": float(a[ev["state"] == 2].mean()) if (ev["state"] == 2).any() else None,
        "auprc": None,
    }
    if ev["score"].notna().any() and pos.any() and (~pos).any() and ev["score"].nunique() > 1:
        from sklearn.metrics import average_precision_score
        out["auprc"] = float(average_precision_score(pos, ev["score"].fillna(0.0)))
    return out


def make_incidents(ev: pd.DataFrame, gap_steps: int = 4) -> pd.DataFrame:
    """포트별로 연속된 알람(간격 gap_steps 이하)을 하나의 인시던트로 묶는다 (알람 폭주를 1건으로 계수)"""
    a = ev[ev["alarm"].astype(bool)].sort_values(KEY + ["occur_date"]).copy()
    if a.empty:
        return pd.DataFrame(columns=KEY + ["incident", "t_first", "t_last", "n_alarms", "on_nuisance"]), a
    gap = a.groupby(KEY)["occur_date"].diff()
    new = gap.isna() | (gap > STEP * gap_steps)
    a["incident"] = new.cumsum()
    inc = a.groupby("incident").agg(
        ip_addr=("ip_addr", "first"), cid=("cid", "first"), lid=("lid", "first"),
        t_first=("occur_date", "min"), t_last=("occur_date", "max"),
        n_alarms=("occur_date", "size"), on_nuisance=("nuisance", "max"),
    ).reset_index()
    return inc, a


def event_metrics(ev: pd.DataFrame, episodes: pd.DataFrame, tol_steps: int = 1,
                  grace_steps: int = 4, gap_steps: int = 4) -> dict:
    """
    에피소드(장애) 단위 + 인시던트(알람 묶음) 단위 지표

    에피소드 활성 구간: [t_start - tol, t_end + grace]  (복구 직후 grace 동안의 알람은 허용)
      - detected : 활성 구간에 알람이 1건이라도 있음
      - early    : failure(t_fail) 이전에 첫 알람이 있음 (= 예지 성공), lead = t_fail - 첫 알람
    false incident: 어떤 에피소드의 활성 구간에도 속하지 않는 알람들을 묶은 인시던트 (오탐)
    incident_precision = 탐지 에피소드 / (탐지 에피소드 + 오탐 인시던트)
    """
    inc, alarms = make_incidents(ev, gap_steps)
    tol, grace = STEP * tol_steps, STEP * grace_steps
    ep = episodes.copy()
    ep["span_start"], ep["span_end"] = ep["t_start"] - tol, ep["t_end"] + grace

    # --- 알람이 에피소드 활성 구간 안/밖 어디에 있는지 ---
    if len(alarms):
        al = alarms.reset_index(drop=True)
        al["_rid"] = np.arange(len(al))
        m = al[["_rid"] + KEY + ["occur_date"]].merge(
            ep[["episode_id"] + KEY + ["span_start", "span_end"]], on=KEY, how="left")
        m["inside"] = (m["occur_date"] >= m["span_start"]) & (m["occur_date"] <= m["span_end"])
        first = m[m["inside"]].groupby("episode_id")["occur_date"].min()
        outside = al[~al["_rid"].isin(set(m.loc[m["inside"], "_rid"]))]
    else:
        first = pd.Series(dtype="datetime64[ns]")
        outside = alarms
    # 활성 구간 안의 알람이 없으면 비어 있는 시각형 Series 를 map 에 쓸 수 없으므로 dict 로 변환
    ep["first_alarm"] = pd.to_datetime(ep["episode_id"].map(first.to_dict()))
    ep["detected"] = ep["first_alarm"].notna()
    ep["early"] = ep["detected"] & (ep["first_alarm"] <= ep["t_fail"])
    ep["lead_min"] = np.where(ep["early"], (ep["t_fail"] - ep["first_alarm"]).dt.total_seconds() / 60, np.nan)
    ep["delay_min"] = np.where(ep["detected"], (ep["first_alarm"] - ep["t_start"]).dt.total_seconds() / 60, np.nan)

    # --- 오탐 인시던트: '활성 구간 밖'의 알람만 모아 묶는다 (거대 인시던트가 에피소드와 겹쳐 오탐이 가려지는 것을 방지) ---
    if len(outside):
        false_inc, _ = make_incidents(outside.assign(alarm=True), gap_steps)
    else:
        false_inc = pd.DataFrame(columns=KEY + ["on_nuisance"])
    n_false = len(false_inc)

    n_ports = ev[KEY].drop_duplicates().shape[0]
    total_port_days = len(ev) / 96.0
    n_ep = len(ep)
    tp, fn = int(ep["detected"].sum()), int((~ep["detected"]).sum())
    tp_e = int(ep["early"].sum())
    normal_steps = int((ev["state"] == 0).sum())

    def f1(tp_, fp_, fn_):
        return 2 * tp_ / (2 * tp_ + fp_ + fn_) if (2 * tp_ + fp_ + fn_) else 0.0

    lead = ep["lead_min"].dropna()
    out = {
        "episodes": n_ep, "ports": int(n_ports), "port_days": float(total_port_days),
        "incidents": int(len(inc)), "false_incidents": int(n_false),
        "false_incidents_on_nuisance": int(false_inc["on_nuisance"].sum()) if n_false else 0,
        "false_incidents_per_port_day": n_false / total_port_days if total_port_days else 0.0,
        "false_alarm_step_ratio": float(len(outside) / normal_steps) if normal_steps else 0.0,  # 정상 구간 중 알람 상태 비율
        "incident_precision": tp / (tp + n_false) if (tp + n_false) else 0.0,                  # 이벤트 단위 정밀도
        "episode_detection_rate": tp / n_ep if n_ep else 0.0,
        "early_detection_rate": tp_e / n_ep if n_ep else 0.0,       # failure 이전에 알람 = 예지
        "late_only_rate": (tp - tp_e) / n_ep if n_ep else 0.0,       # failure 이후에야 알람
        "missed_rate": fn / n_ep if n_ep else 0.0,
        "lead_time_median_min": float(lead.median()) if len(lead) else None,
        "lead_time_p25_min": float(lead.quantile(0.25)) if len(lead) else None,
        "lead_time_p75_min": float(lead.quantile(0.75)) if len(lead) else None,
        "lead_ge_60min_rate": float((lead >= 60).sum() / n_ep) if n_ep else 0.0,
        "detection_delay_median_min": float(ep["delay_min"].median()) if ep["delay_min"].notna().any() else None,
        # 이벤트 단위 F1 (TP=탐지 에피소드, FP=오탐 인시던트, FN=놓친 에피소드) — 단위 통일
        "event_f1": f1(tp, n_false, fn),
        "event_f1_early": f1(tp_e, n_false, n_ep - tp_e),
    }
    by_scn = {}
    for s, g in ep.groupby("scenario"):
        gl = g["lead_min"].dropna()
        by_scn[s] = {"episodes": int(len(g)), "detected": float(g["detected"].mean()),
                     "early": float(g["early"].mean()),
                     "lead_median_min": float(gl.median()) if len(gl) else None}
    out["by_scenario"] = by_scn
    out["_episodes_detail"] = ep
    return out


def summarize(ev: pd.DataFrame, episodes: pd.DataFrame, **kw) -> dict:
    res = {"step": step_metrics(ev), "event": event_metrics(ev, episodes, **kw)}
    return res
