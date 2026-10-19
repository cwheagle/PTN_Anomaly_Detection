"""
P1-2 보고 지표 (설계서 9.4~9.6): C1 원값, C3·C4b(1h/3h/6h 분할), 단절 탐지 지연, C6, 겹침, 유형별 태깅.

- 이벤트(8장 F5)와 방법별 알람을 같은 포트 키로 맞춘다. 키 조인은 join_status(6.3)와 함께 표시되고, verified 가 아니면 C3·탐지 지연은 판정에 쓰지 않는다.
- 성능 공백 제외(9.4·9.5)와 '포트 없음' 은 분모에서 빼고 따로 센다. 판정 기준값은 사전 고정이며 바꾸지 않는다.
- 이 모듈은 DB 를 읽지 않고 운영 코드를 수정하지 않는다.
"""
import numpy as np
import pandas as pd

from validation.evaluation.metrics import KEY
from validation.realdata.port_join import KEY_CORRECTION_NOTE
from validation.realdata.replay import METHODS, incidents_of, port_days

STEP = pd.Timedelta(minutes=15)
C3_WINDOW = pd.Timedelta(hours=24)
C3_MIN_EVENTS = 20
C3_PERF_LOOKBACK = pd.Timedelta(hours=6)
C3_PERF_MIN_RATIO = 0.5
DELAY_WINDOW = pd.Timedelta(hours=6)
DELAY_PERF_WINDOW = pd.Timedelta(hours=1)
DELAY_LIST_BELOW = 10
C6_MIN_NON_EVENT_DAYS = 10
C6_MIN_MEDIAN = 10
C6_BAND = (0.5, 2.0)
C6_MAX_OUT_DAYS = 2
LEAD_BUCKETS = ((15, 60), (60, 180), (180, 360), (360, 1440))     # 분, [lo, hi)

PERF_GAP_NOTE = ("단절 정답 알람은 개발 구간에서 '알람 직후 성능 데이터가 사라지는 비율'로 골랐다. 따라서 단절 정답과 성능 공백 지표"
                 "(9.4·9.5 의 perf_gap, J3 의 '행 없음')는 부분적으로 같은 현상을 본다.")
CIRCULAR_NOTE = ("CRC증가: EMS 임계와 규칙 임계가 같은 카운터일 수 있음 — ETH-ERR 의 발생 조건을 모르는 동안 C3 해석 보류 "
                 "(target_definition.md 5.3).")
REV9_NOTE = ("개발 구간 사후 조정: 설계 개정 9 에서 J4 분모를 규칙 9·9s 로 바꾸는 등 판정 기준 일부를 개발 구간 결과(모델 9·21 J4 0.773/0.775 failed)를 본 뒤 수정했다. "
             "판정 구간은 열지 않았고 J4 기준값 0.8 은 바꾸지 않았다.")
EXPLORE_NOTE = ("설계 9.7.5a R5: 개발 기간 탐색 모델(explore_seed0, 학습 [S, X)·평가 [X, J0))의 결과는 도구 점검용이며 판정 모델·정책·설정 선택에 쓰지 않았다. "
                "탐색 결과는 header.explore_reference 에 표기 전용으로 함께 적는다 (설정 해시에 들어가지 않음).")
EXPOSURE_NOTE = ("판정 기간은 과거 데이터라 완전히 처음 보는 데이터가 아니다: 2026-10-16 lead 의 실DB 집계로 07-10~10-12 의 원 알람 월별 건수와 "
                 "상위 (system_type, alarm_type) 조합이 이미 알려졌고 사용자가 EMS 화면으로 그 기간 사건을 기억할 수 있다. 판정 이벤트·AI·베이스라인 "
                 "결과는 노출되지 않았다 (설계 9.7.2).")


def floor15(t):
    return pd.Timestamp(t).floor("15min")


def bin_end(t):
    """t0 를 처음 담을 수 있는 구간의 끝: floor15(t0) + 15분 (t0 가 정각 경계이면 다음 구간, 설계 9.5)."""
    return floor15(t) + STEP


class PerfIndex:
    """포트 키 → 정상 성능 슬롯(구간 끝) 정렬 배열. 슬롯 수 질의는 이분 탐색."""

    def __init__(self, eth):
        col = "bin_end" if "bin_end" in eth.columns else "occur_date"
        e = eth.assign(_t=pd.to_datetime(eth[col]), ip_addr=eth["ip_addr"].astype(str).str.strip())
        self.slots = {k: np.sort(g["_t"].to_numpy(dtype="datetime64[ns]")) for k, g in e.groupby(KEY)}
        self.devices = set(e["ip_addr"])

    def has(self, key):
        return key in self.slots

    def count(self, key, lo, hi_exclusive):
        a = self.slots.get(key)
        if a is None:
            return 0
        return int(np.searchsorted(a, np.datetime64(pd.Timestamp(hi_exclusive)), side="left")
                   - np.searchsorted(a, np.datetime64(pd.Timestamp(lo)), side="left"))

    def count_closed(self, key, lo, hi):
        a = self.slots.get(key)
        if a is None:
            return 0
        return int(np.searchsorted(a, np.datetime64(pd.Timestamp(hi)), side="right")
                   - np.searchsorted(a, np.datetime64(pd.Timestamp(lo)), side="left"))


class AlarmIndex:
    """방법별 알람 시각 (포트 키 → 정렬 배열)."""

    def __init__(self, frame):
        a = frame[frame["alarm"].astype(bool)]
        self.times = {k: np.sort(g["occur_date"].to_numpy(dtype="datetime64[ns]")) for k, g in a.groupby(KEY)}

    def first_in(self, key, lo, hi):
        """[lo, hi] 안의 첫 알람 시각 (없으면 None)."""
        a = self.times.get(key)
        if a is None:
            return None
        i = np.searchsorted(a, np.datetime64(pd.Timestamp(lo)), side="left")
        if i < len(a) and a[i] <= np.datetime64(pd.Timestamp(hi)):
            return pd.Timestamp(a[i])
        return None


def _key(r, shift):
    return (r["ip_addr"], int(r["cid"]) + shift, int(r["lid"]))


# ---- C1 원값 ------------------------------------------------------------------------------------
def c1_raw(frames):
    out = {}
    for m in METHODS:
        f = frames[m]
        inc, _ = incidents_of(f)
        pdays = port_days(f)
        out[m] = {"alarm_rows": int(f["alarm"].astype(bool).sum()), "incidents": int(len(inc)), "port_days": pdays,
                  "incidents_per_port_day": (len(inc) / pdays) if pdays else float("nan"),
                  "slot_basis_port_days": (len(f) / 96.0),
                  "incidents_per_slot_day": (len(inc) / (len(f) / 96.0)) if len(f) else float("nan"),
                  "severity": (inc["severity_max"].value_counts().to_dict() if "severity_max" in inc.columns and len(inc) else {})}
    return out


# ---- C3 ----------------------------------------------------------------------------------------
def c3_events(events, frames, perf, shift=0):
    """이벤트별 상태·방법별 선행 탐지. 반환 DataFrame (행당 이벤트)."""
    idx = {m: AlarmIndex(frames[m]) for m in METHODS}
    rows = []
    for _, r in events.iterrows():
        key, t0 = _key(r, shift), pd.Timestamp(r["t0"])
        rec = {"ip_addr": r["ip_addr"], "cid": r["cid"], "lid": r["lid"], "target_type": r["target_type"], "grade": r["grade"], "t0": t0,
               "alarm_names": r["alarm_names"] if "alarm_names" in r.index else ""}
        if not perf.has(key):
            rec["status"] = "no_port"
        else:
            slots = perf.count(key, t0 - C3_PERF_LOOKBACK, t0)
            rec["perf_slots_6h"] = slots
            rec["status"] = "perf_gap" if slots < C3_PERF_MIN_RATIO * 24 else "ok"
        if rec["status"] == "ok":
            for m in METHODS:
                first = idx[m].first_in(key, t0 - C3_WINDOW, t0 - STEP)
                rec[f"{m}_lead_min"] = (t0 - first).total_seconds() / 60 if first is not None else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def c3_summary(ev_rows, hold_reason=None):
    """include = status ok. 방법별 선행 탐지율·1h/3h/6h 누적 비율·구간별 건수. 20건 미만이면 판정 보류(값은 참고)."""
    inc = ev_rows[ev_rows["status"] == "ok"] if len(ev_rows) else ev_rows
    n = int(len(inc))
    only = ev_rows["alarm_names"] == "ETH-LINK-FAIL" if len(ev_rows) and "alarm_names" in ev_rows.columns else pd.Series([], dtype=bool)
    out = {"notes": [PERF_GAP_NOTE], "eth_link_fail_only": {"total": int(only.sum()), "included": int((only & (ev_rows["status"] == "ok")).sum()) if len(only) else 0,
                                                         "note": "ETH-LINK-FAIL 은 성능 직후 공백 비율이 낮은데도 사용자 결정으로 1차 (7.4) — 단독 이벤트 수를 따로 적는다"},
           "events_total": int(len(ev_rows)), "included": n,
           "excluded": {"no_port": int((ev_rows["status"] == "no_port").sum()) if len(ev_rows) else 0,
                        "perf_gap": int((ev_rows["status"] == "perf_gap").sum()) if len(ev_rows) else 0},
           "hold": bool(n < C3_MIN_EVENTS or hold_reason), "hold_reasons": [], "methods": {}}
    if n < C3_MIN_EVENTS:
        out["hold_reasons"].append(f"판정 이벤트 {n} < {C3_MIN_EVENTS}")
    if hold_reason:
        out["hold_reasons"].append(hold_reason)
    for m in METHODS:
        lead = inc[f"{m}_lead_min"] if n else pd.Series(dtype=float)
        det = lead.notna()
        res = {"detected": int(det.sum()), "rate": (float(det.mean()) if n else float("nan"))}
        for h in (60, 180, 360):
            res[f"lead_ge_{h // 60}h"] = float((lead >= h).sum() / n) if n else float("nan")
        last = LEAD_BUCKETS[-1]
        res["buckets"] = {(f"[{lo},{hi}]min" if (lo, hi) == last else f"[{lo},{hi})min"):    # 설계 9.4: 마지막 구간 [6h,24h] 는 닫힌 구간
                          int(((lead >= lo) & ((lead <= hi) if (lo, hi) == last else (lead < hi))).sum()) for lo, hi in LEAD_BUCKETS}
        out["methods"][m] = res
    return out


def c4b(summary):
    if summary["hold"]:
        return {"status": "hold", "reason": "C3 판정 보류"}
    ai, fx = summary["methods"]["ai"]["rate"], summary["methods"]["fixed"]["rate"]
    return {"status": "ok", "ai_gt_fixed": bool(ai > fx), "ai_rate": ai, "fixed_rate": fx}


# ---- 단절 탐지 지연 (R4) -------------------------------------------------------------------------------
def delay_events(events, frames, perf, d_end, shift=0):
    idx = {m: AlarmIndex(frames[m]) for m in METHODS}
    rows = []
    for _, r in events.iterrows():
        key, t0 = _key(r, shift), pd.Timestamp(r["t0"])
        be = bin_end(t0)
        rec = {"ip_addr": r["ip_addr"], "cid": r["cid"], "lid": r["lid"], "t0": t0, "bin_end": be,
               "alarm_names": r["alarm_names"] if "alarm_names" in r.index else ""}
        if t0 + DELAY_WINDOW > pd.Timestamp(d_end):
            rec["status"] = "truncated"
        elif not perf.has(key):
            rec["status"] = "no_port"
        elif perf.count_closed(key, be, t0 + DELAY_PERF_WINDOW) == 0:
            rec["status"] = "perf_gap"
        else:
            rec["status"] = "ok"
        if rec["status"] == "ok":
            for m in METHODS:
                det = idx[m].first_in(key, be, t0 + DELAY_WINDOW)
                pre = idx[m].first_in(key, t0 - STEP, be - pd.Timedelta(seconds=1))
                if pre is not None and (det is None or pre < det):
                    rec[f"{m}_pre15"] = True
                    rec[f"{m}_wall_min"] = (be - t0).total_seconds() / 60            # 구조적 하한(15분 데이터)
                    rec[f"{m}_step"] = 0
                elif det is not None:
                    rec[f"{m}_pre15"] = False
                    rec[f"{m}_wall_min"] = (det - t0).total_seconds() / 60
                    rec[f"{m}_step"] = int((det - be) / STEP)
                else:
                    rec[f"{m}_pre15"] = False
                    rec[f"{m}_wall_min"], rec[f"{m}_step"] = np.nan, np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def delay_summary(dr, join_status):
    inc = dr[dr["status"] == "ok"] if len(dr) else dr
    n = int(len(inc))
    only = dr["alarm_names"] == "ETH-LINK-FAIL" if len(dr) and "alarm_names" in dr.columns else pd.Series([], dtype=bool)
    total = int(len(dr))
    gap = int((dr["status"] == "perf_gap").sum()) if total else 0
    out = {"first_line": f"perf_gap {gap}/{total} ({(gap / total) if total else float('nan'):.1%}) — 사건 직후 성능 행이 없어 지연 분모에서 제외된 비율", "perf_gap_ratio": (gap / total) if total else float("nan"),
           "notes": [PERF_GAP_NOTE, "남은 이벤트(행이 남아 있는 단절)는 대표성이 낮은 소수일 수 있다 (8.7)"],
           "eth_link_fail_only": {"total": int(only.sum()), "included": int((only & (dr["status"] == "ok")).sum()) if len(only) else 0},
           "events_total": total, "included": n, "join_status": join_status,
           "excluded": {k: int((dr["status"] == k).sum()) if len(dr) else 0 for k in ("truncated", "no_port", "perf_gap")},
           "list_individually": n < DELAY_LIST_BELOW, "methods": {}}
    for m in METHODS:
        w = inc[f"{m}_wall_min"] if n else pd.Series(dtype=float)
        st = inc[f"{m}_step"] if n else pd.Series(dtype=float)
        det = w.notna()
        res = {"detected": int(det.sum()), "missed_rate": float(1 - det.mean()) if n else float("nan"),
               "pre15": int(inc[f"{m}_pre15"].sum()) if n else 0}
        if not out["list_individually"]:
            res["median_wall_min"] = float(w.median()) if det.any() else float("nan")
            for lim in (15, 30, 60):
                res[f"le_{lim}min"] = float((w <= lim).sum() / n)
            res["steps"] = {"0": int((st == 0).sum()), "1": int((st == 1).sum()), "2": int((st == 2).sum()), "3+": int((st >= 3).sum())}
        out["methods"][m] = res
    if out["list_individually"]:
        out["individual"] = inc.to_dict("records")
    return out


# ---- C6 ----------------------------------------------------------------------------------------
def c6(frames, data_days, start, end, network_event_days=()):
    """일별 인시던트 수 (데이터일만). 망 이벤트일 = 작업 CSV 의 날(원칙)만 판정에 쓴다."""
    days = [d for d in data_days if pd.Timestamp(start) <= d < pd.Timestamp(end)]
    ev_days = {pd.Timestamp(d).normalize() for d in network_event_days}
    out = {"data_days": len(days), "network_event_days": len([d for d in days if d in ev_days]), "methods": {}}
    for m in METHODS:
        inc, _ = incidents_of(frames[m])
        per = (inc["t_first"].dt.normalize().value_counts() if len(inc) else pd.Series(dtype=int)).reindex(days, fill_value=0)
        med = float(per.median()) if len(per) else float("nan")
        non_event = per[[d for d in days if d not in ev_days]]
        res = {"median": med, "non_event_days": int(len(non_event))}
        hold = []
        if len(non_event) < C6_MIN_NON_EVENT_DAYS:
            hold.append(f"망 이벤트 없는 날 {len(non_event)} < {C6_MIN_NON_EVENT_DAYS}")
        if not med >= C6_MIN_MEDIAN:
            hold.append(f"일별 중앙값 {med:.1f} < {C6_MIN_MEDIAN}")
        res["hold"], res["hold_reasons"] = bool(hold), hold
        if med == med:
            out_days = non_event[(non_event < C6_BAND[0] * med) | (non_event > C6_BAND[1] * med)]
            res["out_of_band_days"] = int(len(out_days))
            res["pass"] = None if hold else bool(len(out_days) <= C6_MAX_OUT_DAYS)
        res["daily"] = {str(d.date()): int(v) for d, v in per.items()}
        out["methods"][m] = res
    return out


# ---- 겹침·유형 태깅 ------------------------------------------------------------------------------
def overlap(frames):
    a, _ = incidents_of(frames["ai"])
    r, _ = incidents_of(frames["rolling"])
    rd = {k: g for k, g in r.groupby(KEY)}

    def hit(row, other):
        g = other.get(tuple(row[k] for k in KEY))
        return g is not None and bool(((g["t_first"] <= row["t_last"]) & (g["t_last"] >= row["t_first"])).any())

    ai_hit = [hit(row, rd) for _, row in a.iterrows()]
    ad = {k: g for k, g in a.groupby(KEY)}
    r_hit = [hit(row, ad) for _, row in r.iterrows()]
    return {"ai_incidents": int(len(a)), "ai_overlapping_rolling": int(sum(ai_hit)), "ai_only": int(len(a) - sum(ai_hit)),
            "rolling_incidents": int(len(r)), "rolling_overlapping_ai": int(sum(r_hit)), "rolling_only": int(len(r) - sum(r_hit))}


def tag_ai_incidents(frames, episodes, shift=0, margin=pd.Timedelta(minutes=60)):
    """AI 인시던트를 같은 포트 ±60분의 EMS 에피소드 타겟 유형으로 태깅한 분포 (참고, 판정 아님)."""
    a, _ = incidents_of(frames["ai"])
    if a.empty:
        return {"incidents": 0, "tags": {}}
    ep = episodes.assign(_cid=episodes["cid"].astype(int) + shift)
    by = {k: g for k, g in ep.groupby(["ip_addr", "_cid", "lid"])}
    tags = {}
    for _, row in a.iterrows():
        g = by.get((row["ip_addr"], int(row["cid"]), int(row["lid"])))
        label = "EMS 에피소드 없음"
        if g is not None:
            near = g[(g["start"] <= row["t_last"] + margin) & (g["end"] >= row["t_first"] - margin)]
            if len(near):
                tg = sorted({t for s in near["target_grades"] for t in (s.split(";") if s else ["미분류"])})
                label = ",".join(tg) if tg else "미분류"
        tags[label] = tags.get(label, 0) + 1
    return {"incidents": int(len(a)), "tags": dict(sorted(tags.items(), key=lambda kv: -kv[1]))}


# ---- 조립 ----------------------------------------------------------------------------------------
def build_report(ctx_events, ctx_episodes, ctx_alarms_info, replay_result, eth, d_end, data_days, join_status, purpose,
                 header, shift=0, network_event_days=()):
    frames, (start, end) = replay_result["frames"], replay_result["range"]
    perf = PerfIndex(eth)
    ev = ctx_events[(ctx_events["t0"] >= start) & (ctx_events["t0"] < end) & ctx_events["is_judge"]]
    hold = None if join_status == "verified" else f"join_status = {join_status} (6.3)"
    rep = {"header": {**header, "purpose": purpose, "join_status": join_status, "range": [start, end], "key_shift_diagnostic": shift,
                      "slots": replay_result["slots"], "baselines_use_optical": False,
                      "fixed_notes": [KEY_CORRECTION_NOTE, PERF_GAP_NOTE, CIRCULAR_NOTE] + [REV9_NOTE] + ([EXPOSURE_NOTE, EXPLORE_NOTE] if purpose == "judge" else [])},
           "c1_raw": c1_raw(frames), "overlap": overlap(frames)}
    rep["c3"] = {}
    for tt in ("합계", "CRC증가", "트래픽단절"):
        sub = ev if tt == "합계" else ev[ev["target_type"] == tt]
        rows = c3_events(sub, frames, perf, shift)
        rep["c3"][tt] = c3_summary(rows, hold if tt == "합계" else (hold or "유형별 이벤트 — 판정은 합계로 (참고)"))
        if tt == "합계":
            rep["c3_events"] = rows
    rep["c3"]["합계"]["c4b"] = c4b(rep["c3"]["합계"])
    aux = ctx_events[(ctx_events["t0"] >= start) & (ctx_events["t0"] < end) & (ctx_events["grade"] == "보조")]
    rep["c3_aux_reference"] = c3_summary(c3_events(aux, frames, perf, shift), "보조 등급 — 참고")
    cut = ev[ev["target_type"] == "트래픽단절"]
    dr = delay_events(cut, frames, perf, d_end, shift)
    rep["delay"] = delay_summary(dr, join_status)
    rep["delay_events"] = dr
    rep["c6"] = c6(frames, data_days, start, end, network_event_days)
    rep["ai_incident_tags"] = tag_ai_incidents(frames, ctx_episodes, shift)
    rep["excluded_counts"] = {"short_episodes": ctx_alarms_info.get("short_episodes"), "work_suspect": int(
        ((ctx_events["t0"] >= start) & (ctx_events["t0"] < end) & ctx_events["work_suspect"] & (ctx_events["grade"] == "1차")).sum()),
        "hw_coincident_marked": int(((ctx_events["t0"] >= start) & (ctx_events["t0"] < end) & ctx_events["hw_coincident"]).sum()),
        "no_table_devices": ctx_alarms_info.get("no_table_devices")}
    return rep
