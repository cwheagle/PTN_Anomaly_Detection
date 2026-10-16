"""
P1-2 알람 정답 처리 F1~F8 (설계서 8장, 사전 고정) 와 알람 요약·건수

대상은 6.1 의 포트 알람(row_class == 'port')이다. 장비 단위·슬롯 단위 알람은 F6(작업 대용)·F8(HW 동시 발생)에만 쓴다.
F8 은 표시만 하고 분모에서 빼지 않는다. 규칙 값은 데이터를 보기 전에 고정되었고 결과를 보고 바꾸지 않는다.
시각은 모두 KST naive.
"""
import fnmatch

import numpy as np
import pandas as pd

# 8장 사전 고정 값
FLICKER_MERGE_MIN = 60          # F1
MIN_EPISODE_MIN = 15            # F2
EVENT_MERGE_MIN = 60            # F4
WORK_WINDOW_MIN = 60            # F6
HW_WINDOW_MIN = 15              # F8
JUDGE_TYPES = ("CRC증가", "트래픽단절")
JUDGE_GRADE = "1차"
SEVERITY_LABELS = {2: "CRITICAL", 3: "MAJOR", 4: "MINOR"}

# 8.6 작업 대용 알람 (설정 값, 기본 = target_definition.md 5.3·5.2 D)
WORK_NAMES = ("LINK-FAIL-MANUAL", "BOOTING", "RESET", "SW-UPDATED", "OS-MISMATCH", "FPGA-MISMATCH", "CU-MISMATCH",
              "OPT-REMOVE", "OPT-MIS", "UNIT-REM", "UNIT-MIS")
# F8: HW·전원 제외 유형 (target_definition.md 5.2 D 의 HW·전원 행)
HW_PATTERNS = ("UNIT-FAIL", "UNIT-REM", "UNIT-MIS", "UNIT-HIGHTEMP", "FAN-*", "FUSE-*", "48V-*", "PSU-*", "BATT-*", "REC-*",
               "TEMP-OV*", "SHELF*")


def severity_label(v):
    try:
        return SEVERITY_LABELS.get(int(v), f"기타-{int(v)}")
    except (TypeError, ValueError):
        return "기타-NA"


def prepare_alarms(alarms, names, target_map, d_end=None):
    """스냅샷 알람 → 키 정리, 시각 파싱, 행 분류, 알람명·타겟 유형 부착. d_end 이후에 발생한 행은 버린다."""
    from validation.realdata.port_join import classify_alarm_rows
    a = classify_alarm_rows(alarms)
    a["occur_date"] = pd.to_datetime(a["occur_date"], errors="coerce")
    a["release_date"] = pd.to_datetime(a["release_date"], errors="coerce")
    a = a.dropna(subset=["occur_date"])
    if d_end is not None:
        a = a[a["occur_date"] < pd.Timestamp(d_end)]
    a = names.resolve_frame(a)
    a = target_map.classify_frame(a)
    a["severity"] = a["severity_type"].map(severity_label) if "severity_type" in a.columns else "기타-NA"
    return a.reset_index(drop=True)


def _targets_of(target_map, name, status):
    if status not in ("ok", "override"):
        return []
    return [(t, g) for t, g, _ in target_map.classify(name) if t != "미분류"]


def build_episodes(port_alarms, target_map, d_end, merge_min=FLICKER_MERGE_MIN, min_minutes=MIN_EPISODE_MIN):
    """F1·F2·F3·F7. 같은 (ip, cid, lid, alarm_name) 의 발생을 시간순으로 놓고 다음 occur ≤ 병합 끝 + merge_min 이면 한 에피소드.
    F3: release 가 NULL 이거나 d_end 보다 늦으면 끝을 d_end 로 자르고 ongoing=True (추출 날짜에 결과가 좌우되지 않게 — 4.7)."""
    cols = ["ip_addr", "cid", "lid", "alarm_name", "system_type", "model", "name_status", "start", "end", "duration_min",
            "ongoing", "n_alarms", "is_short", "target_grades", "n_CRITICAL", "n_MAJOR", "n_MINOR", "n_other"]
    if port_alarms.empty:
        return pd.DataFrame(columns=cols)
    d_end = pd.Timestamp(d_end)
    a = port_alarms.copy()
    rel = a["release_date"]
    a["ongoing"] = rel.isna() | (rel > d_end)
    a["end_eff"] = rel.where(~a["ongoing"], d_end)
    a["end_eff"] = a[["end_eff", "occur_date"]].max(axis=1)
    a = a.sort_values(["ip_addr", "cid", "lid", "alarm_name", "occur_date"])
    gap = pd.Timedelta(minutes=merge_min)
    out = []
    for (ip, cid, lid, name), g in a.groupby(["ip_addr", "cid", "lid", "alarm_name"], sort=False):
        cur = None
        for r in g.itertuples():
            if cur is not None and r.occur_date <= cur["end"] + gap:
                cur["end"] = max(cur["end"], r.end_eff)
                cur["ongoing"] = cur["ongoing"] or bool(r.ongoing)
                cur["n_alarms"] += 1
                cur["sev"].append(r.severity)
                continue
            if cur is not None:
                out.append(cur)
            cur = {"ip_addr": ip, "cid": cid, "lid": lid, "alarm_name": name, "system_type": r.system_type, "model": r.model,
                   "name_status": r.name_status, "start": r.occur_date, "end": r.end_eff, "ongoing": bool(r.ongoing),
                   "n_alarms": 1, "sev": [r.severity]}
        out.append(cur)
    ep = pd.DataFrame(out)
    ep["duration_min"] = (ep["end"] - ep["start"]).dt.total_seconds() / 60
    ep["is_short"] = ep["duration_min"] < min_minutes
    for lab in ("CRITICAL", "MAJOR", "MINOR"):
        ep[f"n_{lab}"] = ep["sev"].map(lambda s: sum(1 for x in s if x == lab))
    ep["n_other"] = ep["sev"].map(lambda s: sum(1 for x in s if x not in ("CRITICAL", "MAJOR", "MINOR")))
    ep["target_grades"] = [";".join(f"{t}/{g}" for t, g in _targets_of(target_map, n, s))
                           for n, s in zip(ep["alarm_name"], ep["name_status"])]
    return ep.drop(columns=["sev"])[cols].sort_values(["ip_addr", "cid", "lid", "start"]).reset_index(drop=True)


def build_target_events(episodes, merge_min=EVENT_MERGE_MIN, types=JUDGE_TYPES):
    """F4: 짧지 않은 에피소드를 (ip, cid, lid, 타겟 유형, 등급)별로 겹치거나 60분 이내면 한 타겟 이벤트로. t0 = 가장 이른 시작."""
    cols = ["ip_addr", "cid", "lid", "target_type", "grade", "t0", "end", "ongoing", "n_episodes", "alarm_names", "system_type",
            "model", "n_CRITICAL", "n_MAJOR", "n_MINOR", "n_other"]
    rows = []
    for r in episodes[~episodes["is_short"]].itertuples():
        for tg in filter(None, r.target_grades.split(";")):
            t, g = tg.split("/")
            if t in types:
                rows.append({"ip_addr": r.ip_addr, "cid": r.cid, "lid": r.lid, "target_type": t, "grade": g, "start": r.start,
                             "end": r.end, "ongoing": r.ongoing, "alarm_name": r.alarm_name, "system_type": r.system_type,
                             "model": r.model, "n_CRITICAL": r.n_CRITICAL, "n_MAJOR": r.n_MAJOR, "n_MINOR": r.n_MINOR,
                             "n_other": r.n_other})
    if not rows:
        return pd.DataFrame(columns=cols)
    x = pd.DataFrame(rows).sort_values(["ip_addr", "cid", "lid", "target_type", "grade", "start"])
    gap = pd.Timedelta(minutes=merge_min)
    out = []
    for _, g in x.groupby(["ip_addr", "cid", "lid", "target_type", "grade"], sort=False):
        cur = None
        for r in g.itertuples():
            if cur is not None and r.start <= cur["end"] + gap:
                cur["end"] = max(cur["end"], r.end)
                cur["ongoing"] = cur["ongoing"] or bool(r.ongoing)
                cur["n_episodes"] += 1
                cur["names"].add(r.alarm_name)
                for k in ("n_CRITICAL", "n_MAJOR", "n_MINOR", "n_other"):
                    cur[k] += getattr(r, k)
                continue
            if cur is not None:
                out.append(cur)
            cur = {"ip_addr": r.ip_addr, "cid": r.cid, "lid": r.lid, "target_type": r.target_type, "grade": r.grade,
                   "t0": r.start, "end": r.end, "ongoing": bool(r.ongoing), "n_episodes": 1, "names": {r.alarm_name},
                   "system_type": r.system_type, "model": r.model, "n_CRITICAL": r.n_CRITICAL, "n_MAJOR": r.n_MAJOR,
                   "n_MINOR": r.n_MINOR, "n_other": r.n_other}
        out.append(cur)
    ev = pd.DataFrame(out)
    ev["alarm_names"] = ev["names"].map(lambda s: ";".join(sorted(s)))
    return ev.drop(columns=["names"])[cols].sort_values("t0").reset_index(drop=True)


def load_maintenance(path):
    """작업 CSV(열 start, end, ip_addr, cid(선택), lid(선택), note). 키 공간은 알람 테이블의 (cid, lid) 이다."""
    m = pd.read_csv(path, dtype={"ip_addr": str})
    m["start"], m["end"] = pd.to_datetime(m["start"]), pd.to_datetime(m["end"])
    for c in ("cid", "lid"):
        if c not in m.columns:
            m[c] = np.nan
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["ip_addr"] = m["ip_addr"].str.strip()
    return m


def mark_work_and_hw(events, all_alarms, maintenance=None, work_names=WORK_NAMES, hw_patterns=HW_PATTERNS):
    """F6·F8: 트래픽단절 이벤트에만 적용. work_suspect(작업 대용 알람 또는 작업 CSV, t0 ±60분) 와 hw_coincident(같은 장비 HW·전원 알람
    ±15분, 포트·장비 단위 무관). F8 은 표시만 한다. 작업 대용 알람의 범위(8.6): 포트 알람→그 포트, 슬롯 단위→같은 (ip, cid),
    장비 단위→그 장비의 모든 포트. 판정은 알람 발생 시각이 창 안에 있는지로 한다."""
    ev = events.copy()
    ev["work_suspect"], ev["work_source"], ev["hw_coincident"], ev["hw_names"] = False, "", False, ""
    if ev.empty:
        return ev
    work = all_alarms[all_alarms["alarm_name"].isin(work_names) & all_alarms["name_status"].isin(["ok", "override"])]
    is_hw = np.array([any(fnmatch.fnmatchcase(n, p) for p in hw_patterns) for n in all_alarms["alarm_name"]], dtype=bool)
    hw = all_alarms[is_hw & all_alarms["name_status"].isin(["ok", "override"]).to_numpy()]
    w_win, h_win = pd.Timedelta(minutes=WORK_WINDOW_MIN), pd.Timedelta(minutes=HW_WINDOW_MIN)
    work_by_ip = {ip: g for ip, g in work.groupby("ip_addr")}
    hw_by_ip = {ip: g for ip, g in hw.groupby("ip_addr")}
    for i, r in ev[ev["target_type"] == "트래픽단절"].iterrows():
        srcs = []
        w = work_by_ip.get(r["ip_addr"])
        if w is not None:
            near = w[(w["occur_date"] >= r["t0"] - w_win) & (w["occur_date"] <= r["t0"] + w_win)]
            if len(near):
                scope = np.where(near["row_class"] == "device", True,
                                 np.where(near["row_class"] == "slot", near["cid"] == r["cid"],
                                          (near["cid"] == r["cid"]) & (near["lid"] == r["lid"])))
                near = near[scope.astype(bool)]
                if len(near):
                    srcs.append("alarm:" + ",".join(sorted(set(near["alarm_name"]))))
        if maintenance is not None and len(maintenance):
            m = maintenance[(maintenance["ip_addr"] == r["ip_addr"]) & (maintenance["start"] <= r["t0"] + w_win) & (maintenance["end"] >= r["t0"] - w_win)]
            m = m[(m["cid"].isna() | (m["cid"] == r["cid"])) & (m["lid"].isna() | (m["lid"] == r["lid"]))]
            if len(m):
                srcs.append("csv")
        if srcs:
            ev.at[i, "work_suspect"], ev.at[i, "work_source"] = True, ";".join(srcs)
        h = hw_by_ip.get(r["ip_addr"])
        if h is not None:
            near = h[(h["occur_date"] >= r["t0"] - h_win) & (h["occur_date"] <= r["t0"] + h_win)]
            if len(near):
                ev.at[i, "hw_coincident"], ev.at[i, "hw_names"] = True, ",".join(sorted(set(near["alarm_name"])))
    ev["is_judge"] = (ev["grade"] == JUDGE_GRADE) & ev["target_type"].isin(JUDGE_TYPES) & ~ev["work_suspect"]
    return ev


# ---- 요약 ----------------------------------------------------------------------------
def slot_only_counts(alarms):
    """주소가 슬롯만(`S.n`) 있어 포트 알람이 아니라 slot 으로 재분류된 행(cid>0, lid>0)의 system_type 별 건수 — 6.1 의 cid/lid=0 규칙과 다른 실데이터 보정."""
    from validation.realdata.port_join import norm_port
    if alarms.empty or "address" not in alarms.columns:
        return {}
    sl = alarms[(alarms["row_class"] == "slot") & (alarms["lid"] > 0)]
    if sl.empty:
        return {}
    sl = sl[sl["address"].map(lambda v: (lambda a: a != "" and "-P" not in a)(norm_port(v)))]
    return {str(k): int(v) for k, v in sl.groupby("system_type").size().items()}


def no_table_share(alarms):
    """표 없는 모델(또는 모르는 system_type) 장비 비중 — 7.3 첫 줄."""
    dev = alarms.drop_duplicates("ip_addr")
    bad = dev["name_status"].isin(["no_table", "unknown_model"])
    # 장비 단위로: 그 장비에 no_table/unknown 알람이 있으면 해당
    bad_ips = set(alarms.loc[alarms["name_status"].isin(["no_table", "unknown_model"]), "ip_addr"])
    return {"devices": int(dev.shape[0]), "devices_without_table": len(bad_ips), "share": (len(bad_ips) / dev.shape[0]) if len(dev) else 0.0,
            "models": sorted(set(alarms.loc[alarms["name_status"].isin(["no_table", "unknown_model"]), "model"].dropna()))}


def summarize_alarms(alarms, episodes, events, top=30):
    """`alarms --summary` 내용. 반환 dict (CLI 가 출력/저장)."""
    nts = no_table_share(alarms)
    names = (alarms.groupby(["alarm_name", "name_status", "target_types"]).size().rename("count").reset_index()
             .sort_values("count", ascending=False))
    unmapped = names[(names["target_types"] == "미분류") & (names["name_status"] == "ok")]
    by_class = alarms["row_class"].value_counts().to_dict()
    ports = episodes
    sens = {}
    return {
        "first_line": f"표 없는 모델 장비 {nts['devices_without_table']}/{nts['devices']} ({nts['share']:.0%}) {nts['models']}"
                      + (" — 비중 ≥ 1/3: C3 판정 보류 사유로 보고 (7.3)" if nts["share"] >= 1 / 3 else ""),
        "no_table": nts, "alarm_rows": int(len(alarms)), "slot_only_address_reclassified": slot_only_counts(alarms),
 "row_classes": by_class,
        "name_status": alarms["name_status"].value_counts().to_dict(),
        "top_names": names.head(top).to_dict("records"),
        "unclassified_top": unmapped.head(top).to_dict("records"),
        "episodes": {"merged": int(len(ports)), "short": int(ports["is_short"].sum()) if len(ports) else 0,
                     "ongoing": int(ports["ongoing"].sum()) if len(ports) else 0},
        "severity": {lab: int(alarms["severity"].eq(lab).sum()) for lab in sorted(alarms["severity"].unique())},
        "target_events": ({f"{t}/{g}": int(n) for (t, g), n in events.groupby(["target_type", "grade"]).size().items()} if len(events) else {}),
        "work_alarms": alarms[alarms["alarm_name"].isin(WORK_NAMES)]["alarm_name"].value_counts().to_dict(),
        "optical_observation_names": alarms[alarms["target_types"].str.contains("광관찰")]["alarm_name"].value_counts().head(top).to_dict(),
    }


def merge_sensitivity(port_alarms, target_map, d_end, windows=(15, 240)):
    """참고: 깜빡임 병합 간격 15분·4시간일 때의 에피소드 건수 (판정에 쓰지 않음, F1 한정)."""
    return {w: int(len(build_episodes(port_alarms, target_map, d_end, merge_min=w))) for w in windows}


def unmapped_alarm_counts(alarms):
    """표 없는 모델·모르는 system_type 의 (model, system_type, alarm_type) 별 건수 → unmapped_alarms.csv."""
    x = alarms[alarms["name_status"].isin(["no_table", "unknown_model"])]
    if x.empty:
        return pd.DataFrame(columns=["model", "system_type", "alarm_type", "count"])
    return (x.groupby([x["model"].fillna("UNKNOWN"), "system_type", "alarm_type"]).size().rename("count").reset_index()
            .rename(columns={"model": "model"}).sort_values("count", ascending=False))


# ---- 건수 (count-only) -----------------------------------------------------------------
def count_judge_events(events, start, end, data_days):
    """재생 없이 센 판정 이벤트(F5: CRC증가·트래픽단절 1차, F2·F6 제외 후) — 주별(월요일 시작)·유형별과 데이터일당 비율 r.
    성능 공백 제외(9.4)는 재생 입력(성능 스냅샷)이 필요해 여기서는 적용하지 않는다(보고에 명시)."""
    ev = events[(events["t0"] >= pd.Timestamp(start)) & (events["t0"] < pd.Timestamp(end)) & events["is_judge"]] if len(events) else events
    days_in = [d for d in data_days if pd.Timestamp(start) <= d < pd.Timestamp(end)]
    weekly = {}
    if len(ev):
        wk = ev["t0"].dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
        weekly = {w: int(n) for w, n in wk.value_counts().sort_index().items()}
    by_type = {t: int((ev["target_type"] == t).sum()) for t in JUDGE_TYPES} if len(ev) else {t: 0 for t in JUDGE_TYPES}
    return {"events": int(len(ev)), "by_type": by_type, "weekly": weekly, "data_days": len(days_in),
            "r_per_data_day": (len(ev) / len(days_in)) if days_in else float("nan"),
            "excluded_work_suspect": int(((events["t0"] >= pd.Timestamp(start)) & (events["t0"] < pd.Timestamp(end)) & events["work_suspect"]
                                          & (events["grade"] == JUDGE_GRADE)).sum()) if len(events) else 0,
            "note": "성능 공백 제외(9.4)는 적용하지 않은 건수 — 재생 입력이 필요함"}
