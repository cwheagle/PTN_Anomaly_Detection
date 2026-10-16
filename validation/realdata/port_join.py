"""
P1-2 포트 조인과 안전 검사 J1~J4 (설계서 6장)

조인 키 = (ip_addr, cid, lid) (사용자 결정 D13). address 의 `S.n-Pm` 은 진단용 보조 열(addr_port)일 뿐 키가 아니다.
J 판정이 failed 여도 address 조인으로 자동 전환하지 않는다(사용자 결정 사항). 기준값은 설계서 6.3 에 사전 고정되어 있고
결과를 본 뒤 바꾸지 않는다.
"""
import re

import numpy as np
import pandas as pd

# 6.3 사전 고정 값
J1_MIN_RATIO = 0.95
J3_MIN_MATCH = 0.8
J3_MIN_GAP = 0.5            # (a) − (c)
J3_MIN_EPISODES = 5
J3_MIN_MINUTES = 30
J3_MAX_HOURS = 24
J3_SEED = 0
BIN = pd.Timedelta(minutes=15)

_ADDR_RE = re.compile(r"S\.\s*(\d+)\s*-\s*P\s*(\d+)", re.IGNORECASE)
_SLOT_RE = re.compile(r"S\.\s*(\d+)", re.IGNORECASE)


def norm_port(address):
    """address → 'S.<n>-P<m>' (정규식의 마지막 일치), 없으면 'S.<n>' 또는 ''."""
    if address is None or (isinstance(address, float) and np.isnan(address)) or pd.isna(address):
        return ""
    s = str(address)
    m = list(_ADDR_RE.finditer(s))
    if m:
        return f"S.{int(m[-1].group(1))}-P{int(m[-1].group(2))}"
    m = list(_SLOT_RE.finditer(s))
    return f"S.{int(m[-1].group(1))}" if m else ""


def prep_keys(df):
    """ip_addr 공백 제거, cid/lid 를 nullable 정수로. 사본 반환."""
    out = df.copy()
    out["ip_addr"] = out["ip_addr"].astype(str).str.strip()
    for c in ("cid", "lid"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


DEVICE_CID, DEVICE_LID = 256, 255      # 실데이터(2026-10-16 스냅샷): 장비 단위 알람은 cid=256, lid=255, address='SYSTEM'


def classify_alarm_rows(alarms):
    """6.1 + 실데이터 보정: row_class = port | slot | device | bad_key.
    설계서 6.1 의 cid=0/lid=0 규칙에 더해, 실데이터에서 확인된 장비 단위 표기(cid=256 또는 lid=255 또는 address='SYSTEM')를
    device 로, 포트 없이 슬롯만 있는 address(`S.n`, lid 가 0 이 아니어도)를 slot 으로 분류한다. 판정(J)에는 쓰지 않는 분류다."""
    out = prep_keys(alarms)
    cid, lid = out["cid"], out["lid"]
    addr = out["address"].astype(str).str.strip().str.upper() if "address" in out.columns else pd.Series("", index=out.index)
    slot_only = out["address"].map(norm_port).map(lambda v: v != "" and "-P" not in v) if "address" in out.columns \
        else pd.Series(False, index=out.index)
    cls = pd.Series("bad_key", index=out.index, dtype="object")
    ok = cid.notna() & lid.notna() & (cid >= 0) & (lid >= 0)
    cls[ok & (cid > 0) & (lid > 0)] = "port"
    cls[ok & (cid > 0) & (lid == 0)] = "slot"
    cls[ok & (cid > 0) & (lid > 0) & slot_only] = "slot"
    cls[ok & (cid == 0)] = "device"
    cls[ok & ((cid == DEVICE_CID) | (lid == DEVICE_LID) | (addr == "SYSTEM"))] = "device"
    out["row_class"] = cls
    return out


def add_diag_columns(df, alarm=False):
    """addr_port 진단 열 (+ 알람은 p5_same)."""
    out = df.copy()
    out["addr_port"] = out["address"].map(norm_port) if "address" in out.columns else ""
    if alarm and {"p5_cid", "p5_lid"} <= set(out.columns):
        out["p5_same"] = (pd.to_numeric(out["p5_cid"], errors="coerce") == pd.to_numeric(out["cid"], errors="coerce")) & \
                         (pd.to_numeric(out["p5_lid"], errors="coerce") == pd.to_numeric(out["lid"], errors="coerce"))
    return out


def key_table(eth):
    """ETH 스냅샷 → 키별 한 행 (ip_addr, cid, lid, system_type, addr_port[최빈값])."""
    e = add_diag_columns(prep_keys(eth))
    if "system_type" not in e.columns:
        e["system_type"] = np.nan
    cols = ["ip_addr", "cid", "lid"]
    g = e.groupby(cols, dropna=False)
    kt = g.agg(system_type=("system_type", "first"),
               addr_port=("addr_port", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
               n_addr=("addr_port", lambda s: s[s != ""].nunique()),
               rows=("cid", "size")).reset_index()
    return kt


# ---- J1 ------------------------------------------------------------------------------
def j1_key_consistency(eth, optic=None):
    """장비별 (cid,lid) ↔ addr_port 1:1 비율. 키가 둘 이상의 addr_port 를 갖거나, addr_port 가 둘 이상의 키에 걸리면 1:1 아님.
    address 가 비어 있는 키는 분모에서 뺀다(no_address). 광이 있으면 ETH 키와의 겹침률."""
    kt = key_table(eth)
    kt_addr = kt[kt["addr_port"] != ""].copy()
    fan = kt_addr.groupby(["ip_addr", "addr_port"])["cid"].transform("size")
    kt_addr["keys_per_addr"] = fan
    kt_addr["one_to_one"] = (kt_addr["n_addr"] == 1) & (kt_addr["keys_per_addr"] == 1)
    n = len(kt_addr)
    out = {"keys": len(kt), "keys_with_address": n, "no_address": len(kt) - n,
           "one_to_one_ratio": (float(kt_addr["one_to_one"].mean()) if n else float("nan")),
           "not_one_to_one": kt_addr.loc[~kt_addr["one_to_one"], ["ip_addr", "cid", "lid", "addr_port"]].head(50).to_dict("records"),
           "not_one_to_one_count": int((~kt_addr["one_to_one"]).sum())}
    if optic is not None and len(optic):
        ok = key_table(optic)
        ek = set(zip(kt["ip_addr"], kt["cid"], kt["lid"]))
        ov = [(k in ek) for k in zip(ok["ip_addr"], ok["cid"], ok["lid"])]
        out["optic_eth_overlap"] = float(np.mean(ov)) if ov else float("nan")
    else:
        out["optic_eth_overlap"] = None
    return out


# ---- J2 (진단, 판정 아님) -----------------------------------------------------------------
def j2_diagnosis(alarms_cls, eth):
    """키로 조인된 알람·성능 쌍의 addr_port 일치를 system_type 별로. 장비 단위 알람의 address 형태, p5_same 비율."""
    kt = key_table(eth)[["ip_addr", "cid", "lid", "addr_port"]].rename(columns={"addr_port": "perf_addr_port"})
    a = add_diag_columns(alarms_cls, alarm=True)
    ports = a[a["row_class"] == "port"].merge(kt, on=["ip_addr", "cid", "lid"], how="inner")
    ports = ports[(ports["addr_port"] != "") & (ports["perf_addr_port"] != "")]
    ports["same"] = ports["addr_port"] == ports["perf_addr_port"]
    by_model = ports.groupby("system_type").agg(pairs=("same", "size"), same=("same", "sum")).reset_index()
    by_model["diff_rate"] = 1 - by_model["same"] / by_model["pairs"]

    def slot_shift(r):
        ma, mp = _SLOT_RE.search(r["addr_port"]), _SLOT_RE.search(r["perf_addr_port"])
        return int(ma.group(1)) - int(mp.group(1)) if ma and mp else None

    diff = ports[~ports["same"]]
    shifts = diff.apply(slot_shift, axis=1).dropna() if len(diff) else pd.Series(dtype=float)
    by_model["slot_shift_mode"] = by_model["system_type"].map(
        lambda st: (lambda s: int(s.mode().iloc[0]) if len(s) else None)(shifts[diff.loc[shifts.index, "system_type"] == st]))
    dev = a[a["row_class"].isin(["device", "slot"])]
    shape = dev.groupby(["row_class", dev["addr_port"].map(lambda v: "empty" if v == "" else ("slot-only" if "-P" not in v else "port-form"))]).size()
    p5 = a.groupby("system_type")["p5_same"].mean() if "p5_same" in a.columns else pd.Series(dtype=float)
    return {"by_model": by_model.to_dict("records"), "row_class_reclassification": reclass_report(alarms_cls),
            "device_alarm_address_shape": {f"{k[0]}/{k[1]}": int(v) for k, v in shape.items()},
            "p5_same_ratio": {str(k): float(v) for k, v in p5.items()},
            "diff_samples": diff[["ip_addr", "system_type", "cid", "lid", "addr_port", "perf_addr_port"]].head(20).to_dict("records")}


# ---- J3 ------------------------------------------------------------------------------
def _grid_bins(start, end):
    """[start, end] 에 완전히 포함된 15분 구간 (T−15분, T] 의 T 목록 (D11: occur_date = 구간 끝)."""
    first = (pd.Timestamp(start) + BIN).ceil("15min")
    last = pd.Timestamp(end).floor("15min")
    return pd.date_range(first, last, freq="15min") if first <= last else pd.DatetimeIndex([])


def _alive_set(eth):
    """(ip_addr, bin_end) — 그 장비의 어떤 포트든 그 구간에 행이 있으면 '장비 생존'(진단용)."""
    e = prep_keys(eth)
    b = pd.to_datetime(e["bin_end"] if "bin_end" in e.columns else e["occur_date"])
    return set(zip(e["ip_addr"], b))


def _rx_lookup(eth):
    e = prep_keys(eth)
    e["bin_end"] = pd.to_datetime(e["bin_end"] if "bin_end" in e.columns else e["occur_date"])
    e = e.drop_duplicates(subset=["ip_addr", "cid", "lid", "bin_end"], keep="last")
    return {k: v for k, v in zip(zip(e["ip_addr"], e["cid"], e["lid"], e["bin_end"]), pd.to_numeric(e["rx_packet"], errors="coerce"))}


def _match_stats(lookup, key, bins):
    """key 의 bins 중 rx==0 또는 행 없음. 반환 (matched, zero, missing, total). rx NaN(행은 있으나 값 없음)은 일치로 세지 않는다."""
    zero = missing = 0
    for t in bins:
        v = lookup.get((key[0], key[1], key[2], t), "absent")
        if isinstance(v, str):
            missing += 1
        elif v == 0:
            zero += 1
    return zero + missing, zero, missing, len(bins)


def j3_physical_match(episodes, eth, seed=J3_SEED, min_minutes=J3_MIN_MINUTES, max_hours=J3_MAX_HOURS):
    """단절 1차 포트 알람 에피소드(열: ip_addr, cid, lid, start, end)마다 구간(시작부터 최대 24시간)에 완전히 포함된 15분 구간의
    rx_packet == 0 또는 행 없음 비율. (a) 키 조인, (c) 같은 장비의 다른 임의 포트(고정 시드), (d) addr_port 조인(참고)."""
    lookup = _rx_lookup(eth)
    alive = _alive_set(eth)
    kt = key_table(eth)
    by_ip = {ip: list(zip(g["cid"], g["lid"])) for ip, g in kt.groupby("ip_addr")}
    addr_of = {(r.ip_addr, r.cid, r.lid): r.addr_port for r in kt.itertuples()}
    by_addr = {}
    for r in kt.itertuples():
        if r.addr_port:
            by_addr.setdefault((r.ip_addr, r.addr_port), []).append((r.cid, r.lid))
    sys_of = {(r.ip_addr, r.cid, r.lid): r.system_type for r in kt.itertuples()}
    rng = np.random.RandomState(seed)
    ep = prep_keys(episodes).sort_values(["ip_addr", "cid", "lid", "start"]).reset_index(drop=True)
    rows = []
    for r in ep.itertuples():
        start, end = pd.Timestamp(r.start), pd.Timestamp(r.end)
        if (end - start) < pd.Timedelta(minutes=min_minutes):
            continue
        end = min(end, start + pd.Timedelta(hours=max_hours))
        bins = _grid_bins(start, end)
        key = (r.ip_addr, r.cid, r.lid)
        rec = {"ip_addr": r.ip_addr, "cid": r.cid, "lid": r.lid, "start": start, "bins": len(bins), "in_snapshot": key in addr_of,
               "system_type": sys_of.get(key)}
        if len(bins) == 0 or key not in addr_of:        # 성능 스냅샷에 없는 키는 J4 몫 — 전부 '행 없음'으로 일치 처리하지 않는다
            rows.append({**rec, "a": np.nan, "c": np.nan, "d": np.nan, "a_alive": np.nan, "c_alive": np.nan})
            continue
        m, z, ms, tot = _match_stats(lookup, key, bins)
        rec.update(a=m / tot, a_zero=z / tot, a_missing=ms / tot)
        live_bins = [t for t in bins if (r.ip_addr, t) in alive]                  # 진단: 장비가 살아 있는 구간만
        rec["n_alive_bins"] = len(live_bins)
        rec["a_alive"] = _match_stats(lookup, key, live_bins)[0] / len(live_bins) if live_bins else np.nan
        others = [k for k in by_ip.get(r.ip_addr, []) if k != (r.cid, r.lid)]
        if others:
            ck = others[rng.randint(len(others))]
            rec["c"] = _match_stats(lookup, (r.ip_addr, ck[0], ck[1]), bins)[0] / tot
            rec["c_alive"] = _match_stats(lookup, (r.ip_addr, ck[0], ck[1]), live_bins)[0] / len(live_bins) if live_bins else np.nan
        else:
            rec["c"] = np.nan
            rec["c_alive"] = np.nan
        alarm_addr = getattr(r, "addr_port", "") if hasattr(r, "addr_port") else ""
        cand = by_addr.get((r.ip_addr, alarm_addr), []) if alarm_addr else []
        rec["d"] = _match_stats(lookup, (r.ip_addr, cand[0][0], cand[0][1]), bins)[0] / tot if cand else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    valid = df[df["bins"].gt(0) & df["in_snapshot"]] if len(df) else df
    out = {"episodes": len(df), "with_bins": len(valid),
           "not_in_snapshot": int((~df["in_snapshot"]).sum()) if len(df) else 0}
    for k in ("a", "c", "d"):
        out[f"match_{k}"] = float(valid[k].mean()) if len(valid) and valid[k].notna().any() else float("nan")
    out["a_zero"] = float(valid["a_zero"].mean()) if len(valid) else float("nan")
    va = valid[valid["a_alive"].notna()] if len(valid) else valid
    out["diag_alive"] = {"episodes": int(len(va)), "match_a_alive": float(va["a_alive"].mean()) if len(va) else float("nan"),
                         "match_c_alive": float(va["c_alive"].mean()) if len(va) and va["c_alive"].notna().any() else float("nan"),
                         "note": "장비의 다른 포트에 행이 있는 구간만 본 진단값 — 판정에 쓰지 않음"}
    out["a_missing"] = float(valid["a_missing"].mean()) if len(valid) else float("nan")
    out["n_control"] = int(valid["c"].notna().sum()) if len(valid) else 0
    out["by_model"] = ({str(k): {"episodes": int(len(g)), "match_a": float(g["a"].mean())}
                        for k, g in valid.groupby("system_type", dropna=False)} if len(valid) else {})
    out["detail"] = df
    return out


# ---- J4 ------------------------------------------------------------------------------
def j4_coverage(alarms_cls, eth):
    """포트 알람 중 같은 키의 성능 포트가 있는 비율 (모델별), 못 찾은 알람 표본 20건."""
    kt = key_table(eth)
    keys = set(zip(kt["ip_addr"], kt["cid"], kt["lid"]))
    p = alarms_cls[alarms_cls["row_class"] == "port"].copy()
    p["matched"] = [(k in keys) for k in zip(p["ip_addr"], p["cid"], p["lid"])]
    ips = set(kt["ip_addr"])
    p["device_in_perf"] = p["ip_addr"].isin(ips)
    by_model = p.groupby("system_type").agg(alarms=("matched", "size"), matched=("matched", "mean")).reset_index()
    return {"port_alarms": len(p), "matched_ratio": float(p["matched"].mean()) if len(p) else float("nan"),
            "device_coverage": float(p["device_in_perf"].mean()) if len(p) else float("nan"),
            "by_model": by_model.to_dict("records"),
            "unmatched_samples": p.loc[~p["matched"], ["ip_addr", "system_type", "cid", "lid", "alarm_type", "occur_date"]].head(20).to_dict("records")}


# ---- 판정 ----------------------------------------------------------------------------
def judge_join(j1, j3, min_episodes=J3_MIN_EPISODES):
    """6.3 판정(사전 고정). 반환 dict(status = verified | failed | unverified, reasons).
    address 조인(d)이 기준을 만족해도 자동으로 바꾸지 않는다(address_join_would_pass 는 참고 표시)."""
    reasons = []
    n = j3.get("with_bins", 0)
    a, c, d = j3.get("match_a"), j3.get("match_c"), j3.get("match_d")
    r1 = j1.get("one_to_one_ratio")
    d_pass = bool(d is not None and not np.isnan(d) and d >= J3_MIN_MATCH and c is not None and not np.isnan(c) and d - c >= J3_MIN_GAP)
    if n < min_episodes:
        return {"status": "unverified", "reasons": [f"J3 대상 에피소드 {n}건 < {min_episodes} — 판정 불가 (C3·탐지 지연 출력에 join_status=unverified 표시, 판정에 쓰지 않음)"],
                "address_join_would_pass": d_pass}
    if r1 is None or np.isnan(r1) or c is None or np.isnan(c) or a is None or np.isnan(a):
        return {"status": "unverified", "reasons": ["J1 비율 또는 J3 대조군을 계산할 수 없음"], "address_join_would_pass": d_pass}
    if r1 < J1_MIN_RATIO:
        reasons.append(f"J1 1:1 비율 {r1:.3f} < {J1_MIN_RATIO}")
    if a < J3_MIN_MATCH:
        reasons.append(f"J3 일치율 {a:.3f} < {J3_MIN_MATCH}")
    if a - c < J3_MIN_GAP:
        reasons.append(f"(a)−(c) {a - c:.3f} < {J3_MIN_GAP}")
    status = "verified" if not reasons else "failed"
    return {"status": status, "reasons": reasons, "address_join_would_pass": d_pass,
            "stop": status == "failed"}


# ---- M1 ------------------------------------------------------------------------------
def load_manual(path):
    return pd.read_csv(path, dtype={"ip_addr": str})


def check_manual(df):
    """join_manual.csv(ip_addr, system_type, perf_cid, perf_lid, alarm_cid, alarm_lid, note): 한 건이라도 키가 다르면 stop."""
    need = {"ip_addr", "system_type", "perf_cid", "perf_lid", "alarm_cid", "alarm_lid"}
    if not need <= set(df.columns):
        raise ValueError(f"join_manual.csv 열 부족: {sorted(need - set(df.columns))}")
    d = df.copy()
    d["same_key"] = (d["perf_cid"] == d["alarm_cid"]) & (d["perf_lid"] == d["alarm_lid"])
    warns = []
    if len(d) < 3:
        warns.append(f"M1 은 3개 포트 이상 권장 (현재 {len(d)})")
    if d["system_type"].nunique() < 2:
        warns.append("M1 은 서로 다른 장비 모델 2종 이상 권장")
    bad = d[~d["same_key"]]
    return {"rows": len(d), "mismatches": bad.to_dict("records"), "stop": len(bad) > 0, "warnings": warns}


# ---- 진단: 알람 cid 어긋남 탐색 (판정 아님, 자동 전환 없음) ------------------------------------------
def cid_offset_scan(alarms_cls, eth, offsets=(-2, -1, 0, 1, 2)):
    """포트 알람의 cid 에 offset 을 더한 키가 성능 키에 있는 비율을 모델별로. 키 조인이 틀렸을 때 어긋남의 모양을 보여 주는 진단이며
    J 판정과 무관하고 이 결과로 조인 키를 바꾸지 않는다(바꾸는 것은 사용자 결정)."""
    kt = key_table(eth)
    keys = set(zip(kt["ip_addr"], kt["cid"], kt["lid"]))
    p = alarms_cls[alarms_cls["row_class"] == "port"]
    res = {}
    for off in offsets:
        m = pd.Series([(i, c + off, l) in keys for i, c, l in zip(p["ip_addr"], p["cid"], p["lid"])], index=p.index)
        res[off] = {"all": float(m.mean()) if len(m) else float("nan"),
                    "by_model": {str(k): float(v) for k, v in m.groupby(p["system_type"]).mean().items()}}
    return res


def cid_slot_offsets(df):
    """address 의 슬롯 번호 대비 cid 의 차이(cid − slot) 분포를 system_type 별로. 성능·알람의 cid 체계 차이(K1a)를 보인다."""
    d = add_diag_columns(prep_keys(df))
    slot = d["addr_port"].str.extract(r"S\.(\d+)")[0].astype(float)
    d = d.assign(off=d["cid"].astype(float) - slot)[slot.notna()]
    return {str(k): {str(int(o)): int(n) for o, n in g["off"].value_counts().items()} for k, g in d.groupby("system_type")}


def reclass_report(alarms_cls):
    """고정 출력: 모델별 device/slot 로 분류된 행 수, 그중 ②(cid>0·lid>0 인데 address 가 'S.n' 이라 slot 이 된 행)의 건수와 cid/lid 분포.
    휴리스틱이 포트 알람을 조용히 빼는지 실데이터로 확인하기 위한 출력이다."""
    a = add_diag_columns(alarms_cls)
    slot_only = a["addr_port"].map(lambda v: v != "" and "-P" not in v)
    rule2 = (a["row_class"] == "slot") & (a["lid"] > 0) & slot_only
    by_model = {}
    for st, g in a.groupby("system_type"):
        by_model[str(st)] = {"rows": int(len(g)), "port": int((g["row_class"] == "port").sum()), "device": int((g["row_class"] == "device").sum()),
                             "slot": int((g["row_class"] == "slot").sum()), "slot_by_rule2": int(rule2[g.index].sum()),
                             "bad_key": int((g["row_class"] == "bad_key").sum())}
    r2 = a[rule2]
    dist = {"cid": {str(k): int(v) for k, v in r2["cid"].value_counts().head(15).items()},
            "lid": {str(k): int(v) for k, v in r2["lid"].value_counts().head(15).items()},
            "cid_lid_top": {f"{c}/{l}": int(n) for (c, l), n in r2.groupby(["cid", "lid"]).size().sort_values(ascending=False).head(15).items()}}
    return {"by_model": by_model, "rule2_total": int(rule2.sum()), "rule2_cid_lid": dist,
            "rule2_samples": r2[["ip_addr", "system_type", "cid", "lid", "address", "alarm_type"]].head(20).to_dict("records") if len(r2) else []}
