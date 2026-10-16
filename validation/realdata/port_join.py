"""
P1-2 포트 조인과 안전 검사 J1~J4 (설계서 6장)

조인 키 = (ip_addr, cid, lid) (사용자 결정 D13). address 의 `S.n-Pm` 은 진단용 보조 열(addr_port)일 뿐 키가 아니다.
J 판정이 failed 여도 address 조인으로 자동 전환하지 않는다(사용자 결정 사항). 기준값은 설계서 6.3 에 사전 고정되어 있고
결과를 본 뒤 바꾸지 않는다.
"""
import os
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


def norm_port_base(address):
    """address → 'S.<n>-P<m>' (정규식의 마지막 일치), 없으면 'S.<n>' 또는 ''. 행 분류 규칙 7·8·9 의 판단 기준(MAIN 확장 전)."""
    if address is None or (isinstance(address, float) and np.isnan(address)) or pd.isna(address):
        return ""
    s = str(address)
    m = list(_ADDR_RE.finditer(s))
    if m:
        return f"S.{int(m[-1].group(1))}-P{int(m[-1].group(2))}"
    m = list(_SLOT_RE.finditer(s))
    return f"S.{int(m[-1].group(1))}" if m else ""


_MAIN_FIND_RE = re.compile(r"MAIN-P\s*(\d+)", re.IGNORECASE)


def norm_port(address):
    """진단 열 addr_port (설계 6.1, 개정 6): norm_port_base 결과가 빈 값일 때만 `MAIN-P<n>` 의 마지막 일치를 `MAIN-P<n>` 으로.
    성능 address 안의 MAIN 포트와 알람 `MAIN-Pn` 을 J1·J2 에서 대조하기 위함이다. 행 분류 규칙 8 의 '빈 값' 판단은 확장 전(base) 기준."""
    b = norm_port_base(address)
    if b != "":
        return b
    if address is None or pd.isna(address):
        return ""
    m = list(_MAIN_FIND_RE.finditer(str(address)))
    return f"MAIN-P{int(m[-1].group(1))}" if m else ""


def prep_keys(df):
    """ip_addr 공백 제거, cid/lid 를 nullable 정수로. 사본 반환."""
    out = df.copy()
    out["ip_addr"] = out["ip_addr"].astype(str).str.strip()
    for c in ("cid", "lid"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


DEVICE_CID, DEVICE_LID = 256, 255      # 실데이터(2026-10-16 스냅샷): 장비 단위 알람은 cid=256, lid=255, address='SYSTEM'
ROW_CLASS_VERSION = "row_class_v4"
# 설계 6.1 (개정 4·6·7) 규칙표 — 위에서부터 처음 맞는 규칙. (규칙 ID, 분류)
ROW_RULES = (("dev_system", "device"), ("dev_256_255", "device"), ("dev_partial", "device"), ("dev_cid0", "device"),
             ("bad_key", "bad_key"), ("slot_lid0", "slot"), ("slot_addr", "slot"),
             ("port_main", "port"), ("service_main", "service"), ("main_lid_mismatch", "unit"),
             ("unit_addr", "unit"), ("port", "port"), ("port_noaddr", "port"))
RULE_IDS = tuple(r for r, _ in ROW_RULES)
_MAIN_PLAIN_RE = re.compile(r"^MAIN-P(\d+)\s*$")               # 7a/7b: 접미 없음 (addr 은 앞뒤 공백 제거·대문자)
_MAIN_SUFFIX_RE = re.compile(r"^MAIN-P(\d+)\s+\S.*$")          # 7s: 접미 변형 (예: MAIN-P9 PW-5021)


def classify_alarm_rows(alarms):
    """설계 6.1 규칙표(row_class_v4, 13개 규칙)를 위에서부터 처음 맞는 규칙으로 적용. 열: row_rule(규칙 ID), row_class(device|slot|unit|service|port|bad_key).
    address 는 행을 판정에서 빼는 방향으로만 쓴다(규칙 1·7·7s·7b·8). 예외 7a(MAIN-Pn, 접미 없음, lid=n)는 키가 이미 포트 모양인 행을 되돌리는 것이다.
    조인 키는 (ip_addr, 알람 cid − k, lid) 이다(apply_offsets)."""
    out = prep_keys(alarms)
    cid, lid = out["cid"], out["lid"]
    n = len(out)
    raw = out["address"] if "address" in out.columns else pd.Series([""] * n, index=out.index)
    addr = raw.map(lambda v: "" if pd.isna(v) else str(v).strip().upper())
    ap = raw.map(norm_port_base)
    plain = addr.map(lambda v: _MAIN_PLAIN_RE.match(v))
    plain_n = plain.map(lambda m: int(m.group(1)) if m else -1)
    suffix = addr.map(lambda v: bool(_MAIN_SUFFIX_RE.match(v)))
    rule = pd.Series("", index=out.index, dtype="object")

    def put(rid, cond):
        nonlocal rule
        rule = rule.mask((rule == "") & cond.fillna(False).astype(bool), rid)

    notnull = cid.notna() & lid.notna()
    put("dev_system", addr == "SYSTEM")
    put("dev_256_255", notnull & (cid == DEVICE_CID) & (lid == DEVICE_LID))
    put("dev_partial", (cid == DEVICE_CID).fillna(False) ^ (lid == DEVICE_LID).fillna(False))
    put("dev_cid0", cid == 0)
    put("bad_key", ~notnull | (cid < 0) | (lid < 0))
    put("slot_lid0", (cid > 0) & (lid == 0))
    pos = (cid > 0) & (lid > 0)
    put("slot_addr", pos & ap.map(lambda v: v != "" and "-P" not in v))
    put("port_main", pos & plain.map(lambda m: m is not None) & (lid == plain_n))
    put("service_main", pos & suffix)
    put("main_lid_mismatch", pos & plain.map(lambda m: m is not None) & (lid != plain_n))
    put("unit_addr", pos & (addr != "") & (ap == ""))
    put("port", pos & ap.map(lambda v: "-P" in v))
    put("port_noaddr", pos & (addr == ""))
    cls_of = dict(ROW_RULES)
    out["row_rule"] = rule.where(rule != "", "unclassified")
    out["row_class"] = out["row_rule"].map(lambda r: cls_of.get(r, "bad_key"))
    return out


# ---- 알람 cid 오프셋 보정 (설계 6.1·6.3, 개정 5 — 사용자 결정 Q16) ------------------------------------------------
DEFAULT_K = 1
OFFSET_MIN_ALARMS, OFFSET_MIN_J4, OFFSET_MIN_GAIN = 30, 0.8, 0.2          # 모델별 예외 기준 (사전 고정)
J4_MIN, J2_MIN = 0.8, 0.9
SCAN_KS = (0, 1, 2, 3)
OFFSET_FILE = "alarm_cid_offset.csv"
OFFSET_COLUMNS = ["system_type", "k", "n_alarms", "j4_at_k", "j4_at_1", "basis"]
KEY_CORRECTION_NOTE = "키 보정은 데이터 일관성으로만 확인, 현장 대조 미실시"


def load_offsets(path):
    """오프셋 표 CSV → {system_type: k}. 파일이 없으면 빈 표(모든 모델 기본 k=1)."""
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {int(r.system_type): int(r.k) for r in df.itertuples()}


def k_for(system_type, offsets):
    try:
        return int((offsets or {}).get(int(system_type), DEFAULT_K))
    except (TypeError, ValueError):
        return DEFAULT_K


def apply_offsets(cls_df, offsets=None):
    """행 분류(원 cid 기준) 뒤에 포트·slot 행에만 cid_j = cid − k(system_type) 를 만들어 `cid` 열을 보정 값으로 바꾼다.
    원 값은 `cid_raw`, 모든 행의 cid − k 는 `cid_j`. cid_j ≤ 0 인 포트·slot 행은 bad_key(판정 제외, `cid_nonpositive`=True)."""
    out = cls_df.copy()
    out["cid_raw"] = out["cid"]
    ks = out["system_type"].map(lambda v: k_for(v, offsets)) if "system_type" in out.columns else pd.Series(DEFAULT_K, index=out.index)
    out["k_applied"] = ks
    out["cid_j"] = out["cid"] - ks.astype("Int64")
    mask = out["row_class"].isin(["port", "slot"])
    bad = mask & (out["cid_j"] <= 0).fillna(False)
    ok = mask & ~bad
    out.loc[ok, "cid"] = out.loc[ok, "cid_j"]                # service 행의 cid_j 는 열로만 남기고(포함 규칙이 켜질 때 승격), cid 는 원 값 유지
    out.loc[bad, "row_class"] = "bad_key"
    out["cid_nonpositive"] = bad
    return out


def classify_rev1(alarms):
    """개정 1 규칙(cid=0 → device, cid>0·lid=0 → slot, 키 이상 → bad_key, 나머지 port) — 재분류 건수 비교용."""
    o = prep_keys(alarms)
    cid, lid = o["cid"], o["lid"]
    ok = cid.notna() & lid.notna() & (cid >= 0) & (lid >= 0)
    cls = pd.Series("port", index=o.index, dtype="object")
    cls[~ok] = "bad_key"
    cls[ok & (cid > 0) & (lid == 0)] = "slot"
    cls[ok & (cid == 0)] = "device"
    return cls


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
    ports = ports[ports["addr_port"].str.contains("-P") & ports["perf_addr_port"].str.contains("-P")]       # 설계 6.3: 양쪽이 S.n-Pm 인 쌍만
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
    return {"by_model": by_model.to_dict("records"), "row_class_rules": row_rule_report(alarms_cls, eth),
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
    va = valid[valid["a_alive"].notna() & valid["c_alive"].notna()] if len(valid) else valid        # 대조군이 없으면 방향을 볼 수 없다
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
    """포트 알람 중 보정 키(알람 cid 는 이미 cid_j)의 성능 포트가 있는 비율. 분모는 **성능 스냅샷에 있는 장비(ip_addr)의 포트 알람만**
    (성능이 아예 없는 알람 장비는 따로 셈: no_perf_device)."""
    kt = key_table(eth)
    keys = set(zip(kt["ip_addr"], kt["cid"], kt["lid"]))
    ips = set(kt["ip_addr"])
    p = alarms_cls[alarms_cls["row_class"] == "port"].copy()
    p["device_in_perf"] = p["ip_addr"].isin(ips)
    p["matched"] = [(k in keys) for k in zip(p["ip_addr"], p["cid"], p["lid"])]
    q = p[p["device_in_perf"]]
    by_model = q.groupby("system_type").agg(alarms=("matched", "size"), matched=("matched", "mean")).reset_index()
    return {"port_alarms": len(p), "port_alarms_in_perf_devices": int(len(q)), "no_perf_device": int((~p["device_in_perf"]).sum()),
            "no_perf_device_by_model": {str(k): int(v) for k, v in p[~p["device_in_perf"]].groupby("system_type").size().items()},
            "matched_ratio": float(q["matched"].mean()) if len(q) else float("nan"),
            "by_model": by_model.to_dict("records"),
            "unmatched_samples": q.loc[~q["matched"], ["ip_addr", "system_type", "cid", "lid", "alarm_type", "occur_date"]].head(20).to_dict("records")}


def offset_scan(alarms_cls, eth, ks=SCAN_KS):
    """J0s (개발 구간만): 모델별로 k ∈ ks 각각의 J4 매칭률과 J2 address 일치율. 포트 알람의 원 cid(`cid_raw` 있으면 그것)에 −k 한 키로 본다.
    분모는 성능 스냅샷에 있는 장비의 포트 알람. 반환 {모델: {n_alarms, ks:{k:{j4,j2,j2_pairs}}}}."""
    kt = key_table(eth)
    perf_addr = {(r.ip_addr, r.cid, r.lid): r.addr_port for r in kt.itertuples()}
    ips = set(kt["ip_addr"])
    a = alarms_cls.copy()
    cid_raw = a["cid_raw"] if "cid_raw" in a.columns else a["cid"]
    a = a.assign(_c=cid_raw)
    a = add_diag_columns(a, alarm=False)
    p = a[(a["row_class"].isin(["port", "bad_key"])) & a["_c"].notna() & a["ip_addr"].isin(ips)]
    if "cid_nonpositive" in p.columns:
        p = p[(p["row_class"] == "port") | p["cid_nonpositive"]]
    else:
        p = p[p["row_class"] == "port"]
    out = {}
    for st, g in p.groupby("system_type"):
        res = {}
        for k in ks:
            keys = [(i, c - k, l) for i, c, l in zip(g["ip_addr"], g["_c"], g["lid"])]
            m = [kk in perf_addr for kk in keys]
            pairs = [(ap, perf_addr[kk]) for kk, ap, ok in zip(keys, g["addr_port"], m) if ok and "-P" in ap and "-P" in perf_addr[kk]]
            res[k] = {"j4": float(sum(m) / len(m)) if len(m) else float("nan"),
                      "j2": float(sum(1 for x, y in pairs if x == y) / len(pairs)) if pairs else float("nan"), "j2_pairs": len(pairs)}
        out[int(st)] = {"n_alarms": int(len(g)), "ks": res}
    return out


def derive_offset_table(scan):
    """사전 고정 절차: 포트 알람 ≥ 30 인 모델만 표에 올린다. k 기본 1, 최선 k* 의 J4 ≥ 0.8 이고 k=1 보다 ≥ 0.2 높을 때만 예외.
    반환 DataFrame(OFFSET_COLUMNS). 표에 없는 모델은 판정 실행에서 offset_unverified 검사를 받는다."""
    rows = []
    for st, r in sorted(scan.items()):
        if r["n_alarms"] < OFFSET_MIN_ALARMS:
            continue
        j4 = {k: v["j4"] for k, v in r["ks"].items()}
        best = max(j4, key=lambda k: (j4[k] if j4[k] == j4[k] else -1, -abs(k - DEFAULT_K)))
        j1 = j4.get(DEFAULT_K, float("nan"))
        exc = best != DEFAULT_K and j4[best] >= OFFSET_MIN_J4 and (j4[best] - (j1 if j1 == j1 else 0.0)) >= OFFSET_MIN_GAIN
        k = best if exc else DEFAULT_K
        rows.append({"system_type": st, "k": k, "n_alarms": r["n_alarms"], "j4_at_k": round(j4[k], 4), "j4_at_1": round(j1, 4) if j1 == j1 else "",
                     "basis": "exception" if exc else "default"})
    return pd.DataFrame(rows, columns=OFFSET_COLUMNS)


def write_offsets_csv(df, path):
    df.to_csv(path, index=False)


def model_join_rates(alarms_cls, eth, models=None):
    """보정 키(`cid` 가 이미 cid_j) 기준 모델별 J4·J2 — 판정 구간에서 degraded/offset_unverified 판단용. {모델: {n, j4, j2, j2_pairs}}"""
    j4 = j4_coverage(alarms_cls, eth)
    j2 = j2_diagnosis(alarms_cls, eth)
    r2 = {int(x["system_type"]): x for x in j2["by_model"]}
    out = {}
    for x in j4["by_model"]:
        st = int(x["system_type"])
        if models is not None and st not in models:
            continue
        pairs = r2.get(st, {}).get("pairs", 0)
        out[st] = {"n": int(x["alarms"]), "j4": float(x["matched"]), "j2": (1 - float(r2[st]["diff_rate"])) if pairs else float("nan"), "j2_pairs": int(pairs)}
    return out


# ---- 판정 ----------------------------------------------------------------------------
def judge_join(j1, j3, table_rates, min_episodes=J3_MIN_EPISODES):
    """6.3 판정(사전 고정, 개정 5). table_rates = {오프셋 표 모델: {j4, j2, ...}} (보정 키 기준).
    verified = (i) J1 1:1 ≥ 95% (ii) 표의 모든 모델 J4 ≥ 0.8 (iii) 같은 모델 J2 ≥ 0.9 (iv) J3 대상(장비 생존 구간) ≥ 5 이면 a > c (방향만).
    J3 대상 < 5 이면 j3_insufficient 로 표시하고 막지 않는다. 구 J3 기준(0.8/0.5)은 폐기. 어떤 경우에도 address·다른 k 로 자동 전환하지 않는다."""
    reasons, flags = [], []
    r1 = j1.get("one_to_one_ratio")
    if not table_rates:
        return {"status": "unverified", "reasons": ["오프셋 표에 모델이 없음(성능 장비의 포트 알람 ≥ 30 인 모델 0개) — 검증할 수 없음"], "flags": flags,
                "stop": False, "note": KEY_CORRECTION_NOTE}
    if r1 is None or r1 != r1 or r1 < J1_MIN_RATIO:
        reasons.append(f"J1 1:1 비율 {r1} < {J1_MIN_RATIO}")
    for st, r in sorted(table_rates.items()):
        if not r["j4"] >= J4_MIN:
            reasons.append(f"모델 {st} J4 {r['j4']:.3f} < {J4_MIN}")
        if not r["j2"] >= J2_MIN:                                  # J2 가 없으면(쌍 0) 검증 불가 → 미충족 (기준을 낮추지 않는다)
            reasons.append(f"모델 {st} J2 {r['j2']} < {J2_MIN} (쌍 {r.get('j2_pairs')})")
    alive = j3.get("diag_alive", {})
    n_alive = alive.get("episodes", 0)
    a, c = alive.get("match_a_alive"), alive.get("match_c_alive")
    if n_alive >= min_episodes:
        if not (a is not None and c is not None and a == a and c == c and a > c):
            reasons.append(f"J3(장비 생존 구간) a={a} ≤ c={c} — 방향 불일치")
    else:
        flags.append("j3_insufficient")
    status = "verified" if not reasons else "failed"
    d = j3.get("match_d")
    return {"status": status, "reasons": reasons, "flags": flags, "stop": status == "failed", "j3_alive_episodes": n_alive,
            "address_join_diag": d, "note": KEY_CORRECTION_NOTE}


# ---- M1 ------------------------------------------------------------------------------
def load_manual(path):
    return pd.read_csv(path, dtype={"ip_addr": str})


def check_manual(df, offsets=None):
    """join_manual.csv(ip_addr, system_type, perf_cid, perf_lid, alarm_cid, alarm_lid, note): 알람 cid − k(system_type) 가 성능 cid 와 같아야 한다.
    한 건이라도 다르면 J 판정과 무관하게 stop. (개정 5: 현장 대조는 미실시 — 나중에 입력이 들어왔을 때 쓰는 경로)"""
    need = {"ip_addr", "system_type", "perf_cid", "perf_lid", "alarm_cid", "alarm_lid"}
    if not need <= set(df.columns):
        raise ValueError(f"join_manual.csv 열 부족: {sorted(need - set(df.columns))}")
    d = df.copy()
    d["k"] = d["system_type"].map(lambda v: k_for(v, offsets))
    d["same_key"] = (d["perf_cid"] == d["alarm_cid"] - d["k"]) & (d["perf_lid"] == d["alarm_lid"])
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
    base = p["cid_raw"] if "cid_raw" in p.columns else p["cid"]
    res = {}
    for off in offsets:
        m = pd.Series([(i, c + off, l) in keys for i, c, l in zip(p["ip_addr"], base, p["lid"])], index=p.index)
        res[off] = {"all": float(m.mean()) if len(m) else float("nan"),
                    "by_model": {str(k): float(v) for k, v in m.groupby(p["system_type"]).mean().items()}}
    return res


def cid_slot_offsets(df):
    """address 의 슬롯 번호 대비 cid 의 차이(cid − slot) 분포를 system_type 별로. 성능·알람의 cid 체계 차이(K1a)를 보인다."""
    d = add_diag_columns(prep_keys(df))
    slot = d["addr_port"].str.extract(r"S\.(\d+)")[0].astype(float)
    d = d.assign(off=d["cid"].astype(float) - slot)[slot.notna()]
    return {str(k): {str(int(o)): int(n) for o, n in g["off"].value_counts().items()} for k, g in d.groupby("system_type")}


_TRAIL_RE = re.compile(r"S\.\s*\d+\s*-\s*P\s*\d+(.*)$", re.IGNORECASE)


def trailing_tokens_report(alarms_cls, sample=20):
    """규칙 9·10 행 중 address 에 `S.n-Pm` 뒤로 다른 토큰(서비스 접미 등)이 남는 행의 모델별 건수와 표본 (설계 6.1 확인 필요 항목)."""
    p = alarms_cls[alarms_cls["row_rule"].isin(["port", "port_noaddr"])]
    if "address" not in p.columns or p.empty:
        return {"total": 0, "by_model": {}, "samples": {}, "rows_checked": int(len(p))}
    rest = p["address"].map(lambda v: (lambda m: m.group(1).strip() if m else "")(_TRAIL_RE.search(str(v))) if pd.notna(v) else "")
    hit = p[rest != ""].assign(trailing=rest[rest != ""])
    by = {str(st): int(len(g)) for st, g in hit.groupby("system_type")}
    samples = {str(st): g[["ip_addr", "cid", "lid", "address", "trailing", "alarm_type"]].head(sample).to_dict("records") for st, g in hit.groupby("system_type")}
    top = {str(st): g["trailing"].value_counts().head(5).to_dict() for st, g in hit.groupby("system_type")}
    return {"total": int(len(hit)), "rows_checked": int(len(p)), "by_model": by, "top_tokens": top, "samples": samples}


def row_rule_report(alarms_cls, eth=None):
    """고정 출력(설계 6.1 개정 4): system_type × 규칙 ID 건수(규칙 순서 고정), 개정 1 규칙 대비 재분류 건수,
    slot_addr·unit_addr 행 중 같은 키의 성능 포트가 있는 행 수와 표본 20행(모델별). 휴리스틱이 포트 알람을 조용히 빼는지 확인하는 용도."""
    a = add_diag_columns(alarms_cls)
    rev1 = classify_rev1(alarms_cls)
    a = a.assign(_rev1=rev1)
    a["_changed"] = a["_rev1"] != a["row_class"]
    table = {}
    for st, g in a.groupby("system_type"):
        table[str(st)] = {rid: int((g["row_rule"] == rid).sum()) for rid in RULE_IDS}
    keys = set()
    if eth is not None and len(eth):
        kt = key_table(eth)
        keys = set(zip(kt["ip_addr"], kt["cid"], kt["lid"]))
    a["_inperf"] = [(i, c, l) in keys for i, c, l in zip(a["ip_addr"], a["cid"], a["lid"])]
    inperf, samples = {}, {}
    for rid in ("slot_addr", "unit_addr", "service_main", "main_lid_mismatch"):
        sub = a[a["row_rule"] == rid]
        inperf[rid] = {str(st): {"rows": int(len(g)), "with_perf_key": int(g["_inperf"].sum())} for st, g in sub.groupby("system_type")}
        samples[rid] = {str(st): g[["ip_addr", "cid", "lid", "address", "alarm_type", "_inperf"]].rename(columns={"_inperf": "perf_key_exists"})
                        .head(20).to_dict("records") for st, g in sub.groupby("system_type")}
    sub2 = a[a["row_rule"] == "slot_addr"]
    cl = {"cid_lid_top": {f"{c}/{l}": int(n) for (c, l), n in sub2.groupby(["cid", "lid"]).size().sort_values(ascending=False).head(15).items()}}
    trailing = trailing_tokens_report(alarms_cls)
    return {"version": ROW_CLASS_VERSION, "trailing_tokens_after_port": trailing, "rule_order": list(RULE_IDS), "by_model_rule": table,
            "rows": int(len(a)), "reclassified_vs_rev1": int(a["_changed"].sum()),
            "reclassified_vs_rev1_by_model": {str(st): int(g["_changed"].sum()) for st, g in a.groupby("system_type")},
            "classes": a["row_class"].value_counts().to_dict(), "with_perf_key": inperf, "samples": samples, "slot_addr_cid_lid": cl}


def format_row_rule_report(rep):
    """고정 순서 텍스트: 모델별 한 줄 (규칙 ID 순서), 재분류 건수, 성능 키가 있는 slot_addr/unit_addr 행 수."""
    lines = [f"[행 분류 {rep['version']}] 규칙 순서: " + " ".join(rep["rule_order"])]
    for st, row in rep["by_model_rule"].items():
        lines.append(f"  model {st}: " + " ".join(f"{rid}={row[rid]}" for rid in rep["rule_order"]))
    lines.append(f"  개정 1 규칙 대비 재분류 {rep['reclassified_vs_rev1']}건 {rep['reclassified_vs_rev1_by_model']}")
    for rid in ("slot_addr", "unit_addr", "service_main", "main_lid_mismatch"):
        lines.append(f"  {rid} 중 성능 키 있음: {rep['with_perf_key'].get(rid, {})}")
    t = rep.get("trailing_tokens_after_port", {})
    lines.append(f"  S.n-Pm 뒤에 토큰이 남는 address(규칙 9·10 행 {t.get('rows_checked', 0)} 중): {t.get('total', 0)}건 모델별 {t.get('by_model')} 상위 토큰 {t.get('top_tokens')}")
    return "\n".join(lines)
