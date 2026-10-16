"""
P1-2 현장 검토 샘플과 채점 (설계서 9.8, report_package.md 4.3) — C1·C2·C4a

- `build_sample`: 방법별(AI·고정·롤링) 주간 할당량, 심각도(알람 등급) 층화 무작위(시드 필수·기록), 포트당 최대 3건,
  세 방법을 섞은 블라인드 ID, 같은 포트에서 시간이 겹치는 인시던트는 한 행으로 라벨 공유.
  행이 대표하는 방법 목록은 별도 비공개 키 파일에만 둔다. 검토 CSV 에는 방법을 짐작할 단서(심각도 등)를 넣지 않는다.
- `score_review`: 검토 CSV + 키 → 방법별 유효율(Wilson 95%), 판단 불가 비율, C1 추정 오탐 부하, C4a(유효 탐지량 구간 비교).
검토 분류를 미리 채우지 않는다.
"""
import math

import numpy as np
import pandas as pd

from validation.evaluation.metrics import KEY
from validation.realdata.replay import METHODS, incidents_of

QUOTAS = {"dev": {"ai": 30, "fixed": 10, "rolling": 10}, "judge": {"ai": 50, "fixed": 15, "rolling": 10}}
MAX_PER_PORT = 3
LABELS = ("유효-장애", "유효-예방", "정상 변동", "오탐", "판단 불가")
VALID = ("유효-장애", "유효-예방")
Z = 1.96
MIN_JUDGEABLE = 100
UNJUDGEABLE_WARN = 0.30


REVIEW_COLUMNS = ["blind_id", "ip_addr", "cid", "lid", "window_start", "window_end", "duration_min", "ems_alarms_24h", "work_suspect", "분류", "근거 메모"]
KEY_COLUMNS = ["blind_id", "methods", "incidents", "week"]


def wilson(k, n, z=Z):
    """Wilson 점수 구간 (하한, 상한). n=0 이면 (nan, nan)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)         # 부동소수 오차로 구간이 [0, 1] 을 벗어나지 않게


def _week(ts):
    return ts.dt.to_period("W-SUN").dt.start_time


def _allocate(strata_sizes, quota):
    """층 크기에 비례 배분(최대 나머지법), 비어 있지 않은 층은 quota 가 허용하면 최소 1."""
    total = sum(strata_sizes.values())
    if total <= quota:
        return dict(strata_sizes)
    keys = [k for k, v in strata_sizes.items() if v > 0]
    raw = {k: quota * strata_sizes[k] / total for k in keys}
    alloc = {k: int(math.floor(raw[k])) for k in keys}
    if quota >= len(keys):
        for k in keys:
            alloc[k] = max(alloc[k], 1)
    rem = quota - sum(alloc.values())
    for k in sorted(keys, key=lambda k: raw[k] - math.floor(raw[k]), reverse=True):
        if rem <= 0:
            break
        if alloc[k] < strata_sizes[k]:
            alloc[k] += 1
            rem -= 1
    while sum(alloc.values()) > quota:                      # 최소 1 보장 때문에 넘친 경우 가장 큰 층에서 줄임
        k = max(alloc, key=alloc.get)
        alloc[k] -= 1
    return alloc


def pick_incidents(inc, quota, rng, max_per_port=MAX_PER_PORT):
    """한 방법·한 주의 인시던트에서 층화 무작위로 quota 건 (포트당 최대 max_per_port). 부족하면 전수."""
    if inc.empty:
        return inc
    strat = inc["level_max"] if "level_max" in inc.columns else pd.Series(0, index=inc.index)
    strat = strat.fillna(-1)
    sizes = strat.value_counts().to_dict()
    alloc = _allocate(sizes, quota)
    chosen, per_port = [], {}
    pool = {}
    for lv, g in inc.assign(_s=strat).groupby("_s"):
        order = list(g.index)
        rng.shuffle(order)
        pool[lv] = order
    for lv, order in pool.items():                          # 층별 배분
        take = 0
        for i in list(order):
            if take >= alloc.get(lv, 0):
                break
            port = tuple(inc.loc[i, KEY])
            if per_port.get(port, 0) >= max_per_port:
                continue
            chosen.append(i)
            per_port[port] = per_port.get(port, 0) + 1
            order.remove(i)
            take += 1
    if len(chosen) < min(quota, len(inc)):                  # 포트 상한으로 모자라면 다른 층에서 채움
        rest = [i for order in pool.values() for i in order]
        rng.shuffle(rest)
        for i in rest:
            if len(chosen) >= quota:
                break
            port = tuple(inc.loc[i, KEY])
            if per_port.get(port, 0) < max_per_port:
                chosen.append(i)
                per_port[port] = per_port.get(port, 0) + 1
    return inc.loc[chosen]


def build_sample(frames, purpose, seed, ems_lookup=None, work_lookup=None, quotas=None):
    """frames: {방법: 재생 결과 프레임}. 반환 (review DataFrame, key DataFrame, meta dict).
    ems_lookup(ip, cid, lid, t_lo, t_hi) → 알람명 문자열, work_lookup(ip, cid, lid, t_lo, t_hi) → bool 은 참고 열용(없으면 빈 값)."""
    if seed is None:
        raise ValueError("시드가 필요합니다 (기록되어야 재현 가능)")
    q = quotas or QUOTAS[purpose]
    rng = np.random.RandomState(seed)
    picked = []
    meta = {"seed": int(seed), "purpose": purpose, "quotas": q, "weeks": {}}
    for m in METHODS:
        inc, _ = incidents_of(frames[m])
        if inc.empty:
            continue
        inc = inc.assign(method=m, week=_week(inc["t_first"]))
        for wk, g in inc.groupby("week"):
            sel = pick_incidents(g.reset_index(drop=True), q[m], rng)
            meta["weeks"].setdefault(str(wk.date()), {})[m] = {"available": int(len(g)), "sampled": int(len(sel))}
            picked.append(sel)
    if not picked:
        meta.update(rows=0, per_method_incidents={m: 0 for m in METHODS})
        return pd.DataFrame(columns=REVIEW_COLUMNS), pd.DataFrame(columns=KEY_COLUMNS), meta
    allp = pd.concat(picked, ignore_index=True).sort_values(KEY + ["t_first"]).reset_index(drop=True)
    # 같은 포트에서 시간이 겹치는 선택 인시던트는 한 행 (라벨 공유)
    cluster, last_port, last_end, cid_ = [], None, None, -1
    for r in allp.itertuples():
        port = (r.ip_addr, r.cid, r.lid)
        if port != last_port or r.t_first > last_end:
            cid_ += 1
            last_end = r.t_last
        else:
            last_end = max(last_end, r.t_last)
        last_port = port
        cluster.append(cid_)
    allp["cluster"] = cluster
    rows, keys = [], []
    for c, g in allp.groupby("cluster"):
        w0, w1 = g["t_first"].min(), g["t_last"].max()
        r0 = g.iloc[0]
        row = {"_c": c, "ip_addr": r0["ip_addr"], "cid": r0["cid"], "lid": r0["lid"], "window_start": w0, "window_end": w1,
               "duration_min": (w1 - w0).total_seconds() / 60,
               "ems_alarms_24h": ems_lookup(r0["ip_addr"], r0["cid"], r0["lid"], w0 - pd.Timedelta(hours=24), w1 + pd.Timedelta(hours=24)) if ems_lookup else "",
               "work_suspect": bool(work_lookup(r0["ip_addr"], r0["cid"], r0["lid"], w0 - pd.Timedelta(minutes=60), w1 + pd.Timedelta(minutes=60))) if work_lookup else ""}
        rows.append(row)
        keys.append({"_c": c, "methods": ";".join(sorted(set(g["method"]))), "incidents": ";".join(f"{m}:{i}" for m, i in zip(g["method"], g["incident"])),
                     "week": str(g["week"].min().date())})
    review, key = pd.DataFrame(rows), pd.DataFrame(keys)
    ids = np.arange(len(review))
    rng.shuffle(ids)                                        # 시드로 섞은 블라인드 ID (정렬 순서와 방법에 무관)
    review["blind_id"] = [f"R-{i + 1:04d}" for i in ids]
    key["blind_id"] = review["blind_id"].to_numpy()
    review = review.sort_values("blind_id").drop(columns="_c").reset_index(drop=True)
    key = key.sort_values("blind_id").drop(columns="_c").reset_index(drop=True)
    review["분류"], review["근거 메모"] = "", ""
    review = review[REVIEW_COLUMNS]
    meta["rows"] = int(len(review))
    meta["per_method_incidents"] = {m: int((allp["method"] == m).sum()) for m in METHODS}
    return review, key[KEY_COLUMNS], meta


def score_review(review, key, c1_raw=None, ai_gt_fixed_report=None):
    """검토 결과 채점. 분류 값은 LABELS 만 허용(비어 있으면 미검토로 보고)."""
    r = review.merge(key, on="blind_id", how="left")
    if r["methods"].isna().any():
        raise ValueError("키 파일에 없는 blind_id 가 있습니다")
    bad = r[(r["분류"].fillna("") != "") & ~r["분류"].isin(LABELS)]
    if len(bad):
        raise ValueError(f"허용되지 않은 분류 값 {sorted(set(bad['분류']))} (허용: {LABELS})")
    unreviewed = int((r["분류"].fillna("") == "").sum())
    done = r[r["분류"].fillna("") != ""]
    out = {"rows": int(len(r)), "unreviewed": unreviewed, "methods": {}}
    for m in METHODS:
        sub = done[done["methods"].map(lambda s: m in s.split(";"))]
        n_all = len(sub)
        unj = int((sub["분류"] == "판단 불가").sum())
        n = n_all - unj
        k = int(sub["분류"].isin(VALID).sum())
        lo, hi = wilson(k, n)
        res = {"reviewed": n_all, "unjudgeable": unj, "unjudgeable_ratio": (unj / n_all) if n_all else float("nan"),
               "judgeable": n, "valid": k, "valid_rate": (k / n) if n else float("nan"), "wilson_lo": lo, "wilson_hi": hi,
               "counts": sub["분류"].value_counts().to_dict(), "unjudgeable_warning": bool(n_all and unj / n_all > UNJUDGEABLE_WARN)}
        out["methods"][m] = res
    ai = out["methods"]["ai"]
    out["c2"] = {"hold": ai["judgeable"] < MIN_JUDGEABLE, "judgeable": ai["judgeable"], "need": MIN_JUDGEABLE,
                 "point": ai["valid_rate"], "wilson_lo": ai["wilson_lo"], "target_lo": 0.50,
                 "pass": (None if ai["judgeable"] < MIN_JUDGEABLE else bool(ai["wilson_lo"] >= 0.50))}
    if c1_raw:
        a = c1_raw["ai"]
        rate = a["incidents_per_port_day"]
        c1 = {"incidents_per_port_day": rate, "c2_point": ai["valid_rate"]}
        c1["estimated_false_per_port_day"] = rate * (1 - ai["valid_rate"]) if ai["judgeable"] else float("nan")
        c1["pass"] = None if out["c2"]["hold"] else bool(c1["estimated_false_per_port_day"] <= 0.02)
        c1["hold"] = out["c2"]["hold"]
        out["c1"] = c1
        eff = {}
        for m in METHODS:
            res = out["methods"][m]
            inc = c1_raw[m]["incidents"]
            eff[m] = {"incidents": inc, "lo": inc * res["wilson_lo"] if res["judgeable"] else float("nan"),
                      "hi": inc * res["wilson_hi"] if res["judgeable"] else float("nan")}
        a_, f_ = eff["ai"], eff["fixed"]
        if out["c2"]["hold"] or np.isnan(a_["lo"]) or np.isnan(f_["lo"]):
            c4a = {"status": "hold", "reason": "AI 판정 가능 샘플 < 100 또는 고정 임계치 샘플 없음"}
        elif a_["lo"] > f_["hi"]:
            c4a = {"status": "충족"}
        elif a_["hi"] < f_["lo"]:
            c4a = {"status": "미충족"}
        else:
            c4a = {"status": "동률(유의한 차이 없음)"}
        out["c4a"] = {**c4a, "effective_detections": eff}
    return out
