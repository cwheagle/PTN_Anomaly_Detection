"""
P1-2 읽기 전용 추출 → 로컬 스냅샷 (설계서 4.3~4.7, 5장)

DB 는 추출 때 한 번만 읽고 이후 분석은 스냅샷(`raw/<kind>/<YYYY-MM-DD>.csv.gz`)으로 한다.
- 테이블별 `id` 키셋 청크(OFFSET 미사용), 전체 행 상한 초과 시 중단하고 스냅샷을 "불완전"으로 표시.
- SQL 에는 `occur_date` 조건을 넣지 않는다(쓰레기 값 행이 조용히 빠지면 격리 건수를 셀 수 없음, 4.3).
  4.3a 격리 후 정상 행에만 기간 필터를 Python 에서 적용한다. 격리 행은 분석 프레임에서 제외하고 `quarantine/` 에 보관.
- 증분 추출은 구현하지 않는다(설계 4.7 개정 3a). 스냅샷은 추출 시점 상태로 고정된다. run 은 덮어쓰지 않는다.
"""
import json
import os
import subprocess
from datetime import datetime, timedelta

import pandas as pd

from validation.realdata.tables import (
    ALARM, ETH, KINDS, OPTIC, build_where, fetch_columns, list_tables, missing_tables,
    needed_tables, resolve_filters, DEFAULT_LOOKBACK_HOURS,
)

SNAPSHOT_COLS = {
    ETH: ["id", "system_type", "ip_addr", "address", "cid", "lid", "augid", "au3id", "tugid", "tuid",
          "signal_type", "time_type", "bbe_in_error", "es", "ses", "occur_date"],
    OPTIC: ["id", "system_type", "ip_addr", "address", "cid", "lid", "augid", "au3id", "tugid", "tuid",
            "signal_type", "time_type", "rx_min_power", "rx_max_power", "rx_avg_power", "rx_accu_power",
            "tx_min_power", "tx_max_power", "tx_avg_power", "tx_accu_power", "occur_date"],
    ALARM: ["id", "msgid", "tid", "system_type", "meanid", "sid", "cid", "lid", "augid", "au3id", "tugid", "tuid",
            "p5_meanid", "p5_cid", "p5_lid", "p5_augid", "p5_au3id", "p5_tugid", "p5_tuid", "ip_addr", "address",
            "signal_type", "severity_type", "srv_affect_type", "unit_type", "alarm_type", "gen_data",
            "occur_date", "release_date"],
}
REQUIRED_COLS = ("ip_addr", "cid", "lid", "occur_date")

# 5.1: 운영 매핑(db_connector.py:145)과 같다. 광은 컬럼명이 모델 입력과 같다.
COLUMN_MAP = {ETH: {"es": "tx_packet", "ses": "rx_packet", "bbe_in_error": "error_packet"}}
VALUE_COLS = {ETH: ["tx_packet", "rx_packet", "error_packet"],
              OPTIC: ["rx_avg_power", "tx_avg_power", "rx_min_power", "rx_max_power", "tx_min_power", "tx_max_power"]}
OVER_LIMIT = 1e9                 # 5.2: 운영 학습 필터 참고 기준, 변환·제거하지 않고 건수만 보고
IP_CHUNK = 1000                  # 4.4: ip_addr IN 1회 1,000개 이하
QUARANTINE_HEAVY_RATIO = 0.5     # 4.3a
DEFAULT_MAX_ROWS = 2_000_000
DEFAULT_CHUNK = 50_000


class ExtractError(RuntimeError):
    pass


def load_ports_csv(path):
    """열 ip_addr, 선택 cid, lid. cid/lid 가 비면 장비 전체."""
    df = pd.read_csv(path, dtype=str)
    if "ip_addr" not in df.columns:
        raise ExtractError("ports CSV 에 ip_addr 열이 필요합니다")
    df["ip_addr"] = df["ip_addr"].str.strip()
    for c in ("cid", "lid"):
        if c not in df.columns:
            df[c] = None
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["ip_addr", "cid", "lid"]].drop_duplicates().reset_index(drop=True)


def tool_version():
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _ports_mask(df, ports, alarm):
    """Python 쪽 포트 필터. 장비 전체 행(cid/lid 비어 있음)은 ip 만, 그 외는 (ip,cid,lid). 알람의 장비 단위 행은 ip 만 맞으면 유지."""
    if ports is None:
        return pd.Series(True, index=df.index)
    whole = set(ports.loc[ports["cid"].isna() | ports["lid"].isna(), "ip_addr"])
    pairs = {(r.ip_addr, int(r.cid), int(r.lid)) for r in ports.dropna(subset=["cid", "lid"]).itertuples()}
    any_ip = set(ports["ip_addr"])
    ip = df["ip_addr"]
    cid = pd.to_numeric(df["cid"], errors="coerce")
    lid = pd.to_numeric(df["lid"], errors="coerce")
    m = ip.isin(whole)
    if pairs:
        keys = list(zip(ip, cid, lid))
        m = m | pd.Series([(a, b, c) in pairs for a, b, c in keys], index=df.index)
    if alarm:
        m = m | (ip.isin(any_ip) & ((cid == 0) | (lid == 0)))
    return m


def _iter_chunks(db, table, cols, filters, ips, chunk, has_id, row_cap):
    """테이블 한 개의 행 청크 제너레이터 (list[dict]). id 키셋; id 없으면 LIMIT row_cap 로 1회."""
    col_sql = ", ".join(f"`{c}`" for c in cols)
    ip_groups = [ips[i:i + IP_CHUNK] for i in range(0, len(ips), IP_CHUNK)] if ips else [None]
    for grp in ip_groups:
        where, params = build_where(filters, grp)
        if not has_id:
            yield db.query(f"SELECT {col_sql} FROM `{table}` WHERE {where} LIMIT %s", params + [row_cap])
            continue
        last = 0
        while True:
            rows = db.query(f"SELECT {col_sql} FROM `{table}` WHERE {where} AND `id` > %s ORDER BY `id` LIMIT %s",
                            params + [last, chunk])
            if rows:
                yield rows
                last = rows[-1]["id"]
            if len(rows) < chunk:
                break


class _Stats:
    """종류별 누적 통계 (manifest 에 보관)."""

    def __init__(self, saved=None):
        s = saved or {}
        self.quarantine_total = dict(s.get("quarantine_by_device", {}))   # ip -> [quarantined, total]
        self.q_by_sys = dict(s.get("quarantine_by_system_type", {}))      # system_type -> [q, total]
        self.q_years = dict(s.get("quarantine_year_distribution", {}))
        self.q_samples = list(s.get("quarantine_samples", []))
        self.offsets = dict(s.get("normal_offset_minutes", {}))
        self.value_ranges = dict(s.get("value_ranges", {}))
        self.counters = {k: s.get(k, 0) for k in
                         ("fetched", "kept", "quarantined", "out_of_range", "rounding_shifted", "over_1e9",
                          "duplicates", "lookback_boundary_alarms", "filtered_out")}
        self.filtered_out_unknown = s.get("filtered_out_unknown", False)

    def to_manifest(self):
        heavy = sorted(ip for ip, (q, t) in self.quarantine_total.items() if t and q / t > QUARANTINE_HEAVY_RATIO)
        q = self.counters["quarantined"]
        total = self.counters["fetched"]
        return {
            **self.counters,
            "filtered_out_unknown": self.filtered_out_unknown,
            "quarantine_ratio": (q / total) if total else 0.0,
            "quarantine_by_device": self.quarantine_total,
            "quarantine_by_system_type": self.q_by_sys,
            "quarantine_year_distribution": self.q_years,
            "quarantine_samples": self.q_samples[:20],
            "quarantine_heavy": heavy,
            "normal_offset_minutes": self.offsets,
            "value_ranges": self.value_ranges,
        }


def _bump(d, key, q, t):
    cur = d.get(key, [0, 0])
    d[key] = [cur[0] + q, cur[1] + t]


def _process_chunk(rows, table, kind, ports, start, end, stats, ip_filter_in_sql):
    """청크 → (정상 프레임, 격리 프레임). 통계 누적."""
    df = pd.DataFrame(rows)
    stats.counters["fetched"] += len(df)
    df["ip_addr"] = df["ip_addr"].astype(str).str.strip()
    if "address" in df.columns:
        df["address"] = df["address"].astype("string").str.strip()
    df = df[_ports_mask(df, ports, kind == ALARM)].copy()
    if df.empty:
        return df, df
    occ = pd.to_datetime(df["occur_date"], errors="coerce")
    lo, hi = table.trusted_range()
    trusted = occ.notna() & (occ >= pd.Timestamp(lo)) & (occ <= pd.Timestamp(hi))
    reason = pd.Series("range", index=df.index)
    reason[occ.isna()] = "null"
    rel = None
    if kind == ALARM and "release_date" in df.columns:
        rel = pd.to_datetime(df["release_date"], errors="coerce")
        bad_rel = trusted & rel.notna() & (rel < occ)
        reason[bad_rel] = "release_before_occur"
        trusted = trusted & ~bad_rel
    # 장비/시스템 타입별 격리 비율 (포트 필터 후 기준)
    sys_col = df["system_type"] if "system_type" in df.columns else pd.Series("?", index=df.index)
    for ip, grp in df.groupby("ip_addr"):
        _bump(stats.quarantine_total, ip, int((~trusted[grp.index]).sum()), len(grp))
    for st, grp in df.groupby(sys_col.astype(str)):
        _bump(stats.q_by_sys, st, int((~trusted[grp.index]).sum()), len(grp))

    q = df[~trusted].copy()
    q["src_table"] = table.name
    q["quarantine_reason"] = reason[~trusted]
    stats.counters["quarantined"] += len(q)
    for y in occ[~trusted].dt.year.dropna().astype(int):
        stats.q_years[str(y)] = stats.q_years.get(str(y), 0) + 1
    if len(stats.q_samples) < 20 and len(q):
        for r in q.head(20 - len(stats.q_samples)).to_dict("records"):
            stats.q_samples.append({k: (str(v) if not isinstance(v, (int, float, str, type(None))) else v)
                                    for k, v in r.items()})

    n = df[trusted].copy()
    occ_n = occ[trusted]
    for off in ((occ_n - pd.Timestamp(table.start)).dt.total_seconds() // 900 * 15).astype(int):
        stats.offsets[str(off)] = stats.offsets.get(str(off), 0) + 1
    n["occur_date"] = occ_n
    if rel is not None:
        n["release_date"] = rel[trusted]

    # 기간 필터 (정상 행에만)
    if kind == ALARM:
        in_range = (occ_n < pd.Timestamp(end)) & (n["release_date"].isna() | (n["release_date"] >= pd.Timestamp(start)))
    else:
        in_range = (occ_n >= pd.Timestamp(start)) & (occ_n < pd.Timestamp(end))
    stats.counters["out_of_range"] += int((~in_range).sum())
    n = n[in_range].copy()
    if n.empty:
        return n, q
    if kind == ALARM:
        stats.counters["lookback_boundary_alarms"] += int((n["occur_date"] < pd.Timestamp(start)).sum())
    else:
        n["bin_end"] = n["occur_date"].dt.round("15min")
        stats.counters["rounding_shifted"] += int((n["bin_end"] != n["occur_date"]).sum())
        n = n.rename(columns=COLUMN_MAP.get(kind, {}))
    n["src_table"] = table.name
    stats.counters["kept"] += len(n)
    for c in VALUE_COLS.get(kind, []):
        if c in n.columns:
            v = pd.to_numeric(n[c], errors="coerce").dropna()
            if len(v):
                cur = stats.value_ranges.get(c, [float(v.min()), float(v.max())])
                stats.value_ranges[c] = [min(cur[0], float(v.min())), max(cur[1], float(v.max()))]
                if kind == ETH:
                    stats.counters["over_1e9"] += int((v > OVER_LIMIT).sum())
    return n, q


class _DayWriter:
    """일 단위 파일 쓰기. 같은 run 안에서 같은 날이 두 번 flush 되면 (src_table,id) 중복 제거 후 병합(방어)."""

    def __init__(self, kind, raw_dir, days_meta, stats, aborted_ref):
        self.kind, self.dir = kind, os.path.join(raw_dir, kind)
        self.days = days_meta                      # {day: {"rows", "complete"}} (manifest 에 반영됨)
        self.stats, self.aborted_ref = stats, aborted_ref
        self.buf = {}
        self.new_days = 0

    def add(self, df):
        if df.empty:
            return
        key = df["bin_end"] if "bin_end" in df.columns else df["occur_date"]
        for day, grp in df.groupby(key.dt.strftime("%Y-%m-%d")):
            self.buf.setdefault(day, []).append(grp)

    def flush(self, before_ts=None):
        for day in sorted(list(self.buf)):
            day_end = pd.Timestamp(day) + pd.Timedelta(days=1)
            if before_ts is None or day_end <= pd.Timestamp(before_ts):
                self._write(day, pd.concat(self.buf.pop(day), ignore_index=True))

    def _write(self, day, df):
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, f"{day}.csv.gz")
        meta = self.days.get(day)
        if os.path.exists(path):
            old = pd.read_csv(path, compression="gzip", low_memory=False)
            df = pd.concat([old, df], ignore_index=True)
            subset = ["src_table", "id"] if "id" in df.columns else None
            df = df.drop_duplicates(subset=subset, keep="last")
        else:
            self.new_days += 1
        if self.kind in (ETH, OPTIC):
            dup = int(df.duplicated(subset=["ip_addr", "cid", "lid", "bin_end"], keep="last").sum())
        else:
            dup = int(df.duplicated(subset=["ip_addr", "cid", "lid", "alarm_type", "system_type", "occur_date"],
                                    keep="last").sum()) if {"alarm_type", "system_type"} <= set(df.columns) else 0
        old_dup = (meta or {}).get("duplicates", 0)
        self.stats.counters["duplicates"] += dup - old_dup
        complete = not self.aborted_ref[0]            # 중단된 추출의 파일만 불완전 (4.5)
        df.to_csv(path, index=False, compression="gzip")
        self.days[day] = {"rows": int(len(df)), "complete": bool(complete), "duplicates": dup}


def _write_quarantine(run_dir, kind, frames, round_no):
    if not frames:
        return
    d = os.path.join(run_dir, "quarantine")
    os.makedirs(d, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(os.path.join(d, f"{kind}_round{round_no}.csv.gz"),
                                                index=False, compression="gzip")


def load_snapshot(run_dir, kind):
    """스냅샷의 일 파일을 합쳐 프레임으로 (이후 단위·테스트용). 없으면 빈 프레임."""
    d = os.path.join(run_dir, "raw", kind)
    if not os.path.isdir(d):
        return pd.DataFrame()
    parts = [pd.read_csv(os.path.join(d, f), compression="gzip", low_memory=False)
             for f in sorted(os.listdir(d)) if f.endswith(".csv.gz")]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _write_manifest(run_dir, manifest):
    tmp = os.path.join(run_dir, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, os.path.join(run_dir, "manifest.json"))


def extract(db, run_dir, start, end, kinds=KINDS, ports=None, schema="real", max_rows=DEFAULT_MAX_ROWS,
            chunk=DEFAULT_CHUNK, lookback_hours=DEFAULT_LOOKBACK_HOURS, stop_checker=None, now=None):
    """읽기 전용 추출. 반환: manifest dict. 중단(상한·stop)이면 manifest["complete"]=False."""
    now = now or datetime.now()
    start, end = pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()
    if os.path.exists(run_dir) and os.listdir(run_dir):
        raise ExtractError(f"이미 존재하는 run 입니다(덮어쓰지 않음): {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    manifest = {"stats": {}, "days": {k: {} for k in KINDS}, "rounds": [], "tables": {k: {} for k in KINDS},
                "total_rows": 0}
    raw_dir = os.path.join(run_dir, "raw")
    round_no = len(manifest["rounds"]) + 1

    tabs = list_tables(db, kinds)
    needed = {k: needed_tables(tabs[k], k, start, end, lookback_hours) for k in kinds}
    est = sum(t.rows_est for k in kinds for t in needed[k])
    if manifest["total_rows"] + est > max_rows:
        raise ExtractError(f"예상 행 {est:,} (+기존 {manifest['total_rows']:,}) 가 --max-rows {max_rows:,} 를 넘습니다. "
                           f"구간·포트를 줄이거나 --max-rows 를 올리세요 (plan 의 종류별 표 참고)")

    ips = sorted(set(ports["ip_addr"])) if ports is not None else None
    manifest.update({
        "tool_version": tool_version(), "tz": "Asia/Seoul (naive)", "schema": schema,
        "database": (db.query("SELECT DATABASE() AS db") or [{}])[0].get("db"),
        "range_start": str(start), "kinds": list(kinds),
        "ports_filter": None if ports is None else {"devices": len(set(ports["ip_addr"])), "rows": len(ports)},
        "lookback_hours": lookback_hours, "max_rows": max_rows, "complete": False,
        "round_started_at": str(now), **db.privilege.to_manifest(),
    })
    manifest.setdefault("missing_tables", {})
    manifest.setdefault("filters_omitted", {})
    manifest.setdefault("sim_alarm_skipped", False)
    aborted = [None]
    round_info = {"started_at": str(now), "range": [str(start), str(end)], "fetched": 0, "quarantined": 0,
                  "kept": 0, "new_data_days": 0, "aborted": None}
    kind_stats = {}
    try:
        for kind in kinds:
            stats = _Stats(manifest["stats"].get(kind))
            kind_stats[kind] = stats
            before = dict(stats.counters)
            need = needed[kind]
            manifest["missing_tables"][kind] = [str(m) for m in missing_tables(tabs[kind], kind, start, end)]
            if kind == ALARM and schema == "sim" and not tabs[kind]:
                manifest["sim_alarm_skipped"] = True
            cols_by_table = fetch_columns(db, kind, {t.name for t in need}) if need else {}
            writer = _DayWriter(kind, raw_dir, manifest["days"].setdefault(kind, {}), stats, aborted)
            qframes = []
            for t in need:
                if aborted[0]:
                    break
                if stop_checker and stop_checker():
                    aborted[0] = "stopped"
                    break
                avail = cols_by_table.get(t.name, set())
                filters, omitted = resolve_filters(kind, avail, schema)
                if omitted:
                    manifest["filters_omitted"][kind] = sorted(set(manifest["filters_omitted"].get(kind, [])) | set(omitted))
                cols = [c for c in SNAPSHOT_COLS[kind] if c in avail]
                lacking = [c for c in REQUIRED_COLS if c not in cols]
                if lacking:
                    raise ExtractError(f"{t.name} 에 필수 컬럼이 없습니다: {lacking}")
                f0 = stats.counters["fetched"]
                k0 = stats.counters["kept"]
                q0 = stats.counters["quarantined"]
                budget_hit = False
                for rows in _iter_chunks(db, t.name, cols, filters, ips, chunk, "id" in avail,
                                         max(0, max_rows - manifest["total_rows"] - round_info["fetched"]) + 1):
                    n, q = _process_chunk(rows, t, kind, ports, start, end, stats, ips is not None)
                    writer.add(n)
                    if len(q):
                        qframes.append(q)
                    round_info["fetched"] += len(rows)
                    if manifest["total_rows"] + round_info["fetched"] > max_rows:
                        aborted[0] = "max_rows"
                        budget_hit = True
                        break
                fetched = stats.counters["fetched"] - f0
                # 필터 밖 행 수(2.2-2): 포트 필터가 없을 때만 전체 행 수와 비교 가능
                filtered_out = None
                if ips is None and not budget_hit:
                    total = db.query(f"SELECT COUNT(*) AS n FROM `{t.name}`")[0]["n"]
                    filtered_out = int(total) - fetched
                    stats.counters["filtered_out"] += filtered_out
                elif ips is not None:
                    stats.filtered_out_unknown = True
                manifest["tables"][kind][t.name] = {
                    "rows_fetched": fetched, "rows_kept": stats.counters["kept"] - k0,
                    "quarantined": stats.counters["quarantined"] - q0, "filtered_out": filtered_out}
                writer.flush(before_ts=t.start - timedelta(hours=1, minutes=30))
            writer.flush()
            _write_quarantine(run_dir, kind, qframes, round_no)
            round_info["quarantined"] += stats.counters["quarantined"] - before["quarantined"]
            round_info["kept"] += stats.counters["kept"] - before["kept"]
            round_info["new_data_days"] += writer.new_days
            manifest["stats"][kind] = stats.to_manifest()
            if aborted[0]:
                break
        if OPTIC in kinds and not aborted[0]:
            optic_rows = sum(v["rows_kept"] for v in manifest["tables"][OPTIC].values())
            existing = sum(d["rows"] for d in manifest["days"].get(OPTIC, {}).values())
            if optic_rows == 0 and existing == 0:
                manifest["optic"] = {"status": "SKIP", "reason": "광 성능 행 0 — 관찰 SKIP (설계서 D19)"}
    except BaseException:
        manifest["complete"] = False
        manifest["error"] = "exception during extraction"
        _write_manifest(run_dir, manifest)
        raise
    round_info["aborted"] = aborted[0]
    manifest["total_rows"] += round_info["fetched"]
    manifest["rounds"].append(round_info)
    manifest["complete"] = aborted[0] is None
    if not aborted[0]:
        manifest["range_end"] = str(end)
    else:
        manifest["abort_reason"] = aborted[0]
    _write_manifest(run_dir, manifest)
    return manifest
