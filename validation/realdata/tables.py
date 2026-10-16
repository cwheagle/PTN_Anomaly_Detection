"""
P1-2 시간 테이블 탐색·기간 계산·스키마 프로필·plan (설계서 4.2~4.4, 4.6, 4.9)

테이블명은 information_schema 에서 찾아 정규식으로 검증한 이름만 SQL 에 넣는다. 값은 항상 %s 바인딩.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

ETH, OPTIC, ALARM = "eth", "optic", "alarm"
KINDS = (ETH, OPTIC, ALARM)

PREFIX = {ETH: "cowptn_noti_pm_", OPTIC: "cowptn_noti_pm_optic_power_", ALARM: "cowptn_noti_alarm_"}

# 정규식이 최종 판정: LIKE 'cowptn_noti_pm_%' 는 광 테이블도 잡는다
_RE = {
    ETH: re.compile(r"^cowptn_noti_pm_(\d{4})_(\d{2})_(\d{2})_(\d{2})$"),
    OPTIC: re.compile(r"^cowptn_noti_pm_optic_power_(\d{4})_(\d{2})_(\d{2})_(\d{2})$"),
    ALARM: re.compile(r"^cowptn_noti_alarm_(\d{4})_(\d{2})_(\d{2})(?:_(\d{2}))?$"),
}

QUARANTINE_MARGIN = timedelta(hours=1, minutes=30)     # 4.3a [제안: ±1.5시간]
DEFAULT_LOOKBACK_HOURS = 72                            # K5

# 운영 결과 테이블 정의 (src/data/db_connector.py:47 의 컬럼, 4.9 충돌 점검용 상수)
ANOMALY_TABLE_COLUMNS = frozenset("""
id occur_date ip_addr cid lid tx_packet rx_packet error_packet tx_avg_power rx_avg_power
traffic_score traffic_severity traffic_slope traffic_threshold is_traffic_anomaly
optical_score optical_severity optical_slope optical_threshold is_optical_anomaly
anomaly_score severity slope slope_label threshold is_anomaly alarm_level alarm_label
ttf_minutes expected_fatal_time anomaly_reason rca_diagnosis rca_action feature_contribution detect_time
""".split())


@dataclass(frozen=True)
class TableInfo:
    name: str
    kind: str
    start: datetime          # 테이블 시각 H (시간 테이블) 또는 날짜 00:00 (일 테이블)
    daily: bool              # 알람 일 단위 이름
    rows_est: int = 0

    @property
    def window_end(self):
        return self.start + (timedelta(days=1) if self.daily else timedelta(hours=1))

    def trusted_range(self):
        """4.3a 정상 행의 occur_date 범위(양끝 포함): 테이블이 덮는 구간 앞뒤 1.5시간 여유."""
        return self.start - QUARANTINE_MARGIN, self.window_end + QUARANTINE_MARGIN


def classify_table(name, rows_est=0):
    """정규식을 통과하면 TableInfo, 아니면 None."""
    for kind in KINDS:
        m = _RE[kind].match(name or "")
        if not m:
            continue
        y, mo, d, h = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            start = datetime(int(y), int(mo), int(d), int(h) if h is not None else 0)
        except ValueError:
            return None
        return TableInfo(name, kind, start, daily=(kind == ALARM and h is None), rows_est=int(rows_est or 0))
    return None


def list_tables(db, kinds=KINDS):
    """종류별로 information_schema.tables 를 한 번씩 조회해 분류. 반환: {kind: [TableInfo...]} (시각순)."""
    out = {k: [] for k in kinds}
    seen = set()
    for kind in kinds:
        rows = db.query(
            "SELECT table_name AS table_name, table_rows AS table_rows FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name LIKE %s", (PREFIX[kind] + "%",))
        for r in rows:
            info = classify_table(r["table_name"], r.get("table_rows"))
            if info and info.kind == kind and info.name not in seen:
                seen.add(info.name)
                out[kind].append(info)
    for k in out:
        out[k].sort(key=lambda t: t.start)
    return out


def floor_hour(ts):
    return pd.Timestamp(ts).floor("h").to_pydatetime()


def needed_tables(tables, kind, start, end, lookback_hours=DEFAULT_LOOKBACK_HOURS):
    """4.3: 성능은 floor(start)−1h ~ floor(end−1s)+1h, 알람은 floor(start)−lookback ~ floor(end−1s)+1h 의 테이블."""
    lo = floor_hour(start) - (timedelta(hours=lookback_hours) if kind == ALARM else timedelta(hours=1))
    hi = floor_hour(pd.Timestamp(end) - pd.Timedelta(seconds=1)) + timedelta(hours=1)
    sel = []
    for t in tables:
        if t.daily:
            if t.start <= hi and t.window_end > lo:
                sel.append(t)
        elif lo <= t.start <= hi:
            sel.append(t)
    return sel


def missing_tables(tables, kind, start, end):
    """요청 구간에 해당하는 시간(또는 일) 테이블 중 없는 것의 시각 목록 (원본 공백, 4.2)."""
    present = {t.start for t in tables if t.kind == kind}
    first = floor_hour(start)
    last = floor_hour(pd.Timestamp(end) - pd.Timedelta(seconds=1))
    daily_mode = any(t.daily for t in tables if t.kind == kind) and not any(
        (not t.daily) for t in tables if t.kind == kind)
    out = []
    if daily_mode:
        d = first.replace(hour=0)
        while d <= last:
            if d not in present:
                out.append(d)
            d += timedelta(days=1)
    else:
        h = first
        while h <= last:
            if h not in present:
                out.append(h)
            h += timedelta(hours=1)
    return out


# ---- 4.3a 격리 판정 -----------------------------------------------------------

def is_trusted(table: TableInfo, occur_date, release_date=None, is_alarm=False):
    """정상 행 여부. occur_date NULL·범위 밖·(알람) release_date < occur_date 는 격리."""
    if occur_date is None or pd.isna(occur_date):
        return False
    lo, hi = table.trusted_range()
    if not (lo <= pd.Timestamp(occur_date).to_pydatetime() <= hi):
        return False
    if is_alarm and release_date is not None and not pd.isna(release_date):
        if pd.Timestamp(release_date) < pd.Timestamp(occur_date):
            return False
    return True


# ---- 4.4 스키마 프로필 ---------------------------------------------------------

class ProfileError(RuntimeError):
    pass


def filter_spec(kind, schema="real"):
    """종류별 필터 (이름, 값) 목록. 알람은 필터 없음. 설계서 4.4 표."""
    if kind == ETH:
        return [("signal_type", 6), ("time_type", 1)]
    if kind == OPTIC:
        return [("signal_type", 44), ("time_type", 1)]
    return []


def resolve_filters(kind, columns, schema="real"):
    """테이블의 실제 컬럼 집합으로 적용할 필터를 정한다.
    real: 필터 컬럼이 없으면 ProfileError (조용히 필터를 빼지 않음). sim: 없는 컬럼만 생략하고 생략 목록을 돌려준다.
    반환: (적용 필터 [(col, val)], 생략 목록 [col])"""
    if schema not in ("real", "sim"):
        raise ValueError(f"unknown schema profile: {schema}")
    applied, omitted = [], []
    for col, val in filter_spec(kind, schema):
        if col in columns:
            applied.append((col, val))
        elif schema == "sim":
            omitted.append(col)
        else:
            raise ProfileError(f"{kind} 테이블에 필터 컬럼 {col} 이 없습니다 (--schema real). 시뮬레이션 DB 면 --schema sim")
    return applied, omitted


def fetch_columns(db, kind, names=None):
    """종류별 컬럼 집합 {table: set(cols)} — information_schema.columns 한 번 조회."""
    rows = db.query(
        "SELECT table_name AS table_name, column_name AS column_name FROM information_schema.columns "
        "WHERE table_schema = DATABASE() AND table_name LIKE %s", (PREFIX[kind] + "%",))
    cols = {}
    for r in rows:
        if names is not None and r["table_name"] not in names:
            continue
        cols.setdefault(r["table_name"], set()).add(r["column_name"])
    return cols


def build_where(filters, ip_list=None):
    """필터 → (WHERE 조각, 파라미터). 컬럼 이름은 resolve_filters 가 정한 고정 목록에서만 온다."""
    parts, params = [], []
    for col, val in filters:
        parts.append(f"`{col}` = %s")
        params.append(val)
    if ip_list:
        parts.append("`ip_addr` IN (" + ", ".join(["%s"] * len(ip_list)) + ")")
        params.extend(ip_list)
    return (" AND ".join(parts) if parts else "1=1"), params


# ---- 4.9 결과 테이블 충돌 점검 --------------------------------------------------

def check_result_table(db):
    """anomaly_detection 존재/컬럼 점검 (읽기만). 반환 dict: status = none | operational | conflict_suspected"""
    rows = db.query(
        "SELECT column_name AS column_name FROM information_schema.columns "
        "WHERE table_schema = DATABASE() AND table_name = %s", ("anomaly_detection",))
    cols = {r["column_name"] for r in rows}
    if not cols:
        return {"status": "none", "columns": []}
    if ANOMALY_TABLE_COLUMNS <= cols:
        return {"status": "operational", "columns": sorted(cols)}
    return {"status": "conflict_suspected", "columns": sorted(cols),
            "missing": sorted(ANOMALY_TABLE_COLUMNS - cols),
            "warning": "이름 충돌 의심 — 운영 서비스를 이 DB 로 기동하지 말 것"}


# ---- plan --------------------------------------------------------------------

def tz_warning(db_max_occur, now, low_h=8.0, high_h=10.0):
    """4.6: 최신 테이블 MAX(occur_date) 와 실행 호스트 현재 시각 차이가 8~10시간이면 경고 문자열."""
    if db_max_occur is None:
        return None
    diff_h = abs((pd.Timestamp(now) - pd.Timestamp(db_max_occur)).total_seconds()) / 3600
    if low_h <= diff_h <= high_h:
        return f"시간대 의심: DB 최신 occur_date 와 현재 시각 차이 {diff_h:.1f}시간 (UTC/KST 어긋남 가능, lessons #27)"
    return None


def build_plan(db, start, end, kinds=KINDS, schema="real", lookback_hours=DEFAULT_LOOKBACK_HOURS,
               now=None, sample_distribution=True):
    """DB 데이터는 집계 쿼리로만 조회(행 추출 없음). 반환 dict (CLI 가 출력)."""
    now = now or datetime.now()
    tabs = list_tables(db, kinds)
    plan = {"range": [str(start), str(end)], "schema": schema, "kinds": {}, "warnings": []}
    for kind in kinds:
        all_t = tabs[kind]
        need = needed_tables(all_t, kind, start, end, lookback_hours)
        info = {
            "tables_total": len(all_t),
            "first_table": all_t[0].name if all_t else None,
            "last_table": all_t[-1].name if all_t else None,
            "tables_needed": len(need),
            "rows_est": sum(t.rows_est for t in need),
            "missing": [str(m) for m in missing_tables(all_t, kind, start, end)],
        }
        if kind == ALARM and all_t:
            n_daily = sum(1 for t in all_t if t.daily)
            info["name_format"] = "daily" if n_daily == len(all_t) else "hourly" if n_daily == 0 else "mixed"
            if info["name_format"] == "mixed":
                plan["warnings"].append("알람 테이블 이름에 일 단위와 시간 단위가 섞여 있음 (K15) — 둘 다 읽음, 중복은 추출 보고로 확인")
        cols = fetch_columns(db, kind, {t.name for t in need}) if need else {}
        # 컬럼 프로필 점검(필터 컬럼 존재)
        profile_errors, omitted = [], set()
        for t in need:
            try:
                _, om = resolve_filters(kind, cols.get(t.name, set()), schema)
                omitted.update(om)
            except ProfileError as e:
                profile_errors.append(str(e))
                break
        info["profile_error"] = profile_errors[0] if profile_errors else None
        info["filters_omitted"] = sorted(omitted)
        if sample_distribution and need and kind in (ETH, OPTIC):
            info["signal_time_type"] = _signal_distribution(db, [need[0], need[-1]], cols)
        if need:
            info["boundary"] = _boundary_minmax(db, [need[0], need[-1]])
        plan["kinds"][kind] = info
    plan["total_rows_est"] = sum(v["rows_est"] for v in plan["kinds"].values())
    last_max = None
    for kind in kinds:
        b = plan["kinds"][kind].get("boundary")
        if b:
            m = max((x["max"] for x in b if x["max"] is not None), default=None)
            if m is not None and (last_max is None or m > last_max):
                last_max = m
    plan["tz_warning"] = tz_warning(last_max, now)
    if plan["tz_warning"]:
        plan["warnings"].append(plan["tz_warning"])
    plan["result_table"] = check_result_table(db)
    if plan["result_table"]["status"] == "conflict_suspected":
        plan["warnings"].append(f"anomaly_detection: {plan['result_table']['warning']}")
    return plan


def _signal_distribution(db, tables, cols):
    out = []
    for t in tables:
        c = cols.get(t.name, set())
        sel = [x for x in ("signal_type", "time_type") if x in c]
        if not sel:
            out.append({"table": t.name, "groups": None})
            continue
        col_sql = ", ".join(f"`{x}`" for x in sel)
        rows = db.query(f"SELECT {col_sql}, COUNT(*) AS n FROM `{t.name}` GROUP BY {col_sql}")
        out.append({"table": t.name, "groups": rows})
    return out


def _boundary_minmax(db, tables):
    out = []
    for t in tables:
        r = db.query(f"SELECT MIN(`occur_date`) AS min_d, MAX(`occur_date`) AS max_d FROM `{t.name}`")
        row = r[0] if r else {}
        out.append({"table": t.name, "min": row.get("min_d"), "max": row.get("max_d")})
    return out


def format_plan(plan):
    lines = [f"구간 {plan['range'][0]} ~ {plan['range'][1]}  (schema={plan['schema']})"]
    for kind, v in plan["kinds"].items():
        lines.append(f"[{kind}] 테이블 {v['tables_total']}개 (첫 {v['first_table']}, 끝 {v['last_table']}), "
                     f"구간 필요 {v['tables_needed']}개, 예상 행 {v['rows_est']:,}, 없는 시간 {len(v['missing'])}개"
                     + (f", 이름 형식 {v['name_format']}" if v.get("name_format") else ""))
        if v.get("profile_error"):
            lines.append(f"   !! {v['profile_error']}")
        if v.get("filters_omitted"):
            lines.append(f"   필터 생략(sim): {', '.join(v['filters_omitted'])}")
        for d in v.get("signal_time_type", []):
            lines.append(f"   {d['table']} signal/time_type 분포: {d['groups']}")
        for b in v.get("boundary", []):
            lines.append(f"   {b['table']} occur_date min={b['min']} max={b['max']}")
    lines.append(f"합계 예상 행 {plan['total_rows_est']:,}")
    rt = plan["result_table"]
    lines.append(f"anomaly_detection: {rt['status']}")
    for w in plan["warnings"]:
        lines.append(f"[경고] {w}")
    return "\n".join(lines)
