"""P1-2 테스트용 가짜 MySQL: 도구가 보내는 SQL 패턴만 에뮬레이트하고 모든 쿼리를 기록한다 (DB 불필요)."""
import re
from datetime import datetime

from validation.realdata.db_readonly import ReadOnlyDB

ROOT_GRANTS = ["GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION"]
RO_GRANTS = ["GRANT USAGE ON *.* TO 'ptn_ro'@'10.0.0.1'", "GRANT SELECT ON `ems`.* TO 'ptn_ro'@'10.0.0.1'"]

_SELECT_RE = re.compile(r"^SELECT (?P<cols>.+?) FROM (?:`\w+`\.)?`(?P<t>\w+)` WHERE (?P<where>.+?)"
                        r"(?: AND `id` > %s ORDER BY `id` LIMIT %s| LIMIT %s)?$", re.DOTALL)


def _pad(v):
    """MySQL PAD SPACE 비교(VARCHAR 끝 공백 무시)를 흉내 낸다."""
    return v.rstrip() if isinstance(v, str) else v


def _cond_ok(val, op, v):
    if op == "in":
        return _pad(val) in {_pad(x) for x in v}
    if op == "=":
        return _pad(val) == _pad(v)
    if val is None:
        return False
    return {">=": val >= v, "<=": val <= v, "<": val < v, ">": val > v}[op]


class FakeCursor:
    def __init__(self, server):
        self.s, self.description, self._rows = server, None, []

    def execute(self, sql, params=None):
        self.s.log.append((sql, tuple(params or ())))
        self.description, self._rows = self.s.handle(sql, list(params or ()))

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeServer:
    """tables: name -> {"cols": [...], "rows": [dict], "rows_est": int}"""

    def __init__(self, grants=None, tables=None, database="ems", fail_show_grants=False, fail_read_only=False,
                 extra_columns=None):
        self.grants = RO_GRANTS if grants is None else grants
        self.tables = tables or {}
        self.database = database
        self.fail_show_grants, self.fail_read_only = fail_show_grants, fail_read_only
        self.log, self.closed = [], False

    # DB-API 연결 인터페이스
    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True

    def connector(self, **kwargs):
        self.connect_kwargs = kwargs
        return self

    def statements(self):
        return [q for q, _ in self.log]

    def handle(self, sql, params):
        s = " ".join(sql.split())
        if s == "SHOW GRANTS FOR CURRENT_USER()":
            if self.fail_show_grants:
                raise RuntimeError("SHOW GRANTS denied")
            return [("Grants",)], [(g,) for g in self.grants]
        if s.startswith("SET SESSION"):
            if self.fail_read_only and "READ ONLY" in s:
                raise RuntimeError("cannot set read only")
            return None, []
        if s == "SELECT DATABASE() AS db":
            return [("db",)], [(self.database,)]
        if "FROM information_schema.tables" in s:
            like = params[0].rstrip("%")
            rows = [(n, t.get("rows_est", len(t["rows"]))) for n, t in self.tables.items() if n.startswith(like)]
            return [("table_name",), ("table_rows",)], rows
        if "FROM information_schema.columns" in s:
            if "table_name = %s" in s:
                t = self.tables.get(params[0])
                return [("column_name",)], [(c,) for c in (t["cols"] if t else [])]
            like = params[0].rstrip("%")
            rows = [(n, c) for n, t in self.tables.items() if n.startswith(like) for c in t["cols"]]
            return [("table_name",), ("column_name",)], rows
        m = re.match(r"^SELECT COUNT\(\*\) AS n FROM `(\w+)`$", s)
        if m:
            return [("n",)], [(len(self.tables[m.group(1)]["rows"]),)]
        m = re.match(r"^SELECT MIN\(`occur_date`\) AS min_d, MAX\(`occur_date`\) AS max_d FROM `(\w+)`$", s)
        if m:
            ds = [r["occur_date"] for r in self.tables[m.group(1)]["rows"] if isinstance(r["occur_date"], datetime)]
            return [("min_d",), ("max_d",)], [(min(ds) if ds else None, max(ds) if ds else None)]
        m = re.match(r"^SELECT (.+), COUNT\(\*\) AS n FROM `(\w+)` GROUP BY (.+)$", s)
        if m:
            cols = [c.strip().strip("`") for c in m.group(1).split(",")]
            counts = {}
            for r in self.tables[m.group(2)]["rows"]:
                k = tuple(r.get(c) for c in cols)
                counts[k] = counts.get(k, 0) + 1
            return [(c,) for c in cols] + [("n",)], [k + (n,) for k, n in counts.items()]
        m = _SELECT_RE.match(s)
        if m:
            return self._select(m, params)
        raise AssertionError(f"FakeServer: 처리하지 못한 SQL: {sql}")

    def _select(self, m, params):
        cols = [c.strip().strip("`") for c in m.group("cols").split(",")]
        t = self.tables[m.group("t")]
        where, i, conds = m.group("where"), 0, []
        keyset = "AND `id` >" in " ".join([m.string])
        for part in [p for p in where.split(" AND ") if p != "1=1"]:
            if " IN (" in part:
                col = part.split("`")[1]
                n = part.count("%s")
                conds.append((col, "in", set(params[i:i + n])))
                i += n
            else:
                mm = re.match(r"^`(\w+)` (>=|<=|<|>|=) %s$", part)
                conds.append((mm.group(1), mm.group(2), params[i]))
                i += 1
        last_id = limit = None
        if keyset:
            last_id, limit = params[i], params[i + 1]
        elif m.string.rstrip().endswith("LIMIT %s"):
            limit = params[i]
        out = []
        for r in sorted(t["rows"], key=lambda r: r.get("id", 0)):
            if not all(_cond_ok(r.get(c), op, v) for c, op, v in conds):
                continue
            if last_id is not None and r["id"] <= last_id:
                continue
            out.append(r)
            if limit is not None and len(out) >= limit:
                break
        return [(c,) for c in cols], [tuple(r.get(c) for c in cols) for r in out]


def make_db(server, allow_write_account=False, config=None):
    cfg = config or {"host": "h", "port": 3306, "user": "u", "password": "p", "database": server.database}
    return ReadOnlyDB.connect(cfg, allow_write_account=allow_write_account, connector=server.connector)


ETH_COLS = ["id", "system_type", "ip_addr", "address", "cid", "lid", "augid", "au3id", "tugid", "tuid",
            "signal_type", "time_type", "bbe_in_error", "es", "ses", "occur_date"]
ALARM_COLS = ["id", "msgid", "tid", "system_type", "meanid", "sid", "cid", "lid", "ip_addr", "address",
              "signal_type", "severity_type", "alarm_type", "occur_date", "release_date"]
SIM_ETH_COLS = ["id", "occur_date", "ip_addr", "cid", "lid", "signal_type", "es", "ses", "bbe_in_error"]


def eth_row(i, occ, ip="10.0.0.1", cid=1, lid=1, tx=100, rx=90, err=0, sig=6, tt=1):
    return {"id": i, "system_type": 31, "ip_addr": ip, "address": "A-S.1 -P1 ", "cid": cid, "lid": lid,
            "augid": 0, "au3id": 0, "tugid": 0, "tuid": 0, "signal_type": sig, "time_type": tt,
            "bbe_in_error": err, "es": tx, "ses": rx, "occur_date": occ}


def alarm_row(i, occ, rel=None, ip="10.0.0.1", cid=1, lid=1, atype=600):
    return {"id": i, "msgid": i, "tid": "CA237", "system_type": 31, "meanid": 0, "sid": 0, "cid": cid, "lid": lid,
            "ip_addr": ip + " ", "address": "S.2 -P1  ", "signal_type": 0, "severity_type": 3,
            "alarm_type": atype, "occur_date": occ, "release_date": rel}
