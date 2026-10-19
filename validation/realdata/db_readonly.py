"""
P1-2 실데이터 검증 도구의 읽기 전용 DB 연결 (설계서 docs/design/p1_2_real_data_check.md 4.1)

- `DBConnector` 는 생성자에서 CREATE/ALTER 를 실행하므로 쓰지 않는다(설계서 2.2-1).
- 접속 정보는 환경변수/CLI 인자로만 받는다. `src/config.py`·`.env` 는 읽지 않는다.
- 3중 방어: ① 권한 검사(fail closed, 명시 플래그로만 예외) → ② 세션 읽기 전용 → ③ 문장 가드(SELECT/SHOW 만).
- 접속·조회 실패는 빈 결과로 넘기지 않고 예외로 중단한다(lessons #14).
"""
import os
import re
from dataclasses import dataclass, field

DEFAULT_MAX_EXEC_MS = 60_000

ALLOWED_PRIVILEGES = {"SELECT", "USAGE"}

_GRANT_RE = re.compile(r"^\s*GRANT\s+(?P<privs>.+?)\s+ON\s+(?P<obj>.+?)\s+TO\s+(?P<rest>.+)$",
                       re.IGNORECASE | re.DOTALL)
_GRANT_OPTION_RE = re.compile(r"\bWITH\s+GRANT\s+OPTION\b", re.IGNORECASE)
_PAREN_RE = re.compile(r"\([^)]*\)")


class PrivilegeError(RuntimeError):
    """권한 검사 거부 (4.1)."""


class ReadOnlyGuardError(RuntimeError):
    """SELECT/SHOW 이외의 문장 (4.1 문장 가드)."""


class SessionSetupError(RuntimeError):
    """읽기 전용 세션 설정 실패 — 플래그와 무관하게 중단 (4.1 규칙 3)."""


@dataclass
class PrivilegeResult:
    status: str                      # "ok" | "override"
    grant_lines: list = field(default_factory=list)
    denied: list = field(default_factory=list)   # 거부 사유가 된 권한 단어/사유
    account: str = ""

    def to_manifest(self):
        return {"privilege_check": self.status, "grant_lines": list(self.grant_lines),
                "denied": list(self.denied), "account": self.account}


def parse_grant_line(line):
    """GRANT 한 줄 → (권한 단어 집합, with_grant_option). 파싱 불가면 None."""
    m = _GRANT_RE.match(line or "")
    if not m:
        return None
    privs = _PAREN_RE.sub("", m.group("privs"))      # SELECT (col1, col2) 의 열 목록 제거
    words = {p.strip().upper() for p in privs.split(",") if p.strip()}
    if not words:
        return None
    return words, bool(_GRANT_OPTION_RE.search(m.group("rest")))


def evaluate_grants(grant_lines):
    """허용 외 권한 사유 목록을 반환 (빈 목록 = 허용). 화이트리스트 방식이라 모르는 권한 단어도 사유가 된다."""
    denied = []
    for line in grant_lines:
        parsed = parse_grant_line(line)
        if parsed is None:
            denied.append(f"UNPARSEABLE: {line!r}")
            continue
        words, with_option = parsed
        for w in sorted(words - ALLOWED_PRIVILEGES):
            if w not in denied:
                denied.append(w)
        if with_option and "WITH GRANT OPTION" not in denied:
            denied.append("WITH GRANT OPTION")
    return denied


ACCOUNT_GUIDE = """\
읽기 전용 계정을 만들거나(권장), 테스트망 한정으로 --allow-write-account 를 명시하세요.

[읽기 전용 계정 만들기 — 관리자(root)가 1회] (MySQL 5.7)
  CREATE USER 'ptn_ro'@'<분석PC_IP>' IDENTIFIED BY '<충분히 긴 비밀번호>';   -- '%' 호스트는 쓰지 않는다
  GRANT SELECT ON `<EMS_DB>`.* TO 'ptn_ro'@'<분석PC_IP>';
  SHOW GRANTS FOR 'ptn_ro'@'<분석PC_IP>';   -- 기대: GRANT USAGE ON *.* / GRANT SELECT ON `<EMS_DB>`.*
  환경변수: RDC_DB_USER=ptn_ro, RDC_DB_PASS=...   (비밀번호를 파일·명령 이력에 남기지 않는다)
  삭제:     DROP USER 'ptn_ro'@'<분석PC_IP>';

[쓰기 계정 예외] --allow-write-account (명령줄에서 매번 명시, 환경변수로는 켜지지 않음)
  권한 검사만 건너뛰며 세션 READ ONLY·문장 가드는 그대로 적용된다. 납품 현장 DB 에서는 쓰지 않는 것을 기본으로 한다.
"""


def check_privileges(grant_lines, allow_write_account=False, account=""):
    """권한 검사. 거부 사유가 있고 플래그가 없으면 PrivilegeError."""
    denied = evaluate_grants(grant_lines)
    if not denied:
        return PrivilegeResult("ok", list(grant_lines), [], account)
    if not allow_write_account:
        raise PrivilegeError(
            "읽기 전용이 아닌 계정입니다 (허용: SELECT/USAGE 단독). 거부 사유: "
            + ", ".join(denied) + "\n" + ACCOUNT_GUIDE)
    return PrivilegeResult("override", list(grant_lines), denied, account)


# ---- 문장 가드 ----------------------------------------------------------------

_COMMENT_RE = re.compile(r"/\*.*?\*/|--[^\n]*|#[^\n]*", re.DOTALL)
_ALLOWED_FIRST = {"SELECT", "SHOW"}


def guard_statement(sql):
    """SELECT/SHOW 한 문장만 통과. 주석 제거 후 첫 토큰 검사, 여러 문장(;) 거부. 값은 항상 %s 바인딩이라
    SQL 본문에 리터럴 ';' 이 있는 경우도 보수적으로 거부한다."""
    body = _COMMENT_RE.sub(" ", sql or "").strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    if ";" in body:
        raise ReadOnlyGuardError(f"여러 문장은 허용되지 않습니다: {sql!r}")
    tokens = body.split(None, 1)
    first = tokens[0].upper() if tokens else ""
    if first not in _ALLOWED_FIRST:
        raise ReadOnlyGuardError(f"SELECT/SHOW 만 허용됩니다: {sql!r}")
    return body


# ---- 설정 --------------------------------------------------------------------

def load_db_config(env=None, host=None, port=None, database=None, user=None):
    """접속 정보 = CLI 인자 > 환경변수. 비밀번호는 환경변수로만. src/config.py·.env 는 읽지 않는다.
    환경변수: RDC_DB_USER/RDC_DB_PASS (도구 전용) 가 DB_USER/DB_PASS 보다 우선, DB_HOST, DB_PORT, DB_NAME."""
    env = os.environ if env is None else env
    cfg = {
        "host": host or env.get("DB_HOST"),
        "port": int(port or env.get("DB_PORT") or 3306),
        "database": database or env.get("DB_NAME"),
        "user": user or env.get("RDC_DB_USER") or env.get("DB_USER"),
        "password": env.get("RDC_DB_PASS") or env.get("DB_PASS"),
    }
    missing = [k for k in ("host", "database", "user", "password") if not cfg[k]]
    if missing:
        raise RuntimeError("DB 접속 정보 부족: " + ", ".join(missing)
                           + " (환경변수 DB_HOST/DB_NAME/RDC_DB_USER/RDC_DB_PASS 또는 --host/--database/--user)")
    return cfg


# ---- 연결 --------------------------------------------------------------------

class ReadOnlyDB:
    """권한 검사와 읽기 전용 세션을 통과한 연결. 모든 쿼리는 `query()` 한 곳을 지난다."""

    def __init__(self, conn, privilege: PrivilegeResult, max_exec_ms=DEFAULT_MAX_EXEC_MS):
        self._conn = conn
        self.privilege = privilege
        self.max_exec_ms = max_exec_ms
        self.queries = 0

    @classmethod
    def connect(cls, config, allow_write_account=False, max_exec_ms=DEFAULT_MAX_EXEC_MS, connector=None):
        """접속 → 권한 검사(세션 설정 전) → 세션 설정. `connector` 는 테스트에서 가짜 연결을 주입한다."""
        if connector is None:
            import mysql.connector
            connector = mysql.connector.connect
        conn = connector(host=config["host"], port=config["port"], user=config["user"],
                         password=config["password"], database=config["database"])
        try:
            lines, show_error = [], None
            try:
                lines = [str(r[0]) for r in cls._raw(conn, "SHOW GRANTS FOR CURRENT_USER()")]
            except Exception as e:           # 판단 불가 = 거부 (플래그로만 진행)
                show_error = e
            if show_error is not None:
                if not allow_write_account:
                    raise PrivilegeError(f"SHOW GRANTS 실패(판단 불가 = 거부): {show_error}\n" + ACCOUNT_GUIDE)
                priv = PrivilegeResult("override", [], [f"SHOW GRANTS FAILED: {show_error}"], config["user"])
            else:
                priv = check_privileges(lines, allow_write_account, account=config["user"])
            if priv.status == "override":
                import sys
                print("[경고] --allow-write-account: 권한 검사를 건너뜁니다. 사유: " + ", ".join(priv.denied),
                      file=sys.stderr)
            # 세션 설정 2개는 여기서만 허용 (실패는 플래그와 무관하게 중단)
            for stmt in ("SET SESSION TRANSACTION READ ONLY",
                         f"SET SESSION MAX_EXECUTION_TIME = {int(max_exec_ms)}"):
                try:
                    cur = conn.cursor()
                    cur.execute(stmt)
                    cur.close()
                except Exception as e:
                    raise SessionSetupError(f"읽기 전용 세션 설정 실패 ({stmt}): {e}") from e
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise
        return cls(conn, priv, max_exec_ms)

    @staticmethod
    def _raw(conn, sql):
        guard_statement(sql)
        cur = conn.cursor()
        try:
            cur.execute(sql)
            return cur.fetchall()
        finally:
            cur.close()

    def query(self, sql, params=None):
        """SELECT/SHOW 실행 → 행 dict 목록. 실패는 예외 (빈 결과로 넘기지 않음)."""
        guard_statement(sql)
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
        finally:
            cur.close()
        self.queries += 1
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
