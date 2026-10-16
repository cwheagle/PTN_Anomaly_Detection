"""U1: 권한 검사(fail closed)·문장 가드·세션 설정 (설계서 4.1)"""
import pytest

from tests.validation.realdata.fakes import FakeServer, ROOT_GRANTS, RO_GRANTS, make_db
from validation.realdata import db_readonly as dbr
from validation.realdata.db_readonly import (
    PrivilegeError, ReadOnlyDB, ReadOnlyGuardError, SessionSetupError, evaluate_grants, guard_statement,
    load_db_config, parse_grant_line)

CFG = {"host": "h", "port": 3306, "user": "u", "password": "p", "database": "ems"}


# ---- GRANT 파싱·평가 -----------------------------------------------------------
@pytest.mark.parametrize("lines", [
    RO_GRANTS,
    ["GRANT USAGE ON *.* TO 'u'@'h'"],
    ["GRANT SELECT ON `ems`.`t1` TO 'u'@'h'"],
    ["GRANT SELECT (id, ip_addr) ON `ems`.`t1` TO 'u'@'h'"],
])
def test_grants_allowed(lines):
    assert evaluate_grants(lines) == []


@pytest.mark.parametrize("line,word", [
    (ROOT_GRANTS[0], "ALL PRIVILEGES"),
    ("GRANT SELECT, INSERT ON `ems`.* TO 'u'@'h'", "INSERT"),
    ("GRANT PROXY ON ''@'' TO 'u'@'h' WITH GRANT OPTION", "PROXY"),
    ("GRANT SELECT, FROBNICATE ON `ems`.* TO 'u'@'h'", "FROBNICATE"),        # 모르는 권한 단어 → 거부
    ("GRANT SELECT ON `ems`.* TO 'u'@'h' WITH GRANT OPTION", "WITH GRANT OPTION"),
    ("GRANT SELECT, LOCK TABLES ON `ems`.* TO 'u'@'h'", "LOCK TABLES"),
])
def test_grants_denied_words(line, word):
    assert word in evaluate_grants([line])


def test_unparseable_line_is_denied():
    assert parse_grant_line("something odd") is None
    assert evaluate_grants(["something odd"])[0].startswith("UNPARSEABLE")


def test_one_bad_line_among_good_denies():
    assert evaluate_grants(RO_GRANTS + ["GRANT DELETE ON `ems`.* TO 'u'@'h'"]) == ["DELETE"]


# ---- 연결: 거부/통과/예외 플래그 --------------------------------------------------
def test_readonly_account_passes_and_session_is_read_only():
    s = FakeServer(grants=RO_GRANTS)
    db = make_db(s)
    assert db.privilege.status == "ok"
    assert db.privilege.to_manifest()["privilege_check"] == "ok"
    st = s.statements()
    assert st[0] == "SHOW GRANTS FOR CURRENT_USER()"            # 권한 검사가 세션 설정 앞
    assert st[1] == "SET SESSION TRANSACTION READ ONLY"
    assert st[2] == "SET SESSION MAX_EXECUTION_TIME = 60000"


def test_root_denied_by_default_and_session_never_set():
    s = FakeServer(grants=ROOT_GRANTS)
    with pytest.raises(PrivilegeError) as e:
        make_db(s)
    assert "--allow-write-account" in str(e.value) and "GRANT SELECT" in str(e.value)   # 계정 생성 안내 포함
    assert not any(q.startswith("SET SESSION") for q in s.statements())
    assert s.closed


def test_show_grants_failure_is_denied_by_default():
    with pytest.raises(PrivilegeError):
        make_db(FakeServer(fail_show_grants=True))


def test_override_flag_proceeds_and_records(capsys):
    s = FakeServer(grants=ROOT_GRANTS)
    db = make_db(s, allow_write_account=True)
    m = db.privilege.to_manifest()
    assert m["privilege_check"] == "override" and "ALL PRIVILEGES" in m["denied"] and m["account"] == "u"
    assert "password" not in str(m).lower()
    assert "allow-write-account" in capsys.readouterr().err
    assert "SET SESSION TRANSACTION READ ONLY" in s.statements()      # 세션 읽기 전용은 그대로


def test_override_flag_allows_unparseable_and_show_grants_failure():
    assert make_db(FakeServer(fail_show_grants=True), allow_write_account=True).privilege.status == "override"
    assert make_db(FakeServer(grants=["weird"]), allow_write_account=True).privilege.status == "override"


def test_override_does_not_disable_statement_guard():
    db = make_db(FakeServer(grants=ROOT_GRANTS), allow_write_account=True)
    for bad in ("DELETE FROM t", "INSERT INTO t VALUES (1)", "UPDATE t SET a=1", "DROP TABLE t",
                "SET SESSION TRANSACTION READ WRITE"):
        with pytest.raises(ReadOnlyGuardError):
            db.query(bad)


@pytest.mark.parametrize("allow", [False, True])
def test_read_only_setup_failure_stops_regardless_of_flag(allow):
    s = FakeServer(grants=ROOT_GRANTS if allow else RO_GRANTS, fail_read_only=True)
    with pytest.raises(SessionSetupError):
        make_db(s, allow_write_account=allow)
    assert s.closed


def test_env_var_cannot_enable_override(monkeypatch):
    for k in ("RDC_ALLOW_WRITE_ACCOUNT", "ALLOW_WRITE_ACCOUNT", "PTN_ALLOW_WRITE_ACCOUNT"):
        monkeypatch.setenv(k, "1")
    with pytest.raises(PrivilegeError):
        make_db(FakeServer(grants=ROOT_GRANTS))
    from validation.cli.real_data_check import build_parser
    assert build_parser().parse_args(["plan", "--start", "a", "--end", "b"]).allow_write_account is False
    assert build_parser().parse_args(["plan", "--start", "a", "--end", "b", "--allow-write-account"]).allow_write_account


# ---- 문장 가드 -----------------------------------------------------------------
@pytest.mark.parametrize("sql", ["SELECT 1", "select * from t", "  SHOW TABLES", "-- c\nSELECT 1", "/* x */ SELECT 1;",
                                 "SELECT a FROM t WHERE b = %s"])
def test_guard_allows(sql):
    guard_statement(sql)


@pytest.mark.parametrize("sql", ["", "DELETE FROM t", "/* SELECT */ DELETE FROM t", "SELECT 1; DROP TABLE t",
                                 "SELECT 1;SELECT 2", "UPDATE t SET a=1", "INSERT INTO t VALUES(1)", "CREATE TABLE x(a int)",
                                 "ALTER TABLE t ADD c int", "SET SESSION x=1", "WITH a AS (SELECT 1) SELECT * FROM a",
                                 "-- SELECT\nDROP TABLE t"])
def test_guard_denies(sql):
    with pytest.raises(ReadOnlyGuardError):
        guard_statement(sql)


def test_query_returns_dicts_and_counts():
    s = FakeServer(tables={"cowptn_noti_pm_2026_10_01_00": {"cols": ["id"], "rows": []}})
    db = make_db(s)
    rows = db.query("SELECT DATABASE() AS db")
    assert rows == [{"db": "ems"}] and db.queries == 1


# ---- 설정 ----------------------------------------------------------------------
def test_load_db_config_precedence_and_missing():
    env = {"DB_HOST": "h", "DB_NAME": "n", "DB_USER": "root", "DB_PASS": "rootpw", "RDC_DB_USER": "ro", "RDC_DB_PASS": "ropw"}
    c = load_db_config(env=env)
    assert (c["user"], c["password"], c["port"]) == ("ro", "ropw", 3306)
    assert load_db_config(env=env, user="x", port=3307)["user"] == "x"
    with pytest.raises(RuntimeError, match="password"):
        load_db_config(env={"DB_HOST": "h", "DB_NAME": "n", "DB_USER": "u"})


def test_modules_do_not_import_src_config_or_open_env_files():
    import ast, inspect
    from validation.realdata import extract, tables
    for mod in (dbr, tables, extract):
        tree = ast.parse(inspect.getsource(mod))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                assert not (n.module or "").startswith("src"), mod.__name__
            if isinstance(n, ast.Import):
                assert not any(a.name.startswith("src") for a in n.names), mod.__name__
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                assert n.value not in (".env", "src/config.py"), mod.__name__
