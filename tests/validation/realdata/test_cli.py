"""U1/U2: real_data_check CLI 골격 — 거부 코드·plan·extract 연결"""
from datetime import datetime

from tests.validation.realdata.fakes import ETH_COLS, FakeServer, ROOT_GRANTS, eth_row
from validation.cli import real_data_check as cli
from validation.realdata.db_readonly import ReadOnlyDB

ENV = {"DB_HOST": "h", "DB_NAME": "ems", "RDC_DB_USER": "u", "RDC_DB_PASS": "p"}


def _patch(monkeypatch, srv):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    real = ReadOnlyDB.connect.__func__
    monkeypatch.setattr(ReadOnlyDB, "connect", classmethod(
        lambda cls, config, allow_write_account=False, max_exec_ms=60000, connector=None:
        real(cls, config, allow_write_account, max_exec_ms, connector=srv.connector)))


def _srv(**kw):
    return FakeServer(tables={"cowptn_noti_pm_2026_10_01_10": {
        "cols": ETH_COLS, "rows": [eth_row(1, datetime(2026, 10, 1, 10, 15))]}}, **kw)


def test_root_is_refused_with_exit_3_and_guide(monkeypatch, capsys):
    _patch(monkeypatch, _srv(grants=ROOT_GRANTS))
    assert cli.main(["plan", "--start", "2026-10-01", "--end", "2026-10-02"]) == 3
    err = capsys.readouterr().err
    assert "[거부]" in err and "--allow-write-account" in err


def test_plan_runs_for_readonly_and_for_root_with_flag(monkeypatch, capsys):
    _patch(monkeypatch, _srv())
    assert cli.main(["plan", "--start", "2026-10-01 10:00", "--end", "2026-10-01 11:00", "--kinds", "eth"]) == 0
    assert "[eth]" in capsys.readouterr().out
    _patch(monkeypatch, _srv(grants=ROOT_GRANTS))
    assert cli.main(["plan", "--start", "2026-10-01 10:00", "--end", "2026-10-01 11:00", "--kinds", "eth",
                     "--allow-write-account"]) == 0


def test_extract_via_cli_and_no_overwrite(monkeypatch, tmp_path, capsys):
    _patch(monkeypatch, _srv())
    base = ["extract", "--kinds", "eth", "--runs-dir", str(tmp_path)]
    assert cli.main(base + ["--start", "2026-10-01 10:00", "--end", "2026-10-01 11:00", "--run-id", "r1"]) == 0
    assert (tmp_path / "r1" / "manifest.json").exists() and "완료=True" in capsys.readouterr().out
    assert cli.main(base + ["--start", "2026-10-01 10:00", "--end", "2026-10-01 11:00", "--run-id", "r1"]) == 4   # 덮어쓰기 거부
    assert "append" not in cli.build_parser().format_help()          # 증분 추출은 구현하지 않음 (설계 4.7)


def test_unknown_kind_rejected(monkeypatch):
    _patch(monkeypatch, _srv())
    assert cli.main(["plan", "--start", "a", "--end", "b", "--kinds", "foo"]) == 2
