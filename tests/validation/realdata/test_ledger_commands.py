"""개정 4 ③: 분석 장부 기록과 잠금 거부, 데이터일 < 28 explore (설계 9.7.4, F-14)"""
import json

import pandas as pd
import pytest

from tests.validation.realdata.test_lock_judge import GOOD, ctx_of, locked  # noqa: F401
from tests.validation.realdata.test_replay_report import FakeDetector
from tests.validation.realdata.test_snapshot_commands import D0, run  # noqa: F401  (fixture)
from validation.realdata import commands as cmds
from validation.realdata import periods as pr


def small_run(tmp_path, n_days, names):
    runs = tmp_path / "runs_small"
    rd = runs / "r1"
    (rd / "raw" / "eth").mkdir(parents=True)
    (rd / "raw" / "alarm").mkdir(parents=True)
    for i in range(n_days):
        day = D0 + pd.Timedelta(days=i)
        rows = [{"ip_addr": "1.1.1.1", "cid": 1, "lid": 1, "system_type": 31, "address": "X-S.1 -P1", "tx_packet": 5, "rx_packet": 100,
                 "error_packet": 0, "occur_date": day + pd.Timedelta(minutes=15 * k), "bin_end": day + pd.Timedelta(minutes=15 * k), "src_table": "t", "id": k}
                for k in range(1, 97)]
        pd.DataFrame(rows).to_csv(rd / "raw" / "eth" / f"{day:%Y-%m-%d}.csv.gz", index=False, compression="gzip")
    pd.DataFrame(columns=["id", "ip_addr", "cid", "lid", "system_type", "alarm_type", "severity_type", "occur_date", "release_date", "address", "src_table"]
                 ).to_csv(rd / "raw" / "alarm" / "all.csv.gz", index=False, compression="gzip")
    json.dump({"complete": True, "range_start": str(D0), "range_end": str(D0 + pd.Timedelta(days=n_days))}, open(rd / "manifest.json", "w"))
    return {"runs": str(runs), "rd": rd, "names": names}


def test_commands_append_ledger_with_purpose_and_range(run):
    c = ctx_of(run)
    cmds.cmd_alarms(c, "dev", "count")
    cmds.cmd_join_check(c, "dev")
    cmds.cmd_report(c, "dev", "dev", detector=FakeDetector())
    recs = pr.read_ledger(run["runs"])
    assert [r["command"] for r in recs] == ["alarms --count", "join-check", "report"]
    assert {r["purpose"] for r in recs} == {"dev"} and all(pd.Timestamp(r["end"]) <= c.periods["J0"] for r in recs)
    assert recs[0]["S"].startswith("2026-07-10") and recs[0]["run"] == "r1"


def test_refused_analysis_is_not_recorded(run):
    c = ctx_of(run)
    with pytest.raises(pr.PeriodError):
        cmds.cmd_alarms(c, "judge", "summary")
    assert pr.read_ledger(run["runs"]) is None


def test_lock_refused_when_ledger_has_record_overlapping_judge_range(run):
    c = ctx_of(run)
    cmds.cmd_join_check(c, "dev")
    pr.append_ledger(run["runs"], "alarms --summary", "explore", c.periods["J0"] - pd.Timedelta(days=1), c.periods["J0"] + pd.Timedelta(days=2),
                     {"S": "2026-07-01", "D": "2026-08-01"}, "old_snapshot")             # 다른 스냅샷의 explore 기록
    with pytest.raises(pr.PeriodError, match="장부"):
        cmds.cmd_lock(c, model_infos=json.loads(json.dumps(GOOD)))
    assert pr.load_lock(run["runs"], c.periods) is None


def test_lock_records_ledger_missing_when_no_ledger(run, capsys):
    c = ctx_of(run)
    cmds.cmd_join_check(c, "dev")
    import os
    os.remove(pr.ledger_path(run["runs"]))
    cmds.cmd_lock(c, model_infos=json.loads(json.dumps(GOOD)))
    assert pr.load_lock(run["runs"], c.periods)["ledger_missing"] is True and "장부가 없습니다" in capsys.readouterr().err


def test_judge_run_is_recorded_and_blocks_a_second_lock_over_same_range(run):
    c = locked(run)
    cmds.cmd_report(c, "judge", "judge", detector=FakeDetector())
    recs = pr.read_ledger(run["runs"])
    assert recs[-1]["purpose"] == "judge" and pd.Timestamp(recs[-1]["start"]) == c.periods["J0"]
    assert pr.ledger_conflicts(run["runs"], c.periods)                                       # 같은 범위에 D 만 늘린 새 잠금 시도도 막힌다


def test_explore_only_snapshot_allows_only_explore_and_blocks_later_judge_overlap(tmp_path):
    from tests.validation.realdata.test_snapshot_commands import run as _  # noqa: F401
    names = tmp_path / "names.json"
    from tests.validation.realdata.test_alarm_episodes import names as mk
    n = mk()
    json.dump({"code_offset": 367, "sys_id": {str(k): v for k, v in n.sys_id.items()}, "tables": n.tables, "models_without_table": []}, open(names, "w"))
    r = small_run(tmp_path, 20, str(names))
    c = cmds.Context(str(r["rd"]), r["runs"], names_path=r["names"])
    assert c.periods["explore_only"]
    with pytest.raises(pr.PeriodError, match="explore"):
        cmds.cmd_alarms(c, "dev", "count")
    with pytest.raises(pr.PeriodError):
        cmds.cmd_replay(c, "2026-07-12..2026-07-20", "dev", detector=FakeDetector())
    cmds.cmd_replay(c, "2026-07-12..2026-07-20", "explore", detector=FakeDetector())
    rec = pr.read_ledger(r["runs"])[-1]
    assert rec["purpose"] == "explore" and rec["start"].startswith("2026-07-12")
    # 데이터가 늘어난 새 추출(같은 runs 폴더의 장부 공유): 07-12~07-20 은 J0(08-07) 이전이라 충돌 없음, 07-30~ 범위는 충돌
    big = small_run(tmp_path / "b", 50, str(names))
    c2 = cmds.Context(str(big["rd"]), r["runs"], names_path=big["names"])
    assert pr.ledger_conflicts(r["runs"], c2.periods) == []
    cmds.cmd_replay(c, "2026-08-01..2026-08-12", "explore", detector=FakeDetector())
    assert len(pr.ledger_conflicts(r["runs"], c2.periods)) == 1
    with pytest.raises(pr.PeriodError, match="explore"):
        cmds.cmd_replay(c2, "dev", "explore", detector=FakeDetector())                       # 기간을 정할 수 있으면 explore 불가
