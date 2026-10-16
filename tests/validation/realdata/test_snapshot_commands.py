"""U5: 스냅샷 기반 CLI (alarms --summary/--count-only, join-check) — 합성 스냅샷으로 엔드투엔드, 판정 구간 봉인 거부"""
import json
import os

import pandas as pd
import pytest

from tests.validation.realdata.test_alarm_episodes import A, names
from validation.cli import real_data_check as cli
from validation.realdata import periods as pr

D0 = pd.Timestamp("2026-07-10")


@pytest.fixture
def run(tmp_path):
    """50 데이터일(1일 1행 이상) 합성 스냅샷: 한 장비 ip 1.1.1.1 포트 (1,1),(1,2). 알람은 알람 키 (cid=2,lid=1) = 성능 (1,1) + 1 어긋난 구성."""
    runs = tmp_path / "runs"
    rd = runs / "r1"
    (rd / "raw" / "eth").mkdir(parents=True)
    (rd / "raw" / "alarm").mkdir(parents=True)
    for i in range(50):
        day = D0 + pd.Timedelta(days=i)
        rows = []
        for k in range(1, 97):
            t = day + pd.Timedelta(minutes=15 * k)
            for c, l in ((1, 1), (1, 2)):
                rows.append({"ip_addr": "1.1.1.1", "cid": c, "lid": l, "system_type": 31, "address": f"X-S.{c} -P{l}", "tx_packet": 5,
                             "rx_packet": 100, "error_packet": 0, "occur_date": t, "bin_end": t, "src_table": "t", "id": k})
        pd.DataFrame(rows).to_csv(rd / "raw" / "eth" / f"{day:%Y-%m-%d}.csv.gz", index=False, compression="gzip")
    j0 = D0 + pd.Timedelta(days=28)
    al = []
    for i, base in enumerate((D0 + pd.Timedelta(days=3, hours=10), j0 + pd.Timedelta(days=3, hours=10))):          # 개발 1건, 판정 1건
        al.append({"id": i + 1, "ip_addr": "1.1.1.1 ", "cid": 2, "lid": 1, "system_type": 31, "alarm_type": A["ETH-LINK-FAIL"], "severity_type": 3,
                   "occur_date": base, "release_date": base + pd.Timedelta(hours=2), "address": "S.1 -P1", "src_table": "a"})
    pd.DataFrame(al).to_csv(rd / "raw" / "alarm" / "all.csv.gz", index=False, compression="gzip")
    json.dump({"complete": True, "range_start": str(D0), "range_end": str(D0 + pd.Timedelta(days=50))}, open(rd / "manifest.json", "w"))
    nm = tmp_path / "names.json"
    n = names()
    json.dump({"code_offset": 367, "sys_id": {str(k): v for k, v in n.sys_id.items()}, "tables": n.tables, "models_without_table": []},
              open(nm, "w"))
    return {"runs": str(runs), "rd": rd, "names": str(nm), "j0": j0}


def args(run, *extra):
    return [*extra, "--run", "r1", "--runs-dir", run["runs"], "--alarm-names", run["names"]]


def test_summary_dev_works_and_writes_outputs(run, capsys):
    assert cli.main(["alarms", *args(run, "--summary")]) == 0
    out = capsys.readouterr().out
    assert "표 없는 모델 장비 0/1" in out
    files = os.listdir(run["rd"] / "out")
    assert any(f.startswith("alarms_summary_dev") for f in files) and any(f.startswith("unmapped_alarms_dev") for f in files)


def test_judge_range_is_refused_before_lock_and_writes_nothing(run, capsys):
    for extra in (["--summary", "--range", "judge"], ["--count-only", "--range", "judge"]):
        assert cli.main(["alarms", *args(run, *extra)]) == 5
    assert cli.main(["alarms", *args(run, "--summary", "--range", f"{run['j0'] - pd.Timedelta(days=2):%Y-%m-%d}..{run['j0'] + pd.Timedelta(days=1):%Y-%m-%d}")]) == 5
    assert cli.main(["join-check", *args(run, "--range", "judge")]) == 5
    assert "잠금" in capsys.readouterr().err
    assert not (run["rd"] / "out").exists() or not any("judge" in f for f in os.listdir(run["rd"] / "out"))


def test_judge_range_opens_with_lock_file(run):
    ctx_periods = pr.compute_periods(pr.data_days_from_eth(pd.concat([pd.read_csv(run["rd"] / "raw" / "eth" / f) for f in os.listdir(run["rd"] / "raw" / "eth")])), start=str(D0))
    lp = pr.lock_path(run["runs"], ctx_periods)
    os.makedirs(os.path.dirname(lp))
    json.dump({"alternative": "fixed_split_v1"}, open(lp, "w"))
    assert cli.main(["alarms", *args(run, "--count-only", "--range", "judge")]) == 0


def test_count_only_dev_counts_only_dev_events(run, capsys):
    assert cli.main(["alarms", *args(run, "--count-only")]) == 0
    res = json.load(open(next((run["rd"] / "out").glob("count_only_dev*.json")), encoding="utf-8"))
    assert res["events"] == 1 and res["data_days"] == 28 and res["judge_days_planned"] == 23 and res["c3_hold_expected"] is True
    assert res["r_per_data_day"] == pytest.approx(1 / 28) and res["r_needed_for_20"] == pytest.approx(20 / 23)


def test_join_check_dev_reports_failed_and_stops_without_auto_switch(run, capsys):
    assert cli.main(["join-check", *args(run)]) == 0
    out = capsys.readouterr().out
    res = json.load(open(next((run["rd"] / "out").glob("join_check_dev*.json")), encoding="utf-8"))
    assert res["join_status"] in ("failed", "unverified")                 # 에피소드 1건 < 5 → 판정 불가(unverified), 보고에 diagnostics 포함
    assert "diagnostic_cid_offset_scan" in res and "diagnostic_j3_alarm_cid_minus_1" in res
    scan = res["diagnostic_cid_offset_scan"]
    assert scan["-1"]["all"] == 1.0 and scan["0"]["all"] == 0.0           # 알람 cid-1 이 성능 키와 일치하는 진단 — 판정·조인 키는 바뀌지 않음
    assert res["verdict"]["address_join_would_pass"] in (True, False)
    assert "자동 전환" in res["note"]


def test_join_check_manual_mismatch_stops(run):
    man = run["rd"] / "man.csv"
    man.write_text("ip_addr,system_type,perf_cid,perf_lid,alarm_cid,alarm_lid,note\n1.1.1.1,31,1,1,2,1,x\n", encoding="utf-8")
    assert cli.main(["join-check", *args(run, "--manual", str(man))]) == 0
    res = json.load(open(next((run["rd"] / "out").glob("join_check_dev*.json")), encoding="utf-8"))
    assert res["join_status"] == "stopped" and res["manual"]["stop"] is True
