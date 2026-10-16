"""U7: 현장 검토 샘플·채점(Wilson)·C5 라이브 점검 (설계서 9.8·9.9)"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from tests.validation.realdata.fakes import FakeServer, make_db
from tests.validation.realdata.test_replay_report import FakeDetector
from tests.validation.realdata.test_snapshot_commands import run  # noqa: F401  (fixture)
from validation.cli import real_data_check as cli
from validation.realdata import commands as cmds
from validation.realdata import live as lv
from validation.realdata import periods as pr
from validation.realdata import review as rv

T = pd.Timestamp("2026-09-07 00:00")      # 월요일


# ---- Wilson ---------------------------------------------------------------------------------
def test_wilson_known_values():
    lo, hi = rv.wilson(60, 100)
    assert lo == pytest.approx(0.5020, abs=5e-4) and hi == pytest.approx(0.6905, abs=5e-4)       # report_package: n=100, 60 → 하한 약 0.502
    assert rv.wilson(0, 10)[0] == 0.0 and rv.wilson(10, 10)[1] == pytest.approx(1.0)
    assert np.isnan(rv.wilson(0, 0)[0])
    assert rv.wilson(59, 100)[0] < 0.5 <= rv.wilson(60, 100)[0]


# ---- 할당·선택 ------------------------------------------------------------------------------
def test_allocate_proportional_with_minimum_one_and_total_equal_to_quota():
    a = rv._allocate({1: 90, 2: 8, 3: 2}, 20)
    assert sum(a.values()) == 20 and a[2] >= 1 and a[3] >= 1 and a[1] > a[2]
    assert rv._allocate({1: 3, 2: 2}, 10) == {1: 3, 2: 2}                                           # 부족하면 전수
    assert sum(rv._allocate({1: 50, 2: 50, 3: 50, 4: 50}, 3).values()) == 3


def incs(n, port_of, level_of=lambda i: 1):
    return pd.DataFrame({"incident": range(n), "ip_addr": "a", "cid": 1, "lid": [port_of(i) for i in range(n)],
                         "t_first": [T + pd.Timedelta(hours=i) for i in range(n)], "t_last": [T + pd.Timedelta(hours=i, minutes=15) for i in range(n)],
                         "level_max": [level_of(i) for i in range(n)]})


def test_pick_respects_port_cap_quota_and_reproducibility():
    inc = incs(40, lambda i: i % 5)                                                # 포트 5개 × 8건
    sel = rv.pick_incidents(inc, 30, np.random.RandomState(1))
    assert len(sel) == 15 and sel.groupby("lid").size().max() == rv.MAX_PER_PORT         # 포트당 최대 3건 → 5×3=15 (할당량 못 채움)
    again = rv.pick_incidents(inc, 30, np.random.RandomState(1))
    assert list(sel.index) == list(again.index) and list(sel.index) != list(rv.pick_incidents(inc, 30, np.random.RandomState(2)).index)
    small = incs(4, lambda i: i)
    assert len(rv.pick_incidents(small, 30, np.random.RandomState(0))) == 4                  # 전수 검토
    assert rv.pick_incidents(inc.iloc[0:0], 5, np.random.RandomState(0)).empty


def test_pick_is_stratified_by_level():
    inc = incs(60, lambda i: i, level_of=lambda i: 3 if i < 6 else 1)             # 등급3 6건, 등급1 54건
    sel = rv.pick_incidents(inc, 10, np.random.RandomState(3))
    assert len(sel) == 10 and (sel["level_max"] == 3).sum() >= 1 and (sel["level_max"] == 1).sum() >= 5


# ---- 샘플 구성 ------------------------------------------------------------------------------
def method_frame(times_by_port, level=1, with_level=True):
    rows = [{"ip_addr": "a", "cid": 1, "lid": lid, "occur_date": t, "alarm": True, "severity": 60.0, **({"alarm_level": level} if with_level else {})}
            for lid, ts in times_by_port.items() for t in ts]
    return pd.DataFrame(rows)


def frames_basic():
    ai = method_frame({1: [T + pd.Timedelta(hours=1)], 2: [T + pd.Timedelta(hours=1)], 3: [T + pd.Timedelta(days=8, hours=1)]})
    fx = method_frame({4: [T + pd.Timedelta(hours=2)]}, with_level=False)
    rl = method_frame({1: [T + pd.Timedelta(hours=1), T + pd.Timedelta(hours=1, minutes=15)]}, with_level=False)   # AI(포트1) 와 시간이 겹침
    return {"ai": ai, "fixed": fx.drop(columns="severity"), "rolling": rl.drop(columns="severity")}


def test_build_sample_requires_seed_and_merges_overlapping_incidents_across_methods():
    with pytest.raises(ValueError, match="시드"):
        rv.build_sample(frames_basic(), "dev", None)
    review, key, meta = rv.build_sample(frames_basic(), "dev", 5)
    assert len(review) == 4 and meta["seed"] == 5 and meta["per_method_incidents"] == {"ai": 3, "fixed": 1, "rolling": 1}
    shared = key[key["methods"] == "ai;rolling"]
    assert len(shared) == 1                                                                   # 포트1 의 AI·롤링 인시던트 → 한 행, 라벨 공유
    assert set(key["methods"]) == {"ai", "fixed", "ai;rolling"}


def test_review_csv_is_blind_and_labels_are_not_prefilled():
    review, key, _ = rv.build_sample(frames_basic(), "dev", 5)
    assert not ({"method", "methods", "severity", "severity_max", "level_max", "n_alarms", "incident"} & set(review.columns))
    assert (review["분류"] == "").all() and (review["근거 메모"] == "").all()
    assert review["blind_id"].is_unique and review["blind_id"].is_monotonic_increasing and set(key["blind_id"]) == set(review["blind_id"])
    assert review["blind_id"].str.match(r"^R-\d{4}$").all()
    r2, k2, _ = rv.build_sample(frames_basic(), "dev", 5)
    pd.testing.assert_frame_equal(review, r2)
    pd.testing.assert_frame_equal(key, k2)                                                    # 같은 시드 → 같은 샘플
    r3, k3, _ = rv.build_sample(frames_basic(), "dev", 6)
    assert not key.equals(k3)                                                                 # 블라인드 ID 배정은 시드에 의존


def test_build_sample_weekly_quota_and_reference_columns():
    ai = method_frame({lid: [T + pd.Timedelta(days=d, hours=1) for d in (0, 8)] for lid in range(1, 8)})            # 2주 × 7포트
    fr = {"ai": ai, "fixed": ai.iloc[0:0].drop(columns="severity"), "rolling": ai.iloc[0:0].drop(columns="severity")}
    review, key, meta = rv.build_sample(fr, "dev", 1, quotas={"ai": 3, "fixed": 1, "rolling": 1},
                                        ems_lookup=lambda *a: "ETH-ERR", work_lookup=lambda *a: True)
    assert meta["weeks"]["2026-09-07"]["ai"] == {"available": 7, "sampled": 3} and meta["weeks"]["2026-09-14"]["ai"]["sampled"] == 3
    assert len(review) == 6 and (review["ems_alarms_24h"] == "ETH-ERR").all() and review["work_suspect"].all()
    assert rv.QUOTAS["dev"] == {"ai": 30, "fixed": 10, "rolling": 10} and rv.QUOTAS["judge"] == {"ai": 50, "fixed": 15, "rolling": 10}


def test_build_sample_empty_input():
    e = pd.DataFrame(columns=["ip_addr", "cid", "lid", "occur_date", "alarm"])
    r, k, meta = rv.build_sample({"ai": e, "fixed": e, "rolling": e}, "dev", 1)
    assert r.empty and k.empty


# ---- 채점 -----------------------------------------------------------------------------------
def review_and_key(labels, methods):
    n = len(labels)
    review = pd.DataFrame({"blind_id": [f"R-{i:04d}" for i in range(n)], "분류": labels})
    key = pd.DataFrame({"blind_id": review["blind_id"], "methods": methods})
    return review, key


C1 = {"ai": {"incidents": 1000, "port_days": 5000, "incidents_per_port_day": 0.2}, "fixed": {"incidents": 100, "port_days": 5000, "incidents_per_port_day": 0.02},
      "rolling": {"incidents": 300, "port_days": 5000, "incidents_per_port_day": 0.06}}


def test_score_excludes_unjudgeable_and_shares_labels_across_methods():
    labels = ["유효-장애"] * 50 + ["유효-예방"] * 20 + ["오탐"] * 20 + ["정상 변동"] * 10 + ["판단 불가"] * 30
    methods = ["ai"] * 40 + ["ai;fixed"] * 10 + ["ai"] * 80                                       # 라벨이 유효인 앞쪽 10행을 고정 임계치와 공유
    s = rv.score_review(*review_and_key(labels, methods), C1)
    ai, fx = s["methods"]["ai"], s["methods"]["fixed"]
    assert (ai["reviewed"], ai["unjudgeable"], ai["judgeable"], ai["valid"]) == (130, 30, 100, 70) and ai["valid_rate"] == 0.7
    assert ai["unjudgeable_warning"] is False                                                    # 30/130 = 23% ≤ 30%
    assert fx["reviewed"] == 10 and fx["judgeable"] == 10                                         # 공유 라벨은 고정 임계치 샘플에도 반영
    assert s["c2"] == {**s["c2"], "hold": False, "pass": True, "point": 0.7}
    assert s["c1"]["estimated_false_per_port_day"] == pytest.approx(0.2 * 0.3) and s["c1"]["pass"] is False      # 0.06 > 0.02


def test_score_holds_when_fewer_than_100_judgeable_and_flags_unjudgeable_ratio():
    s = rv.score_review(*review_and_key(["유효-장애"] * 60 + ["판단 불가"] * 50, ["ai"] * 110), C1)
    assert s["c2"]["hold"] and s["c2"]["pass"] is None and s["c1"]["hold"] and s["c1"]["pass"] is None and s["c4a"]["status"] == "hold"
    assert s["methods"]["ai"]["unjudgeable_warning"] is True and s["unreviewed"] == 0


@pytest.mark.parametrize("ai_inc,fx_inc,fx_valid,fx_n,status", [
    (1000, 100, 2, 20, "충족"),                       # AI 유효 탐지량 하한 502 > 고정 상한 (100 × Wilson 상한)
    (100, 1000, 9, 10, "미충족"),                     # AI 상한 69 < 고정 하한 596
    (1000, 1000, 7, 10, "동률(유의한 차이 없음)")])   # 구간 겹침
def test_c4a_interval_comparison(ai_inc, fx_inc, fx_valid, fx_n, status):
    ai_labels = ["유효-장애"] * 60 + ["오탐"] * 40
    fx_labels = ["유효-예방"] * fx_valid + ["오탐"] * (fx_n - fx_valid)
    review, key = review_and_key(ai_labels + fx_labels, ["ai"] * 100 + ["fixed"] * fx_n)
    c1 = {**C1, "ai": {**C1["ai"], "incidents": ai_inc}, "fixed": {**C1["fixed"], "incidents": fx_inc}}
    assert rv.score_review(review, key, c1)["c4a"]["status"] == status


def test_score_validates_labels_and_keys():
    review, key = review_and_key(["오탐", "엉뚱한 값", ""], ["ai"] * 3)
    with pytest.raises(ValueError, match="허용되지 않은 분류"):
        rv.score_review(review, key)
    review, key = review_and_key(["오탐", ""], ["ai"] * 2)
    assert rv.score_review(review, key)["unreviewed"] == 1
    with pytest.raises(ValueError, match="키 파일"):
        rv.score_review(review.assign(blind_id=["X", "Y"]), key)


# ---- C5 -------------------------------------------------------------------------------------
def stored_rows(times, lag_min=5, ip="1.1.1.1", cid=1, lid=1, start_id=1, anomaly=0):
    return [{"id": start_id + i, "occur_date": t, "ip_addr": ip, "cid": cid, "lid": lid, "is_anomaly": anomaly,
             "detect_time": t + pd.Timedelta(minutes=lag_min)} for i, t in enumerate(times)]


def test_fetch_results_keyset_chunks_range_and_schema_validation():
    ts = [T + pd.Timedelta(minutes=15 * i) for i in range(10)]
    srv = FakeServer(tables={"anomaly_detection": {"cols": ["id"], "rows": stored_rows(ts)}})
    df = lv.fetch_results(make_db(srv), ts[2], ts[8], chunk=2)
    assert list(df["occur_date"]) == ts[2:8]
    sel = [p for q, p in srv.log if "anomaly_detection" in q and q.startswith("SELECT `id`")]
    assert [p[-2] for p in sel] == [0, 4, 6, 8]                                                  # id 키셋 전진 (3,4 → 5,6 → 7,8 → 빈 청크)
    assert all("OFFSET" not in q.upper() for q, _ in srv.log)
    with pytest.raises(ValueError):
        lv.result_table("a`b")
    assert lv.result_table("ems2") == "`ems2`.`anomaly_detection`"
    srv2 = FakeServer(tables={"anomaly_detection": {"cols": ["id"], "rows": stored_rows(ts)}})
    assert len(lv.fetch_results(make_db(srv2), ts[0], ts[-1] + pd.Timedelta(minutes=15), schema="ems")) == 10


def target_frame(times):
    return pd.DataFrame({"ip_addr": "1.1.1.1", "cid": 1, "lid": 1, "occur_date": times, "alarm": False})


def test_c5_rates_latency_and_gaps():
    ts = [T + pd.Timedelta(minutes=15 * i) for i in range(1000)]
    stored_ts = [t for i, t in enumerate(ts) if i not in range(300, 310)]                         # 10 슬롯 누락 → 2.5시간 공백
    stored = pd.DataFrame(stored_rows(stored_ts))
    r = lv.compute_c5(stored, target_frame(ts))
    assert r["target_rows"] == 1000 and r["matched_rows"] == 990 and r["c5a"] == pytest.approx(0.99) and r["c5a_pass"] is False
    assert r["c5b"]["within_30min"] == 1.0 and r["c5b"]["pass"] is True and r["c5b"]["p50_min"] == 5
    assert len(r["c5c_gaps"]) == 1 and r["c5c_gaps"][0]["hours"] == pytest.approx(2.75)
    full = lv.compute_c5(pd.DataFrame(stored_rows(ts)), target_frame(ts))
    assert full["c5a"] == 1.0 and full["c5a_pass"] and full["c5c_gaps"] == [] and full["alarm_agreement_ref"] == 1.0


def test_c5_latency_threshold_and_interval_limit_and_empty():
    ts = [T + pd.Timedelta(minutes=15 * i) for i in range(200)]
    slow = pd.DataFrame(stored_rows(ts[:100], lag_min=31) + stored_rows(ts[100:], lag_min=5, start_id=1000))
    r = lv.compute_c5(slow, target_frame(ts))
    assert r["c5b"]["within_30min"] == 0.5 and r["c5b"]["pass"] is False
    part = lv.compute_c5(pd.DataFrame(stored_rows(ts[50:150])), target_frame(ts))
    assert part["target_rows"] == 100 and part["c5a"] == 1.0                                       # 라이브 가동 구간만
    assert lv.compute_c5(pd.DataFrame(columns=["id", "occur_date", "ip_addr", "cid", "lid", "is_anomaly", "detect_time"]), target_frame(ts))["hold"]


# ---- CLI 통합 -------------------------------------------------------------------------------
class AlarmingDetector(FakeDetector):
    """매일 03시대 lid=1 포트에 알람을 내는 가짜 탐지기 (샘플링 입력용)."""

    def detect(self, df_traffic, df_optical, latest_only):
        r = super().detect(df_traffic, df_optical, latest_only)
        hit = (r["occur_date"].dt.hour == 3) & (r["lid"] == 1)
        return r.assign(is_anomaly=hit, alarm_level=np.where(hit, 2, 0), severity=np.where(hit, 60.0, 0.0))


def prep_replay(run):
    c = cmds.Context(str(run["rd"]), run["runs"], names_path=run["names"])
    cmds.cmd_join_check(c, "dev")
    cmds.cmd_report(c, "dev", "dev", detector=AlarmingDetector())
    return c


def base_args(run, *extra):
    return [*extra, "--run", "r1", "--runs-dir", run["runs"], "--alarm-names", run["names"]]


def test_sample_cli_end_to_end_and_no_overwrite_and_judge_refused(run, capsys):
    prep_replay(run)
    assert cli.main(["sample", *base_args(run, "--range", "dev", "--purpose", "dev", "--seed", "3")]) in (0,)
    out = run["rd"] / "out"
    assert any(f.startswith("sample_review_dev_dev") for f in os.listdir(out)) and any(f.startswith("sample_key_dev_dev") for f in os.listdir(out))
    meta = json.load(open(next(out.glob("sample_meta_dev_dev*.json")), encoding="utf-8"))
    assert meta["seed"] == 3 and meta["purpose"] == "dev"
    assert cli.main(["sample", *base_args(run, "--range", "dev", "--purpose", "dev", "--seed", "3")]) == 5          # 라벨 보호: 덮어쓰기 거부
    assert cli.main(["sample", *base_args(run, "--range", "judge", "--purpose", "judge", "--seed", "3")]) == 5      # 잠금 전
    with pytest.raises(SystemExit) as e:                                                                              # 시드 필수 (argparse)
        cli.main(["sample", *base_args(run, "--range", "dev", "--purpose", "dev")])
    assert e.value.code == 2


def test_empty_sample_has_headers_and_rows_zero(run):
    c = cmds.Context(str(run["rd"]), run["runs"], names_path=run["names"])
    cmds.cmd_report(c, "dev", "dev", detector=FakeDetector())                         # 알람 없는 탐지기
    meta = cmds.cmd_sample(c, "dev", "dev", 1)
    assert meta["rows"] == 0
    df = pd.read_csv(next((run["rd"] / "out").glob("sample_review_dev_dev*.csv")), encoding="utf-8-sig")
    assert list(df.columns) == rv.REVIEW_COLUMNS and df.empty


def test_sample_requires_replay_first(run):
    c = cmds.Context(str(run["rd"]), run["runs"], names_path=run["names"])
    with pytest.raises(pr.PeriodError, match="재생 결과가 없습니다"):
        cmds.cmd_sample(c, "dev", "dev", 1)


def test_score_review_cli_reads_filled_csv(run, capsys):
    c = prep_replay(run)
    cmds.cmd_sample(c, "dev", "dev", 1)
    out = run["rd"] / "out"
    rp = next(out.glob("sample_review_dev_dev*.csv"))
    df = pd.read_csv(rp, encoding="utf-8-sig")
    df["분류"] = ["유효-장애"] * len(df)
    df.to_csv(rp, index=False, encoding="utf-8-sig")
    assert len(df) > 0
    assert cli.main(["score-review", "--review", str(rp), "--key", str(next(out.glob("sample_key_dev_dev*.csv"))),
                     "--report", str(next(out.glob("report_dev_dev*.json")))]) == 0
    assert "C2 보류=True" in capsys.readouterr().out and any(f.startswith("score_review_") for f in os.listdir(out))


def test_live_cli_path_with_fake_server(run):
    c = prep_replay(run)
    frames = cmds.load_replay_frames(c, "dev", "dev")
    ai = frames["ai"]
    srv = FakeServer(tables={"anomaly_detection": {"cols": ["id"], "rows": [
        {"id": i + 1, "occur_date": r.occur_date, "ip_addr": r.ip_addr, "cid": r.cid, "lid": r.lid, "is_anomaly": 0,
         "detect_time": r.occur_date + pd.Timedelta(minutes=3)} for i, r in enumerate(ai.itertuples())]}})
    res = cmds.cmd_live(c, make_db(srv), "dev", "dev")
    assert res["c5a"] == 1.0 and res["c5b"]["pass"] is True and os.path.exists(run["rd"] / "out" / "live_c5_dev_dev.json")
    assert all(q.split()[0].upper() in ("SELECT", "SHOW", "SET") for q in srv.statements())              # 읽기 전용 경로만
