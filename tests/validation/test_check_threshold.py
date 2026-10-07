"""validation/cli/check_threshold.py 테스트 (P1-5 U1, T-P5-U1a~g, 설계서 8.1)

소형 데이터(2노드 x 3일, 1에폭)로 train -> oracle -> e0 를 끝까지 돌리고, 판정 함수는 손으로 만든 입력과
하드코딩한 경계값으로 고정한다. 활성 모델(규칙 A 입력)은 conftest 의 tiny_model_dir 를 쓴다.
"""
import inspect
import json
import os
import types

import numpy as np
import pandas as pd
import pytest

from src.data import train_window
from src.pipeline.retrain_policy import RetrainPolicy
from validation.cli import check_promotion as cp
from validation.cli import check_threshold as ct

TRACKS = ("traffic", "optical")
# 소형 데이터(시드 5, 2노드 x 3일, 20포트)에서 홀드아웃(10%)이 서로 다르고 비어 있지 않은 salt 쌍: ptn 3포트, p15c 1포트 (서로소)
SALT_A, SALT_B = "ptn", "p15c"


def _targs(out, tiny, **kw):
    base = dict(out=str(out), data_seed=5, train_seed=0, split_salt="ptn", split="port", exclusion="aub", nodes=2, days=3,
                epochs=1, active_models=str(tiny))
    return types.SimpleNamespace(**{**base, **kw})


@pytest.fixture(scope="module")
def runs(tmp_path_factory, tiny_model_dir):
    root = tmp_path_factory.mktemp("p15")
    specs = {"r0": {}, "salt": {"split_salt": SALT_B}, "time": {"split": "time"}, "truth": {"exclusion": "truth"}}
    out = {}
    for name, kw in specs.items():
        out[name] = root / name
        ct.cmd_train(_targs(out[name], tiny_model_dir, **kw))
    ct.cmd_oracle(types.SimpleNamespace(runs=",".join(str(p) for p in out.values()), seed=211, nodes=2, days=3))
    return out


def _ports(path):
    df = pd.read_csv(path, usecols=["ip_addr", "cid", "lid"]).drop_duplicates()
    return {tuple(r) for r in df.itertuples(index=False)}


# ── T-P5-U1a: train -> oracle -> e0 끝까지, 출력 형식 ──
def test_run_json_has_the_fixed_schema(runs):
    run = json.load(open(runs["r0"] / "run.json", encoding="utf-8"))
    assert set(run) == {"data_seed", "train_seed", "split", "split_salt", "exclusion", "nodes", "days", "epochs", "code",
                        "started_at", "elapsed_sec", "tracks"}
    assert (run["data_seed"], run["train_seed"], run["split"], run["split_salt"], run["exclusion"]) == (5, 0, "port", "ptn", "aub")
    assert (run["nodes"], run["days"], run["epochs"]) == (2, 3, 1) and run["elapsed_sec"] > 0
    for ft in TRACKS:
        assert set(run["tracks"][ft]) == {"threshold", "final_val_loss", "samples_used", "val_ports", "train_ports",
                                          "threshold_recomputed", "recompute_rel_err"}
        assert run["tracks"][ft]["val_ports"] > 0 and run["tracks"][ft]["train_ports"] > 0
    assert os.path.exists(runs["r0"] / "train_data" / "traffic_train.csv") and os.path.exists(runs["r0"] / "traffic_registry.json")


def test_oracle_json_has_the_fixed_schema(runs):
    orc = json.load(open(runs["r0"] / "oracle.json", encoding="utf-8"))
    assert set(orc) == {"seed", "tracks"} and orc["seed"] == 211
    for ft in TRACKS:
        t = orc["tracks"][ft]
        assert set(t) == {"th_star", "th_in", "bias", "n_scores"} and t["n_scores"] > 0
        assert t["bias"] == pytest.approx(t["th_in"] / t["th_star"])


def test_e0_runs_end_to_end_and_writes_fixed_csv_columns_and_json_keys(runs, tmp_path, capsys):
    a = types.SimpleNamespace(runs=",".join(str(p) for p in runs.values()), seeds="7", nodes=2, days=4,
                              out_csv=str(tmp_path / "e0.csv"), out_json=str(tmp_path / "e0.json"))
    ct.cmd_e0(a)
    df = pd.read_csv(a.out_csv)
    assert list(df.columns) == ["run", "data_seed", "train_seed", "split", "split_salt", "exclusion", "seed", "policy", "th_kind",
                                "th_traffic", "th_optical", "auprc", "f1", "early", "fp", "prec", "lead"]
    real = df[~df.run.str.startswith("[기준]")]
    assert set(real.policy) == {"기본", "오탐억제형"} and set(real.th_kind) == {"own", "oracle"} and set(real.run) == set(runs)
    assert len(real) == 4 * 2 * 2                                              # 실행 4 x 정책 2 x 임계치 종류 2 (시드 1개)
    assert {"[기준] rolling_rule", "[기준] fixed_threshold", "[기준] always_alarm"} == set(df[df.run.str.startswith("[기준]")].run)
    own, orc = real[real.th_kind == "own"], real[real.th_kind == "oracle"]
    run0 = json.load(open(runs["r0"] / "run.json"))
    orc0 = json.load(open(runs["r0"] / "oracle.json"))
    r0_own = own[(own.run == "r0") & (own.policy == "오탐억제형")].iloc[0]
    r0_orc = orc[(orc.run == "r0") & (orc.policy == "오탐억제형")].iloc[0]
    assert r0_own.th_traffic == pytest.approx(run0["tracks"]["traffic"]["threshold"])      # own = 메타 임계치
    assert r0_orc.th_traffic == pytest.approx(orc0["tracks"]["traffic"]["th_star"])        # oracle = th*
    out = json.load(open(a.out_json, encoding="utf-8"))
    assert set(out) == {"seeds", "runs", "R0", "QA", "QB", "QC", "decision", "criteria"}
    assert out["seeds"] == [7] and out["R0"] == ["r0"] and out["decision"] in {"T1T3T2", "T2", "stop"}
    assert set(out["QA"]) == set(TRACKS) and set(out["QA"]["traffic"]) == {"max_min", "cv", "met"}
    assert set(out["QB"]) == {"sd_own", "sd_oracle", "explained", "fp_oracle_mean", "met"}
    assert set(out["QC"]) == {"salt", "split", "exclusion"} and set(out["QC"]["salt"]) == {"pairs", "mean_diff", "same_sign", "met"}
    assert all(out["QC"][f]["pairs"] == 1 for f in out["QC"])                  # r0 와 요인 하나만 다른 실행이 각각 1개
    assert out["criteria"] == ct.CRITERIA
    assert "[결정]" in capsys.readouterr().out


# ── T-P5-U1b: 재계산 임계치 = 메타 임계치 (#31) ──
def test_recomputed_threshold_equals_meta_threshold(runs):
    for name, path in runs.items():
        run = json.load(open(path / "run.json"))
        for ft in TRACKS:
            t = run["tracks"][ft]
            assert t["recompute_rel_err"] <= 1e-6, (name, ft)
            assert t["threshold_recomputed"] == pytest.approx(t["threshold"], rel=1e-6)


# ── T-P5-U1c: salt 별 홀드아웃 / 시간 분할 ──
def test_same_salt_same_holdout_ports_and_different_salt_differs(runs):
    from validation.simulator.scenario_generator import ScenarioConfig, generate
    d = generate(ScenarioConfig(seed=5, nodes=2, days=3))
    allp = {tuple(r) for r in d["traffic"][train_window.PORT_KEYS].drop_duplicates().itertuples(index=False)}
    expect = {s: {p for p in allp if train_window.is_val_port(*p, 0.10, s)} for s in (SALT_A, SALT_B)}
    assert expect[SALT_A] and expect[SALT_B] and expect[SALT_A] != expect[SALT_B]          # 테스트에 고정한 salt 쌍의 전제
    for ft in TRACKS:
        a = _ports(runs["r0"] / "train_data" / f"{ft}_test.csv")
        b = _ports(runs["salt"] / "train_data" / f"{ft}_test.csv")
        assert a == expect[SALT_A] and b == expect[SALT_B] and a != b
        assert not (a & _ports(runs["r0"] / "train_data" / f"{ft}_train.csv"))             # 홀드아웃 포트는 학습에 없음


def test_time_split_validation_csv_has_only_the_last_20_percent_rows(runs):
    run = json.load(open(runs["time"] / "run.json"))
    assert run["split"] == "time" and run["split_salt"] is None                           # time 에서는 salt 무시, null 기록
    cut = pd.Timestamp(cp.ScenarioConfig(seed=5).start) + pd.Timedelta(days=3 * 0.8)
    for ft in TRACKS:
        tr = pd.read_csv(runs["time"] / "train_data" / f"{ft}_train.csv", parse_dates=["occur_date"])
        va = pd.read_csv(runs["time"] / "train_data" / f"{ft}_test.csv", parse_dates=["occur_date"])
        assert len(tr) and len(va) and tr["occur_date"].max() < cut <= va["occur_date"].min()
        assert _ports(runs["time"] / "train_data" / f"{ft}_test.csv") == _ports(runs["time"] / "train_data" / f"{ft}_train.csv")   # 전 포트
        assert run["tracks"][ft]["val_ports"] == run["tracks"][ft]["train_ports"]


def test_truth_exclusion_removes_more_rows_than_nothing_and_skips_suspect_stats(runs):
    run_t = json.load(open(runs["truth"] / "run.json"))
    assert run_t["exclusion"] == "truth"
    meta = json.load(open(runs["truth"] / "traffic_ae_v1.json"))
    assert meta["suspect_stats"] is None                                                   # 규칙·자기 알람 없이 정답 구간만 제외


# ── T-P5-U1d: 시드 가드 ──
@pytest.mark.parametrize("seed", [7, 11, 53, 59, 61, 211, 23, 31, 47, 102, 900, 33, 34, 35])
def test_train_rejects_evaluation_validation_oracle_and_forbidden_data_seeds(tmp_path, tiny_model_dir, seed):
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match="학습에 쓸 수 없습니다"):
        ct.cmd_train(_targs(out, tiny_model_dir, data_seed=seed))
    assert not out.exists()                                                                # 거부는 측정(학습) 전, 폴더도 안 만듦


@pytest.mark.parametrize("seeds", ["23", "7,23", "53", "7,11,53", "5"])
def test_e0_rejects_seeds_outside_7_11(tmp_path, runs, seeds):
    a = types.SimpleNamespace(runs=str(runs["r0"]), seeds=seeds, nodes=2, days=3, out_csv=str(tmp_path / "x.csv"), out_json=str(tmp_path / "x.json"))
    with pytest.raises(SystemExit, match="부분집합"):
        ct.cmd_e0(a)
    assert not (tmp_path / "x.csv").exists()


@pytest.mark.parametrize("seed", [7, 5, 53, 212, 0])
def test_oracle_rejects_any_seed_except_211(runs, seed):
    with pytest.raises(SystemExit, match="211"):
        ct.cmd_oracle(types.SimpleNamespace(runs=str(runs["r0"]), seed=seed, nodes=2, days=3))


def test_seed_constants_are_the_fixed_values():
    assert ct.DEV_SEEDS == (7, 11) and ct.P15_VAL_SEEDS == (53, 59, 61) and ct.TRAIN_DATA_SEEDS == (101, 103, 105)
    assert ct.ORACLE_SEED == 211 and ct.FORBIDDEN_SEEDS == (23, 31, 47, 102, 900, 33, 34, 35)
    assert not set(ct.TRAIN_DATA_SEEDS) & set(ct.NOT_FOR_TRAINING)                         # 학습 데이터 시드는 가드에 걸리지 않음
    assert {7, 11, 53, 59, 61, 211, 23, 31, 47, 102, 900, 33, 34, 35} == set(ct.NOT_FOR_TRAINING)


# ── T-P5-U1e: 덮어쓰기 거부 (측정 전) ──
def test_train_refuses_existing_out_folder_and_leaves_it_untouched(tmp_path, tiny_model_dir):
    out = tmp_path / "exists"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(SystemExit, match="이미 있습니다"):
        ct.cmd_train(_targs(out, tiny_model_dir))
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_oracle_refuses_existing_oracle_json_before_measuring_and_writes_nothing_else(tmp_path, runs, monkeypatch):
    monkeypatch.setattr(cp, "make_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("측정이 실행되면 안 됨")))
    before = (runs["r0"] / "oracle.json").read_text()
    with pytest.raises(SystemExit, match="이미 있습니다"):
        ct.cmd_oracle(types.SimpleNamespace(runs=str(runs["r0"]), seed=211, nodes=2, days=3))
    assert (runs["r0"] / "oracle.json").read_text() == before


def test_oracle_refuses_folder_without_run_json(tmp_path):
    (tmp_path / "plain").mkdir()
    with pytest.raises(SystemExit, match="run.json"):
        ct.cmd_oracle(types.SimpleNamespace(runs=str(tmp_path / "plain"), seed=211, nodes=2, days=3))


def test_e0_refuses_existing_outputs_missing_folders_and_missing_oracle_before_measuring(tmp_path, runs, monkeypatch):
    monkeypatch.setattr(cp, "make_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("측정이 실행되면 안 됨")))
    base = dict(runs=str(runs["r0"]), seeds="7", nodes=2, days=3)
    (tmp_path / "old.csv").write_text("ORIGINAL")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        ct.cmd_e0(types.SimpleNamespace(**base, out_csv=str(tmp_path / "old.csv"), out_json=str(tmp_path / "n.json")))
    assert (tmp_path / "old.csv").read_text() == "ORIGINAL"
    (tmp_path / "old.json").write_text("ORIGINAL")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        ct.cmd_e0(types.SimpleNamespace(**base, out_csv=str(tmp_path / "n.csv"), out_json=str(tmp_path / "old.json")))
    with pytest.raises(SystemExit, match="폴더"):
        ct.cmd_e0(types.SimpleNamespace(**base, out_csv=str(tmp_path / "nodir" / "n.csv"), out_json=str(tmp_path / "n.json")))
    with pytest.raises(SystemExit, match="서로 다른"):
        ct.cmd_e0(types.SimpleNamespace(**base, out_csv=str(tmp_path / "same.x"), out_json=str(tmp_path / "same.x")))
    (tmp_path / "noorc").mkdir()
    (tmp_path / "noorc" / "run.json").write_text("{}")
    with pytest.raises(SystemExit, match="oracle.json"):
        ct.cmd_e0(types.SimpleNamespace(**{**base, "runs": str(tmp_path / "noorc")}, out_csv=str(tmp_path / "n.csv"), out_json=str(tmp_path / "n.json")))
    assert not (tmp_path / "n.csv").exists() and not (tmp_path / "n.json").exists()


# ── T-P5-U1f: 판정 함수 경계값과 결정표 ──
def test_criteria_are_the_pre_registered_values():
    assert (ct.QA_MAX_MIN, ct.QB_EXPLAINED, ct.QB_FP_ORACLE, ct.QC_MIN_DIFF, ct.QC_SAME_SIGN) == (1.25, 0.50, 0.02, 0.008, 2 / 3)
    assert ct.RECOMPUTE_TOL == 1e-6


@pytest.mark.parametrize("vals,met", [([1.0, 1.25], True), ([1.0, 1.2499], False), ([2.0, 1.0, 1.5], True)])
def test_qa_boundary(vals, met):
    assert ct.judge_qa({"traffic": vals, "optical": [1.0, 1.0]})["traffic"]["met"] is met
    assert ct.judge_qa({"traffic": [1.0, 1.0], "optical": vals})["optical"]["met"] is met      # 어느 한 트랙이라도


def test_qa_reports_max_min_and_cv():
    q = ct.judge_qa({"traffic": [1.0, 1.5], "optical": [2.0, 2.0]})
    assert q["traffic"]["max_min"] == pytest.approx(1.5) and q["traffic"]["cv"] == pytest.approx(0.25 / 1.25)
    assert q["optical"]["max_min"] == 1.0 and q["optical"]["cv"] == 0.0


def test_qb_explained_share_boundary_with_binary_exact_values():
    own = [0.0, 2 ** -5]                                                      # SD 2^-6
    at = ct.judge_qb(own, [2 ** -7, 2 ** -7 + 2 ** -6])                       # SD 2^-7: 설명 몫 정확히 50%, th* 오탐 평균 0.0156 <= 0.02
    assert at["explained"] == 0.5 and at["fp_oracle_mean"] <= 0.02 and at["met"] is True
    below = ct.judge_qb(own, [2 ** -7, 2 ** -7 + 2 ** -6 + 2 ** -10])         # 설명 몫 46.9% < 50%
    assert below["explained"] == pytest.approx(0.46875) and below["met"] is False


def test_qb_fp_mean_boundary_and_zero_variance():
    assert ct.judge_qb([0.0, 0.04], [0.02, 0.02])["met"] is True              # 설명 몫 100%, 평균 정확히 0.02 -> 충족 (이하)
    assert ct.judge_qb([0.0, 0.04], [0.0201, 0.0201])["met"] is False         # 분산은 사라졌지만 평균 0.0201 > 0.02
    assert ct.judge_qb([0.0, 0.04], [0.0, 0.04])["met"] is False              # 설명 몫 0%
    assert ct.judge_qb([0.03, 0.03], [0.0, 0.0])["explained"] == 0.0          # 원래 분산이 0 이면 설명 몫 0 으로 처리
    q = ct.judge_qb([0.0, 0.04], [0.01, 0.03])
    assert q["sd_own"] == pytest.approx(0.02) and q["sd_oracle"] == pytest.approx(0.01) and q["fp_oracle_mean"] == pytest.approx(0.02)


@pytest.mark.parametrize("diffs,met", [
    ([0.008, 0.008], True),                       # |평균| 정확히 0.008, 모두 같은 부호
    ([0.0079, 0.0079], False),
    ([0.02, 0.01, -0.003], True),                 # 평균 0.009, 같은 부호 2/3 정확히
    ([0.02, -0.001, -0.002], False),              # 평균 0.0057 (< 0.008)
    ([0.05, -0.001, -0.002, -0.001], False),      # 평균 0.0115 이지만 같은 부호 1/4 < 2/3
    ([-0.02, -0.02], True),                       # 음의 방향도 효과
    ([], False),
])
def test_qc_boundary(diffs, met):
    assert ct.judge_qc(diffs)["met"] is met


def test_qc_reports_pairs_mean_and_same_sign_count():
    q = ct.judge_qc([0.02, 0.01, -0.003])
    assert q["pairs"] == 3 and q["same_sign"] == 2 and q["mean_diff"] == pytest.approx(0.009)
    assert ct.judge_qc([]) == {"pairs": 0, "mean_diff": None, "same_sign": 0, "met": False}


def test_factor_of_identifies_single_factor_runs_and_ignores_combinations():
    mk = lambda split="port", salt="ptn", excl="aub": {"split": split, "split_salt": salt, "exclusion": excl}
    assert ct.factor_of(mk()) == "R0"
    assert ct.factor_of(mk(salt="p15a")) == "salt"
    assert ct.factor_of(mk(split="time", salt=None)) == "split"
    assert ct.factor_of(mk(excl="truth")) == "exclusion"
    assert ct.factor_of(mk(split="time", salt=None, excl="truth")) is None               # 조합은 무시
    assert ct.factor_of(mk(salt="p15a", excl="truth")) is None


def _row(name, d, t, own, oracle=0.019, th=(1.0, 1.0), split="port", salt="ptn", excl="aub"):
    return {"run": name, "data_seed": d, "train_seed": t, "split": split, "split_salt": salt, "exclusion": excl,
            "th": {"traffic": th[0], "optical": th[1]}, "fp_own": own, "fp_oracle": oracle}


def test_decision_table_all_three_paths():
    r0 = [_row("a", 101, 0, 0.01), _row("b", 103, 0, 0.05)]
    # Q-B 충족(SD 가 th* 대입으로 사라지고 평균 0.019) -> T1T3T2 (Q-C 와 무관)
    assert ct.judge_e0(r0)["decision"] == "T1T3T2"
    # Q-B 미충족(oracle 오탐이 own 과 같음) + salt 효과 충족 -> T2
    r0n = [_row("a", 101, 0, 0.01, oracle=0.01), _row("b", 103, 0, 0.05, oracle=0.05)]
    salt = [_row("a_s", 101, 0, 0.03, oracle=0.03, salt="p15a"), _row("b_s", 103, 0, 0.07, oracle=0.07, salt="p15a")]   # 짝 차이 +0.02, +0.02
    j = ct.judge_e0(r0n + salt)
    assert j["QB"]["met"] is False and j["QC"]["salt"]["met"] is True and j["decision"] == "T2"
    # 분할 효과만 충족 -> T2
    tsplit = [_row("a_t", 101, 0, 0.03, oracle=0.03, split="time", salt=None), _row("b_t", 103, 0, 0.07, oracle=0.07, split="time", salt=None)]
    assert ct.judge_e0(r0n + tsplit)["decision"] == "T2"
    # 정제 효과만 충족 -> 대안 대상 아님 -> stop (보고만)
    truth = [_row("a_x", 101, 0, 0.03, oracle=0.03, excl="truth"), _row("b_x", 103, 0, 0.07, oracle=0.07, excl="truth")]
    j = ct.judge_e0(r0n + truth)
    assert j["QC"]["exclusion"]["met"] is True and j["decision"] == "stop"
    assert ct.judge_e0(r0n)["decision"] == "stop"                                       # 아무 요인도 없음


def test_decision_pairs_are_matched_by_data_and_train_seed():
    r0 = [_row("a", 101, 0, 0.01, oracle=0.01), _row("b", 103, 0, 0.05, oracle=0.05)]
    v = [_row("a_s", 101, 1, 0.5, salt="p15a")]                                         # 짝(101, 1) 이 R0 에 없음 -> 무시
    assert ct.judge_e0(r0 + v)["QC"]["salt"]["pairs"] == 0


def test_e0_decision_wiring_in_decide():
    met, unmet = {"met": True}, {"met": False}
    assert ct.decide(met, {"salt": unmet, "split": unmet, "exclusion": met}) == "T1T3T2"
    assert ct.decide(unmet, {"salt": met, "split": unmet, "exclusion": unmet}) == "T2"
    assert ct.decide(unmet, {"salt": unmet, "split": met, "exclusion": unmet}) == "T2"
    assert ct.decide(unmet, {"salt": unmet, "split": unmet, "exclusion": met}) == "stop"


# ── T-P5-U1g: train_operational 기본 인자 = 현행 ──
def test_train_operational_defaults_keep_current_recipe():
    sig = inspect.signature(cp.train_operational).parameters
    assert sig["split_salt"].default is None and sig["split"].default == "port" and sig["exclusion"].default == "aub"
    assert RetrainPolicy().split_salt == "ptn" and RetrainPolicy().val_port_fraction == 0.10        # salt None -> 정책값(ptn)
    assert list(sig)[:5] == ["out_dir", "data_seed", "train_seed", "nodes", "days"]                  # 기존 위치 인자 순서 불변
    with pytest.raises(ValueError):
        cp.train_operational("x", 5, 0, split="bogus")
    with pytest.raises(ValueError):
        cp.train_operational("x", 5, 0, exclusion="bogus")


def test_default_run_uses_policy_salt_ptn_and_cli_defaults_are_r0(runs, monkeypatch):
    run = json.load(open(runs["r0"] / "run.json"))
    assert run["split_salt"] == "ptn"                                                                # 실제로 돌린 기본 실행의 salt = 운영 기본
    parsed = {}
    monkeypatch.setattr(ct, "cmd_train", lambda a: parsed.update(vars(a)))
    monkeypatch.setattr("sys.argv", ["check_threshold.py", "train", "--out", "o"])
    ct.main()
    assert parsed["split_salt"] == "ptn" and parsed["split"] == "port" and parsed["exclusion"] == "aub"
    assert (parsed["nodes"], parsed["days"], parsed["epochs"], parsed["data_seed"], parsed["train_seed"]) == (4, 21, 30, 101, 0)
    assert parsed["active_models"] == "models"
