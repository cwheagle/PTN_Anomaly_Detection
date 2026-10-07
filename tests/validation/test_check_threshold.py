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
                epochs=1, active_models=str(tiny), code_rev=None, val_fraction=0.10, min_val_ports=None)
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
    assert set(run) == {"data_seed", "train_seed", "split", "split_salt", "exclusion", "nodes", "days", "epochs", "val_fraction",
                        "min_val_ports", "code", "started_at", "elapsed_sec", "tracks"}
    assert run["val_fraction"] == 0.10 and run["min_val_ports"] is None
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


# ── run.json code 필드: --code-rev > PTN_CODE_REV > unknown ──
def test_code_rev_precedence_argument_then_env_then_unknown(monkeypatch):
    monkeypatch.delenv("PTN_CODE_REV", raising=False)
    assert ct._code_rev(None) == "unknown" and ct._code_rev("") == "unknown"
    monkeypatch.setenv("PTN_CODE_REV", "envrev")
    assert ct._code_rev(None) == "envrev"
    assert ct._code_rev("argrev") == "argrev"                                   # 인자가 환경변수보다 우선


def test_train_records_code_rev_in_run_json(tmp_path, tiny_model_dir, monkeypatch):
    monkeypatch.delenv("PTN_CODE_REV", raising=False)
    run = ct.cmd_train(_targs(tmp_path / "a", tiny_model_dir, code_rev="abc1234"))
    assert run["code"] == "abc1234" and json.load(open(tmp_path / "a" / "run.json"))["code"] == "abc1234"
    monkeypatch.setenv("PTN_CODE_REV", "fromenv")
    assert ct.cmd_train(_targs(tmp_path / "b", tiny_model_dir))["code"] == "fromenv"


def test_cli_exposes_code_rev_option_default_none(monkeypatch):
    parsed = {}
    monkeypatch.setattr(ct, "cmd_train", lambda a: parsed.update(vars(a)))
    monkeypatch.setattr("sys.argv", ["check_threshold.py", "train", "--out", "o", "--code-rev", "deadbee"])
    ct.main()
    assert parsed["code_rev"] == "deadbee"
    monkeypatch.setattr("sys.argv", ["check_threshold.py", "train", "--out", "o"])
    ct.main()
    assert parsed["code_rev"] is None


# ══════════════════════════════════════════════
# U2 (T-P5-U2a~h): qd / alt T1 / T2 학습 옵션 / evalalt / select / verify
# 검증 시드 53·59·61 의 데이터는 어떤 테스트에서도 만들지 않는다 (make_data 를 가로채 시드 5 로 바꿈).
# ══════════════════════════════════════════════
import shutil
import hashlib


@pytest.fixture
def alt_env(monkeypatch):
    """소형 데이터(시드 5, 난수 0)를 2단계 R0 로 취급하도록 대상 상수만 바꾼다 (설계 값은 test_u2_constants 가 고정)"""
    monkeypatch.setattr(ct, "ALT_DATA_SEEDS", (5,))
    monkeypatch.setattr(ct, "ALT_TRAIN_SEEDS", (0,))


@pytest.fixture
def r0_copy(runs, tmp_path):
    dest = tmp_path / "r0"
    shutil.copytree(runs["r0"], dest)
    return dest


def _qd_file(tmp_path, cands=("T2", "T1")):
    f = tmp_path / "qd.json"
    f.write_text(json.dumps({"candidates": list(cands)}))
    return f


def _hashes(folder):
    return {str(p.relative_to(folder)): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}


def _no_53_59_61(monkeypatch):
    real = cp.make_data
    seen = []

    def fake(seed, nodes, days, **k):
        seen.append(seed)
        return real(5 if seed in (53, 59, 61) else seed, nodes, days, **k)
    monkeypatch.setattr(cp, "make_data", fake)
    return seen


def test_u2_constants_are_the_fixed_values():
    assert (ct.QD_BIAS, ct.QD_FP_ORACLE, ct.QD_FP_OWN) == (0.90, 0.02, 0.02)
    assert (ct.SEL_FP_MAX, ct.SEL_F1_MARGIN, ct.SEL_EARLY_MARGIN, ct.SEL_AUPRC_MARGIN, ct.SEL_TIE) == (0.02, 0.03, 0.05, 0.03, 0.003)
    assert ct.ALT_DATA_SEEDS == (101, 103, 105) and ct.ALT_TRAIN_SEEDS == (0, 1) and (ct.T2_FRACTION, ct.T2_MIN_PORTS) == (0.20, 8)
    assert ct.VERIFY_SEEDS == (53, 59, 61) and ct.EVIDENCE_LOCK == "p1_5_evidence.json"
    assert ct.CODE_CHANGE_ORDER == {"T2": 0, "T1": 1, "T3": 2}


# ── T-P5-U2a: alt T1 은 alt_T1.json 만 추가, 원본 불변 ──
def test_alt_t1_adds_only_alt_json_and_leaves_everything_else_byte_identical(r0_copy, tmp_path, alt_env):
    before = _hashes(r0_copy)
    ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), qd=str(_qd_file(tmp_path))))
    after = _hashes(r0_copy)
    assert set(after) - set(before) == {"alt_T1.json"} and not set(before) - set(after)
    assert {k: v for k, v in after.items() if k != "alt_T1.json"} == before                # 모델·스케일러·메타·레지스트리·run/oracle.json 불변
    alt = json.load(open(r0_copy / "alt_T1.json", encoding="utf-8"))
    assert alt["kind"] == "T1" and set(alt["tracks"]) == set(TRACKS)
    for ft in TRACKS:
        t = alt["tracks"][ft]
        assert set(t) == {"th_t1", "th_t1_infer", "infer_rel_diff", "n_seq", "val_ports", "th_star", "bias_t1", "bias_in"}
        assert t["bias_t1"] == pytest.approx(t["th_t1"] / t["th_star"]) and t["n_seq"] > 0 and t["val_ports"] > 0
    with pytest.raises(SystemExit, match="이미 있습니다"):                                   # 기존 alt_T1.json 이면 거부
        ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), qd=str(_qd_file(tmp_path))))
    assert _hashes(r0_copy) == after


# ── T-P5-U2e: th_t1 = Trainer 경로의 홀드아웃 백분위 ──
def test_th_t1_equals_trainer_percentile_on_holdout_and_same_function_reproduces_train_threshold(r0_copy, tmp_path, alt_env):
    from src.models.trainer import Trainer
    ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), qd=str(_qd_file(tmp_path))))
    alt = json.load(open(r0_copy / "alt_T1.json", encoding="utf-8"))
    run = json.load(open(r0_copy / "run.json"))
    for ft in TRACKS:
        entry = ct.active_entry(str(r0_copy), ft)
        meta = json.load(open(r0_copy / entry["config_path"]))
        tr = Trainer(ft, config_override=meta["config"])                                      # Trainer 자신의 _save_metadata 로 직접 계산
        tr.processor.load_scaler(str(r0_copy / entry["scaler_path"]))
        import torch
        state = torch.load(r0_copy / entry["model_path"], map_location=tr.device, weights_only=True)
        tr.model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
        clean = tr.processor.preprocess(pd.read_csv(r0_copy / "train_data" / f"{ft}_test.csv"), is_train=True)
        seqs = tr.processor.create_sequences(clean, is_train=True, fit_scaler=False)
        tr.paths = {"model": str(tmp_path / f"{ft}_x.pth"), "scaler": str(tmp_path / f"{ft}_x.joblib")}
        direct = tr._save_metadata(seqs)["threshold"]
        assert alt["tracks"][ft]["th_t1"] == pytest.approx(direct, rel=1e-6)
        assert alt["tracks"][ft]["n_seq"] == len(seqs)
        # 같은 함수를 학습 CSV 에 쓰면 run.json 의 threshold_recomputed 와 같다
        th_train, _, _ = ct.threshold_from_csv(str(r0_copy), ft, str(r0_copy / "train_data" / f"{ft}_train.csv"))
        assert th_train == pytest.approx(run["tracks"][ft]["threshold_recomputed"], rel=1e-6)


# ── T-P5-U2f: alt·evalalt 대상 가드 ──
def test_alt_and_evalalt_reject_non_r0_runs_and_t3(runs, r0_copy, tmp_path, alt_env, monkeypatch):
    qd = _qd_file(tmp_path)
    for name in ("salt", "time", "truth"):                                                   # salt·split·exclusion 이 R0 와 다른 실행
        with pytest.raises(SystemExit, match="대상 실행이 아닙니다"):
            ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(runs[name]), qd=str(qd)))
    monkeypatch.setattr(ct, "ALT_TRAIN_SEEDS", (1,))                                         # 난수가 다른 실행
    with pytest.raises(SystemExit, match="대상 실행이 아닙니다"):
        ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), qd=str(qd)))
    monkeypatch.setattr(ct, "ALT_TRAIN_SEEDS", (0,))
    monkeypatch.setattr(ct, "ALT_DATA_SEEDS", (101, 103, 105))                               # 데이터 시드가 다른 실행
    with pytest.raises(SystemExit, match="대상 실행이 아닙니다"):
        ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), qd=str(qd)))
    assert not (r0_copy / "alt_T1.json").exists()
    monkeypatch.setattr(ct, "ALT_DATA_SEEDS", (5,))
    for kind in ("T3", "T2", "x"):
        with pytest.raises(SystemExit, match="T1 만 허용"):
            ct.cmd_alt(types.SimpleNamespace(kind=kind, runs=str(r0_copy), qd=str(qd)))
    ev = lambda kind, run: types.SimpleNamespace(kind=kind, runs=str(run), seeds="7", nodes=2, days=3, out_csv=str(tmp_path / "ea.csv"))
    with pytest.raises(SystemExit, match="T1 또는 T2"):
        ct.cmd_evalalt(ev("T3", r0_copy))
    with pytest.raises(SystemExit, match="대상 실행이 아닙니다"):
        ct.cmd_evalalt(ev("T2", r0_copy))                                                    # 비율 0.10 실행은 T2 대상이 아님
    with pytest.raises(SystemExit, match="대상 실행이 아닙니다"):
        ct.cmd_evalalt(ev("T1", runs["salt"]))
    assert not (tmp_path / "ea.csv").exists()


def test_evalalt_rejects_bad_seeds_missing_alt_json_and_existing_csv(r0_copy, tmp_path, alt_env):
    base = dict(kind="T1", runs=str(r0_copy), nodes=2, days=3)
    for seeds in ("53", "7,23", "5"):
        with pytest.raises(SystemExit, match="부분집합"):
            ct.cmd_evalalt(types.SimpleNamespace(**base, seeds=seeds, out_csv=str(tmp_path / "e.csv")))
    with pytest.raises(SystemExit, match="alt_T1.json"):                                     # alt 를 안 돌린 실행
        ct.cmd_evalalt(types.SimpleNamespace(**base, seeds="7", out_csv=str(tmp_path / "e.csv")))
    (tmp_path / "old.csv").write_text("ORIGINAL")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        ct.cmd_evalalt(types.SimpleNamespace(**base, seeds="7", out_csv=str(tmp_path / "old.csv")))
    assert (tmp_path / "old.csv").read_text() == "ORIGINAL"


def test_e0_rejects_runs_with_other_holdout_fraction(runs, tmp_path):
    odd = tmp_path / "odd"
    shutil.copytree(runs["r0"], odd)
    run = json.load(open(odd / "run.json"))
    run["val_fraction"] = 0.20
    json.dump(run, open(odd / "run.json", "w"))
    a = types.SimpleNamespace(runs=str(odd), seeds="7", nodes=2, days=3, out_csv=str(tmp_path / "x.csv"), out_json=str(tmp_path / "x.json"))
    with pytest.raises(SystemExit, match="0.10"):
        ct.cmd_e0(a)


# ── T-P5-U2g: T2 학습 인자 ──
def test_holdout_port_count_matches_split_ports_and_default_is_unchanged(runs):
    from validation.simulator.scenario_generator import ScenarioConfig, generate
    d = generate(ScenarioConfig(seed=5, nodes=2, days=3))
    for frac in (0.10, 0.20, 0.5):
        _, val = train_window.split_ports(d["traffic"], frac, "ptn")
        n, total = ct.holdout_port_count(5, 2, 3, frac, "ptn")
        assert n == len(val[train_window.PORT_KEYS].drop_duplicates()) and total == 20
    assert ct.holdout_port_count(5, 2, 3, 0.10, "ptn")[0] == json.load(open(runs["r0"] / "run.json"))["tracks"]["traffic"]["val_ports"]   # 기본 = 8.1 분할


def test_train_min_val_ports_rejects_before_training_without_creating_folder(tmp_path, tiny_model_dir):
    n20, _ = ct.holdout_port_count(5, 2, 3, 0.20, "ptn")
    out = tmp_path / "f20"
    with pytest.raises(SystemExit, match="구조적 SKIP"):
        ct.cmd_train(_targs(out, tiny_model_dir, val_fraction=0.20, min_val_ports=n20 + 1))
    assert not out.exists()                                                                  # 학습 전 거부, 폴더 미생성
    with pytest.raises(SystemExit, match="0 과 1"):
        ct.cmd_train(_targs(out, tiny_model_dir, val_fraction=1.5))
    assert not out.exists()


def test_train_with_val_fraction_uses_that_holdout_and_records_it(tmp_path, tiny_model_dir):
    n, _ = ct.holdout_port_count(5, 2, 3, 0.20, "ptn")
    run = ct.cmd_train(_targs(tmp_path / "f20", tiny_model_dir, val_fraction=0.20, min_val_ports=n))        # 경계: 정확히 최소 포트 수면 학습
    assert run["val_fraction"] == 0.20 and run["min_val_ports"] == n
    for ft in TRACKS:
        assert run["tracks"][ft]["val_ports"] == n
        assert len(_ports(tmp_path / "f20" / "train_data" / f"{ft}_test.csv")) == n


# ── T-P5-U2h: qd ──
@pytest.mark.parametrize("bias,d1", [
    ({"traffic": [0.95, 0.91, 0.99], "optical": [0.95, 0.95, 0.95]}, False),      # 중앙값 0.95/0.95 -> 미충족
    ({"traffic": [0.90], "optical": [0.99]}, False),                              # 중앙값이 정확히 0.90 -> '미만'이 아님
    ({"traffic": [0.8999], "optical": [0.99]}, True),
    ({"traffic": [0.99, 0.99], "optical": [0.5, 0.99, 0.1]}, True),               # 어느 한 트랙이라도 (optical 중앙값 0.5)
    ({"traffic": [0.5, 0.99, 0.99], "optical": [0.99, 0.99]}, False)])            # 중앙값이라 이상치 하나로는 충족되지 않음
def test_qd_d1_boundary(bias, d1):
    assert ct.judge_qd(bias, 0.01, 0.03, "T2")["QD"]["D1"] is d1


def test_qd_d2_d3_boundaries_and_met_requires_all():
    ok = {"traffic": [0.8], "optical": [0.4]}
    assert ct.judge_qd(ok, 0.02, 0.0201, "T2")["QD"]["met"] is True                                         # D-2 는 이하, D-3 은 초과
    assert ct.judge_qd(ok, 0.0201, 0.05, "T2")["QD"]["D2"] is False
    assert ct.judge_qd(ok, 0.01, 0.02, "T2")["QD"]["D3"] is False                                           # own 오탐 평균 정확히 0.02 -> 초과 아님
    assert ct.judge_qd({"traffic": [0.9], "optical": [0.95]}, 0.01, 0.05, "T2")["QD"]["met"] is False


@pytest.mark.parametrize("decision,qd_met,expected", [
    ("T2", True, ["T2", "T1"]), ("T2", False, ["T2"]), ("T1T3T2", True, ["T1", "T3", "T2"]), ("T1T3T2", False, ["T1", "T3", "T2"]),
    ("stop", True, ["T1"]), ("stop", False, [])])
def test_qd_candidates_combination(decision, qd_met, expected):
    bias = {"traffic": [0.8], "optical": [0.4]} if qd_met else {"traffic": [0.99], "optical": [0.99]}
    assert ct.judge_qd(bias, 0.01, 0.03, decision)["candidates"] == expected
    with pytest.raises(SystemExit):
        ct.judge_qd(bias, 0.01, 0.03, "bogus")


def _e0_fixture(tmp_path, names, fp_own, fp_orc, qb_mean=None, bias=(0.8, 0.4)):
    rows = []
    for i, n in enumerate(names):
        for kind, fp in (("own", fp_own[i]), ("oracle", fp_orc[i])):
            for seed in (7, 11):
                rows.append({"run": n, "data_seed": 101, "train_seed": i, "split": "port", "split_salt": "ptn", "exclusion": "aub", "seed": seed,
                             "policy": ct.FP_POLICY, "th_kind": kind, "th_traffic": 0.5, "th_optical": 0.2, "auprc": 0.6, "f1": 0.8,
                             "early": 0.7, "fp": fp, "prec": 0.9, "lead": 100})
        d = tmp_path / n
        d.mkdir()
        json.dump({"seed": 211, "tracks": {ft: {"th_star": 1.0, "th_in": b, "bias": b, "n_scores": 10} for ft, b in zip(TRACKS, bias)}}, open(d / "oracle.json", "w"))
    pd.DataFrame(rows).to_csv(tmp_path / "e0.csv", index=False)
    json.dump({"R0": list(names), "decision": "T2", "QB": {"fp_oracle_mean": float(np.mean(fp_orc)) if qb_mean is None else qb_mean}}, open(tmp_path / "e0.json", "w"))
    return types.SimpleNamespace(e0_json=str(tmp_path / "e0.json"), e0_csv=str(tmp_path / "e0.csv"), runs=",".join(str(tmp_path / n) for n in names),
                                 out_json=str(tmp_path / "qd.json"))


def test_qd_command_computes_from_files_and_expects_t2_then_t1(tmp_path):
    a = _e0_fixture(tmp_path, ["a", "b", "c"], fp_own=[0.03, 0.04, 0.05], fp_orc=[0.01, 0.01, 0.01])
    out = ct.cmd_qd(a)
    assert out["candidates"] == ["T2", "T1"] and out["QD"]["met"] is True
    assert out["QD"]["fp_own_mean"] == pytest.approx(0.04) and out["QD"]["fp_oracle_mean"] == pytest.approx(0.01)
    assert out["QD"]["bias_median"] == {"traffic": 0.8, "optical": 0.4} and out["criteria"] == ct.QD_CRITERIA and out["e0_decision"] == "T2"
    assert json.load(open(a.out_json, encoding="utf-8")) == json.loads(json.dumps(out))


def test_qd_command_rejects_mismatched_r0_set_and_oracle_mean_and_existing_output(tmp_path):
    a = _e0_fixture(tmp_path, ["a", "b"], fp_own=[0.03, 0.04], fp_orc=[0.01, 0.01])
    (tmp_path / "z").mkdir()
    bad = types.SimpleNamespace(**{**vars(a), "runs": f"{tmp_path / 'a'},{tmp_path / 'z'}"})
    with pytest.raises(SystemExit, match="R0 집합"):
        ct.cmd_qd(bad)
    mism = tmp_path / "m"
    mism.mkdir()
    a2 = _e0_fixture(mism, ["a", "b"], fp_own=[0.03, 0.04], fp_orc=[0.01, 0.01], qb_mean=0.5)
    with pytest.raises(SystemExit, match="QB.fp_oracle_mean"):
        ct.cmd_qd(a2)
    (tmp_path / "qd.json").write_text("ORIGINAL")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        ct.cmd_qd(a)
    assert (tmp_path / "qd.json").read_text() == "ORIGINAL"


# ── T-P5-U2b: select 판정 ──
R0S = {"fp": 0.0364, "f1": 0.77, "early": 0.70, "auprc": 0.62, "th_max_min": {"traffic": 1.3, "optical": 1.9}, "runs": ["x"]}


def _alt(**kw):
    base = {"fp": 0.01, "f1": 0.75, "early": 0.66, "auprc": 0.60, "th_max_min": {"traffic": 1.3, "optical": 1.9},
            "bias": {"t1": {"traffic": 0.1, "optical": 0.2}, "in": {"traffic": 0.3, "optical": 0.4}}}
    return {**base, **kw}


@pytest.mark.parametrize("field,value,rule,ok", [
    ("fp", 0.02, "1", True), ("fp", 0.0201, "1", False),
    ("f1", 0.74, "2", True), ("f1", 0.7399, "2", False),                  # R0 0.77 - 0.03
    ("early", 0.65, "3", True), ("early", 0.6499, "3", False),            # R0 0.70 - 0.05
    ("auprc", 0.59, "4", True), ("auprc", 0.5899, "4", False),            # R0 0.62 - 0.03
    ("th_max_min", {"traffic": 1.3, "optical": 1.9}, "5", True),          # R0 와 같으면 충족
    ("th_max_min", {"traffic": 1.3001, "optical": 1.9}, "5", False), ("th_max_min", {"traffic": 1.0, "optical": 1.9001}, "5", False)])
def test_select_rules_1_to_5_boundaries(field, value, rule, ok):
    assert ct.judge_alt(R0S, _alt(**{field: value}), "T2")[rule] is ok


def test_select_rule_6_applies_to_t1_only_and_needs_both_tracks_strictly_smaller():
    assert ct.judge_alt(R0S, _alt(), "T1")["6"] is True
    same = _alt(bias={"t1": {"traffic": 0.3, "optical": 0.2}, "in": {"traffic": 0.3, "optical": 0.4}})       # 같으면 충족 아님
    assert ct.judge_alt(R0S, same, "T1")["6"] is False
    one = _alt(bias={"t1": {"traffic": 0.1, "optical": 0.5}, "in": {"traffic": 0.3, "optical": 0.4}})       # 한 트랙만 개선
    assert ct.judge_alt(R0S, one, "T1")["6"] is False
    assert "6" not in ct.judge_alt(R0S, one, "T2")                                                          # T2 에는 규칙 6 이 없다


def test_select_tie_rule_prefers_smaller_code_change_within_0_003_and_fails_with_no_candidates():
    cands = {"T1": {"fp": 0.010}, "T2": {"fp": 0.012}}
    assert ct.pick_candidate(cands) == "T2"                              # 차이 0.002 <= 0.003 -> T2 < T1
    assert ct.pick_candidate({"T1": {"fp": 0.010}, "T2": {"fp": 0.0131}}) == "T1"      # 차이 0.0031 -> 오탐 최소
    assert ct.pick_candidate({"T1": {"fp": 0.010}, "T2": {"fp": 0.013}}) == "T2"       # 정확히 0.003 은 이내
    assert ct.pick_candidate({}) is None


def test_judge_select_outputs_selected_fail_skip_and_exploratory_flag():
    t1, t2 = _alt(), _alt(bias=None)
    out = ct.judge_select(R0S, {"T1": t1, "T2": t2}, ["T2", "T1"])
    assert out["selected"] == "T2" and out["result"] == "PASS" and out["exploratory"] is True and out["criteria"] == ct.SEL_CRITERIA
    assert out["alts"]["T1"]["candidate"] and set(out["alts"]["T1"]["rules"]) == {"1", "2", "3", "4", "5", "6"}
    assert set(out["alts"]["T2"]["rules"]) == {"1", "2", "3", "4", "5"}
    bad = ct.judge_select(R0S, {"T1": _alt(early=0.5), "T2": _alt(bias=None, fp=0.03)}, ["T2", "T1"])      # 규칙 3·1 위반 -> 후보 0개
    assert bad["selected"] is None and bad["result"] == "FAIL" and bad["alts"]["T1"]["rules"]["3"] is False
    skip = ct.judge_select(R0S, {"T1": t1}, ["T2", "T1"], t2_skip=True)
    assert skip["alts"]["T2"] == {"skip": "structural"} and skip["selected"] == "T1"
    with pytest.raises(SystemExit, match="evalalt 결과"):
        ct.judge_select(R0S, {"T1": t1}, ["T2", "T1"])                                                       # T2 결과도 skip 도 없으면 거부


def test_select_command_end_to_end_on_tiny_runs(runs, r0_copy, tmp_path, alt_env):
    qd = _qd_file(tmp_path, ("T2", "T1"))
    e0_csv, e0_json = tmp_path / "e0.csv", tmp_path / "e0.json"
    ct.cmd_e0(types.SimpleNamespace(runs=",".join(str(p) for p in runs.values()), seeds="7", nodes=2, days=4, out_csv=str(e0_csv), out_json=str(e0_json)))
    ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), qd=str(qd)))
    ea = tmp_path / "t1.csv"
    df = ct.cmd_evalalt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), seeds="7", nodes=2, days=4, out_csv=str(ea)))
    assert list(df.columns) == ct.CSV_COLUMNS and set(df.th_kind) == {"T1"} and not df.run.str.startswith("[기준]").any()    # 베이스라인 행 없음
    alt = json.load(open(r0_copy / "alt_T1.json"))
    assert df[df.policy == ct.FP_POLICY].iloc[0].th_traffic == pytest.approx(alt["tracks"]["traffic"]["th_t1"])
    a = types.SimpleNamespace(e0_csv=str(e0_csv), qd=str(qd), alt_csv=[str(ea)], t1_runs=str(r0_copy), t2_skip=True, out_json=str(tmp_path / "sel.json"))
    out = ct.cmd_select(a)
    assert set(out) == {"R0", "alts", "selected", "result", "criteria", "exploratory"} and out["exploratory"] is True
    assert out["alts"]["T2"] == {"skip": "structural"} and set(out["alts"]["T1"]["rules"]) == {"1", "2", "3", "4", "5", "6"}
    assert out["result"] in ("PASS", "FAIL") and json.load(open(a.out_json))["selected"] == out["selected"]
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        ct.cmd_select(a)
    with pytest.raises(SystemExit, match="t1-runs"):
        ct.cmd_select(types.SimpleNamespace(**{**vars(a), "t1_runs": None, "out_json": str(tmp_path / "s2.json")}))
    with pytest.raises(SystemExit, match="evalalt 결과"):                                   # T2 후보인데 결과도 skip 도 없음
        ct.cmd_select(types.SimpleNamespace(**{**vars(a), "t2_skip": False, "out_json": str(tmp_path / "s3.json")}))


def test_select_rejects_t3_candidate_and_missing_files(tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"candidates": ["T1", "T3", "T2"]}))
    (tmp_path / "e.csv").write_text("x")
    with pytest.raises(SystemExit, match="T3"):
        ct.cmd_select(types.SimpleNamespace(e0_csv=str(tmp_path / "e.csv"), qd=str(q), alt_csv=[str(tmp_path / "e.csv")], t1_runs=None,
                                            t2_skip=False, out_json=str(tmp_path / "o.json")))
    with pytest.raises(SystemExit, match="가 없습니다"):
        ct.cmd_select(types.SimpleNamespace(e0_csv=str(tmp_path / "nope.csv"), qd=str(q), alt_csv=[], t1_runs=None, t2_skip=False,
                                            out_json=str(tmp_path / "o.json")))


# ── T-P5-U2c / U2d: verify 거부와 검증 시드 ──
def _sel(tmp_path, result="PASS", selected="T1"):
    f = tmp_path / "sel.json"
    f.write_text(json.dumps({"result": result, "selected": selected}))
    return f


def _vargs(tmp_path, runs, sel, **kw):
    base = dict(select=str(sel), runs=str(runs), seeds="53,59,61", nodes=2, days=3,
                record_csv=str(tmp_path / "res" / "v.csv"), record_json=str(tmp_path / "res" / "v.json"))
    return types.SimpleNamespace(**{**base, **kw})


def test_verify_rejects_fail_or_missing_selection_before_lock_and_measurement(r0_copy, tmp_path, alt_env, monkeypatch):
    seen = _no_53_59_61(monkeypatch)
    monkeypatch.setattr(cp, "make_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("측정이 실행되면 안 됨")))
    (tmp_path / "res").mkdir()
    for sel in (_sel(tmp_path, "FAIL", None), _sel(tmp_path, "FAIL", "T1"), _sel(tmp_path, "PASS", None)):
        with pytest.raises(SystemExit, match="PASS 가 아니거나"):
            ct.cmd_verify(_vargs(tmp_path, r0_copy, sel))
    assert not (tmp_path / "res" / ct.EVIDENCE_LOCK).exists() and seen == []
    with pytest.raises(SystemExit, match="select 파일"):
        ct.cmd_verify(_vargs(tmp_path, r0_copy, tmp_path / "none.json"))


@pytest.mark.parametrize("seeds", ["53", "53,59", "7,11", "53,59,61,7", "23,31,47", "5"])
def test_verify_requires_exactly_53_59_61(r0_copy, tmp_path, alt_env, monkeypatch, seeds):
    monkeypatch.setattr(cp, "make_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("측정이 실행되면 안 됨")))
    (tmp_path / "res").mkdir()
    with pytest.raises(SystemExit, match="전체만"):
        ct.cmd_verify(_vargs(tmp_path, r0_copy, _sel(tmp_path), seeds=seeds))
    assert not (tmp_path / "res" / ct.EVIDENCE_LOCK).exists()


def test_verify_rejects_existing_lock_records_and_missing_folder_before_measuring(r0_copy, tmp_path, alt_env, monkeypatch):
    ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), qd=str(_qd_file(tmp_path))))
    monkeypatch.setattr(cp, "make_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("측정이 실행되면 안 됨")))
    res = tmp_path / "res"
    res.mkdir()
    sel = _sel(tmp_path)
    (res / ct.EVIDENCE_LOCK).write_text("{}")                                                 # 이미 연 근거 실행
    with pytest.raises(SystemExit, match="1회만"):
        ct.cmd_verify(_vargs(tmp_path, r0_copy, sel))
    (res / ct.EVIDENCE_LOCK).unlink()
    (res / "v.csv").write_text("ORIGINAL")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        ct.cmd_verify(_vargs(tmp_path, r0_copy, sel))
    assert (res / "v.csv").read_text() == "ORIGINAL" and not (res / ct.EVIDENCE_LOCK).exists()
    (res / "v.csv").unlink()
    with pytest.raises(SystemExit, match="폴더"):
        ct.cmd_verify(_vargs(tmp_path, r0_copy, sel, record_csv=str(tmp_path / "nodir" / "v.csv"), record_json=str(tmp_path / "nodir" / "v.json")))
    with pytest.raises(SystemExit, match="잠금 파일과 같은 경로"):
        ct.cmd_verify(_vargs(tmp_path, r0_copy, sel, record_json=str(res / ct.EVIDENCE_LOCK)))
    with pytest.raises(SystemExit, match="서로 다른"):
        ct.cmd_verify(_vargs(tmp_path, r0_copy, sel, record_csv=str(res / "same"), record_json=str(res / "same")))


def test_verify_requires_alt_json_for_t1_and_rejects_non_r0_runs(runs, r0_copy, tmp_path, alt_env, monkeypatch):
    monkeypatch.setattr(cp, "make_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("측정이 실행되면 안 됨")))
    (tmp_path / "res").mkdir()
    with pytest.raises(SystemExit, match="alt_T1.json"):
        ct.cmd_verify(_vargs(tmp_path, r0_copy, _sel(tmp_path)))
    with pytest.raises(SystemExit, match="대상 실행이 아닙니다"):
        ct.cmd_verify(_vargs(tmp_path, runs["salt"], _sel(tmp_path)))
    assert not (tmp_path / "res" / ct.EVIDENCE_LOCK).exists()


def test_verify_t1_runs_once_with_synthetic_data_writes_lock_and_refuses_second_run(r0_copy, tmp_path, alt_env, monkeypatch):
    ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), qd=str(_qd_file(tmp_path))))
    seen = _no_53_59_61(monkeypatch)                                                           # 시드 53·59·61 은 5 로 바꿔치기 (실제 시드 데이터는 만들지 않음)
    (tmp_path / "res").mkdir()
    a = _vargs(tmp_path, r0_copy, _sel(tmp_path, selected="T1"))
    out = ct.cmd_verify(a)
    assert sorted(seen) == [53, 59, 61]                                                        # 시드 집합 전체만 요청됨 (데이터는 합성)
    lock = tmp_path / "res" / ct.EVIDENCE_LOCK
    assert lock.exists() and json.load(open(lock))["status"] == "done" and json.load(open(a.record_json))["status"] == "done"
    assert out["result"] in ("PASS", "FAIL") and set(out["rules"]) == {"1", "2", "3", "4"} and "5" in out["report_only"] and out["selected"] == "T1"
    df = pd.read_csv(a.record_csv)
    assert list(df.columns) == ct.CSV_COLUMNS and {"own", "T1", "-"} == set(df.th_kind) and df.run.str.startswith("[기준]").sum() == 9      # 3종 x 시드 3
    assert set(df.seed) == {53, 59, 61}
    assert set(out["baselines"]) == {"[기준] rolling_rule", "[기준] fixed_threshold", "[기준] always_alarm"}
    with pytest.raises(SystemExit, match="1회만|덮어쓰지 않습니다"):                                  # 같은 select 로 두 번째는 거부
        ct.cmd_verify(_vargs(tmp_path, r0_copy, _sel(tmp_path, selected="T1"), record_csv=str(tmp_path / "res" / "v2.csv"),
                             record_json=str(tmp_path / "res" / "v2.json")))
    assert not (tmp_path / "res" / "v2.json").exists()


def test_judge_verify_checks_rules_1_to_4_and_only_reports_rule_5():
    ok = ct.judge_verify(R0S, _alt(th_max_min={"traffic": 9.0, "optical": 9.0}))                # 규칙 5 위반이어도 PASS (보고만)
    assert ok["pass"] is True and ok["report_only"] == {"5": False} and set(ok["rules"]) == {"1", "2", "3", "4"}
    for field, value in (("fp", 0.03), ("f1", 0.70), ("early", 0.60), ("auprc", 0.55)):
        assert ct.judge_verify(R0S, _alt(**{field: value}))["pass"] is False


def test_verify_t2_selection_needs_matching_f20_runs(runs, r0_copy, tmp_path, alt_env, monkeypatch, tiny_model_dir):
    monkeypatch.setattr(ct, "T2_FRACTION", 0.5)
    monkeypatch.setattr(ct, "T2_MIN_PORTS", 1)
    f20 = tmp_path / "f20"
    ct.cmd_train(_targs(f20, tiny_model_dir, val_fraction=0.5, min_val_ports=1))
    ct.cmd_oracle(types.SimpleNamespace(runs=str(f20), seed=211, nodes=2, days=3))
    _no_53_59_61(monkeypatch)
    (tmp_path / "res").mkdir()
    out = ct.cmd_verify(_vargs(tmp_path, f"{r0_copy},{f20}", _sel(tmp_path, selected="T2")))
    assert out["selected"] == "T2" and out["alt"]["runs"] == ["f20"] and out["R0"]["runs"] == ["r0"] and "6" not in out["report_only"]
    with pytest.raises(SystemExit, match="1회만|덮어쓰지 않습니다"):
        ct.cmd_verify(_vargs(tmp_path, f"{r0_copy},{f20}", _sel(tmp_path, selected="T2"), record_csv=str(tmp_path / "res" / "v2.csv"),
                             record_json=str(tmp_path / "res" / "v2.json")))
    # T1 선택인데 비율 0.20 실행이 섞여 있으면 거부 (다른 폴더에서 새로 확인)
    res2 = tmp_path / "res2"
    res2.mkdir()
    ct.cmd_alt(types.SimpleNamespace(kind="T1", runs=str(r0_copy), qd=str(_qd_file(tmp_path))))
    with pytest.raises(SystemExit, match="비율 0.20"):
        ct.cmd_verify(_vargs(tmp_path, f"{r0_copy},{f20}", _sel(tmp_path, selected="T1"), record_csv=str(res2 / "a.csv"), record_json=str(res2 / "a.json")))
