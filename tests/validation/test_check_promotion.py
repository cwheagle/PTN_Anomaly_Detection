"""validation/cli/check_promotion.py 테스트 (P1-3 U2, T-P3-V1)

판정 기준값(설계서 6장)과 C3 계산은 손으로 만든 입력과 하드코딩한 기대값으로 고정하고,
도구가 소형 시드로 학습 -> 평가 -> 게이트 표 -> v1 대비까지 끝까지 동작하는지 확인한다.
"""
import json
import os

import pandas as pd
import pytest

from src.models import promotion_gate
from src.pipeline.retrain_policy import RetrainPolicy
from validation.cli import check_promotion as cp


# ── 합격 기준은 설계서 6장의 사전 고정값 그대로 ──
def test_acceptance_constants_are_the_pre_registered_values():
    assert (cp.V1_MIN_AUPRC, cp.V1_MIN_F1) == (0.587, 0.755)          # P1-1 7.2 auto 기준선(0.637 / 0.805) - 0.05
    assert cp.V2_MAX_FP == 0.02 and cp.V4_MIN_DIFF == 0.03
    assert cp.RECIPE == {"epochs": 30, "batch_size": 64, "patience": 5}
    assert cp.DEV_SEEDS == (7, 11) and cp.VAL_SEEDS == (23, 31, 47) and cp.TRAIN_SEEDS == (101, 103)
    assert cp.GATE_DAYS == 3
    assert (cp.V3_EXCESS_FACTOR, cp.V3_EXCESS_MIN) == (3.0, 0.04)             # 과다 = max(3 x 후보①, 0.04) (설계서 6장 2차 재정의)
    assert cp.POLICIES["오탐억제형"].dampening_steps == {1: 6, 2: 4, 3: 3} and cp.POLICIES["기본"].dampening_steps == {1: 3, 2: 2, 3: 1}


def _rows(model, policy, **kw):
    base = {"auprc": 0.6, "f1": 0.8, "early": 0.7, "fp": 0.01, "prec": 0.9, "lead": 150.0}
    return {"model": model, "policy": policy, "seed": 1, **{**base, **kw}}


# ── V1·V2·V4 판정 경계 ──
@pytest.mark.parametrize("auprc,f1,expected", [(0.587, 0.755, True), (0.586, 0.80, False), (0.60, 0.754, False)])
def test_v1_judgement_boundaries(auprc, f1, expected):
    df = pd.DataFrame([_rows("m", "기본", auprc=auprc), _rows("m", "오탐억제형", f1=f1)])
    assert cp.judge_v1(df, ["m"])["pass"] is expected


@pytest.mark.parametrize("fp,f1_p,f1_d,expected", [
    (0.02, 0.81, 0.70, True), (0.021, 0.81, 0.70, False),     # 오탐 상한 경계
    (0.01, 0.70, 0.70, False), (0.01, 0.701, 0.70, True),     # 기본 정책보다 F1 이 '높아야' 함 (같으면 FAIL)
])
def test_v2_judgement_boundaries(fp, f1_p, f1_d, expected):
    df = pd.DataFrame([_rows("m", "오탐억제형", fp=fp, f1=f1_p), _rows("m", "기본", f1=f1_d, fp=0.1)])
    assert cp.judge_v2(df, ["m"])["pass"] is expected


def test_v4_requires_all_three_improvements_over_0_03():
    old = _rows("v1", "기본", auprc=0.30, f1=0.32, fp=0.50)
    new_p, new_d = _rows("v2", "오탐억제형", f1=0.36, fp=0.46), _rows("v2", "기본", auprc=0.34)
    assert cp.judge_v4(pd.DataFrame([old, new_p, new_d]), "v1", ["v2"])["pass"] is True
    new_p["f1"] = 0.35                                              # 차이 0.03 은 '초과'가 아님
    assert cp.judge_v4(pd.DataFrame([old, new_p, new_d]), "v1", ["v2"])["pass"] is False
    new_p["f1"], new_p["fp"] = 0.36, 0.48                           # 오탐 감소 0.02
    assert cp.judge_v4(pd.DataFrame([old, new_p, new_d]), "v1", ["v2"])["pass"] is False


# ── V3: C3 계산 ──
def _port_frame(n=200):
    return pd.DataFrame({"occur_date": pd.date_range("2026-03-01", periods=n, freq="15min"),
                         "ip_addr": "10.0.0.1", "cid": 1, "lid": 1,
                         "tx_packet": 1000, "rx_packet": 1000, "error_packet": 0})


def test_fp_incidents_exclude_alarms_inside_rule_b_intervals():
    """규칙 B 구간(에러 12스텝 연속 -> 앞 16·뒤 4스텝 확장) 안의 알람은 오탐에서 제외, 밖의 알람 인시던트만 센다"""
    raw = _port_frame()
    raw.loc[100:111, "error_packet"] = 50
    alarms = raw[["occur_date", "ip_addr", "cid", "lid"]].copy()
    alarms["alarm"] = False
    alarms.loc[[30, 105, 190], "alarm"] = True                      # 구간 밖 2건(30, 190) + 구간 안 1건(105)
    assert cp.fp_incidents_per_port_day(alarms, raw, port_days=2.0) == pytest.approx(1.0)      # 2건 / 2포트·일
    alarms.loc[[30, 190], "alarm"] = False
    assert cp.fp_incidents_per_port_day(alarms, raw, port_days=2.0) == 0.0
    assert cp.fp_incidents_per_port_day(alarms.assign(alarm=False), raw, port_days=2.0) == 0.0


@pytest.mark.parametrize("cand_total,cand_fp,active,port_days,expected", [
    (0.060, 0.050, 0.100, 180, "PASS"),    # 전체는 상한(0.05) 초과이지만 추정 오탐이 상한 이내 -> PASS (C3 의 목적)
    (0.060, 0.051, 0.100, 180, "FAIL"),    # 추정 오탐 상한 초과
    (0.160, 0.010, 0.100, 180, "FAIL"),    # 상대 기준(활성 x1.5 = 0.15) 초과는 전체 인시던트로 판정
    (0.150, 0.010, 0.100, 180, "PASS"),
    (0.060, 0.010, 0.010, 149, "SKIP"),    # 최소 포트·일 가드(150) 유지
])
def test_c3_status_rule(cand_total, cand_fp, active, port_days, expected):
    cand = {"incidents_per_port_day": cand_total, "port_days": port_days}
    assert cp.c3_status(cand, {"incidents_per_port_day": active}, cand_fp, RetrainPolicy()) == expected


# ── 끝까지 동작 (소형 시드) ──
@pytest.fixture(scope="module")
def trained(tmp_path_factory, tiny_model_dir):
    out = tmp_path_factory.mktemp("promo_model")
    res = cp.train_operational(str(out), data_seed=5, train_seed=0, nodes=2, days=6, epochs=1, active_models_dir=str(tiny_model_dir))
    return out, res


def test_operational_recipe_trains_both_tracks_with_suspect_exclusion_and_port_holdout(trained):
    out, res = trained
    for ft in cp.TRACKS:
        assert res[ft]["train"] > 0 and res[ft]["test"] > 0                   # 포트 홀드아웃: 학습/검증 포트 모두 존재
        assert 0 <= res[ft]["suspect_stats"]["fraction"] < RetrainPolicy().max_excluded_fraction
        assert os.path.exists(out / f"{ft}_ae_v1.pth") and os.path.exists(out / f"{ft}_registry.json")
        meta = json.load(open(out / f"{ft}_ae_v1.json"))
        assert meta["suspect_stats"]["rows"] > 0 and meta["config"]["batch_size"] == 64 and meta["config"]["patience"] == 5


def test_evaluate_models_and_judgements_run_end_to_end(trained, tiny_model_dir):
    out, _ = trained
    df = cp.evaluate_models({"new": str(out), "old": str(tiny_model_dir)}, seeds=[23], nodes=1, days=4)
    assert {"new", "old"} <= set(df.model) and {"기본", "오탐억제형"} <= set(df.policy)
    assert {"[기준] rolling_rule", "[기준] fixed_threshold", "[기준] always_alarm"} <= set(df.model)       # 기준선 병기
    for judged in (cp.judge_v1(df, ["new"]), cp.judge_v2(df, ["new"]), cp.judge_v4(df, "old", ["new"])):
        assert isinstance(judged["pass"], bool)
    # 오탐 억제형은 같은 점수에서 알람이 줄어드는 정책 -> 오탐이 늘어나지 않는다
    new = df[df.model == "new"].set_index("policy")
    assert new.loc["오탐억제형", "fp"] <= new.loc["기본", "fp"]


# ── V2 후속: 정책 재선택 ──
def _cand(f1, early, fp, scale=1.0):
    return {"scale": scale, "sigma": 3.0, "damping": "default(3/2/1)", "f1": f1, "early": early, "fp": fp, "auprc": 0.6, "prec": 0.5, "lead": 100.0}


def test_pick_policy_applies_the_fixed_rule_and_never_relaxes_it():
    t = pd.DataFrame([_cand(0.90, 0.69, 0.01, 1), _cand(0.85, 0.80, 0.021, 2),        # F1 은 높지만 규칙 위반(조기 < 70% / 오탐 > 0.02)
                      _cand(0.80, 0.70, 0.02, 3), _cand(0.75, 0.90, 0.00, 4)])          # 경계값(조기 0.70, 오탐 0.02)은 충족
    assert cp.pick_policy(t)["scale"] == 3                                               # 충족 중 F1 최대
    assert cp.pick_policy(t.iloc[[0, 1]]) is None                                         # 충족 없음 -> 완화하지 않고 None


def test_grid_is_the_pre_registered_p0_2_selection_space():
    assert cp.GRID_SCALES == (0.8, 1.0, 1.25, 1.5, 2.0, 3.0) and cp.GRID_SIGMAS == (2.0, 3.0, 4.0)
    assert (cp.SELECT_MIN_EARLY, cp.SELECT_MAX_FP) == (0.70, 0.02)
    c = cp.policy_candidates()
    assert len(c) == 6 * 3 * 7
    assert any(x["threshold_scale"] == 3.0 and x["sigma_k"] == 2.0 and x["damping"] == "heavy(6/4/3)" for x in c)   # 현재 precision 포함


def test_reselect_refuses_validation_seeds_before_any_work(tiny_model_dir):
    with pytest.raises(ValueError, match="개발 시드"):
        cp.reselect_policy({"m": str(tiny_model_dir)}, seeds=[7, 23], nodes=1, days=3)


def test_reselect_runs_on_dev_seeds_and_chosen_row_satisfies_the_rule(tiny_model_dir):
    cands = [c for c in cp.policy_candidates() if c["sigma_k"] == 3.0 and c["damping"] == "default(3/2/1)"][:3]
    table, chosen = cp.reselect_policy({"m": str(tiny_model_dir)}, seeds=[7], nodes=1, days=4, candidates=cands)
    assert len(table) == 3 and list(table["f1"]) == sorted(table["f1"], reverse=True)
    if chosen is not None:
        assert chosen["early"] >= 0.70 and chosen["fp"] <= 0.02
        assert cp.policy_from_row(chosen).threshold_scale == chosen["scale"]


# ──────────────────────────────────────────────
# V3 (설계서 6장 2차 재정의): truth-FP 라벨, 후보 풀 5개 x 활성 2가지, 판정은 src check_g3(C4')
# ──────────────────────────────────────────────
@pytest.mark.parametrize("fp,base,label", [
    (0.000, 0.000, "정상"), (0.039, 0.000, "중간"), (0.040, 0.000, "과다"),       # 기준 0 이면 하한 0.04 가 과다 경계
    (0.030, 0.030, "정상"), (0.020, 0.030, "정상"), (0.031, 0.030, "중간"),         # 정상 = 기준 이하(같으면 정상)
    (0.089, 0.030, "중간"), (0.090, 0.030, "과다"),                                  # 3 x 0.03 = 0.09
    (0.039, 0.005, "중간"), (0.040, 0.005, "과다"),                                  # 3 x 0.005 = 0.015 < 0.04 -> 하한 0.04
])
def test_truth_fp_label_boundaries(fp, base, label):
    assert cp.truth_fp_label(fp, base) == label


def test_pool_and_actives_are_the_fixed_five_and_two():
    names = [c[0] for c in cp.V3_POOL]
    assert names == ["①V1모델+최종정책", "②V1모델+기본정책", "③V1모델x0.25+최종정책", "④V1모델x0.1+최종정책", "⑤v1+기본정책"]
    assert [(c[1], c[2], c[3]) for c in cp.V3_POOL] == [("new", 1.0, "final"), ("new", 1.0, "default"), ("new", 0.25, "final"),
                                                         ("new", 0.1, "final"), ("v1", 1.0, "default")]
    assert [a[1] for a in cp.V3_ACTIVES] == ["⑤v1+기본정책", "①V1모델+최종정책"]       # (a) 첫 승격 = v1+기본, (b) 이후 = 정상 v2+최종 정책


def test_v3_phase_guards_dev_vs_evidence_seeds():
    assert cp.v3_phase([7, 11]) == "탐색" and cp.v3_phase([7]) == "탐색" and cp.v3_phase([47, 23, 31]) == "근거"
    for bad in ([23], [23, 31], [31, 47], [7, 23], [7, 11, 23, 31, 47], [5], []):    # 부분집합·혼입·임의 시드는 거부
        with pytest.raises(ValueError):
            cp.v3_phase(bad)


def _v3_rows(label, active, g3, cond=(7, "기준", "traffic"), name="후보"):
    return {"seed": cond[0], "density": cond[1], "track": cond[2], "candidate": name, "active": active, "label": label, "G3": g3}


A, B = cp.V3_ACTIVES[0][0], cp.V3_ACTIVES[1][0]


def test_judge_v3_criteria():
    ok = pd.DataFrame([_v3_rows("정상", A, "PASS"), _v3_rows("정상", B, "WARN"),            # 정상: FAIL 이 아니면 됨(PASS·WARN)
                       _v3_rows("과다", A, "WARN"), _v3_rows("과다", A, "FAIL"), _v3_rows("과다", B, "FAIL")])
    j = cp.judge_v3(ok)
    assert j["pass"] is True and (j["normal_violations"], j["excess_a_violations"], j["excess_b_violations"]) == (0, 0, 0)
    bad_normal = pd.concat([ok, pd.DataFrame([_v3_rows("정상", B, "FAIL")])])
    assert cp.judge_v3(bad_normal)["pass"] is False and cp.judge_v3(bad_normal)["normal_violations"] == 1
    bad_a = pd.concat([ok, pd.DataFrame([_v3_rows("과다", A, "PASS")])])                      # 과다가 첫 승격에서 PASS
    assert cp.judge_v3(bad_a)["excess_a_violations"] == 1 and cp.judge_v3(bad_a)["pass"] is False
    bad_b = pd.concat([ok, pd.DataFrame([_v3_rows("과다", B, "WARN")])])                      # 과다가 이후 재학습에서 FAIL 이 아님
    assert cp.judge_v3(bad_b)["excess_b_violations"] == 1 and cp.judge_v3(bad_b)["pass"] is False
    mid = pd.concat([ok, pd.DataFrame([_v3_rows("중간", A, "PASS"), _v3_rows("중간", B, "FAIL")])])   # 중간은 판정 제외
    assert cp.judge_v3(mid)["pass"] is True


def test_judge_v3_records_conditions_without_excess_and_undecidable_when_none():
    t = pd.DataFrame([_v3_rows("정상", A, "PASS", cond=(7, "기준", "traffic")), _v3_rows("과다", B, "FAIL", cond=(7, "잦음", "optical")),
                      _v3_rows("정상", B, "PASS", cond=(7, "잦음", "optical"))])
    j = cp.judge_v3(t)
    assert j["conditions"] == 2 and j["conditions_without_excess"] == [(7, "기준", "traffic")]
    none = t[t.label != "과다"]
    assert cp.judge_v3(none)["pass"] is None                                                # 과다 라벨 전무 -> 판정 불가


def test_c0_reference_column_is_the_pre_c4prime_rule():
    c = lambda x: {"incidents_per_port_day": x, "port_days": 180}
    assert cp.c0_status(c(0.06), c(0.20)) == "FAIL"           # 구 식: 절대 상한 초과는 FAIL (C4' 에서는 WARN)
    assert cp.c0_status(c(0.05), c(0.20)) == "PASS"
    assert cp.c0_status({"incidents_per_port_day": 0.0, "port_days": 149}, c(0.0)) == "SKIP"


def test_v3_table_structure_and_g3_equals_src_check_g3_independently(tiny_env):
    """5노드(50포트) x 3일 = 150포트·일: 후보 5 x 활성 2 x 트랙 2 = 20행. G3 열과 truth-FP 는 src·평가 함수로 따로 계산한 값과 같다"""
    from src.pipeline.inference import AnomalyDetector
    from validation.evaluation.tuning import evaluate_alarms, simulate_alarms
    from validation.simulator.scenario_generator import ScenarioConfig

    t = cp.v3_table(str(tiny_env), str(tiny_env), seeds=[7], nodes=5, densities={"기준": 7.0})
    assert len(t) == 20 and (t.port_days == 150).all()
    assert set(t.candidate) == {c[0] for c in cp.V3_POOL} and set(t.active) == {a[0] for a in cp.V3_ACTIVES}
    assert (t[t.candidate == cp.V3_BASE].label == "정상").all()                              # ① 은 정의상 정상

    data = cp.make_data(7, 5, cp.GATE_DAYS + 1, mean_gap_days=7.0)
    since = pd.Timestamp(ScenarioConfig().start) + pd.Timedelta(days=1)
    det = AnomalyDetector()                                                                  # tiny_env 가 PATHS 를 그 폴더로 돌려 둠
    rp = RetrainPolicy()
    for ft in cp.TRACKS:
        scores, th = det.track_scores(data[ft], ft)
        scores = scores[pd.to_datetime(scores["occur_date"]) >= since]
        built = {}
        for name, which, factor, pol in cp.V3_POOL:
            policy = cp.POLICIES["오탐억제형"] if pol == "final" else cp.POLICIES["기본"]
            stats, alarms = promotion_gate.alarm_stats(scores, th * factor, policy, cp.GATE_DAYS, ft)
            n_false = evaluate_alarms(alarms, data)["event"]["false_incidents"]
            built[name] = (stats, n_false / stats["port_days"])
        for name, *_ in cp.V3_POOL:
            for aname, aref in cp.V3_ACTIVES:
                row = t[(t.track == ft) & (t.candidate == name) & (t.active == aname)].iloc[0]
                want = promotion_gate.check_g3(built[name][0], built[aref][0], rp)
                assert (row["G3"], row["G3_value"], row["G3_limit"]) == (want.status, want.value, want.limit)
                assert row["truth_fp"] == pytest.approx(built[name][1]) and row["base_fp"] == pytest.approx(built[cp.V3_BASE][1])
                assert row["label"] == ("정상" if name == cp.V3_BASE else cp.truth_fp_label(built[name][1], built[cp.V3_BASE][1]))
                assert row["C0"] in ("PASS", "FAIL") and row["C3"] in ("PASS", "FAIL", "SKIP")


def _gate_args(model, seeds, record=None, nodes=1):
    import types
    return types.SimpleNamespace(model=str(model), active_models=str(model), seeds=seeds, nodes=nodes, traffic_drop_only=False,
                                 final_policy="오탐억제형", record=record)


def _boom(*a, **k):
    raise AssertionError("측정이 실행되면 안 됨")


@pytest.mark.parametrize("seeds", ["23", "23,31", "31,47", "7,23", "7,11,23,31,47", "5"])
def test_gate_evidence_run_rejects_partial_or_mixed_seeds_before_any_measurement(tmp_path, monkeypatch, seeds):
    monkeypatch.setattr(cp, "v3_table", _boom)
    with pytest.raises(ValueError):
        cp.cmd_gate(_gate_args(tmp_path, seeds))
    assert not (tmp_path / cp.V3_LOCK).exists()                        # 거부된 요청은 잠금도 남기지 않음


def test_gate_evidence_lock_is_fixed_per_model_and_cannot_be_bypassed_with_another_record_name(tmp_path, monkeypatch):
    """1회 잠금은 모델 폴더의 v3_evidence.json — --record 이름을 바꿔도 우회 불가, 잠금이 있으면 측정 전에 거부"""
    monkeypatch.setattr(cp, "v3_table", _boom)
    (tmp_path / cp.V3_LOCK).write_text("{}")                          # 이미 근거 실행을 한 모델
    (tmp_path / "x").mkdir()                                           # 사본 폴더 검사(check_record)를 통과시켜 잠금 거부를 확인
    for record in (None, str(tmp_path / "other_name.json"), str(tmp_path / "x" / "y.json")):
        with pytest.raises(SystemExit, match="1회만"):
            cp.cmd_gate(_gate_args(tmp_path, "23,31,47", record=record))
    assert not (tmp_path / "other_name.json").exists()


def test_gate_evidence_run_creates_the_lock_before_measuring_and_keeps_it_when_the_run_fails(tmp_path, monkeypatch):
    seen = {}

    def failing(*a, **k):
        seen["lock_exists_at_measure_time"] = (tmp_path / cp.V3_LOCK).exists()
        raise RuntimeError("중간 실패")
    monkeypatch.setattr(cp, "v3_table", failing)
    with pytest.raises(RuntimeError):
        cp.cmd_gate(_gate_args(tmp_path, "23,31,47"))
    assert seen["lock_exists_at_measure_time"] is True                 # 측정 시작 전에 잠금 생성
    assert (tmp_path / cp.V3_LOCK).exists()                            # 실패해도 잠금은 남음 (검증 시드를 한 번 열었음)
    monkeypatch.setattr(cp, "v3_table", _boom)
    with pytest.raises(SystemExit, match="1회만"):
        cp.cmd_gate(_gate_args(tmp_path, "23,31,47"))                  # 재시도 거부


def test_gate_evidence_success_writes_result_into_lock_and_record_copy(tmp_path, monkeypatch, capsys):
    t = pd.DataFrame([{"seed": 23, "density": "기준", "track": "traffic", "candidate": c[0], "active": a[0], "label": "정상",
                       "truth_fp": 0.0, "base_fp": 0.0, "total": 0.0, "active_total": 0.0, "port_days": 180, "G3": "PASS",
                       "G3_value": 0.0, "G3_limit": 0.05, "C0": "PASS", "C3": "PASS"} for c in cp.V3_POOL for a in cp.V3_ACTIVES])
    monkeypatch.setattr(cp, "v3_table", lambda *a, **k: t)
    copy = tmp_path / "copy.json"
    cp.cmd_gate(_gate_args(tmp_path, "23,31,47", record=str(copy)))
    out = capsys.readouterr().out
    assert "근거 판정(1회)" in out and "탐색 결과이며" not in out
    lock = json.load(open(tmp_path / cp.V3_LOCK, encoding="utf-8"))
    assert lock["phase"] == "근거" and lock["seeds"] == [23, 31, 47] and json.load(open(copy, encoding="utf-8")) == lock


def _confirm_args(tmp_path, seeds, record=None):
    import types
    pj = tmp_path / "selected_policy.json"
    pj.write_text(json.dumps({"scale": 3.0, "sigma": 2.0, "damping": "heavy(6/4/3)"}))
    return types.SimpleNamespace(models=f"m={tmp_path}", policy_json=str(pj), seeds=seeds, nodes=1, days=4, record=record)


@pytest.mark.parametrize("seeds", ["23", "23,31", "7,11", "7,23,31,47", "5"])
def test_confirm_rejects_partial_dev_or_mixed_seeds_before_measuring(tmp_path, monkeypatch, seeds):
    monkeypatch.setattr(cp, "evaluate_models", _boom)
    with pytest.raises(SystemExit, match="전체로만"):
        cp.cmd_confirm(_confirm_args(tmp_path, seeds))
    assert not (tmp_path / cp.CONFIRM_LOCK).exists()


def test_confirm_runs_once_lock_beside_policy_json_and_record_name_is_no_bypass(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_eval(models, seeds, nodes, days, policies=None, baselines=True):
        calls.append(seeds)
        return pd.DataFrame([_rows("m", n, fp=0.01) for n in ("기본", "오탐억제형(기존)", "후보")])
    monkeypatch.setattr(cp, "evaluate_models", fake_eval)
    cp.cmd_confirm(_confirm_args(tmp_path, "23,31,47", record=str(tmp_path / "copy1.json")))
    assert calls == [[23, 31, 47]] and (tmp_path / cp.CONFIRM_LOCK).exists() and (tmp_path / "copy1.json").exists()
    assert json.load(open(tmp_path / cp.CONFIRM_LOCK, encoding="utf-8"))["status"] == "done"
    for record in (None, str(tmp_path / "copy2.json")):                # 두 번째 실행은 다른 --record 이름으로도 거부
        with pytest.raises(SystemExit, match="1회만"):
            cp.cmd_confirm(_confirm_args(tmp_path, "23,31,47", record=record))
    assert calls == [[23, 31, 47]] and not (tmp_path / "copy2.json").exists()


def test_gate_cli_explore_mode_prints_judgement_and_writes_record(tiny_env, tmp_path, capsys):
    """소형 임시 모델로 탐색(개발 시드) 경로의 출력·기록 파일을 확인 (실제 모델·시드 측정이 아님 — 코드 경로가 끝까지 동작하는지만)"""
    import types
    rec = tmp_path / "explore.json"
    cp.cmd_gate(types.SimpleNamespace(model=str(tiny_env), active_models=str(tiny_env), seeds="7", nodes=5,
                                      traffic_drop_only=False, final_policy="오탐억제형", record=str(rec)))
    out = capsys.readouterr().out
    assert "[탐색 — 판정 근거 아님]" in out and "C0·C3 열은 참고용입니다 (판정은 src check_g3 = C4')" in out
    assert "[V3 탐색]" in out and "과다 라벨이 없는 조건(판정 불가)" in out
    assert "※ 탐색 결과이며 판정 근거가 아닙니다" in out and "traffic: 정상" in out and "optical: 정상" in out     # 탐색 표시, 트랙별 라벨 수
    assert not (tiny_env / cp.V3_LOCK).exists()                                                              # 탐색은 잠금을 만들지 않음 (반복 가능)
    saved = json.load(open(rec, encoding="utf-8"))
    assert saved["phase"] == "탐색" and saved["seeds"] == [7] and saved["final_policy"] == "오탐억제형"
    assert "pass" in saved["judgement"] and os.path.exists(str(rec) + ".csv")


# ── V4 근거 실행 가드 (vs-v1) ──
def _v4_args(models, seeds="23,31,47", record=None, v1="models"):
    import types
    return types.SimpleNamespace(models=",".join(f"{k}={v}" for k, v in models.items()), v1_models=v1, seeds=seeds, nodes=1, days=4, record=record)


def _fake_v4_df():
    rows = []
    for seed in (23, 31, 47):
        rows += [_rows("v1", "기본", seed=seed, auprc=0.30, f1=0.30, fp=0.50)]
        for m, dd in (("s101", 0.0), ("s103", 0.02)):
            rows += [_rows(m, "오탐억제형", seed=seed, f1=0.60 + dd, fp=0.04), _rows(m, "기본", seed=seed, auprc=0.60 + dd)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seeds", ["23", "23,31", "7,11", "7,23,31,47", "5"])
def test_v4_rejects_partial_dev_or_mixed_seeds_before_any_measurement(tmp_path, monkeypatch, seeds):
    m = {"a": tmp_path / "a", "b": tmp_path / "b"}
    for d in m.values():
        d.mkdir()
    monkeypatch.setattr(cp, "evaluate_models", _boom)
    with pytest.raises(SystemExit, match="전체로만"):
        cp.cmd_vs_v1(_v4_args({k: str(v) for k, v in m.items()}, seeds))
    assert not any((d / cp.V4_LOCK).exists() for d in m.values())


def test_v4_runs_once_locks_each_v2_model_folder_and_never_touches_the_v1_folder(tmp_path, monkeypatch, capsys):
    a, b, v1 = tmp_path / "a", tmp_path / "b", tmp_path / "v1models"
    for d in (a, b, v1):
        d.mkdir()
    calls = []
    monkeypatch.setattr(cp, "evaluate_models", lambda models, seeds, nodes, days, **k: calls.append(list(models)) or _fake_v4_df())
    args = _v4_args({"s101": str(a), "s103": str(b)}, record=str(tmp_path / "copy.json"), v1=str(v1))
    cp.cmd_vs_v1(args)
    out = capsys.readouterr().out
    assert calls == [["v1", "s101", "s103"]] and "근거 판정(1회)" in out and "모델별 판정 아님" in out
    for d in (a, b):
        rec = json.load(open(d / cp.V4_LOCK, encoding="utf-8"))
        assert rec["status"] == "done" and rec["judgement"]["pass"] is True and (d / (cp.V4_LOCK + ".csv")).exists()
    assert list(v1.iterdir()) == []                                              # v1(운영) 폴더에는 아무것도 쓰지 않음
    assert "[개별 값 — 모델 x 시드" in out and out.count("v2+오탐억제형") >= 6      # 모델 x 시드 개별 값이 출력됨
    for rec_name in (None, str(tmp_path / "other.json")):                        # 같은 모델로 다시, 다른 --record 로도 거부
        with pytest.raises(SystemExit, match="1회만"):
            cp.cmd_vs_v1(_v4_args({"s101": str(a), "s103": str(b)}, record=rec_name, v1=str(v1)))
    assert len(calls) == 1 and not (tmp_path / "other.json").exists()


def test_v4_refusal_when_only_one_model_folder_is_locked_leaves_no_new_lock(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    (b / cp.V4_LOCK).write_text("{}")                                           # b 는 이미 근거 실행을 한 모델
    monkeypatch.setattr(cp, "evaluate_models", _boom)
    with pytest.raises(SystemExit, match="1회만"):
        cp.cmd_vs_v1(_v4_args({"a": str(a), "b": str(b)}))
    assert not (a / cp.V4_LOCK).exists()                                         # 거부 시 a 에 새 잠금을 남기지 않음


def test_v4_failure_during_measurement_keeps_the_locks(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    monkeypatch.setattr(cp, "evaluate_models", lambda *x, **k: (_ for _ in ()).throw(RuntimeError("중간 실패")))
    with pytest.raises(RuntimeError):
        cp.cmd_vs_v1(_v4_args({"a": str(a)}))
    assert (a / cp.V4_LOCK).exists()                                             # 측정 시작 후 실패해도 잠금 유지 -> 재시도 거부
    monkeypatch.setattr(cp, "evaluate_models", _boom)
    with pytest.raises(SystemExit, match="1회만"):
        cp.cmd_vs_v1(_v4_args({"a": str(a)}))


def test_v4_individual_rows_report_keeps_each_model_and_seed_visible():
    r = cp.v4_rows_report(_fake_v4_df(), "v1", ["s101", "s103"])
    assert len(r) == 3 + 6 and set(r.arm) == {"v1+기본", "v2+오탐억제형"}
    v2 = r[r.arm == "v2+오탐억제형"].set_index(["model", "seed"])
    assert v2.loc[("s101", 23), "f1"] == pytest.approx(0.60) and v2.loc[("s103", 47), "f1"] == pytest.approx(0.62)
    assert v2.loc[("s103", 31), "auprc_rank"] == pytest.approx(0.62)             # v2 의 AUPRC(순위 품질)는 기본 정책 행 값
    d = cp.judge_v4(_fake_v4_df(), "v1", ["s101", "s103"])                       # 판정 상수·방식 불변: v2 6개 평균 vs v1 3개 평균
    assert d["f1"] == pytest.approx(0.31 - 0.0) and d["pass"] is True


# ──────────────────────────────────────────────
# --record 사본 덮어쓰기 거부 / evidence_lock 폴더 미생성 거부 (reviewer 보류 2건)
# ──────────────────────────────────────────────
def test_check_record_rejects_existing_copy_or_csv_missing_folder_and_lock_path(tmp_path):
    cp.check_record(None)                                                       # 사본 없음 = 통과
    cp.check_record(str(tmp_path / "new.json"))                                 # 새 사본 = 통과
    (tmp_path / "a.json").write_text("old")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        cp.check_record(str(tmp_path / "a.json"))
    (tmp_path / "b.json.csv").write_text("old")                                 # .csv 사본만 있어도 거부
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        cp.check_record(str(tmp_path / "b.json"))
    with pytest.raises(SystemExit, match="폴더"):
        cp.check_record(str(tmp_path / "nodir" / "c.json"))
    with pytest.raises(SystemExit, match="잠금 파일과 같은 경로"):
        cp.check_record(str(tmp_path / "lock.json"), reserved=[str(tmp_path / "lock.json")])


def test_evidence_lock_refuses_missing_folder_without_creating_it(tmp_path):
    target = tmp_path / "typo_model" / cp.V3_LOCK
    with pytest.raises(SystemExit, match="폴더"):
        cp.evidence_lock(str(target), {"kind": "V3"})
    assert not (tmp_path / "typo_model").exists()                               # 폴더도 잠금도 만들지 않음


def test_gate_evidence_existing_record_is_rejected_before_lock_and_measurement_and_left_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "v3_table", _boom)
    rec = tmp_path / "copy.json"
    rec.write_text("ORIGINAL")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        cp.cmd_gate(_gate_args(tmp_path, "23,31,47", record=str(rec)))
    assert rec.read_text() == "ORIGINAL" and not (tmp_path / cp.V3_LOCK).exists()   # 사본 보존, 잠금 안 남김(검증 시드를 열지 않음)


def test_gate_explore_existing_record_is_also_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "v3_table", _boom)
    rec = tmp_path / "explore.json"
    rec.write_text("ORIGINAL")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        cp.cmd_gate(_gate_args(tmp_path, "7,11", record=str(rec)))
    assert rec.read_text() == "ORIGINAL"


def test_gate_evidence_with_missing_model_folder_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "v3_table", _boom)
    with pytest.raises(SystemExit, match="폴더"):
        cp.cmd_gate(_gate_args(tmp_path / "typo", "23,31,47"))
    assert not (tmp_path / "typo").exists()


def test_confirm_existing_record_is_rejected_before_lock_and_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "evaluate_models", _boom)
    rec = tmp_path / "copy.json"
    rec.write_text("ORIGINAL")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        cp.cmd_confirm(_confirm_args(tmp_path, "23,31,47", record=str(rec)))
    assert rec.read_text() == "ORIGINAL" and not (tmp_path / cp.CONFIRM_LOCK).exists()


def test_v4_existing_record_or_missing_model_folder_is_rejected_without_any_lock(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    monkeypatch.setattr(cp, "evaluate_models", _boom)
    rec = tmp_path / "copy.json"
    rec.write_text("ORIGINAL")
    with pytest.raises(SystemExit, match="덮어쓰지 않습니다"):
        cp.cmd_vs_v1(_v4_args({"s101": str(a), "s103": str(b)}, record=str(rec)))
    assert rec.read_text() == "ORIGINAL"
    with pytest.raises(SystemExit, match="폴더"):                                # 두 번째 모델 폴더가 없으면 첫 폴더에도 잠금을 만들지 않음
        cp.cmd_vs_v1(_v4_args({"s101": str(a), "s103": str(tmp_path / "typo")}))
    assert not (a / cp.V4_LOCK).exists() and not (b / cp.V4_LOCK).exists() and not (tmp_path / "typo").exists()
