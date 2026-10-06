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
    assert cp.EXCESS_THRESHOLD_FACTOR == 0.5 and cp.GATE_DAYS == 3
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


def test_judge_v3_requires_normal_pass_and_excess_fail_everywhere():
    t = pd.DataFrame({"candidate": ["정상", "정상", "과다", "과다"], "C0": ["PASS", "FAIL", "FAIL", "FAIL"],
                      "C3": ["PASS", "PASS", "FAIL", "FAIL"]})
    assert cp.judge_v3(t, "C0") == {"normal_all_pass": False, "excess_all_fail": True, "pass": False}
    assert cp.judge_v3(t, "C3")["pass"] is True


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


def test_gate_table_skips_below_min_port_days(trained, tiny_model_dir):
    """1노드 = 10포트 x 3일 = 30포트·일 < 150 -> 두 식 모두 최소 포트·일 가드로 SKIP"""
    out, _ = trained
    t = cp.gate_table(str(out), str(tiny_model_dir), seeds=[7], nodes=1, densities={"기준": 7.0})
    assert len(t) == 4 and set(t.candidate) == {"정상", "과다"} and set(t.track) == set(cp.TRACKS)
    assert (t.port_days == 30).all() and (t.C0 == "SKIP").all() and (t.C3 == "SKIP").all()


def test_gate_table_values_equal_independent_src_calculation(tiny_env):
    """5노드(50포트) x 3일 = 150포트·일: 표의 인시던트율·판정이 src 함수로 따로 계산한 값과 같다 (T-P3-V1)"""
    import pandas as pd
    from src.pipeline.inference import AnomalyDetector
    from validation.simulator.scenario_generator import ScenarioConfig

    seed, gap = 7, 7.0
    t = cp.gate_table(str(tiny_env), str(tiny_env), seeds=[seed], nodes=5, densities={"기준": gap})
    assert len(t) == 4 and (t.port_days == 150).all()

    data = cp.make_data(seed, 5, cp.GATE_DAYS + 1, mean_gap_days=gap)
    since = pd.Timestamp(ScenarioConfig().start) + pd.Timedelta(days=1)
    det = AnomalyDetector()                                         # tiny_env 가 PATHS 를 그 폴더로 돌려 둠
    rp = RetrainPolicy()
    for ft in cp.TRACKS:
        scores, th = det.track_scores(data[ft], ft)
        scores = scores[pd.to_datetime(scores["occur_date"]) >= since]
        active, _ = promotion_gate.alarm_stats(scores, th, cp.POLICIES["기본"], cp.GATE_DAYS, ft)
        for kind, factor in (("정상", 1.0), ("과다", 0.5)):
            cand, _ = promotion_gate.alarm_stats(scores, th * factor, cp.POLICIES["오탐억제형"], cp.GATE_DAYS, ft)
            row = t[(t.track == ft) & (t.candidate == kind)].iloc[0]
            assert row["total"] == pytest.approx(cand["incidents_per_port_day"])
            assert row["active"] == pytest.approx(active["incidents_per_port_day"])
            assert row["C0"] == promotion_gate.check_g3(cand, active, rp).status
            assert row["C0"] in ("PASS", "FAIL") and row["C3"] in ("PASS", "FAIL")       # 150포트·일 -> 가드 통과, 판정이 나옴
            assert 0 <= row["fp"] <= row["total"] + 1e-12                                  # 추정 오탐은 전체 인시던트의 부분
