"""
P1-3 v2 승격 검증 도구 (설계서 docs/design/p1_3_model_promotion.md 6장 V1~V4)

바꾸는 것은 단계마다 한 가지뿐이다 (lessons #29). 합격 기준·시드는 데이터를 보기 전에 설계서에 고정된 값이며
이 파일의 상수로 옮겨 두었다 — 결과를 본 뒤 바꾸지 않는다. 판정에 쓰는 함수(알람 계산, 인시던트 집계, 게이트 G3,
의심 구간 규칙 B, 포트 분할, 수집·학습)는 모두 운영 코드(src)를 그대로 호출한다 (lessons #31).
모든 수치는 시뮬레이션 기준이며 실데이터 검증이 아니다.

주의 (V3 의 C3 열): C0(현재 식)은 src 의 promotion_gate.check_g3 를 그대로 호출하지만, C3(추정 오탐 보정 식,
`fp_incidents_per_port_day`·`c3_status`)는 **이 도구의 구현값**이며 src 와 같은 코드라는 증거가 아니다 (lessons #31).
U4(C3 를 게이트에 구현)를 하게 되면 두 함수를 src/models/promotion_gate.py 로 옮기고 도구는 import 하며 동등성 테스트를 추가한다.

  train  (V1 학습)   운영 레시피로 학습: 규칙 B + 자기 알람(활성 모델 v1 의 알람) 의심 구간 제외 + 포트 홀드아웃 10%,
                     DataCollector·train_window·Trainer 를 그대로 사용 (DB 대신 시뮬레이터 데이터를 주입)
  eval   (V1·V2)     학습한 모델을 검증 시드에서 평가: AUPRC·이벤트 지표, 정책 기본 ↔ 오탐 억제형
  gate   (V3)        게이트 G3(C4') 판별력: 후보 풀 5개 x 활성 2가지, truth-FP(시뮬레이터 정답) 기준 정상/과다 라벨.
                     C0·C3 는 참고 열. 개발 시드 7·11 = 탐색(근거 아님), 검증 시드 23·31·47 = 근거 판정 1회
  vs-v1  (V4)        v1 + 기본 정책 ↔ 새 모델 + 오탐 억제형, 기준선 병기
  select (V2 후속)   V2 미달 시 개발 시드 7·11 로만 정책 재선택 (선택 규칙: 조기 탐지 >= 70% 이고 오탐 <= 0.02 중 F1 최대)
  confirm (V2 후속)  선택된 후보 정책을 검증 시드에서 **한 번만** 확인 (기본 / 기존 오탐억제형 / 후보 병기)

근거 실행(검증 시드 23·31·47)의 1회 보장: 시드는 **전체 집합만** 허용(부분집합·개발 시드 혼입 거부)하고, 측정 전에 잠금 파일을
만든다 (V3 = <모델 폴더>/v3_evidence.json, confirm = <정책 JSON 폴더>/v2_confirm.json). --record 를 바꿔도 우회할 수 없다.
**한계**: 잠금 파일을 직접 지우거나 다른 모델 폴더를 쓰는 것은 막을 수 없다 — 운영자 규율이다 (lessons #30).

사용법:
  python validation/cli/check_promotion.py train --out validation/runs/promo_s101 --data-seed 101 --train-seed 0
  python validation/cli/check_promotion.py eval --models s101=validation/runs/promo_s101,s103=validation/runs/promo_s103
  python validation/cli/check_promotion.py gate --model validation/runs/promo_s101 --seeds 7,11            # 탐색
  python validation/cli/check_promotion.py gate --model validation/runs/promo_s101 --seeds 23,31,47                     # 근거(1회, 전체 시드)
  python validation/cli/check_promotion.py vs-v1 --models s101=...,s103=... [--v1-models models]
"""
import argparse
import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from src.data import train_window
from src.data.data_collector import DataCollector
from src.models import promotion_gate
from src.pipeline import alerting
from src.pipeline.alerting import AlertPolicy
from src.pipeline.retrain_policy import RetrainPolicy
from validation.evaluation.tuning import baseline_alarms, compute_track_scores, evaluate_alarms, simulate_alarms
from validation.simulator.scenario_generator import ScenarioConfig, generate

TRACKS = ("traffic", "optical")
KEY = ["ip_addr", "cid", "lid"]
# 근거 실행(검증 시드 23·31·47 전체)의 1회 잠금 파일: 사용자 지정 파일명에 의존하지 않도록 위치를 고정한다.
#   V3 근거  = <모델 폴더>/v3_evidence.json,  confirm = <정책 JSON 과 같은 폴더>/v2_confirm.json
# 근거 실행은 측정을 시작하기 전에 이 파일을 배타적으로 만들고(이미 있으면 거부), 실행이 도중에 실패해도 잠금은 남는다
# (검증 시드를 한 번 열었기 때문, lessons #30). **한계: 잠금 파일을 직접 지우는 것은 막을 수 없다 — 운영자 규율.**
V3_LOCK, CONFIRM_LOCK = "v3_evidence.json", "v2_confirm.json"
DEV_SEEDS = (7, 11)                 # 선택(정책·식)은 개발 시드로만
VAL_SEEDS = (23, 31, 47)            # 확인은 검증 전용 시드로만
TRAIN_SEEDS = (101, 103)            # 학습 시드 (학습 난수·데이터 영향)
RECIPE = {"epochs": 30, "batch_size": 64, "patience": 5}     # D1: 실험 레시피

# 합격 기준 (설계서 6장, 2026-10-06 고정). 기준선 = P1-1 7.2 `auto`(AUPRC 0.637, 오탐 억제형 이벤트 F1 0.805)의 -0.05
V1_MIN_AUPRC, V1_MIN_F1 = 0.587, 0.755
V2_MAX_FP = 0.02                    # 오탐 억제형의 오탐 평균 상한 (건/포트·일)
V4_MIN_DIFF = 0.03                  # v1 대비 AUPRC·F1 상승, 오탐 감소 폭 (각각 > 0.03)
GATE_DAYS = 3                       # 판정 3일 + 규칙 B 기준선 1일 (설계서 V3 세부)

# 정책 재선택(V2 후속): 선택 규칙은 performance_report.md 3장과 같고, 탐색 공간은 그 선택에 쓴 P0-2 격자
# (tune_alerting.py 기본값: 배율 6 x σ 3 x 댐프닝 7). 이 공간 밖으로 넓히지 않는다.
SELECT_MIN_EARLY, SELECT_MAX_FP = 0.70, 0.02
GRID_SCALES, GRID_SIGMAS = (0.8, 1.0, 1.25, 1.5, 2.0, 3.0), (2.0, 3.0, 4.0)

POLICIES = {"기본": alerting.get_preset("default"), "오탐억제형": alerting.get_preset("precision")}


# ─────────────────────────────────────────────
# 공통: 모델 폴더 선택, 점수 계산
# ─────────────────────────────────────────────
@contextmanager
def use_models(models_dir):
    """추론기가 읽는 모델 경로(PATHS)를 임시로 models_dir 로 돌린다 (끝나면 복구)"""
    from src.config import PATHS
    saved = {ft: dict(PATHS[ft]) for ft in TRACKS}
    for ft in TRACKS:
        PATHS[ft] = {"model": os.path.join(models_dir, f"{ft}_ae.pth"),
                     "scaler": os.path.join(models_dir, f"{ft}_scaler.joblib")}
    try:
        yield
    finally:
        for ft in TRACKS:
            PATHS[ft] = saved[ft]


def score_data(data, models_dir):
    """모델 추론을 한 번 수행해 트랙별 원시 점수 {track: (df, 전역 임계치)} (정책 비교는 이 점수를 재사용)"""
    from src.pipeline.inference import AnomalyDetector
    with use_models(models_dir):
        return compute_track_scores(AnomalyDetector(), data["traffic"], data["optical"])


def make_data(seed, nodes, days, **cfg):
    return generate(ScenarioConfig(seed=seed, nodes=nodes, days=days, **cfg))


# ─────────────────────────────────────────────
# 학습 (V1 운영 레시피)
# ─────────────────────────────────────────────
class _SimDB:
    """DataCollector 가 쓰는 DB 인터페이스를 시뮬레이터 데이터로 대신한다 (수집·정제 코드는 운영 그대로)"""

    def __init__(self, data, alarm_rows):
        self.data, self.alarm_rows = data, alarm_rows

    def fetch_traffic(self, start, end, stop_checker=None):
        return self.data["traffic"].copy()

    def fetch_optical(self, start, end, stop_checker=None):
        return self.data["optical"].copy()

    def fetch_alarm_rows(self, start, end, min_level=1):
        return self.alarm_rows


def active_model_alarm_rows(data, active_models_dir, policy=None):
    """규칙 A 입력: 활성 모델(v1)이 이 데이터에서 냈을 알람 행 (운영의 anomaly_detection 이력에 해당)"""
    scores = score_data(data, active_models_dir)
    sim = simulate_alarms(scores, policy or alerting.load_policy())
    return sim[sim["alarm"]][["occur_date"] + KEY]


def train_operational(out_dir, data_seed, train_seed, nodes=4, days=21, epochs=None, active_models_dir="models",
                      batch_size=None, patience=None, alert_policy=None):
    """운영 레시피로 두 트랙을 학습해 out_dir 에 저장한다. Returns: {track: {train, test, suspect_stats, threshold}}"""
    import torch
    from src.config import PATHS
    from src.models.trainer import Trainer

    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    cfg = ScenarioConfig(seed=data_seed, nodes=nodes, days=days)
    data = generate(cfg)
    policy = RetrainPolicy()
    alarms = active_model_alarm_rows(data, active_models_dir)
    collector = DataCollector.__new__(DataCollector)          # __init__ 은 DB 풀을 만들므로 건너뛰고 시뮬레이터 DB 주입
    collector.db = _SimDB(data, alarms)
    os.makedirs(out_dir, exist_ok=True)
    start = pd.Timestamp(cfg.start)
    results = collector.collect_and_save(
        train_start=start, train_end=start + pd.Timedelta(days=days), output_dir=os.path.join(out_dir, "train_data"),
        val_port_fraction=policy.val_port_fraction, split_salt=policy.split_salt, suspect_policy=policy)

    saved = {ft: dict(PATHS[ft]) for ft in TRACKS}
    out = {}
    try:
        for ft in TRACKS:
            PATHS[ft] = {"model": os.path.join(out_dir, f"{ft}_ae.pth"), "scaler": os.path.join(out_dir, f"{ft}_scaler.joblib")}
            res = results.get(ft, {})
            if "skipped" in res or "train" not in res:
                raise RuntimeError(f"{ft}: 학습 데이터를 만들지 못함 ({res})")
            override = {"epochs": epochs or RECIPE["epochs"], "batch_size": batch_size or RECIPE["batch_size"],
                        "patience": patience or RECIPE["patience"]}
            d = os.path.join(out_dir, "train_data")
            trainer = Trainer(ft, config_override=override, activate=True, trigger="manual",
                              suspect_stats=res.get("suspect_stats"), alert_policy=alert_policy)
            if not trainer.train(train_path=os.path.join(d, f"{ft}_train.csv"), val_path=os.path.join(d, f"{ft}_test.csv")):
                raise RuntimeError(f"{ft} 학습 실패")
            out[ft] = {**res, "version": trainer.version}
    finally:
        for ft in TRACKS:
            PATHS[ft] = saved[ft]
    return out


# ─────────────────────────────────────────────
# 평가 (V1·V2·V4)
# ─────────────────────────────────────────────
def metrics_row(res):
    """evaluate_alarms 결과 -> 비교에 쓰는 지표 한 줄"""
    ev = res["event"]
    return {"auprc": res["step"]["auprc"], "f1": ev["event_f1"], "early": ev["early_detection_rate"],
            "fp": ev["false_incidents_per_port_day"], "prec": ev["incident_precision"], "lead": ev["lead_time_median_min"]}


def evaluate_models(models, seeds, nodes=6, days=14, policies=None, baselines=True):
    """models: {이름: 모델 폴더}. 검증 시드마다 모델×정책의 지표 행을 만든다 (기준선은 model='기준', 오탐억제형 알람 스텝 기준)."""
    policies = policies or POLICIES
    rows = []
    for seed in seeds:
        data = make_data(seed, nodes, days)
        for name, mdir in models.items():
            scores = score_data(data, mdir)
            for pname, pol in policies.items():
                sim = simulate_alarms(scores, pol)
                rows.append({"model": name, "policy": pname, "seed": seed, **metrics_row(evaluate_alarms(sim, data))})
                if baselines and pname == next(iter(policies)) and name == next(iter(models)):
                    for b in ("rolling_rule", "fixed_threshold", "always_alarm"):
                        r = evaluate_alarms(baseline_alarms(data, sim, b), data)
                        rows.append({"model": f"[기준] {b}", "policy": "-", "seed": seed, **metrics_row(r)})
    return pd.DataFrame(rows)


def judge_v1(df, models):
    """V1: 운영 레시피 모델의 AUPRC 평균 >= 0.587 그리고 오탐억제형 이벤트 F1 평균 >= 0.755 (정책 재선택 없음)"""
    m = df[df.model.isin(models)]
    auprc = m[m.policy == "기본"]["auprc"].mean()
    f1 = m[m.policy == "오탐억제형"]["f1"].mean()
    return {"auprc": float(auprc), "f1_precision": float(f1), "pass": bool(auprc >= V1_MIN_AUPRC and f1 >= V1_MIN_F1)}


def judge_v2(df, models, policy="오탐억제형"):
    """V2: 오탐억제형 오탐 평균 <= 0.02 그리고 이벤트 F1 평균 > 기본 정책 F1 평균 (조기 탐지·리드타임은 병기).
    policy: 판정할 정책 이름 (재선택 후보를 같은 기준으로 판정할 때 지정)"""
    m = df[df.model.isin(models)]
    p, d = m[m.policy == policy], m[m.policy == "기본"]
    out = {"fp_precision": float(p["fp"].mean()), "f1_precision": float(p["f1"].mean()), "f1_default": float(d["f1"].mean()),
           "early_precision": float(p["early"].mean()), "early_default": float(d["early"].mean()),
           "lead_precision": float(p["lead"].median()), "lead_default": float(d["lead"].median())}
    out["pass"] = bool(out["fp_precision"] <= V2_MAX_FP and out["f1_precision"] > out["f1_default"])
    return out


def judge_v4(df, v1_name, v2_models):
    """V4: v1+기본정책 대비 v2+오탐억제형이 AUPRC·이벤트 F1 은 > 0.03 높고 오탐은 > 0.03 낮음 (세 가지 모두)"""
    old = df[(df.model == v1_name) & (df.policy == "기본")]
    new = df[df.model.isin(v2_models) & (df.policy == "오탐억제형")]
    # AUPRC 는 정책과 무관한 순위 품질이라 v2 도 기본 정책 행에서 읽는다
    new_auprc = df[df.model.isin(v2_models) & (df.policy == "기본")]["auprc"].mean()
    d = {"auprc": float(new_auprc - old["auprc"].mean()), "f1": float(new["f1"].mean() - old["f1"].mean()),
         "fp": float(old["fp"].mean() - new["fp"].mean())}        # 오탐은 줄어든 폭
    d["pass"] = bool(all(v > V4_MIN_DIFF for v in d.values()))
    return d


# ─────────────────────────────────────────────
# V2 후속: 정책 재선택 (개발 시드로만) / 검증 시드 1회 확인
# ─────────────────────────────────────────────
def policy_candidates():
    """선택 공간: P0-2 격자 (배율 x σ x 댐프닝 프리셋). 이름 열로 부분집합을 구분할 수 있게 값을 함께 반환"""
    from validation.evaluation.tuning import DAMPING_PRESETS, policy_grid
    return list(policy_grid(threshold_scales=GRID_SCALES, sigma_ks=GRID_SIGMAS, damping_presets=DAMPING_PRESETS))


def reselect_policy(models, seeds, nodes=6, days=14, candidates=None, log=None):
    """개발 시드에서 후보 정책마다 (모델 x 시드) 평균 지표를 구하고 선택 규칙으로 하나를 고른다.
    검증 시드가 섞이면 거부한다(lessons #30). Returns: (표, 선택된 행 또는 None)"""
    if set(int(x) for x in seeds) & set(VAL_SEEDS):
        raise ValueError(f"재선택은 개발 시드 {DEV_SEEDS} 로만 한다 (검증 시드 {VAL_SEEDS} 는 선택 이후 확인용): {list(seeds)}")
    candidates = candidates or policy_candidates()
    rows = []
    for seed in seeds:
        data = make_data(seed, nodes, days)
        for name, mdir in models.items():
            scores = score_data(data, mdir)
            for c in candidates:
                m = metrics_row(evaluate_alarms(simulate_alarms(scores, c["policy"]), data))
                rows.append({"scale": c["threshold_scale"], "sigma": c["sigma_k"], "damping": c["damping"],
                             "model": name, "seed": seed, **m})
            if log:
                log(f"seed {seed} model {name}: {len(candidates)} policies done")
    df = pd.DataFrame(rows)
    table = (df.groupby(["scale", "sigma", "damping"])[["auprc", "f1", "early", "fp", "prec", "lead"]]
             .mean().reset_index().sort_values("f1", ascending=False).reset_index(drop=True))
    return table, pick_policy(table)


def pick_policy(table):
    """선택 규칙(사전 고정): 조기 탐지 >= 70% 이고 오탐 <= 0.02 인 정책 중 이벤트 F1 최대. 없으면 None (기준을 완화하지 않음).
    table 은 F1 내림차순으로 정렬되어 있어야 한다 (동률이면 앞선 행)."""
    ok = table[(table["early"] >= SELECT_MIN_EARLY) & (table["fp"] <= SELECT_MAX_FP)]
    return None if ok.empty else ok.iloc[0]


def policy_from_row(row):
    from validation.evaluation.tuning import DAMPING_PRESETS
    return AlertPolicy(threshold_scale=float(row["scale"]), sigma_k=float(row["sigma"]),
                       dampening_steps=dict(DAMPING_PRESETS[row["damping"]]))


# ─────────────────────────────────────────────
# V3: 게이트 G3 판정식 (C0 현재 식 / C3 추정 오탐 보정 식)
# ─────────────────────────────────────────────
def gate_stats(scores, th, policy, ft):
    """승격 게이트와 같은 계산: (통계 dict, 알람 프레임). 판정 3일 구간만 센다."""
    return promotion_gate.alarm_stats(scores, th, policy, GATE_DAYS, ft)


def fp_incidents_per_port_day(alarms, raw, port_days, rp=None):
    """C3 의 추정 오탐: 모델과 무관한 규칙 B 의심 구간(학습 정제와 같은 함수) 밖의 알람 인시던트 / 포트·일.
    규칙 B 의 직전 24h 중앙값 계산을 위해 raw 는 판정 구간 앞 1일을 포함해 넘긴다."""
    rp = rp or RetrainPolicy()
    iv = train_window.intervals_from_rules(raw, rp)
    a = alarms[alarms["alarm"]].copy()
    if a.empty:
        return 0.0
    inside = train_window.exclusion_mask(a, iv)
    return alerting.count_incidents(a[~inside]) / port_days if port_days else 0.0


def c3_status(cand, active, fp_rate, rp=None):
    """C3 판정: 상대 기준은 전체 인시던트(기존과 같음), 절대 상한(0.05)은 추정 오탐에. 최소 포트·일 가드 포함."""
    rp = rp or RetrainPolicy()
    if cand["port_days"] < rp.gate_min_port_days:
        return "SKIP"
    limit = max(active["incidents_per_port_day"] * rp.gate_max_alarm_ratio, rp.gate_alarm_floor)
    ok = cand["incidents_per_port_day"] <= limit and fp_rate <= rp.gate_max_incidents_per_port_day
    return "PASS" if ok else "FAIL"


# V3 정상·과다 라벨 (설계서 6장, 2026-10-06 2차 재정의): 게이트 식과 독립인 시뮬레이터 정답 기준(truth-FP)
V3_EXCESS_FACTOR, V3_EXCESS_MIN = 3.0, 0.04       # 과다 = truth-FP >= max(3 x 후보① 의 truth-FP, 0.04)
# 후보 풀 5개 (이름, 모델: new=새 모델 / v1=현재 활성, 임계치 배율, 정책: final=최종 정책 / default=기본 정책)
V3_POOL = (("①V1모델+최종정책", "new", 1.0, "final"), ("②V1모델+기본정책", "new", 1.0, "default"),
           ("③V1모델x0.25+최종정책", "new", 0.25, "final"), ("④V1모델x0.1+최종정책", "new", 0.1, "final"),
           ("⑤v1+기본정책", "v1", 1.0, "default"))
V3_ACTIVES = (("a 첫승격(v1+기본)", "⑤v1+기본정책"), ("b 이후(정상v2+최종정책)", "①V1모델+최종정책"))
V3_BASE = V3_POOL[0][0]


def truth_fp_label(fp, base_fp):
    """정상 = truth-FP <= 후보①(기준 구성)의 truth-FP, 과다 = >= max(3 x 기준, 0.04), 그 외 중간(판정 제외, 값만 기록).
    게이트의 상대 기준(1.5배)은 일부러 쓰지 않는다 — 라벨이 판정 식을 그대로 따라가지 않도록."""
    if fp <= base_fp:
        return "정상"
    if fp >= max(V3_EXCESS_FACTOR * base_fp, V3_EXCESS_MIN):
        return "과다"
    return "중간"


def truth_fp_per_port_day(alarms, data, port_days):
    """시뮬레이터 정답 에피소드의 활성 구간 밖 알람 인시던트 / 포트·일 (정답은 검증에서만 쓰고 게이트는 쓰지 않음).
    정의·묶음 규칙은 평가 도구(event_metrics)의 false incident 와 같고, 분모는 게이트와 같은 포트·일이다."""
    n = evaluate_alarms(alarms, data)["event"]["false_incidents"]
    return n / port_days if port_days else 0.0


def c0_status(cand, active, rp=None):
    """참고 열(판정 아님): P1-1 의 식 — 후보 <= min(0.05, max(활성 x 1.5, 0.01)) 이면 PASS, 아니면 FAIL (U4' 이전의 check_g3)"""
    rp = rp or RetrainPolicy()
    if cand["port_days"] < rp.gate_min_port_days:
        return "SKIP"
    limit = min(rp.gate_max_incidents_per_port_day, max(active["incidents_per_port_day"] * rp.gate_max_alarm_ratio, rp.gate_alarm_floor))
    return "PASS" if cand["incidents_per_port_day"] <= limit else "FAIL"


def v3_phase(seeds):
    """개발 시드 7·11(부분집합 가능)은 식을 확인하는 '탐색'(판정 근거 아님), 검증 시드 23·31·47 **전체**는 1회만 쓰는 '근거'.
    섞거나 검증 시드의 일부만 주면 거부한다 (시드를 쪼개 여러 번 보는 우회 차단)."""
    ss = {int(x) for x in seeds}
    if ss and ss <= set(DEV_SEEDS):
        return "탐색"
    if ss == set(VAL_SEEDS):
        return "근거"
    raise ValueError(f"V3 시드는 개발 {DEV_SEEDS}(탐색, 일부 가능) 또는 검증 {VAL_SEEDS} 전체(근거, 1회) 중 하나여야 합니다: {sorted(ss)}")


def evidence_lock(path, info):
    """근거 실행의 1회 잠금을 측정 전에 만든다 (고정 위치, 배타적 생성). 이미 있으면 SystemExit — 측정은 시작되지 않는다."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        fh = open(path, "x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"[!] {path} 가 이미 있습니다 — 검증 시드(근거) 실행은 1회만 허용합니다. 기준·식·정책을 다시 고르지 않습니다.")
    import json
    with fh:
        json.dump({"status": "started", **info}, fh, ensure_ascii=False)
    return path


def v3_table(model_dir, active_models_dir, seeds, nodes=6, densities=None, tracks=TRACKS, weights=None, final="오탐억제형"):
    """V3 표: (시드 x 장애 밀도 x 트랙) 게이트 데이터마다 후보 풀 5개 x 활성 2가지의 G3 판정.
    판정 열(G3)은 src 의 promotion_gate.check_g3(C4') 를 그대로 호출한다. truth-FP 라벨은 게이트와 독립(시뮬레이터 정답).
    C0·C3 는 참고 열이다 (C3 는 도구 구현값)."""
    densities = densities or {"기준(7일 간격)": 7.0, "잦음(2일 간격)": 2.0}
    rp = RetrainPolicy()
    policies = {"final": POLICIES[final], "default": POLICIES["기본"]}
    rows = []
    for seed in seeds:
        for dname, gap in densities.items():
            extra = {"scenario_weights": weights} if weights else {}
            data = make_data(seed, nodes, GATE_DAYS + 1, mean_gap_days=gap, **extra)
            since = pd.Timestamp(ScenarioConfig().start) + pd.Timedelta(days=1)
            scored = {"new": score_data(data, model_dir), "v1": score_data(data, active_models_dir)}
            for ft in tracks:
                cand = {}
                for name, which, factor, pol in V3_POOL:
                    sc, th = scored[which][ft]
                    sc = sc[pd.to_datetime(sc["occur_date"]) >= since]
                    stats, alarms = gate_stats(sc, th * factor, policies[pol], ft)
                    fp = truth_fp_per_port_day(alarms, data, stats["port_days"])
                    cand[name] = {"stats": stats, "alarms": alarms, "truth_fp": fp}
                base = cand[V3_BASE]["truth_fp"]
                for name, *_ in V3_POOL:
                    c = cand[name]
                    label = "정상" if name == V3_BASE else truth_fp_label(c["truth_fp"], base)
                    cfp = fp_incidents_per_port_day(c["alarms"], data[ft], c["stats"]["port_days"], rp)
                    for aname, aref in V3_ACTIVES:
                        act = cand[aref]["stats"]
                        g3 = promotion_gate.check_g3(c["stats"], act, rp)
                        rows.append({"seed": seed, "density": dname, "track": ft, "candidate": name, "active": aname,
                                     "label": label, "truth_fp": c["truth_fp"], "base_fp": base,
                                     "total": c["stats"]["incidents_per_port_day"], "active_total": act["incidents_per_port_day"],
                                     "port_days": c["stats"]["port_days"], "G3": g3.status, "G3_value": g3.value, "G3_limit": g3.limit,
                                     "C0": c0_status(c["stats"], act, rp), "C3": c3_status(c["stats"], act, cfp, rp)})
    return pd.DataFrame(rows)


def judge_v3(table):
    """V3 합격 기준 (설계서 6장). 정상 후보: 모든 조건·활성에서 G3 가 FAIL 이 아님. 과다 후보: 활성 a(첫 승격)에서 PASS 가 아님,
    활성 b(이후 재학습)에서 FAIL. 조건(시드 x 밀도 x 트랙)에 과다 라벨이 하나도 없으면 그 조건은 과다 기준의 판정 불가로 기록하고
    정상 기준은 그대로 적용한다. 과다 라벨이 전 조건에 없으면 pass=None(판정 불가)."""
    normal = table[table.label == "정상"]
    excess = table[table.label == "과다"]
    a, b = excess[excess.active == V3_ACTIVES[0][0]], excess[excess.active == V3_ACTIVES[1][0]]
    out = {"normal_rows": len(normal), "excess_rows_a": len(a), "excess_rows_b": len(b),
           "normal_violations": int((normal.G3 == "FAIL").sum()),
           "excess_a_violations": int((a.G3 == "PASS").sum()),
           "excess_b_violations": int((b.G3 != "FAIL").sum())}
    cond = ["seed", "density", "track"]
    has_excess = table.groupby(cond)["label"].apply(lambda s: bool((s == "과다").any()))
    out["conditions"] = int(len(has_excess))
    out["conditions_without_excess"] = [tuple(k) for k, v in has_excess.items() if not v]
    out["normal_ok"] = out["normal_violations"] == 0
    out["excess_ok"] = (out["excess_a_violations"] == 0 and out["excess_b_violations"] == 0) if len(excess) else None
    out["pass"] = None if not len(excess) else bool(out["normal_ok"] and out["excess_ok"])
    return out


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _fmt(df, cols):
    pd.set_option("display.width", 200)
    g = df.groupby(["model", "policy"])[cols]
    return g.agg(lambda s: f"{s.mean():.3f}±{s.std(ddof=0):.3f}" if s.notna().any() else "-").to_string()


def _models_arg(text):
    return dict(kv.split("=", 1) for kv in text.split(","))


def cmd_train(a):
    res = train_operational(a.out, a.data_seed, a.train_seed, a.nodes, a.days, a.epochs, a.active_models)
    for ft, r in res.items():
        print(f"[OK] {ft}: train {r['train']:,} / val {r['test']:,} rows, 의심 비율 {r['suspect_stats']['fraction']:.1%} -> {a.out}")


def cmd_eval(a):
    models = _models_arg(a.models)
    df = evaluate_models(models, [int(s) for s in a.seeds.split(",")], a.nodes, a.days)
    cols = ["auprc", "f1", "early", "fp", "prec", "lead"]
    print(f"== 검증 시드 {a.seeds} 평균±표준편차 (모델 {list(models)}) ==")
    print(_fmt(df, cols))
    v1, v2 = judge_v1(df, list(models)), judge_v2(df, list(models))
    print(f"\n[V1 운영 레시피] AUPRC {v1['auprc']:.3f} (>= {V1_MIN_AUPRC}), 오탐억제형 F1 {v1['f1_precision']:.3f} (>= {V1_MIN_F1}) -> {'PASS' if v1['pass'] else 'FAIL'}")
    print(f"[V2 정책] 오탐억제형 오탐 {v2['fp_precision']:.3f} (<= {V2_MAX_FP}), F1 {v2['f1_precision']:.3f} > 기본 {v2['f1_default']:.3f} -> {'PASS' if v2['pass'] else 'FAIL'}")
    print(f"        병기(기준 없음): 조기 탐지 {v2['early_precision']:.1%} (기본 {v2['early_default']:.1%}), 리드타임 중앙값 {v2['lead_precision']:.0f}분 (기본 {v2['lead_default']:.0f}분)")


def cmd_gate(a):
    import json
    seeds = [int(x) for x in a.seeds.split(",")]
    phase = v3_phase(seeds)
    lock = None
    if phase == "근거":                                   # 검증 시드: 전체 집합만, 잠금은 모델 폴더 안 고정 파일(--record 와 무관), 측정 전에 잠금
        lock = evidence_lock(os.path.join(a.model, V3_LOCK), {"kind": "V3", "seeds": seeds, "model": a.model, "final_policy": a.final_policy})
    weights = (0.0, 0.0, 1.0) if a.traffic_drop_only else None
    t = v3_table(a.model, a.active_models, seeds, a.nodes, weights=weights, final=a.final_policy)
    pd.set_option("display.width", 220)
    show = ["seed", "density", "track", "candidate", "active", "label", "truth_fp", "base_fp", "total", "G3", "C0", "C3"]
    print(f"== V3 [{phase}{' — 판정 근거 아님' if phase == '탐색' else ' — 근거 판정(1회)'}] 시드 {seeds}, 최종 정책 = {a.final_policy} ==")
    print(t[show].round(4).to_string(index=False))
    j = judge_v3(t)
    print(f"\n[V3 {phase}] 정상 행 {j['normal_rows']}건 중 G3 FAIL {j['normal_violations']}건 / 과다 행(a) {j['excess_rows_a']}건 중 PASS {j['excess_a_violations']}건 / "
          f"과다 행(b) {j['excess_rows_b']}건 중 FAIL 아님 {j['excess_b_violations']}건")
    for ft in sorted(t.track.unique()):                  # 트랙별 정상·과다 라벨 수 (후보 x 활성 행 기준이 아니라 후보 단위)
        lab = t[(t.track == ft) & (t.active == V3_ACTIVES[0][0])].label.value_counts().to_dict()
        print(f"        {ft}: 정상 {lab.get('정상', 0)} / 과다 {lab.get('과다', 0)} / 중간 {lab.get('중간', 0)} (후보 x 시드 x 밀도 단위)")
    print(f"        과다 라벨이 없는 조건(판정 불가) {len(j['conditions_without_excess'])}/{j['conditions']}: {j['conditions_without_excess']}")
    print("        판정: " + ("판정 불가(과다 라벨 없음)" if j["pass"] is None else ("PASS" if j["pass"] else "FAIL")) +
          ("   ※ 탐색 결과이며 판정 근거가 아닙니다" if phase == "탐색" else ""))
    print("C0·C3 열은 참고용입니다 (판정은 src check_g3 = C4').")
    rec = {"phase": phase, "seeds": seeds, "final_policy": a.final_policy, "judgement": {k: v for k, v in j.items() if k != "conditions_without_excess"},
           "conditions_without_excess": [list(map(str, x)) for x in j["conditions_without_excess"]]}
    for out in (lock, a.record):                          # 잠금 파일에 결과를 기록하고, --record 는 사본 출력용
        if out:
            json.dump(rec, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
            t.to_csv(out + ".csv", index=False)


def cmd_vs_v1(a):
    models = _models_arg(a.models)
    v1 = {"v1": a.v1_models}
    df = evaluate_models({**v1, **models}, [int(s) for s in a.seeds.split(",")], a.nodes, a.days)
    print(_fmt(df, ["auprc", "f1", "early", "fp", "prec", "lead"]))
    d = judge_v4(df, "v1", list(models))
    print(f"\n[V4] AUPRC +{d['auprc']:.3f}, 이벤트 F1 +{d['f1']:.3f}, 오탐 -{d['fp']:.3f} (각 > {V4_MIN_DIFF}) -> {'PASS' if d['pass'] else 'FAIL'}")


def cmd_select(a):
    import json
    models = _models_arg(a.models)
    seeds = [int(s) for s in a.seeds.split(",")]
    table, chosen = reselect_policy(models, seeds, a.nodes, a.days, log=lambda m: print(f"[*] {m}", flush=True))
    pd.set_option("display.width", 200)
    if a.out_csv:
        table.to_csv(a.out_csv, index=False)
    print(f"== 개발 시드 {seeds} x 모델 {list(models)} 평균, 선택 규칙: 조기 >= {SELECT_MIN_EARLY:.0%} & 오탐 <= {SELECT_MAX_FP} 중 F1 최대 ==")
    print(f"후보 {len(table)}개 중 규칙 충족 {int(((table['early'] >= SELECT_MIN_EARLY) & (table['fp'] <= SELECT_MAX_FP)).sum())}개")
    print("\n[F1 상위 8 (규칙 무관)]"); print(table.head(8).round(3).to_string(index=False))
    ok = table[(table["early"] >= SELECT_MIN_EARLY) & (table["fp"] <= SELECT_MAX_FP)]
    print("\n[규칙 충족 후보 상위 8]"); print(ok.head(8).round(3).to_string(index=False) if len(ok) else "(없음)")
    # 설계 4.4 의 부분집합: 현재 프리셋 2종(default, precision)의 임계치 배율 격자
    sub = table[((table.sigma == 3.0) & (table.damping == "default(3/2/1)")) | ((table.sigma == 2.0) & (table.damping == "heavy(6/4/3)"))]
    sub_ok = sub[(sub["early"] >= SELECT_MIN_EARLY) & (sub["fp"] <= SELECT_MAX_FP)]
    print("\n[부분집합: 프리셋 2종 x 배율 격자 — 규칙 충족]"); print(sub_ok.round(3).to_string(index=False) if len(sub_ok) else "(없음)")
    cur = table[(table.scale == 3.0) & (table.sigma == 2.0) & (table.damping == "heavy(6/4/3)")]
    print("\n[현재 오탐억제형 프리셋(3.0/σ2/heavy)의 개발 시드 값]"); print(cur.round(3).to_string(index=False))
    if chosen is None:
        print("\n[결과] 선택 규칙을 만족하는 정책이 없음 — 기준을 완화하지 않고 보고 후 중단")
        return
    print(f"\n[선택] 배율 {chosen['scale']:g} / σ {chosen['sigma']:g} / 댐프닝 {chosen['damping']}  "
          f"(F1 {chosen['f1']:.3f}, 조기 {chosen['early']:.1%}, 오탐 {chosen['fp']:.3f}, 리드 {chosen['lead']:.0f}분)")
    if a.out_json:
        json.dump({"scale": float(chosen["scale"]), "sigma": float(chosen["sigma"]), "damping": chosen["damping"]},
                  open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False)


def cmd_confirm(a):
    import json
    seeds = [int(x) for x in a.seeds.split(",")]
    if set(seeds) != set(VAL_SEEDS):                      # 근거 실행: 검증 시드 전체만 (부분집합·개발 시드 혼입 거부)
        raise SystemExit(f"[!] confirm 은 검증 시드 {VAL_SEEDS} 전체로만 실행합니다 (받은 값: {sorted(seeds)}).")
    models = _models_arg(a.models)
    sel = json.load(open(a.policy_json, encoding="utf-8"))
    lock = evidence_lock(os.path.join(os.path.dirname(os.path.abspath(a.policy_json)), CONFIRM_LOCK),
                         {"kind": "confirm", "seeds": seeds, "models": a.models, "policy": sel})
    cand = policy_from_row(sel)
    policies = {"기본": POLICIES["기본"], "오탐억제형(기존)": POLICIES["오탐억제형"], "후보": cand}
    df = evaluate_models(models, seeds, a.nodes, a.days, policies=policies)
    print(f"== 검증 시드 {a.seeds} 평균±표준편차 (모델 {list(models)}) | 후보 = 배율 {sel['scale']:g}/σ {sel['sigma']:g}/{sel['damping']} ==")
    print(_fmt(df, ["auprc", "f1", "early", "fp", "prec", "lead"]))
    judged = {}
    for name in ("오탐억제형(기존)", "후보"):
        j = judge_v2(df, list(models), policy=name)
        judged[name] = j
        print(f"[V2 판정 — {name}] 오탐 {j['fp_precision']:.3f} (<= {V2_MAX_FP}), F1 {j['f1_precision']:.3f} > 기본 {j['f1_default']:.3f}; "
              f"조기 {j['early_precision']:.1%} (기본 {j['early_default']:.1%}), 리드 {j['lead_precision']:.0f}분 (기본 {j['lead_default']:.0f}분) -> {'PASS' if j['pass'] else 'FAIL'}")
    for out in (lock, a.record):                          # 잠금 파일에 결과를 기록하고, --record 는 사본 출력용
        if out:
            json.dump({"status": "done", "kind": "confirm", "seeds": seeds, "policy": sel, "judgement": judged},
                      open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sl = sub.add_parser("select")
    sl.add_argument("--models", required=True)
    sl.add_argument("--seeds", default=",".join(map(str, DEV_SEEDS)))
    sl.add_argument("--nodes", type=int, default=6)
    sl.add_argument("--days", type=int, default=14)
    sl.add_argument("--out-csv", default=None)
    sl.add_argument("--out-json", default=None, help="선택된 정책 저장 (confirm 입력)")
    cf = sub.add_parser("confirm")
    cf.add_argument("--models", required=True)
    cf.add_argument("--policy-json", required=True)
    cf.add_argument("--seeds", default=",".join(map(str, VAL_SEEDS)))
    cf.add_argument("--record", default=None, help="결과 사본 저장 파일(선택). 1회 잠금은 정책 JSON 옆 v2_confirm.json")
    cf.add_argument("--nodes", type=int, default=6)
    cf.add_argument("--days", type=int, default=14)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--data-seed", type=int, default=TRAIN_SEEDS[0])
    t.add_argument("--train-seed", type=int, default=0)
    t.add_argument("--nodes", type=int, default=4)
    t.add_argument("--days", type=int, default=21)
    t.add_argument("--epochs", type=int, default=None, help="기본 30 (설계서 레시피). 시험용으로만 줄임")
    t.add_argument("--active-models", default="models", help="규칙 A 의 활성 모델(v1) 폴더")
    e = sub.add_parser("eval")
    e.add_argument("--models", required=True)
    e.add_argument("--seeds", default=",".join(map(str, VAL_SEEDS)))
    e.add_argument("--nodes", type=int, default=6)
    e.add_argument("--days", type=int, default=14)
    g = sub.add_parser("gate")
    g.add_argument("--model", required=True)
    g.add_argument("--active-models", default="models")
    g.add_argument("--seeds", required=True, help=f"개발 {DEV_SEEDS} 로 식을 고르고, 확인 {VAL_SEEDS} 로 확인")
    g.add_argument("--nodes", type=int, default=6)
    g.add_argument("--traffic-drop-only", action="store_true", help="추가 기록용(판정 아님): traffic_drop 위주 조건")
    g.add_argument("--final-policy", default="오탐억제형", choices=list(POLICIES), help="후보 풀의 '최종 정책' (V2 에서 확정된 정책)")
    g.add_argument("--record", default=None, help="결과 사본을 추가로 저장할 파일(선택). 1회 잠금은 --record 와 무관하게 모델 폴더의 v3_evidence.json")
    v = sub.add_parser("vs-v1")
    v.add_argument("--models", required=True)
    v.add_argument("--v1-models", default="models")
    v.add_argument("--seeds", default=",".join(map(str, VAL_SEEDS)))
    v.add_argument("--nodes", type=int, default=6)
    v.add_argument("--days", type=int, default=14)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "gate": cmd_gate, "vs-v1": cmd_vs_v1,
     "select": cmd_select, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
