"""
P1-3 v2 승격 검증 도구 (설계서 docs/design/p1_3_model_promotion.md 6장 V1~V4)

바꾸는 것은 단계마다 한 가지뿐이다 (lessons #29). 합격 기준·시드는 데이터를 보기 전에 설계서에 고정된 값이며
이 파일의 상수로 옮겨 두었다 — 결과를 본 뒤 바꾸지 않는다. 판정에 쓰는 함수(알람 계산, 인시던트 집계, 게이트 G3,
의심 구간 규칙 B, 포트 분할, 수집·학습)는 모두 운영 코드(src)를 그대로 호출한다 (lessons #31).
모든 수치는 시뮬레이션 기준이며 실데이터 검증이 아니다.

  train  (V1 학습)   운영 레시피로 학습: 규칙 B + 자기 알람(활성 모델 v1 의 알람) 의심 구간 제외 + 포트 홀드아웃 10%,
                     DataCollector·train_window·Trainer 를 그대로 사용 (DB 대신 시뮬레이터 데이터를 주입)
  eval   (V1·V2)     학습한 모델을 검증 시드에서 평가: AUPRC·이벤트 지표, 정책 기본 ↔ 오탐 억제형
  gate   (V3)        게이트 G3 판정식 비교: 현재 식(C0) ↔ 추정 오탐 보정 식(C3), 정상/과다 알람 후보
  vs-v1  (V4)        v1 + 기본 정책 ↔ 새 모델 + 오탐 억제형, 기준선 병기

사용법:
  python validation/cli/check_promotion.py train --out validation/runs/promo_s101 --data-seed 101 --train-seed 0
  python validation/cli/check_promotion.py eval --models s101=validation/runs/promo_s101,s103=validation/runs/promo_s103
  python validation/cli/check_promotion.py gate --model validation/runs/promo_s101 --seeds 7,11
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
DEV_SEEDS = (7, 11)                 # 선택(정책·식)은 개발 시드로만
VAL_SEEDS = (23, 31, 47)            # 확인은 검증 전용 시드로만
TRAIN_SEEDS = (101, 103)            # 학습 시드 (학습 난수·데이터 영향)
RECIPE = {"epochs": 30, "batch_size": 64, "patience": 5}     # D1: 실험 레시피

# 합격 기준 (설계서 6장, 2026-10-06 고정). 기준선 = P1-1 7.2 `auto`(AUPRC 0.637, 오탐 억제형 이벤트 F1 0.805)의 -0.05
V1_MIN_AUPRC, V1_MIN_F1 = 0.587, 0.755
V2_MAX_FP = 0.02                    # 오탐 억제형의 오탐 평균 상한 (건/포트·일)
V4_MIN_DIFF = 0.03                  # v1 대비 AUPRC·F1 상승, 오탐 감소 폭 (각각 > 0.03)
EXCESS_THRESHOLD_FACTOR = 0.5       # V3 과다 알람 후보 = 임계치 x 0.5
GATE_DAYS = 3                       # 판정 3일 + 규칙 B 기준선 1일 (설계서 V3 세부)

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


def judge_v2(df, models):
    """V2: 오탐억제형 오탐 평균 <= 0.02 그리고 이벤트 F1 평균 > 기본 정책 F1 평균 (조기 탐지·리드타임은 병기)"""
    m = df[df.model.isin(models)]
    p, d = m[m.policy == "오탐억제형"], m[m.policy == "기본"]
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
# V3: 게이트 G3 판정식 (C0 현재 식 / C3 추정 오탐 보정 식)
# ─────────────────────────────────────────────
def gate_stats(scores, th, policy, ft):
    """승격 게이트와 같은 계산: (통계 dict, 알람 프레임). 판정 3일 구간만 센다."""
    return promotion_gate._alarm_stats(scores, th, policy, GATE_DAYS, ft)


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


def gate_table(model_dir, active_models_dir, seeds, nodes=6, densities=None, tracks=TRACKS, weights=None):
    """V3 표: 시드 x 장애 밀도 x 트랙 x 후보(정상/과다)의 C0·C3 판정.
    정상 후보 = 새 모델 + 오탐억제형, 과다 알람 후보 = 같은 모델의 임계치 x0.5 + 오탐억제형, 활성 = v1 + 기본 정책."""
    densities = densities or {"기준(7일 간격)": 7.0, "잦음(2일 간격)": 2.0}
    rp = RetrainPolicy()
    rows = []
    for seed in seeds:
        for dname, gap in densities.items():
            extra = {"scenario_weights": weights} if weights else {}
            data = make_data(seed, nodes, GATE_DAYS + 1, mean_gap_days=gap, **extra)
            since = pd.Timestamp(ScenarioConfig().start) + pd.Timedelta(days=1)
            cand_scores, act_scores = score_data(data, model_dir), score_data(data, active_models_dir)
            for ft in tracks:
                raw = data[ft]
                for kind, factor in (("정상", 1.0), ("과다", EXCESS_THRESHOLD_FACTOR)):
                    sc, th = cand_scores[ft]
                    sc = sc[pd.to_datetime(sc["occur_date"]) >= since]
                    cand, alarms = gate_stats(sc, th * factor, POLICIES["오탐억제형"], ft)
                    a_sc, a_th = act_scores[ft]
                    a_sc = a_sc[pd.to_datetime(a_sc["occur_date"]) >= since]
                    active, _ = gate_stats(a_sc, a_th, POLICIES["기본"], ft)
                    fp = fp_incidents_per_port_day(alarms, raw, cand["port_days"], rp)
                    rows.append({"seed": seed, "density": dname, "track": ft, "candidate": kind,
                                 "total": cand["incidents_per_port_day"], "fp": fp, "active": active["incidents_per_port_day"],
                                 "port_days": cand["port_days"],
                                 "C0": promotion_gate.check_g3(cand, active, rp).status, "C3": c3_status(cand, active, fp, rp)})
    return pd.DataFrame(rows)


def judge_v3(table, formula):
    """식(C0 또는 C3)이 모든 조건에서 정상 후보 PASS, 과다 알람 후보 FAIL 인가"""
    ok_normal = bool((table[table.candidate == "정상"][formula] == "PASS").all())
    ok_excess = bool((table[table.candidate == "과다"][formula] == "FAIL").all())
    return {"normal_all_pass": ok_normal, "excess_all_fail": ok_excess, "pass": ok_normal and ok_excess}


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
    seeds = [int(s) for s in a.seeds.split(",")]
    weights = (0.0, 0.0, 1.0) if a.traffic_drop_only else None
    t = gate_table(a.model, a.active_models, seeds, a.nodes, weights=weights)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    for f in ("C0", "C3"):
        j = judge_v3(t, f)
        print(f"[V3 {f}] 정상 후보 전부 PASS: {j['normal_all_pass']}, 과다 알람 후보 전부 FAIL: {j['excess_all_fail']} -> {'만족' if j['pass'] else '불만족'}")
    print("선택 규칙: 개발 시드(7,11)에서 C0 가 만족하면 C0(보정 없음), 아니면 C3. 확인 시드(23,31,47)로 식을 다시 고르지 않는다.")


def cmd_vs_v1(a):
    models = _models_arg(a.models)
    v1 = {"v1": a.v1_models}
    df = evaluate_models({**v1, **models}, [int(s) for s in a.seeds.split(",")], a.nodes, a.days)
    print(_fmt(df, ["auprc", "f1", "early", "fp", "prec", "lead"]))
    d = judge_v4(df, "v1", list(models))
    print(f"\n[V4] AUPRC +{d['auprc']:.3f}, 이벤트 F1 +{d['f1']:.3f}, 오탐 -{d['fp']:.3f} (각 > {V4_MIN_DIFF}) -> {'PASS' if d['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
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
    v = sub.add_parser("vs-v1")
    v.add_argument("--models", required=True)
    v.add_argument("--v1-models", default="models")
    v.add_argument("--seeds", default=",".join(map(str, VAL_SEEDS)))
    v.add_argument("--nodes", type=int, default=6)
    v.add_argument("--days", type=int, default=14)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "gate": cmd_gate, "vs-v1": cmd_vs_v1}[a.cmd](a)


if __name__ == "__main__":
    main()
