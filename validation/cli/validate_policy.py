"""
검증 전용 시드 평가: 정책을 개발에 쓰지 않은 데이터셋에서 평가하고 기본 정책/베이스라인과 비교

튜닝(tune_alerting.py)은 개발용 데이터셋에서만 수행하고, 고른 정책은 여기서 처음 보는 시드로 검증한다.
(같은 데이터로 고르고 같은 데이터로 평가하면 과대평가되므로 분리)

사용법:
  python validation/cli/validate_policy.py --models-dir validation/runs/models_noisy_e100 --tag e100 \\
      --datasets seed23_n6_d14,seed31_n6_d14,seed47_n6_d14 --scale 3 --sigma 2 --damping "heavy(6/4/3)"
출력: 데이터셋별/평균±표준편차 표 (이벤트 지표), 기본 정책·기준선과 같은 채점
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from src.pipeline.alerting import AlertPolicy
from validation.evaluation.tuning import DAMPING_PRESETS, baseline_alarms, evaluate_alarms, get_scores, simulate_alarms

METRICS = [("early_detection_rate", "조기탐지", "pct"), ("episode_detection_rate", "탐지", "pct"),
           ("lead_time_median_min", "리드(분)", "num"), ("false_incidents_per_port_day", "오탐/포트일", "f3"),
           ("incident_precision", "이벤트P", "pct"), ("event_f1", "이벤트F1", "f3")]


def fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return {"pct": f"{v:.1%}", "num": f"{v:.0f}", "f3": f"{v:.3f}"}[kind]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", required=True, help="검증 전용 데이터셋(쉼표 구분)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--damping", default="default(3/2/1)", choices=list(DAMPING_PRESETS))
    ap.add_argument("--json", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    chosen = AlertPolicy(threshold_scale=args.scale, sigma_k=args.sigma, dampening_steps=dict(DAMPING_PRESETS[args.damping]))
    label = f"AI[배율{args.scale:g}/σ{args.sigma:g}/{args.damping}]"
    methods = {label: chosen, "AI[현재 기본값]": AlertPolicy()}

    per_ds = {}
    for ds in [x for x in args.datasets.split(",") if x]:
        data, scores = get_scores(ds, args.models_dir, args.tag)
        res = {}
        sims = {}
        for name, pol in methods.items():
            sims[name] = simulate_alarms(scores, pol)
            res[name] = evaluate_alarms(sims[name], data)["event"]
        rows = sims[label]
        for b in ("rolling_rule", "fixed_threshold", "always_alarm"):
            res[f"[기준] {b}"] = evaluate_alarms(baseline_alarms(data, rows, b), data)["event"]
        per_ds[ds] = res
        e = data["episodes"]
        print(f"\n== {ds} (에피소드 {len(e)}건) ==")
        print(f"{'방법':<34}" + "".join(f"{m[1]:>11}" for m in METRICS))
        for name, ev in res.items():
            print(f"{name:<34}" + "".join(f"{fmt(ev[m[0]], m[2]):>11}" for m in METRICS))

    names = list(next(iter(per_ds.values())).keys())
    print(f"\n==== 검증 전용 데이터셋 {len(per_ds)}개 평균 ± 표준편차 ====")
    print(f"{'방법':<34}" + "".join(f"{m[1]:>16}" for m in METRICS))
    summary = {}
    for name in names:
        cells, summary[name] = [], {}
        for key, _, kind in METRICS:
            vals = [per_ds[ds][name][key] for ds in per_ds if per_ds[ds][name][key] is not None]
            mu, sd = (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)
            summary[name][key] = {"mean": mu, "std": sd}
            cells.append("-" if mu is None else (f"{mu:.1%}±{sd:.1%}" if kind == "pct" else (f"{mu:.0f}±{sd:.0f}" if kind == "num" else f"{mu:.3f}±{sd:.3f}")))
        print(f"{name:<34}" + "".join(f"{c:>16}" for c in cells))
    if args.json:
        json.dump({"policy": {"scale": args.scale, "sigma": args.sigma, "damping": args.damping},
                   "datasets": list(per_ds), "summary": summary}, open(args.json, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
        print(f"\n[*] 저장: {args.json}")


if __name__ == "__main__":
    main()
