"""
시뮬레이션 가정 민감도 평가: 노이즈 / 결측 / 장애 강도 / 유병률 / 장애 유형 비율을 바꿔 성능 범위를 측정

실데이터가 없으면 점수는 시뮬레이터의 가정(장애 모양, 노이즈 크기)의 함수이므로, 단일 수치 대신
"어떤 가정에서 성능이 어떻게 변하는가"를 보고한다. 선택된 모델/정책과 기준선을 같은 채점으로 비교한다.

사용법:
  python validation/cli/sensitivity.py --models-dir validation/runs/models_noisy --scale 3 --sigma 2 --damping "heavy(6/4/3)"
  python validation/cli/sensitivity.py ... --seeds 101,102 --json validation/runs/sensitivity.json
시드는 튜닝(7, 11)과 검증(23, 31, 47)에 쓰지 않은 값을 쓴다 (기본 101, 102).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from src.pipeline.alerting import AlertPolicy
from validation.evaluation.tuning import (DAMPING_PRESETS, baseline_alarms, compute_track_scores, evaluate_alarms,
                                          simulate_alarms)
from validation.simulator.scenario_generator import ScenarioConfig, generate

BASE = dict(nodes=6, days=14)


def variants():
    """이름 -> ScenarioConfig 덮어쓰기. 기본(baseline) 대비 한 번에 한 가지 가정만 바꾼다."""
    d = ScenarioConfig()
    scale_noise = lambda k: dict(benign_error_prob=min(0.5, d.benign_error_prob * k), burst_prob=d.burst_prob * k,
                                 blip_prob=d.blip_prob * k)
    return {
        "baseline": {},
        "노이즈 없음 (0x)": scale_noise(0),
        "노이즈 2배": scale_noise(2),
        "노이즈 4배": scale_noise(4),
        "결측 5% (기본 0.5%)": dict(missing_prob=0.05),
        "장애 약함 (CRC 20~200, 광 2~6dB, 감소 40~70%)": dict(crc_final_errors=(20, 200), optical_final_drop_db=(2.0, 6.0),
                                                       traffic_drop_fraction=(0.4, 0.7)),
        "장애 강함 (CRC 500~5000, 광 10~25dB, 감소 80~95%)": dict(crc_final_errors=(500, 5000), optical_final_drop_db=(10.0, 25.0),
                                                           traffic_drop_fraction=(0.8, 0.95)),
        "장애 드묾 (평균 21일 간격)": dict(mean_gap_days=21.0),
        "장애 잦음 (평균 2일 간격)": dict(mean_gap_days=2.0),
        "광 열화 위주 (10/80/10)": dict(scenario_weights=(0.1, 0.8, 0.1)),
        "트래픽 감소 위주 (20/10/70)": dict(scenario_weights=(0.2, 0.1, 0.7)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--damping", default="heavy(6/4/3)", choices=list(DAMPING_PRESETS))
    ap.add_argument("--seeds", default="101,102")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.models_dir:
        from src.config import PATHS
        for ft in ("traffic", "optical"):
            PATHS[ft] = {"model": os.path.join(args.models_dir, f"{ft}_ae.pth"),
                         "scaler": os.path.join(args.models_dir, f"{ft}_scaler.joblib")}
    from src.pipeline.inference import AnomalyDetector
    detector = AnomalyDetector()
    policy = AlertPolicy(threshold_scale=args.scale, sigma_k=args.sigma, dampening_steps=dict(DAMPING_PRESETS[args.damping]))
    default_policy = AlertPolicy()
    seeds = [int(x) for x in args.seeds.split(",")]

    methods = ["AI(선택 정책)", "AI(기본 정책)", "rolling_rule", "fixed_threshold"]
    results = {}
    t0 = time.time()
    for name, over in variants().items():
        per = {m: [] for m in methods}
        info = []
        for seed in seeds:
            cfg = ScenarioConfig(seed=seed, **BASE, **over)
            data = generate(cfg)
            scores = compute_track_scores(detector, data["traffic"], data["optical"])
            sim = simulate_alarms(scores, policy)
            per["AI(선택 정책)"].append(evaluate_alarms(sim, data)["event"])
            per["AI(기본 정책)"].append(evaluate_alarms(simulate_alarms(scores, default_policy), data)["event"])
            for b in ("rolling_rule", "fixed_threshold"):
                per[b].append(evaluate_alarms(baseline_alarms(data, sim, b), data)["event"])
            info.append((len(data["episodes"]), float((data["labels"]["state"] > 0).mean())))
        agg = {}
        for m in methods:
            agg[m] = {k: float(np.mean([e[k] for e in per[m] if e[k] is not None])) if any(e[k] is not None for e in per[m]) else None
                      for k in ("early_detection_rate", "episode_detection_rate", "false_incidents_per_port_day", "event_f1")}
        results[name] = {"episodes": float(np.mean([i[0] for i in info])), "prevalence": float(np.mean([i[1] for i in info])), "methods": agg}
        print(f"[*] {name}: 완료 ({time.time() - t0:.0f}s)", flush=True)

    # ---- 보고 ----
    print(f"\n정책: 배율 {args.scale:g} / σ {args.sigma:g} / {args.damping} | 시드 {seeds} | 데이터셋당 6노드x10포트 x 14일")
    print(f"\n{'조건':<44}{'에피소드':>7}{'유병률':>7} | {'AI F1':>6}{'조기':>7}{'오탐/일':>8} | {'기본정책 F1':>11} | {'롤링 F1':>8}{'조기':>7} | {'고정 F1':>8}")
    print("-" * 140)
    for name, r in results.items():
        a, d0, rr, ff = (r["methods"][m] for m in methods)
        print(f"{name:<44}{r['episodes']:>7.0f}{r['prevalence']:>7.1%} | {a['event_f1']:>6.3f}{a['early_detection_rate']:>7.1%}{a['false_incidents_per_port_day']:>8.3f}"
              f" | {d0['event_f1']:>11.3f} | {rr['event_f1']:>8.3f}{rr['early_detection_rate']:>7.1%} | {ff['event_f1']:>8.3f}")
    f1s = [r["methods"]["AI(선택 정책)"]["event_f1"] for r in results.values()]
    rf = [r["methods"]["rolling_rule"]["event_f1"] for r in results.values()]
    ef = [r["methods"]["AI(선택 정책)"]["early_detection_rate"] for r in results.values()]
    print("-" * 140)
    print(f"AI(선택 정책) 이벤트 F1 범위 {min(f1s):.3f} ~ {max(f1s):.3f} (중앙 {np.median(f1s):.3f}) | 조기 탐지율 범위 {min(ef):.1%} ~ {max(ef):.1%}")
    print(f"롤링 규칙     이벤트 F1 범위 {min(rf):.3f} ~ {max(rf):.3f} (중앙 {np.median(rf):.3f})")
    worst = min(results.items(), key=lambda kv: kv[1]["methods"]["AI(선택 정책)"]["event_f1"])
    print(f"AI 가 가장 취약한 조건: {worst[0]} (F1 {worst[1]['methods']['AI(선택 정책)']['event_f1']:.3f})")
    if args.json:
        json.dump({"policy": {"scale": args.scale, "sigma": args.sigma, "damping": args.damping}, "seeds": seeds, "results": results},
                  open(args.json, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
        print(f"[*] 저장: {args.json}")


if __name__ == "__main__":
    main()
