"""
알람 정책 스윕 (재학습 없이): 임계치 배율 x σ 배수 x 댐프닝 강도 -> 이벤트 지표

모델 추론은 데이터셋마다 한 번만 하고(점수 캐시), 정책 조합은 저장된 점수에서 수 초 만에 다시 계산한다.
(튜닝 도구와 운영 추론의 동등성은 tests/pipeline/test_tuning_equivalence.py 가 검증)

사용법:
  python validation/cli/tune_alerting.py --models-dir validation/runs/models_noisy --tag noisy30
  python validation/cli/tune_alerting.py --models-dir <dir> --datasets seed7_n6_d14,seed7_n6_d14_clean
출력: 정책별 지표 표 (이벤트 F1 순) + CSV (validation/runs/tuning_<tag>.csv)
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from validation.evaluation.tuning import DAMPING_PRESETS, evaluate_policy, get_scores, policy_grid, select_policy

RUNS = os.path.join("validation", "runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=None, help="평가할 모델 폴더 (기본: 활성 모델)")
    ap.add_argument("--tag", required=True, help="점수 캐시/결과 파일 접미사")
    ap.add_argument("--datasets", default="seed7_n6_d14,seed7_n6_d14_clean", help="validation/runs 아래 데이터셋 폴더(쉼표 구분)")
    ap.add_argument("--scales", default="0.8,1.0,1.25,1.5,2.0,3.0")
    ap.add_argument("--sigmas", default="2,3,4")
    ap.add_argument("--dampings", default=",".join(DAMPING_PRESETS))
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-early", type=float, default=0.70, help="선택 규칙: 조기 탐지율 하한")
    ap.add_argument("--max-fp", type=float, default=0.02, help="선택 규칙: 오탐(건/포트·일) 상한")
    args = ap.parse_args()

    datasets = [x for x in args.datasets.split(",") if x]
    loaded = {ds: get_scores(ds, args.models_dir, args.tag) for ds in datasets}
    grid = list(policy_grid(
        threshold_scales=[float(x) for x in args.scales.split(",")],
        sigma_ks=[float(x) for x in args.sigmas.split(",")],
        damping_presets={k: DAMPING_PRESETS[k] for k in args.dampings.split(",")}))
    print(f"[*] 정책 {len(grid)}개 x 데이터셋 {len(datasets)}개 평가")

    rows = []
    t = time.time()
    for g in grid:
        r = {"scale": g["threshold_scale"], "sigma_k": g["sigma_k"], "damping": g["damping"]}
        per = []
        for ds, (data, scores) in loaded.items():
            res = evaluate_policy(data, scores, g["policy"])
            e = res["event"]
            per.append(e)
        for key in ("early_detection_rate", "episode_detection_rate", "false_incidents_per_port_day",
                    "incident_precision", "event_f1", "lead_time_median_min"):
            vals = [p[key] for p in per if p[key] is not None]
            r[key] = float(np.mean(vals)) if vals else None
        r["f1_min"] = min(p["event_f1"] for p in per)            # 데이터셋 간 최악값 (노이즈 환경 등)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("event_f1", ascending=False)
    out = os.path.join(RUNS, f"tuning_{args.tag}.csv")
    df.to_csv(out, index=False)
    print(f"[*] 평가 {time.time() - t:.0f}s | 저장: {out}")

    base = df[(df.scale == 1.0) & (df.sigma_k == 3.0) & (df.damping == "default(3/2/1)")]
    show = pd.concat([df.head(args.top), base]).drop_duplicates()
    fmt = lambda c: show[c].map(lambda v: f"{v:.3f}" if v is not None and not pd.isna(v) else "-")
    table = pd.DataFrame({"배율": show.scale, "σ": show.sigma_k, "댐프닝": show.damping,
                          "조기탐지": show.early_detection_rate.map(lambda v: f"{v:.1%}"),
                          "탐지": show.episode_detection_rate.map(lambda v: f"{v:.1%}"),
                          "오탐/포트일": fmt("false_incidents_per_port_day"),
                          "이벤트P": show.incident_precision.map(lambda v: f"{v:.1%}"),
                          "F1(평균)": fmt("event_f1"), "F1(최악)": fmt("f1_min")})
    print("\n[상위 정책 + 현재 기본값(배율 1.0 / σ 3 / default)]  * 데이터셋 평균 기준")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
