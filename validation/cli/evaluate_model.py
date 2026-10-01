"""
오프라인 모델 평가 (이벤트 중심, 시드 고정, 학습과 분리된 데이터, 베이스라인 병기)

기존 방식(시뮬레이터 DB + TTF 필터 + 관대한 정답 창)의 신뢰성 문제를 해결한 평가 스크립트.
  - DB를 쓰지 않는다: validation/simulator/scenario_generator.py 가 시드 고정으로 데이터와 정답을 생성
  - 평가 데이터는 학습에 쓰인 적 없는 포트/에피소드 (학습과 다른 seed 사용)
  - 지표는 이벤트 단위: 장애 전 조기 탐지율, 리드타임, 오탐 인시던트/포트·일, 이벤트 F1, AUPRC
  - 같은 채점으로 자명한 베이스라인(항상 알람/고정 임계/포트별 롤링 규칙)과 비교
  - AI 트랙별(traffic/optical) 진단과 댐프닝 영향도 함께 보고

사용법:
  python validation/cli/evaluate_model.py                       # seed=7, 6노드x10포트, 14일
  python validation/cli/evaluate_model.py --seed 11 --days 21
  python validation/cli/evaluate_model.py --reuse-predictions   # 저장된 추론 결과로 지표만 재계산
  python validation/cli/evaluate_model.py --models-dir <dir>    # 활성 모델 대신 다른 모델 폴더 평가
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

from validation.simulator.scenario_generator import ScenarioConfig, generate, save, load
from validation.evaluation.metrics import summarize, KEY
from validation.evaluation.baselines import BASELINES

PRED_COLS = KEY + ["occur_date", "is_anomaly", "severity", "is_traffic_anomaly", "is_optical_anomaly"]


def get_dataset(args):
    out_dir = args.out or os.path.join("validation", "runs", f"seed{args.seed}_n{args.nodes}_d{args.days}{'_clean' if args.clean else ''}")
    if os.path.exists(os.path.join(out_dir, "episodes.csv")) and not args.regenerate:
        print(f"[*] Loading existing dataset: {out_dir}")
        return load(out_dir), out_dir
    noise = dict(benign_error_prob=0.0, burst_prob=0.0, blip_prob=0.0, missing_prob=0.0) if args.clean else {}
    cfg = ScenarioConfig(seed=args.seed, nodes=args.nodes, days=args.days, **noise)
    print(f"[*] Generating dataset (seed={cfg.seed}, ports={cfg.nodes * cfg.ports_per_node}, days={cfg.days})")
    data = generate(cfg)
    save(data, cfg, out_dir)
    print(f"[*] Saved to {out_dir}: episodes={len(data['episodes'])}, "
          f"fault_step_ratio={(data['labels']['state'] > 0).mean():.2%}")
    return data, out_dir


def run_inference(data, out_dir, models_dir=None, tag="active"):
    from src.config import PATHS
    if models_dir:
        for ft in ("traffic", "optical"):
            PATHS[ft] = {"model": os.path.join(models_dir, f"{ft}_ae.pth"),
                         "scaler": os.path.join(models_dir, f"{ft}_scaler.joblib")}
    from src.pipeline.inference import AnomalyDetector
    detector = AnomalyDetector()
    t = time.time()
    res = detector.detect(df_traffic=data["traffic"], df_optical=data["optical"], latest_only=False)
    print(f"[*] Inference finished in {time.time() - t:.0f}s ({0 if res is None else len(res):,} scored steps)")
    if res is None or res.empty:
        raise SystemExit("[!] No inference results.")
    pred = res[PRED_COLS].copy()
    pred["occur_date"] = pd.to_datetime(pred["occur_date"])
    pred.to_csv(os.path.join(out_dir, f"predictions_{tag}.csv"), index=False)
    return pred


def build_eval_frame(base, alarm, score, labels):
    """base: 평가할 스텝 키(occur_date + KEY). 방법별 alarm/score 를 붙이고 정답과 조인"""
    ev = base[KEY + ["occur_date"]].copy()
    ev["alarm"] = np.asarray(alarm).astype(bool)
    ev["score"] = np.asarray(score, dtype=float)
    return ev.merge(labels[KEY + ["occur_date", "state", "episode_id", "scenario", "nuisance"]],
                    on=KEY + ["occur_date"], how="inner")


def fmt(v, kind="pct"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   -  "
    return f"{v * 100:5.1f}%" if kind == "pct" else (f"{v:6.0f}" if kind == "min" else f"{v:6.3f}")


def print_report(results, meta):
    print("\n" + "=" * 112)
    print(f" 평가 리포트 | 포트 {meta['ports']}개 | 에피소드 {meta['episodes']}건 | 유병률 {meta['prevalence']:.2%} | seed={meta['seed']}")
    print("=" * 112)
    head = f"{'방법':<22}{'조기탐지':>8}{'탐지(전체)':>10}{'리드중앙(분)':>13}{'오탐/포트·일':>13}{'정상구간알람':>11}{'이벤트P':>9}{'이벤트F1':>9}{'AUPRC':>8}"
    print(head)
    print("-" * 112)
    for name, r in results.items():
        e, s = r["event"], r["step"]
        print(f"{name:<22}{fmt(e['early_detection_rate']):>8}{fmt(e['episode_detection_rate']):>10}"
              f"{fmt(e['lead_time_median_min'], 'min'):>13}{e['false_incidents_per_port_day']:>13.3f}"
              f"{fmt(e['false_alarm_step_ratio']):>11}{fmt(e['incident_precision']):>9}{e['event_f1']:>9.3f}{fmt(s['auprc'], 'f'):>8}")
    print("-" * 112)
    print(" * 조기탐지: failure 이전에 첫 알람이 울린 에피소드 비율(예지) / 탐지(전체): 복구 전 어느 시점이든 알람")
    print(" * 오탐: 에피소드 활성 구간 밖 알람을 묶은 인시던트의 포트·일당 건수 / 정상구간알람: 정상 스텝 중 알람 상태인 비율")
    print(" * 이벤트P: 탐지 에피소드 / (탐지 에피소드 + 오탐 인시던트)")


def print_scenarios(results, names):
    print("\n[시나리오별 조기탐지율 / 탐지율 / 리드 중앙값(분)]")
    scns = sorted({s for n in names for s in results[n]["event"]["by_scenario"]})
    print(f"{'방법':<22}" + "".join(f"{s:>34}" for s in scns))
    for n in names:
        row = f"{n:<22}"
        for s in scns:
            d = results[n]["event"]["by_scenario"].get(s)
            row += f"{(fmt(d['early']) + ' / ' + fmt(d['detected']) + ' / ' + fmt(d['lead_median_min'], 'min')) if d else '-':>34}"
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nodes", type=int, default=6, help="노드 수 (노드당 10포트)")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out", default=None, help="데이터/결과 저장 폴더 (기본 validation/runs/...)")
    ap.add_argument("--clean", action="store_true", help="정상 노이즈(산발 에러/버스트/광 흔들림/결측)를 끈 데이터로 평가 (분포 이동 영향 분리용)")
    ap.add_argument("--regenerate", action="store_true", help="기존 폴더가 있어도 데이터를 다시 생성")
    ap.add_argument("--reuse-predictions", action="store_true", help="저장된 predictions.csv 로 지표만 재계산")
    ap.add_argument("--tag", default="active", help="결과 파일 접미사 (predictions_<tag>.csv, report_<tag>.json). 모델 비교 시 구분용")
    ap.add_argument("--models-dir", default=None, help="평가할 모델 폴더 (기본: 활성 모델)")
    args = ap.parse_args()

    data, out_dir = get_dataset(args)
    labels, episodes = data["labels"], data["episodes"]

    pred_path = os.path.join(out_dir, f"predictions_{args.tag}.csv")
    if args.reuse_predictions and os.path.exists(pred_path):
        pred = pd.read_csv(pred_path, parse_dates=["occur_date"])
        print(f"[*] Reusing predictions: {pred_path} ({len(pred):,} rows)")
    else:
        pred = run_inference(data, out_dir, args.models_dir, args.tag)

    # 모든 방법을 'AI가 채점한 스텝'(윈도우 워밍업 제외)에서 동일하게 평가
    raw = data["traffic"].merge(data["optical"], on=KEY + ["occur_date"], how="outer")
    results = {}

    def add(name, base, alarm, score):
        ev = build_eval_frame(base, alarm, score, labels)
        results[name] = summarize(ev, episodes)

    add("AI (최종 알람)", pred, pred["is_anomaly"], pred["severity"])
    add("AI (댐프닝 전)", pred, pred["is_traffic_anomaly"].astype(bool) | pred["is_optical_anomaly"].astype(bool), pred["severity"])
    add("AI traffic 트랙", pred, pred["is_traffic_anomaly"].astype(bool), pred["severity"])
    add("AI optical 트랙", pred, pred["is_optical_anomaly"].astype(bool), pred["severity"])
    for name, fn in BASELINES.items():
        b = fn(raw)
        merged = pred[KEY + ["occur_date"]].merge(
            pd.concat([raw[KEY + ["occur_date"]], b], axis=1), on=KEY + ["occur_date"], how="left")
        add(f"[기준] {name}", merged, merged["alarm"].fillna(False), merged["score"].fillna(0.0))

    ev0 = build_eval_frame(pred, pred["is_anomaly"], pred["severity"], labels)
    meta = {"ports": results["AI (최종 알람)"]["event"]["ports"], "episodes": int(len(episodes)),
            "prevalence": float((ev0["state"] > 0).mean()), "seed": args.seed}

    print_report(results, meta)
    print_scenarios(results, ["AI (최종 알람)", "AI traffic 트랙", "AI optical 트랙", "[기준] fixed_threshold", "[기준] rolling_rule"])
    s = results["AI (최종 알람)"]["step"]
    print(f"\n[AI 스텝 단위] precision={s['precision']:.3f} recall={s['recall']:.3f} f1={s['f1']:.3f} "
          f"| 열화구간 재현율={fmt(s['recall_ramp'])} 장애지속구간 재현율={fmt(s['recall_plateau'])}")

    report = {"meta": meta, "results": {k: {"step": v["step"], "event": {kk: vv for kk, vv in v["event"].items() if kk != "_episodes_detail"}}
                                         for k, v in results.items()}}
    with open(os.path.join(out_dir, f"report_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    results["AI (최종 알람)"]["event"]["_episodes_detail"].to_csv(os.path.join(out_dir, f"ai_episodes_detail_{args.tag}.csv"), index=False)
    print(f"\n[*] Report saved: {os.path.join(out_dir, f'report_{args.tag}.json')}")


if __name__ == "__main__":
    main()
