"""
학습 데이터 장애 제외 방식 비교 실험 (P1-1 단위 D, 설계서 6.2)

바꾸는 것은 '학습 데이터에서 장애 구간을 어떻게 제외하느냐' 하나뿐이다 (lessons #29). 같은 원본 학습 데이터(시드·노이즈 포함),
같은 시간 분할(앞 80% 학습/뒤 20% 검증), 같은 하이퍼파라미터·torch 시드로 세 가지를 학습한다.
  truth : 정답 에피소드 기반 제외 (현재 train_isolated.py 방식, 전 1h / 후 4h)
  auto  : 자동 의심 구간 제외 = 규칙 B (train_window.intervals_from_rules + apply_suspect_filter, RetrainPolicy 기본값)
  none  : 제외 없음
합격 기준(설계서 6.2, 사전 정의): auto 가 truth 대비 AUPRC -0.05, 이벤트 F1 -0.05 이내이고 none 보다 낫다.
평가는 학습에 쓰지 않은 검증 시드(23,31,47)에서 AUPRC(순위 품질)와 이벤트 지표(운영점)를 분리해 본다 (lessons #29, #30).

사용법:
  python validation/cli/compare_train_exclusion.py train --variant auto --out validation/runs/excl_auto_s101 [--train-seed 101]
  python validation/cli/compare_train_exclusion.py eval --models truth=DIR,auto=DIR,none=DIR [--seeds 23,31,47]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

from src.data import train_window
from src.data.data_collector import DataCollector
from src.pipeline.alerting import AlertPolicy
from src.pipeline.retrain_policy import RetrainPolicy
from validation.simulator.scenario_generator import ScenarioConfig, generate, save

TRACKS = ("traffic", "optical")


def exclude(variant, data, policy):
    """variant 별 학습 데이터와 제외 통계 반환: ({track: df}, {track: stats})"""
    out, stats = {}, {}
    ep = data["episodes"]
    truth_iv = pd.DataFrame({"ip_addr": ep["ip_addr"], "cid": ep["cid"], "lid": ep["lid"],
                             "start_time": ep["t_start"] - pd.Timedelta(hours=1),
                             "failure_time": ep["t_end"] + pd.Timedelta(hours=4)})
    for ft in TRACKS:
        raw = data[ft].copy()
        raw["occur_date"] = pd.to_datetime(raw["occur_date"])
        if variant == "none":
            out[ft] = raw
        elif variant == "truth":
            out[ft] = DataCollector.filter_excluded(raw, truth_iv)
        elif variant == "auto":
            iv = train_window.intervals_from_rules(raw, policy)
            st = train_window.suspect_stats(raw, iv, policy.port_drop_fraction)
            if train_window.exceeds_suspect_limit(st, policy):
                raise SystemExit(f"[!] {ft}: 의심 비율 {st['fraction']:.1%} > 한도 — 학습 중단")
            out[ft], _ = train_window.apply_suspect_filter(raw, iv, policy.port_drop_fraction, st)
        else:
            raise ValueError(variant)
        stats[ft] = {"rows": len(raw), "kept": len(out[ft])}
    return out, stats


def cmd_train(a):
    import torch
    torch.manual_seed(a.train_seed)
    np.random.seed(a.train_seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = ScenarioConfig(seed=a.data_seed, nodes=a.nodes, days=a.days)
    data = generate(cfg)
    frames, stats = exclude(a.variant, data, RetrainPolicy())
    print(f"[*] variant={a.variant} data_seed={a.data_seed} train_seed={a.train_seed} rows kept: "
          + ", ".join(f"{ft} {s['kept']:,}/{s['rows']:,}" for ft, s in stats.items()))
    os.makedirs(os.path.join(a.out, "train_data"), exist_ok=True)
    split = pd.Timestamp(cfg.start) + pd.Timedelta(days=a.days * 0.8)
    paths = {}
    for ft, df in frames.items():
        df = df.sort_values(train_window.PORT_KEYS + ["occur_date"])
        tr, va = df[df["occur_date"] < split], df[df["occur_date"] >= split]
        paths[ft] = (os.path.join(a.out, "train_data", f"{ft}_train.csv"), os.path.join(a.out, "train_data", f"{ft}_val.csv"))
        tr.to_csv(paths[ft][0], index=False)
        va.to_csv(paths[ft][1], index=False)
    from src.config import PATHS
    from src.models.trainer import Trainer
    for ft in TRACKS:
        PATHS[ft] = {"model": os.path.join(a.out, f"{ft}_ae.pth"), "scaler": os.path.join(a.out, f"{ft}_scaler.joblib")}
    for ft in TRACKS:
        t = time.time()
        ok = Trainer(ft, config_override={"epochs": a.epochs, "batch_size": 64, "patience": 5}).train(
            train_path=paths[ft][0], val_path=paths[ft][1])
        print(f"[{'OK' if ok else 'FAIL'}] {ft} {time.time() - t:.0f}s -> {a.out}")
        if not ok:
            raise SystemExit(1)


def _ensure_dataset(seed, nodes, days):
    name = f"seed{seed}_n{nodes}_d{days}"
    d = os.path.join("validation", "runs", name)
    if not os.path.exists(os.path.join(d, "labels.csv")):
        cfg = ScenarioConfig(seed=seed, nodes=nodes, days=days)
        save(generate(cfg), cfg, d)
    return name


def cmd_eval(a):
    from validation.cli import check_retrain_policy as crp
    from validation.evaluation.tuning import DAMPING_PRESETS, evaluate_alarms, get_scores, simulate_alarms
    pols = {"기본정책": AlertPolicy(),
            "선택정책(3/σ2/heavy)": AlertPolicy(threshold_scale=3.0, sigma_k=2.0, dampening_steps=dict(DAMPING_PRESETS["heavy(6/4/3)"]))}
    models = dict(kv.split("=", 1) for kv in a.models.split(","))
    seeds = [int(s) for s in a.seeds.split(",")]
    rows, cover = [], []
    for seed in seeds:
        ds = _ensure_dataset(seed, a.nodes, a.days)
        for name, mdir in models.items():
            data, scores = get_scores(ds, mdir, f"excl_{name}_{os.path.basename(mdir.rstrip('/'))}")
            for pname, pol in pols.items():
                sim = simulate_alarms(scores, pol)
                res = evaluate_alarms(sim, data)
                rows.append({"variant": name, "policy": pname, "seed": seed, "auprc": res["step"]["auprc"],
                             "f1": res["event"]["event_f1"], "early": res["event"]["early_detection_rate"],
                             "fp": res["event"]["false_incidents_per_port_day"], "prec": res["event"]["incident_precision"]})
            # 자기 알람 이력(A) 측정: 기본정책 알람 행으로 규칙 A, A∪B 의 장애 제외율
            sim = simulate_alarms(scores, AlertPolicy())
            alarm_rows = sim[sim["alarm"]][["occur_date", "ip_addr", "cid", "lid"]]
            for m, c in crp.evaluate_methods(data, RetrainPolicy(), alarm_rows).items():
                cover.append({"variant": name, "seed": seed, "method": m, **c})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for pname in pols:
        print(f"\n== {pname}: 검증 시드 {seeds} 평균±표준편차 ==")
        g = df[df.policy == pname].groupby("variant")
        print(g[["auprc", "f1", "early", "fp", "prec"]].agg(lambda s: f"{s.mean():.3f}±{s.std(ddof=0):.3f}").to_string())
    print("\n== 시드별 상세 ==")
    print(df.round(3).to_string(index=False))
    c = pd.DataFrame(cover)
    print("\n== 의심 구간 제외율 (모델 알람 이력 기반 A, A∪B 포함; 모델별 알람) ==")
    print(c.groupby(["variant", "method"])[["fault_excl", "nuisance_excl", "total_excl"]].mean().round(3).to_string())
    # 합격 판정 (설계서 6.2): auto vs truth 차 <= 0.05, auto > none
    for pname in pols:
        m = df[df.policy == pname].groupby("variant")[["auprc", "f1"]].mean()
        if {"auto", "truth", "none"} <= set(m.index):
            ok = (m.loc["auto", "auprc"] >= m.loc["truth", "auprc"] - 0.05 and m.loc["auto", "f1"] >= m.loc["truth", "f1"] - 0.05
                  and m.loc["auto", "auprc"] > m.loc["none", "auprc"] and m.loc["auto", "f1"] > m.loc["none", "f1"])
            print(f"[판정] {pname}: auto-truth AUPRC {m.loc['auto','auprc']-m.loc['truth','auprc']:+.3f}, F1 {m.loc['auto','f1']-m.loc['truth','f1']:+.3f}; "
                  f"auto-none AUPRC {m.loc['auto','auprc']-m.loc['none','auprc']:+.3f}, F1 {m.loc['auto','f1']-m.loc['none','f1']:+.3f} -> {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--variant", choices=["truth", "auto", "none"], required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--data-seed", type=int, default=101)
    t.add_argument("--train-seed", type=int, default=0)
    t.add_argument("--nodes", type=int, default=4)
    t.add_argument("--days", type=int, default=21)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--threads", type=int, default=None)
    e = sub.add_parser("eval")
    e.add_argument("--models", required=True)
    e.add_argument("--seeds", default="23,31,47")
    e.add_argument("--nodes", type=int, default=6)
    e.add_argument("--days", type=int, default=14)
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_eval(a)


if __name__ == "__main__":
    main()
