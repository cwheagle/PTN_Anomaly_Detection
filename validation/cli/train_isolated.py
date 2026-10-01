"""
격리된 재학습 실험 (활성 모델/DB를 건드리지 않음)

시드 고정 시나리오 생성기로 학습 데이터를 만들고, 장애 구간을 제거한 '정상 데이터'로 AE 를 학습해
별도 폴더(--out)에 저장한다. 저장된 모델은 validation/cli/evaluate_model.py --models-dir 로 평가한다.

  clean : 정상 노이즈 없는 정상 데이터만 (이상적인 정상)
  noisy : 장애 구간만 제거하고 정상적인 변동(버스트/산발 에러/광 흔들림/결측)은 유지 (현실적인 정상)

사용법:
  python validation/cli/train_isolated.py --variant clean --out validation/runs/models_clean
  python validation/cli/train_isolated.py --variant noisy --out validation/runs/models_noisy --epochs 30
"""
import argparse
import os
import sys
import time

import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
os.chdir(root_dir)

from validation.simulator.scenario_generator import ScenarioConfig, generate, STEPS_PER_DAY
from src.data.data_collector import DataCollector


def build_normal_frames(variant, seed, nodes, days, margin_before_h, margin_after_h):
    noise = dict(benign_error_prob=0.0, burst_prob=0.0, blip_prob=0.0, missing_prob=0.0) if variant == "clean" else {}
    cfg = ScenarioConfig(seed=seed, nodes=nodes, days=days, **noise)
    data = generate(cfg)
    ep = data["episodes"]
    # 장애 에피소드 전후(파생 변수의 영향 구간 포함)를 학습에서 제외
    exclusions = pd.DataFrame({
        "ip_addr": ep["ip_addr"], "cid": ep["cid"], "lid": ep["lid"],
        "start_time": ep["t_start"] - pd.Timedelta(hours=margin_before_h),
        "failure_time": ep["t_end"] + pd.Timedelta(hours=margin_after_h),
    })
    normal = {k: DataCollector.filter_excluded(data[k], exclusions) for k in ("traffic", "optical")}
    removed = {k: len(data[k]) - len(normal[k]) for k in normal}
    print(f"[*] variant={variant} seed={seed} ports={nodes * cfg.ports_per_node} days={days} episodes={len(ep)}")
    print(f"[*] rows removed as fault-related: traffic={removed['traffic']:,} optical={removed['optical']:,} "
          f"(of {len(data['traffic']):,} / {len(data['optical']):,})")
    return normal, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["clean", "noisy"], required=True)
    ap.add_argument("--out", required=True, help="모델 저장 폴더 (예: validation/runs/models_clean)")
    ap.add_argument("--seed", type=int, default=101, help="학습 데이터 시드 (평가 시드와 달라야 함)")
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--margin-before-h", type=float, default=1.0)
    ap.add_argument("--margin-after-h", type=float, default=4.0, help="복구 후 제외 시간 (ma_16 등 파생 변수 영향 구간)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--tracks", choices=["both", "traffic", "optical", "none"], default="both",
                    help="학습할 트랙. none 이면 학습 데이터 준비만 수행 (병렬 학습 전 데이터 생성용)")
    ap.add_argument("--reuse-data", action="store_true", help="out/train_data 에 이미 있는 CSV 를 재사용 (병렬 학습 프로세스용)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU 스레드 수 (병렬 학습 시 코어 분배)")
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    if args.seed == 7:
        raise SystemExit("[!] seed 7 은 평가용 시드입니다. 학습 데이터는 다른 시드를 사용하세요.")

    os.makedirs(args.out, exist_ok=True)
    data_dir = os.path.join(args.out, "train_data")
    os.makedirs(data_dir, exist_ok=True)
    paths = {ft: (os.path.join(data_dir, f"{ft}_train.csv"), os.path.join(data_dir, f"{ft}_val.csv"))
             for ft in ("traffic", "optical")}

    if args.reuse_data and all(os.path.exists(p) for pair in paths.values() for p in pair):
        print(f"[*] 기존 학습 데이터를 재사용합니다: {data_dir}")
    else:
        normal, cfg = build_normal_frames(args.variant, args.seed, args.nodes, args.days,
                                          args.margin_before_h, args.margin_after_h)
        # 시간 기준 분할: 앞쪽 (1 - val_ratio) 학습 / 뒤쪽 검증 (둘 다 정상 데이터)
        split = pd.Timestamp(cfg.start) + pd.Timedelta(days=args.days * (1 - args.val_ratio))
        for ft in ("traffic", "optical"):
            df = normal[ft]
            tr, va = df[df["occur_date"] < split], df[df["occur_date"] >= split]
            tr.to_csv(paths[ft][0], index=False)
            va.to_csv(paths[ft][1], index=False)
            print(f"[*] {ft}: train rows={len(tr):,}, val rows={len(va):,}")

    if args.tracks == "none":
        print("[*] 데이터 준비만 완료했습니다.")
        return
    tracks = ("traffic", "optical") if args.tracks == "both" else (args.tracks,)

    # 활성 모델 경로(PATHS)를 격리 폴더로 교체 (이 프로세스 안에서만 유효)
    from src.config import PATHS
    from src.models.trainer import Trainer
    for ft in tracks:
        PATHS[ft] = {"model": os.path.join(args.out, f"{ft}_ae.pth"),
                     "scaler": os.path.join(args.out, f"{ft}_scaler.joblib")}

    for ft in tracks:
        t = time.time()
        trainer = Trainer(ft, config_override={"epochs": args.epochs, "batch_size": args.batch_size,
                                               "patience": args.patience})
        ok = trainer.train(train_path=paths[ft][0], val_path=paths[ft][1])
        print(f"[{'OK' if ok else 'FAIL'}] {ft} trained in {time.time() - t:.0f}s -> {args.out}")
        if not ok:
            raise SystemExit(f"[!] {ft} training failed")


if __name__ == "__main__":
    main()
