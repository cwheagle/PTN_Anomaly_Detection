"""
P1-5 임계치 산출 방식 검토 — 1단계(E0) 원인 분해 도구 (설계서 docs/design/p1_5_threshold.md 3장·6.1·8.1)

운영 레시피 오탐 0.029(P1-3 V2 FAIL)의 원인이 학습 난수·데이터·홀드아웃 구성·분할 방식·정제 방식 중 무엇인지 분해한다.
바꾸는 것은 실험마다 한 가지뿐이다 (lessons #29). 판정 기준 값(6.1)은 설계서에 데이터를 열기 전 고정된 값이며 이 파일의
상수다 — 인자로 바꿀 수 없고 결과 JSON 의 `criteria` 에 그대로 기록된다 (lessons #37). 모든 수치는 시뮬레이션 기준이다.
`src/` 는 바꾸지 않고, 학습·점수·알람·지표는 check_promotion 의 함수를 그대로 재사용한다 (lessons #31).
오라클 임계치(th*)와 truth 정제는 정답을 쓰는 **진단 전용**이며 운영 코드로 옮기지 않는다 (lessons #22).

  train   학습 1회(두 트랙) = 실행 폴더 1개 + run.json (임계치 재계산 검증·소요 시간 포함)
  oracle  실행 폴더마다 시드 211 정상 기준 데이터의 out-of-sample 임계치 th* 와 in-sample 편향 → oracle.json
  e0      개발 시드 7·11 평가(own / oracle 임계치) + Q-A·Q-B·Q-C 판정 + 결정표 → --out-csv, --out-json

사용법:
  python validation/cli/check_threshold.py train --out validation/runs/p1_5/e0/d101_t0_sptn_port_aub --data-seed 101 --train-seed 0
  python validation/cli/check_threshold.py oracle --runs "validation/runs/p1_5/e0/*"
  python validation/cli/check_threshold.py e0 --runs "validation/runs/p1_5/e0/*" --out-csv e0.csv --out-json e0.json
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from validation.cli import check_promotion as cp
from validation.cli.check_promotion import TRACKS, check_record
from validation.evaluation.tuning import baseline_alarms, evaluate_alarms, simulate_alarms

DEV_SEEDS = (7, 11)                         # e0 평가 (여러 번 열린 개발용 — 선택에만)
P15_VAL_SEEDS = (53, 59, 61)                # 새 검증 시드: 이 단계(U1)에서는 열지 않는다
TRAIN_DATA_SEEDS = (101, 103, 105)          # 학습 데이터 시드
ORACLE_SEED = 211                           # 오라클 기준(정상) 데이터 — 진단 전용
FORBIDDEN_SEEDS = (23, 31, 47, 102, 900, 33, 34, 35)    # 검증 소진·민감도·카나리·DB 대형 시드
# train --data-seed 로 쓸 수 없는 시드 = 평가·검증·오라클 + 금지
NOT_FOR_TRAINING = tuple(sorted(set(DEV_SEEDS) | set(P15_VAL_SEEDS) | {ORACLE_SEED} | set(FORBIDDEN_SEEDS)))

# 판정 기준 (설계서 6.1, 2026-10-07 확정 — 데이터를 열기 전 고정). 변경은 설계서 6.4 규칙만.
QA_MAX_MIN = 1.25                           # Q-A: R0 세트에서 어느 한 트랙이라도 임계치 max/min >= 1.25
QB_EXPLAINED, QB_FP_ORACLE = 0.50, 0.02     # Q-B: 설명 몫 >= 50% 그리고 th* 대입 오탐 평균 <= 0.02
QC_MIN_DIFF, QC_SAME_SIGN = 0.008, 2 / 3    # Q-C: |짝 오탐 차이 평균| >= 0.008 그리고 짝의 2/3 이상 같은 부호
CRITERIA = {"QA_max_min": QA_MAX_MIN, "QB_explained": QB_EXPLAINED, "QB_fp_oracle_mean": QB_FP_ORACLE,
            "QC_min_abs_diff": QC_MIN_DIFF, "QC_same_sign_fraction": QC_SAME_SIGN}
RECOMPUTE_TOL = 1e-6                        # 재계산 임계치와 메타 임계치의 상대 오차 허용 (3.3-5)
R0 = {"split": "port", "split_salt": "ptn", "exclusion": "aub"}
FP_POLICY = "오탐억제형"


# ─────────────────────────────────────────────
# 공통
# ─────────────────────────────────────────────
def _code_rev(arg):
    """run.json 의 code: --code-rev 인자 > 환경변수 PTN_CODE_REV > 'unknown' (컨테이너에서는 git 을 읽지 못하므로 호출자가 넘긴다)"""
    return arg or os.environ.get("PTN_CODE_REV") or "unknown"


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def expand_runs(text):
    """쉼표로 구분한 폴더/글롭 목록 -> 정렬된 실행 폴더 목록 (없으면 거부)"""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        hits = sorted(glob.glob(part)) if any(c in part for c in "*?[") else [part]
        out += [h for h in hits if os.path.isdir(h)]
    out = list(dict.fromkeys(out))
    if not out:
        raise SystemExit(f"[!] 실행 폴더를 찾지 못했습니다: {text}")
    return out


def active_entry(model_dir, ft):
    from src.models import registry
    reg = registry.load(model_dir, ft)
    entry = registry.find(reg, reg.get("active_version"))
    if entry is None:
        raise SystemExit(f"[!] {model_dir} 에 {ft} 활성 모델이 없습니다.")
    return entry


def _csv_ports(path):
    df = pd.read_csv(path, usecols=cp.KEY)
    return len(df.drop_duplicates())


def recompute_threshold(run_dir, ft):
    """학습 CSV 를 Trainer 와 같은 경로(저장 스케일러, create_sequences(is_train=True, fit_scaler=False))로 시퀀스화해
    마지막 시점 MSE 의 threshold_percentile 백분위를 다시 계산한다 (운영 산출식과 같다는 증거, lessons #31)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.data.data_processor import DataProcessor
    from src.models.model import LSTMAutoencoder

    entry = active_entry(run_dir, ft)
    meta = _load_json(os.path.join(run_dir, entry["config_path"]))
    cfg = dict(meta["config"])
    proc = DataProcessor(ft, config=cfg)
    if not proc.load_scaler(os.path.join(run_dir, entry["scaler_path"])):
        raise SystemExit(f"[!] {ft} 스케일러를 읽지 못했습니다.")
    cfg["input_dim"] = len(proc.extended_feature_cols)
    clean = proc.preprocess(pd.read_csv(os.path.join(run_dir, "train_data", f"{ft}_train.csv")), is_train=True)
    seqs = proc.create_sequences(clean, is_train=True, fit_scaler=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(cfg).to(device)
    state = torch.load(os.path.join(run_dir, entry["model_path"]), map_location=device, weights_only=True)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    model.eval()
    mses = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.from_numpy(seqs).float()), batch_size=cfg["batch_size"]):
            x = batch.to(device)
            mses.extend(torch.mean((x[:, -1, :] - model(x)[:, -1, :]) ** 2, dim=1).cpu().numpy())
    return float(np.percentile(mses, cfg["threshold_percentile"])), meta


# ─────────────────────────────────────────────
# train
# ─────────────────────────────────────────────
def check_train_args(a):
    if a.data_seed in NOT_FOR_TRAINING:
        raise SystemExit(f"[!] --data-seed {a.data_seed} 는 학습에 쓸 수 없습니다 (평가·검증·오라클·금지 시드: {list(NOT_FOR_TRAINING)}).")
    if os.path.exists(a.out):
        raise SystemExit(f"[!] --out {a.out} 가 이미 있습니다 — 덮어쓰지 않습니다.")


def cmd_train(a):
    check_train_args(a)
    started = datetime.now()
    t0 = time.time()
    res = cp.train_operational(a.out, a.data_seed, a.train_seed, nodes=a.nodes, days=a.days, epochs=a.epochs,
                               active_models_dir=a.active_models, split_salt=a.split_salt, split=a.split, exclusion=a.exclusion)
    elapsed = time.time() - t0
    tracks = {}
    for ft in TRACKS:
        entry = active_entry(a.out, ft)
        meta = _load_json(os.path.join(a.out, entry["config_path"]))
        recomputed, _ = recompute_threshold(a.out, ft)
        rel = abs(recomputed - meta["threshold"]) / max(abs(meta["threshold"]), 1e-12)
        if rel > RECOMPUTE_TOL:
            print(f"[경고] {ft}: 재계산 임계치 {recomputed:.8g} 가 메타 {meta['threshold']:.8g} 와 다릅니다 (상대 오차 {rel:.2e} > {RECOMPUTE_TOL:g}) "
                  "— 기준을 완화하지 않습니다. 원인 미확인이면 architect 에 보고하세요.")
        d = os.path.join(a.out, "train_data")
        tracks[ft] = {"threshold": meta["threshold"], "final_val_loss": meta["final_val_loss"], "samples_used": meta["samples_used"],
                      "val_ports": _csv_ports(os.path.join(d, f"{ft}_test.csv")),
                      "train_ports": _csv_ports(os.path.join(d, f"{ft}_train.csv")),
                      "threshold_recomputed": recomputed, "recompute_rel_err": rel}
    run = {"data_seed": a.data_seed, "train_seed": a.train_seed, "split": a.split,
           "split_salt": a.split_salt if a.split == "port" else None, "exclusion": a.exclusion, "nodes": a.nodes, "days": a.days,
           "epochs": a.epochs, "code": _code_rev(a.code_rev), "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
           "elapsed_sec": elapsed, "tracks": tracks}
    _dump_json(os.path.join(a.out, "run.json"), run)
    print(f"[OK] {a.out}: {elapsed:.0f}s, " + ", ".join(
        f"{ft} th={t['threshold']:.4f} (재계산 오차 {t['recompute_rel_err']:.1e}, 검증 포트 {t['val_ports']})" for ft, t in tracks.items()))
    return run


# ─────────────────────────────────────────────
# oracle
# ─────────────────────────────────────────────
def cmd_oracle(a):
    if a.seed != ORACLE_SEED:
        raise SystemExit(f"[!] oracle 의 --seed 는 {ORACLE_SEED} 만 허용합니다 (받은 값: {a.seed}).")
    runs = expand_runs(a.runs)
    for r in runs:                                          # 측정 전에 모두 검사: 하나라도 문제면 아무것도 쓰지 않는다
        if not os.path.exists(os.path.join(r, "run.json")):
            raise SystemExit(f"[!] {r} 에 run.json 이 없습니다 — train 으로 만든 실행 폴더가 아닙니다.")
        if os.path.exists(os.path.join(r, "oracle.json")):
            raise SystemExit(f"[!] {os.path.join(r, 'oracle.json')} 가 이미 있습니다 — 덮어쓰지 않습니다.")
    from src.data import train_window
    data = cp.make_data(a.seed, a.nodes, a.days)
    truth = cp.truth_exclusions(data)
    for r in runs:
        meta_pct = _load_json(os.path.join(r, active_entry(r, TRACKS[0])["config_path"]))["config"]["threshold_percentile"]
        scores = cp.score_data(data, r)
        own_train = {ft: pd.read_csv(os.path.join(r, "train_data", f"{ft}_train.csv")) for ft in TRACKS}
        in_scores = cp.score_data(own_train, r)
        tracks = {}
        for ft in TRACKS:
            df = scores[ft][0]
            keep = df[~train_window.exclusion_mask(df, truth)]
            th_star = float(np.percentile(keep["mse"], meta_pct))
            th_in = float(np.percentile(in_scores[ft][0]["mse"], meta_pct))
            tracks[ft] = {"th_star": th_star, "th_in": th_in, "bias": th_in / th_star, "n_scores": int(len(keep))}
        _dump_json(os.path.join(r, "oracle.json"), {"seed": a.seed, "tracks": tracks})
        print(f"[OK] {r}: " + ", ".join(f"{ft} th*={t['th_star']:.4f} th_in/th*={t['bias']:.2f}" for ft, t in tracks.items()))


# ─────────────────────────────────────────────
# e0 판정 (순수 함수 — 합성 입력으로 테스트)
# ─────────────────────────────────────────────
def _sd(values):
    v = np.asarray(list(values), dtype=float)
    return float(v.std(ddof=0)) if len(v) else float("nan")      # check_promotion._fmt 와 같은 모집단 SD


def judge_qa(thresholds):
    """thresholds: {track: [R0 세트 실행별 임계치]} -> {track: {max_min, cv, met}}"""
    out = {}
    for ft, vals in thresholds.items():
        v = np.asarray(vals, dtype=float)
        mm = float(v.max() / v.min()) if len(v) and v.min() > 0 else float("nan")
        out[ft] = {"max_min": mm, "cv": float(v.std(ddof=0) / v.mean()) if len(v) and v.mean() > 0 else float("nan"),
                   "met": bool(mm >= QA_MAX_MIN)}
    return out


def judge_qb(fp_own, fp_oracle):
    """fp_own / fp_oracle: R0 세트 실행별 오탐(오탐억제형, 개발 시드 평균) -> 설명 몫과 충족 여부"""
    sd_own, sd_or = _sd(fp_own), _sd(fp_oracle)
    explained = float(1 - sd_or / sd_own) if sd_own and sd_own > 0 else 0.0
    mean_or = float(np.mean(list(fp_oracle))) if len(list(fp_oracle)) else float("nan")
    return {"sd_own": sd_own, "sd_oracle": sd_or, "explained": explained, "fp_oracle_mean": mean_or,
            "met": bool(explained >= QB_EXPLAINED and mean_or <= QB_FP_ORACLE)}


def judge_qc(diffs):
    """diffs: 요인 수준 하나의 R0 대비 짝 오탐 차이 목록(변형 - R0) -> {pairs, mean_diff, same_sign, met}"""
    d = np.asarray(list(diffs), dtype=float)
    if len(d) == 0:
        return {"pairs": 0, "mean_diff": None, "same_sign": 0, "met": False}
    mean = float(d.mean())
    same = int((np.sign(d) == np.sign(mean)).sum()) if mean != 0 else 0
    met = bool(abs(mean) >= QC_MIN_DIFF and same / len(d) >= QC_SAME_SIGN)
    return {"pairs": int(len(d)), "mean_diff": mean, "same_sign": same, "met": met}


def decide(qb, qc):
    """3.4 결정표: Q-B 충족 -> T1T3T2, 아니면 Q-C 의 salt 또는 split 충족 -> T2, 아니면 stop (exclusion 은 보고만)"""
    if qb["met"]:
        return "T1T3T2"
    if qc["salt"]["met"] or qc["split"]["met"]:
        return "T2"
    return "stop"


def factor_of(spec):
    """실행이 R0 와 정확히 한 요인만 다르면 그 요인 이름, R0 이면 'R0', 그 밖은 None (Q-C 에서 무시)"""
    split, salt, excl = spec["split"], spec["split_salt"], spec["exclusion"]
    if (split, salt, excl) == (R0["split"], R0["split_salt"], R0["exclusion"]):
        return "R0"
    if split == "port" and excl == "aub" and salt != R0["split_salt"]:
        return "salt"
    if split == "time" and excl == "aub":
        return "split"
    if split == "port" and salt == R0["split_salt"] and excl == "truth":
        return "exclusion"
    return None


def judge_e0(rows):
    """rows: [{run, data_seed, train_seed, split, split_salt, exclusion, th: {ft: 임계치}, fp_own, fp_oracle}] (오탐억제형, 개발 시드 평균)
    -> QA, QB, QC, decision, R0 (실행 이름)"""
    r0 = [r for r in rows if factor_of(r) == "R0"]
    qa = judge_qa({ft: [r["th"][ft] for r in r0] for ft in TRACKS})
    qb = judge_qb([r["fp_own"] for r in r0], [r["fp_oracle"] for r in r0])
    base = {(r["data_seed"], r["train_seed"]): r for r in r0}
    diffs = {"salt": [], "split": [], "exclusion": []}
    for r in rows:
        f = factor_of(r)
        if f in diffs and (r["data_seed"], r["train_seed"]) in base:
            diffs[f].append(r["fp_own"] - base[(r["data_seed"], r["train_seed"])]["fp_own"])
    qc = {f: judge_qc(v) for f, v in diffs.items()}
    return {"QA": qa, "QB": qb, "QC": qc, "decision": decide(qb, qc), "R0": [r["run"] for r in r0]}


# ─────────────────────────────────────────────
# e0
# ─────────────────────────────────────────────
CSV_COLUMNS = ["run", "data_seed", "train_seed", "split", "split_salt", "exclusion", "seed", "policy", "th_kind",
               "th_traffic", "th_optical", "auprc", "f1", "early", "fp", "prec", "lead"]


def check_e0_args(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    if not set(seeds) <= set(DEV_SEEDS):
        raise SystemExit(f"[!] e0 --seeds 는 {list(DEV_SEEDS)} 의 부분집합만 허용합니다 (받은 값: {sorted(seeds)}).")
    if os.path.abspath(a.out_csv) == os.path.abspath(a.out_json):
        raise SystemExit("[!] --out-csv 와 --out-json 은 서로 다른 파일이어야 합니다.")
    for out in (a.out_csv, a.out_json):
        check_record(out)
    runs = expand_runs(a.runs)
    for r in runs:
        for name in ("run.json", "oracle.json"):
            if not os.path.exists(os.path.join(r, name)):
                raise SystemExit(f"[!] {r} 에 {name} 이 없습니다 — train/oracle 을 먼저 실행하세요.")
    return seeds, runs


def cmd_e0(a):
    seeds, runs = check_e0_args(a)
    rows, summary = [], []
    for seed in seeds:
        data = cp.make_data(seed, a.nodes, a.days)
        first_sim = None
        for r in runs:
            run, orc = _load_json(os.path.join(r, "run.json")), _load_json(os.path.join(r, "oracle.json"))
            name = os.path.basename(os.path.normpath(r))
            scores = cp.score_data(data, r)
            for kind in ("own", "oracle"):
                if kind == "own":
                    sc = scores
                else:                                           # 같은 가중치·같은 점수, 전역 임계치만 교체
                    sc = {ft: (df, orc["tracks"][ft]["th_star"]) for ft, (df, _) in scores.items()}
                ths = {ft: float(sc[ft][1]) for ft in sc}
                for pname, pol in cp.POLICIES.items():
                    sim = simulate_alarms(sc, pol)
                    if first_sim is None:
                        first_sim = sim
                    rows.append({"run": name, "data_seed": run["data_seed"], "train_seed": run["train_seed"], "split": run["split"],
                                 "split_salt": run["split_salt"], "exclusion": run["exclusion"], "seed": seed, "policy": pname,
                                 "th_kind": kind, "th_traffic": ths.get("traffic"), "th_optical": ths.get("optical"),
                                 **cp.metrics_row(evaluate_alarms(sim, data))})
        for b in ("rolling_rule", "fixed_threshold", "always_alarm"):          # 베이스라인은 시드별 1회 병기
            rows.append({"run": f"[기준] {b}", "seed": seed, "policy": "-", "th_kind": "-",
                         **cp.metrics_row(evaluate_alarms(baseline_alarms(data, first_sim, b), data))})
    df = pd.DataFrame(rows).reindex(columns=CSV_COLUMNS)

    judge_rows = []
    for r in runs:
        run = _load_json(os.path.join(r, "run.json"))
        name = os.path.basename(os.path.normpath(r))
        sub = df[(df.run == name) & (df.policy == FP_POLICY)]
        judge_rows.append({"run": name, "data_seed": run["data_seed"], "train_seed": run["train_seed"], "split": run["split"],
                           "split_salt": run["split_salt"], "exclusion": run["exclusion"],
                           "th": {ft: run["tracks"][ft]["threshold"] for ft in TRACKS},
                           "fp_own": float(sub[sub.th_kind == "own"]["fp"].mean()),
                           "fp_oracle": float(sub[sub.th_kind == "oracle"]["fp"].mean())})
    j = judge_e0(judge_rows)
    out = {"seeds": seeds, "runs": [os.path.basename(os.path.normpath(r)) for r in runs], "R0": j["R0"], "QA": j["QA"], "QB": j["QB"],
           "QC": j["QC"], "decision": j["decision"], "criteria": CRITERIA}
    df.to_csv(a.out_csv, index=False)
    _dump_json(a.out_json, out)
    _print_e0(df, out)
    return out


def _print_e0(df, out):
    pd.set_option("display.width", 220)
    sub = df[df.policy.isin([FP_POLICY, "기본"]) & df.th_kind.isin(["own", "oracle"])]
    g = sub.groupby(["run", "policy", "th_kind"])[["auprc", "f1", "early", "fp", "lead"]].mean().round(4)
    print(f"== E0 개발 시드 {out['seeds']} 평균 (실행 x 정책 x 임계치 종류) ==")
    print(g.to_string())
    base = df[df.run.str.startswith("[기준]")].groupby("run")[["auprc", "f1", "early", "fp", "lead"]].mean().round(4)
    print("\n[베이스라인]"); print(base.to_string())
    print(f"\n[Q-A] " + ", ".join(f"{ft}: max/min {v['max_min']:.3f} cv {v['cv']:.3f} -> {'충족' if v['met'] else '미충족'}" for ft, v in out["QA"].items()))
    qb = out["QB"]
    print(f"[Q-B] SD own {qb['sd_own']:.4f} / oracle {qb['sd_oracle']:.4f}, 설명 몫 {qb['explained']:.1%}, th* 대입 오탐 평균 {qb['fp_oracle_mean']:.4f} -> {'충족' if qb['met'] else '미충족'}")
    for f, v in out["QC"].items():
        md = "-" if v["mean_diff"] is None else f"{v['mean_diff']:+.4f}"
        print(f"[Q-C {f}] 짝 {v['pairs']}, 평균 차이 {md}, 같은 부호 {v['same_sign']} -> {'충족' if v['met'] else '미충족'}")
    print(f"[결정] {out['decision']}   ※ 개발 시드 결과이며 모든 수치를 그대로 보고한다 (설계서 6.1)")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True, help="실행 폴더 (이미 있으면 거부)")
    t.add_argument("--data-seed", type=int, default=101)
    t.add_argument("--train-seed", type=int, default=0)
    t.add_argument("--split-salt", default="ptn")
    t.add_argument("--split", choices=["port", "time"], default="port")
    t.add_argument("--exclusion", choices=["aub", "truth"], default="aub")
    t.add_argument("--nodes", type=int, default=4)
    t.add_argument("--days", type=int, default=21)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--active-models", default="models")
    t.add_argument("--code-rev", default=None, help="run.json 의 code 에 기록할 코드 버전(없으면 환경변수 PTN_CODE_REV, 둘 다 없으면 unknown)")
    o = sub.add_parser("oracle")
    o.add_argument("--runs", required=True)
    o.add_argument("--seed", type=int, default=ORACLE_SEED)
    o.add_argument("--nodes", type=int, default=4)
    o.add_argument("--days", type=int, default=14)
    e = sub.add_parser("e0")
    e.add_argument("--runs", required=True)
    e.add_argument("--seeds", default=",".join(map(str, DEV_SEEDS)))
    e.add_argument("--nodes", type=int, default=6)
    e.add_argument("--days", type=int, default=14)
    e.add_argument("--out-csv", required=True)
    e.add_argument("--out-json", required=True)
    a = ap.parse_args()
    {"train": cmd_train, "oracle": cmd_oracle, "e0": cmd_e0}[a.cmd](a)


if __name__ == "__main__":
    main()
