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
  qd      6.5 Q-D(편향) 판정 + 2단계 후보 목록 (기존 파일만 읽음)
  alt     T1: 홀드아웃 백분위 사후 계산 → 실행 폴더에 alt_T1.json 만 추가 (T2 는 train --val-fraction 0.20 --min-val-ports 8)
  evalalt 개발 시드 7·11 에서 T1/T2 평가
  select  6.2 규칙 1~5 (+T1 전용 6) 적용, 탐색 표시(exploratory)
  verify  새 검증 시드 53·59·61 **1회** (select PASS 일 때만, 잠금 p1_5_evidence.json) — 별도 지시 전에는 실행하지 않는다

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


def threshold_from_csv(run_dir, ft, csv_path):
    """CSV 를 Trainer 와 같은 경로(저장 스케일러, create_sequences(is_train=True, fit_scaler=False))로 시퀀스화해
    마지막 시점 MSE 의 threshold_percentile 백분위를 계산한다. Returns: (임계치, 시퀀스 수, 메타).
    학습 CSV 에 쓰면 운영 산출식과 같다는 증거(lessons #31, 3.3-5), 홀드아웃 CSV 에 쓰면 T1 의 임계치(th_t1)다."""
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
    clean = proc.preprocess(pd.read_csv(csv_path), is_train=True)
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
    return float(np.percentile(mses, cfg["threshold_percentile"])), int(len(seqs)), meta


def recompute_threshold(run_dir, ft):
    """학습 CSV 로 임계치 재계산 (3.3-5). Returns: (임계치, 메타)"""
    th, _, meta = threshold_from_csv(run_dir, ft, os.path.join(run_dir, "train_data", f"{ft}_train.csv"))
    return th, meta


# ─────────────────────────────────────────────
# train
# ─────────────────────────────────────────────
def holdout_port_count(data_seed, nodes, days, fraction, salt):
    """시뮬레이터 포트 키에 split_ports 의 해시 규칙을 적용한 홀드아웃 포트 수 (측정·학습 아님, 데이터 생성만)"""
    from src.data import train_window
    data = cp.make_data(data_seed, nodes, days)
    ports = data["traffic"][cp.KEY].drop_duplicates()
    return int(sum(train_window.is_val_port(r.ip_addr, r.cid, r.lid, fraction, salt) for r in ports.itertuples(index=False))), len(ports)


def check_train_args(a):
    val_fraction = getattr(a, "val_fraction", 0.10)
    min_val_ports = getattr(a, "min_val_ports", None)
    if a.data_seed in NOT_FOR_TRAINING:
        raise SystemExit(f"[!] --data-seed {a.data_seed} 는 학습에 쓸 수 없습니다 (평가·검증·오라클·금지 시드: {list(NOT_FOR_TRAINING)}).")
    if os.path.exists(a.out):
        raise SystemExit(f"[!] --out {a.out} 가 이미 있습니다 — 덮어쓰지 않습니다.")
    if not 0 < val_fraction < 1:
        raise SystemExit(f"[!] --val-fraction 은 0 과 1 사이여야 합니다: {val_fraction}")
    if min_val_ports is not None and a.split == "port":      # 홀드아웃이 너무 작으면 학습 전에 거부 (운영 T2 의 SKIP 에 해당, 폴더 안 만듦)
        n, total = holdout_port_count(a.data_seed, a.nodes, a.days, val_fraction, a.split_salt)
        if n < min_val_ports:
            raise SystemExit(f"[!] 홀드아웃 포트 {n}개(전체 {total}, 비율 {val_fraction}, salt {a.split_salt}) < --min-val-ports {min_val_ports} — 학습하지 않습니다 (구조적 SKIP).")


def cmd_train(a):
    check_train_args(a)
    started = datetime.now()
    t0 = time.time()
    res = cp.train_operational(a.out, a.data_seed, a.train_seed, nodes=a.nodes, days=a.days, epochs=a.epochs,
                               active_models_dir=a.active_models, split_salt=a.split_salt, split=a.split, exclusion=a.exclusion,
                               val_fraction=getattr(a, "val_fraction", 0.10))
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
           "epochs": a.epochs, "val_fraction": getattr(a, "val_fraction", 0.10), "min_val_ports": getattr(a, "min_val_ports", None),
           "code": _code_rev(a.code_rev), "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
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
        if _load_json(os.path.join(r, "run.json")).get("val_fraction", 0.10) != 0.10:
            raise SystemExit(f"[!] {r} 는 홀드아웃 비율 0.10 이 아닌 실행입니다 — e0 는 0.10 실행만 받습니다 (T2 실행은 evalalt).")
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
# U2: 2단계 대안 비교 (설계서 6.2·6.5·8.2) — 이번 차수 T1·T2 만
# ─────────────────────────────────────────────
# 판정 기준 (6.2·6.5 확정값, 인자로 바꿀 수 없음)
QD_BIAS, QD_FP_ORACLE, QD_FP_OWN = 0.90, 0.02, 0.02
SEL_FP_MAX, SEL_F1_MARGIN, SEL_EARLY_MARGIN, SEL_AUPRC_MARGIN, SEL_TIE = 0.02, 0.03, 0.05, 0.03, 0.003
SEL_CRITERIA = {"fp_max": SEL_FP_MAX, "f1_margin": SEL_F1_MARGIN, "early_margin": SEL_EARLY_MARGIN, "auprc_margin": SEL_AUPRC_MARGIN,
                "th_max_min": "<= R0 (두 트랙)", "bias_rule6": "중앙값 |ln(th_t1/th*)| < 중앙값 |ln(th_in/th*)| (T1 만, 두 트랙)", "tie": SEL_TIE}
QD_CRITERIA = {"QD_bias_median": QD_BIAS, "QD_fp_oracle_mean": QD_FP_ORACLE, "QD_fp_own_mean": QD_FP_OWN}
EPS = 1e-12                                 # 경계 포함 비교의 부동소수점 오차 허용
ALT_DATA_SEEDS, ALT_TRAIN_SEEDS = TRAIN_DATA_SEEDS, (0, 1)          # 2단계 대상 = R0 6회 (D6)
T2_FRACTION, T2_MIN_PORTS = 0.20, 8
CODE_CHANGE_ORDER = {"T2": 0, "T1": 1, "T3": 2}                      # 동률일 때 운영 코드 변경이 작은 순
VERIFY_SEEDS = P15_VAL_SEEDS
EVIDENCE_LOCK = "p1_5_evidence.json"
METRICS = ("fp", "f1", "early", "auprc")
EXPECTED_KINDS = {"T2": ["T2"], "T1T3T2": ["T1", "T3", "T2"], "stop": []}


def run_spec(run_dir):
    run = _load_json(os.path.join(run_dir, "run.json"))
    run.setdefault("val_fraction", 0.10)
    run.setdefault("min_val_ports", None)
    return run


def is_r0_spec(run, allow_fraction=0.10):
    """2단계 대상 R0 조건: port·ptn·aub, 데이터 101·103·105, 난수 0·1, 홀드아웃 비율 = allow_fraction"""
    return (run["split"] == R0["split"] and run["split_salt"] == R0["split_salt"] and run["exclusion"] == R0["exclusion"]
            and run["data_seed"] in ALT_DATA_SEEDS and run["train_seed"] in ALT_TRAIN_SEEDS and run["val_fraction"] == allow_fraction)


def check_alt_runs(runs, kind):
    """T1: R0 조건(비율 0.10) 실행만 / T2: 같은 조건에 비율 0.20·최소 포트 8 인 실행만. 아니면 측정 전 거부"""
    for r in runs:
        if not os.path.exists(os.path.join(r, "run.json")):
            raise SystemExit(f"[!] {r} 에 run.json 이 없습니다.")
        spec = run_spec(r)
        if kind == "T1":
            ok = is_r0_spec(spec)
        else:
            ok = is_r0_spec(spec, T2_FRACTION) and spec["min_val_ports"] == T2_MIN_PORTS
        if not ok:
            raise SystemExit(f"[!] {r} 는 {kind} 대상 실행이 아닙니다 (조건: split port·salt ptn·exclusion aub, 데이터 {list(ALT_DATA_SEEDS)}, "
                             f"난수 {list(ALT_TRAIN_SEEDS)}, 홀드아웃 비율 {0.10 if kind == 'T1' else T2_FRACTION}"
                             f"{'' if kind == 'T1' else f', min_val_ports {T2_MIN_PORTS}'}).")


# ── qd ──
def judge_qd(bias_by_track, fp_oracle_mean, fp_own_mean, e0_decision):
    """6.5 Q-D: D-1 R0 세트 th_in/th* 중앙값 < 0.90 (어느 한 트랙), D-2 오라클 대입 오탐 평균 <= 0.02, D-3 own 오탐 평균 > 0.02.
    candidates = e0 결정표의 대안 + (Q-D 충족 시) T1 (중복 없음)"""
    if e0_decision not in EXPECTED_KINDS:
        raise SystemExit(f"[!] e0.json 의 decision 이 알 수 없는 값입니다: {e0_decision}")
    med = {ft: float(np.median(v)) for ft, v in bias_by_track.items()}
    d1 = any(m < QD_BIAS for m in med.values())
    d2 = fp_oracle_mean <= QD_FP_ORACLE
    d3 = fp_own_mean > QD_FP_OWN
    met = bool(d1 and d2 and d3)
    cands = list(EXPECTED_KINDS[e0_decision])
    if met and "T1" not in cands:
        cands.append("T1")
    return {"QD": {"bias_median": med, "fp_oracle_mean": float(fp_oracle_mean), "fp_own_mean": float(fp_own_mean),
                   "D1": bool(d1), "D2": bool(d2), "D3": bool(d3), "met": met},
            "criteria": QD_CRITERIA, "e0_decision": e0_decision, "candidates": cands}


def _run_means(df, policy, th_kind, runs=None):
    """실행별 개발 시드 평균(오탐억제형 등) -> DataFrame(index=run)"""
    s = df[(df.policy == policy) & (df.th_kind == th_kind)]
    if runs is not None:
        s = s[s.run.isin(runs)]
    return s.groupby("run").agg(fp=("fp", "mean"), f1=("f1", "mean"), early=("early", "mean"), auprc=("auprc", "mean"),
                                th_traffic=("th_traffic", "first"), th_optical=("th_optical", "first"),
                                data_seed=("data_seed", "first"), train_seed=("train_seed", "first"))


def cmd_qd(a):
    for out in (a.out_json,):
        check_record(out)
    for f in (a.e0_json, a.e0_csv):
        if not os.path.exists(f):
            raise SystemExit(f"[!] {f} 가 없습니다.")
    e0 = _load_json(a.e0_json)
    df = pd.read_csv(a.e0_csv)
    runs = expand_runs(a.runs)
    names = [os.path.basename(os.path.normpath(r)) for r in runs]
    if set(names) != set(e0["R0"]):
        raise SystemExit(f"[!] --runs 가 e0.json 의 R0 집합과 다릅니다: {sorted(set(names) ^ set(e0['R0']))}")
    bias = {ft: [] for ft in TRACKS}
    for r in runs:
        if not os.path.exists(os.path.join(r, "oracle.json")):
            raise SystemExit(f"[!] {r} 에 oracle.json 이 없습니다.")
        for ft in TRACKS:
            bias[ft].append(_load_json(os.path.join(r, "oracle.json"))["tracks"][ft]["bias"])
    own = _run_means(df, FP_POLICY, "own", e0["R0"])
    orc = _run_means(df, FP_POLICY, "oracle", e0["R0"])
    if len(own) != len(e0["R0"]) or len(orc) != len(e0["R0"]):
        raise SystemExit("[!] e0.csv 에 R0 실행의 own/oracle 행이 모두 있지 않습니다.")
    fp_orc, fp_own = float(orc["fp"].mean()), float(own["fp"].mean())
    if abs(fp_orc - e0["QB"]["fp_oracle_mean"]) > 1e-9:
        raise SystemExit(f"[!] e0.csv 의 오라클 오탐 평균 {fp_orc:.6f} 이 e0.json QB.fp_oracle_mean {e0['QB']['fp_oracle_mean']:.6f} 과 다릅니다.")
    out = judge_qd(bias, fp_orc, fp_own, e0["decision"])
    _dump_json(a.out_json, out)
    q = out["QD"]
    print(f"[Q-D] 편향 중앙값 {q['bias_median']} (D-1 {q['D1']}), 오라클 오탐 평균 {q['fp_oracle_mean']:.4f} (D-2 {q['D2']}), "
          f"own 오탐 평균 {q['fp_own_mean']:.4f} (D-3 {q['D3']}) -> {'충족' if q['met'] else '미충족'}; e0 결정 {out['e0_decision']} -> 후보 {out['candidates']}")
    return out


# ── alt T1 ──
def cmd_alt(a):
    if a.kind != "T1":
        raise SystemExit(f"[!] alt --kind 는 T1 만 허용합니다 (받은 값: {a.kind}; T2 는 train 옵션, T3 는 이번 차수 미구현).")
    if not os.path.exists(a.qd):
        raise SystemExit(f"[!] --qd 파일 {a.qd} 가 없습니다.")
    qd = _load_json(a.qd)
    if "T1" not in qd.get("candidates", []):
        raise SystemExit(f"[!] qd 의 candidates {qd.get('candidates')} 에 T1 이 없어 T1 을 실행하지 않습니다.")
    runs = expand_runs(a.runs)
    check_alt_runs(runs, "T1")
    for r in runs:
        for name in ("oracle.json",):
            if not os.path.exists(os.path.join(r, name)):
                raise SystemExit(f"[!] {r} 에 {name} 이 없습니다.")
        if os.path.exists(os.path.join(r, "alt_T1.json")):
            raise SystemExit(f"[!] {os.path.join(r, 'alt_T1.json')} 가 이미 있습니다 — 덮어쓰지 않습니다.")
    for r in runs:                                      # 읽기만: 모델·메타·레지스트리·run.json·oracle.json 은 바꾸지 않고 alt_T1.json 만 추가
        orc = _load_json(os.path.join(r, "oracle.json"))
        test_csv = {ft: os.path.join(r, "train_data", f"{ft}_test.csv") for ft in TRACKS}
        infer = cp.score_data({ft: pd.read_csv(test_csv[ft]) for ft in TRACKS}, r)          # 추론 경로(비교용 th_t1_infer)
        tracks = {}
        for ft in TRACKS:
            th, n_seq, meta = threshold_from_csv(r, ft, test_csv[ft])                          # Trainer 경로 = th_t1
            th_infer = float(np.percentile(infer[ft][0]["mse"], meta["config"]["threshold_percentile"]))
            tracks[ft] = {"th_t1": th, "th_t1_infer": th_infer, "infer_rel_diff": abs(th_infer - th) / max(abs(th), 1e-12), "n_seq": n_seq,
                          "val_ports": _csv_ports(test_csv[ft]), "th_star": orc["tracks"][ft]["th_star"],
                          "bias_t1": th / orc["tracks"][ft]["th_star"], "bias_in": orc["tracks"][ft]["bias"]}
        _dump_json(os.path.join(r, "alt_T1.json"), {"kind": "T1", "tracks": tracks})
        print(f"[OK] {r}: " + ", ".join(f"{ft} th_t1={t['th_t1']:.4f} (th*={t['th_star']:.4f}, 홀드아웃 {t['val_ports']}포트, 추론경로 차이 {t['infer_rel_diff']:.1e})" for ft, t in tracks.items()))


# ── evalalt ──
def eval_run_rows(data, run_dir, name, seed, kind, th_override=None):
    """한 실행을 한 시드에서 평가: 정책(오탐억제형·기본) x 임계치(kind). th_override 가 있으면 트랙별 전역 임계치만 교체"""
    run = run_spec(run_dir)
    scores = cp.score_data(data, run_dir)
    sc = scores if th_override is None else {ft: (df, th_override[ft]) for ft, (df, _) in scores.items()}
    ths = {ft: float(sc[ft][1]) for ft in sc}
    rows = []
    for pname, pol in cp.POLICIES.items():
        sim = simulate_alarms(sc, pol)
        rows.append({"run": name, "data_seed": run["data_seed"], "train_seed": run["train_seed"], "split": run["split"],
                     "split_salt": run["split_salt"], "exclusion": run["exclusion"], "seed": seed, "policy": pname, "th_kind": kind,
                     "th_traffic": ths.get("traffic"), "th_optical": ths.get("optical"), **cp.metrics_row(evaluate_alarms(sim, data))})
    return rows


def check_evalalt_args(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    if not set(seeds) <= set(DEV_SEEDS):
        raise SystemExit(f"[!] evalalt --seeds 는 {list(DEV_SEEDS)} 의 부분집합만 허용합니다 (받은 값: {sorted(seeds)}).")
    if a.kind not in ("T1", "T2"):
        raise SystemExit(f"[!] evalalt --kind 는 T1 또는 T2 만 허용합니다 (받은 값: {a.kind}).")
    check_record(a.out_csv)
    runs = expand_runs(a.runs)
    check_alt_runs(runs, a.kind)
    need = "alt_T1.json" if a.kind == "T1" else "oracle.json"
    for r in runs:
        for name in ("run.json", need):
            if not os.path.exists(os.path.join(r, name)):
                raise SystemExit(f"[!] {r} 에 {name} 이 없습니다.")
    return seeds, runs


def cmd_evalalt(a):
    seeds, runs = check_evalalt_args(a)
    rows = []
    for seed in seeds:
        data = cp.make_data(seed, a.nodes, a.days)
        for r in runs:
            name = os.path.basename(os.path.normpath(r))
            if a.kind == "T1":
                alt = _load_json(os.path.join(r, "alt_T1.json"))
                rows += eval_run_rows(data, r, name, seed, "T1", {ft: alt["tracks"][ft]["th_t1"] for ft in TRACKS})
            else:
                rows += eval_run_rows(data, r, name, seed, "own")
    df = pd.DataFrame(rows).reindex(columns=CSV_COLUMNS)
    df.to_csv(a.out_csv, index=False)
    s = df[df.policy == FP_POLICY].groupby("run")[["auprc", "f1", "early", "fp", "lead"]].mean().round(4)
    print(f"== evalalt {a.kind} 개발 시드 {seeds} 평균 (오탐억제형) =="); print(s.to_string())
    return df


# ── select ──
def summarize_runs(df, policy, th_kind, runs=None):
    """실행별 개발 시드 평균을 먼저 낸 뒤 실행 평균 (8.1 (c) 와 같은 순서) + 트랙별 임계치 max/min"""
    m = _run_means(df, policy, th_kind, runs)
    if m.empty:
        raise SystemExit(f"[!] 행이 없습니다 (policy {policy}, th_kind {th_kind}).")
    out = {k: float(m[k].mean()) for k in METRICS}
    out["th_max_min"] = {"traffic": float(m["th_traffic"].max() / m["th_traffic"].min()),
                         "optical": float(m["th_optical"].max() / m["th_optical"].min())}
    out["runs"] = sorted(m.index)
    out["pairs"] = sorted(zip(m.data_seed.astype(int), m.train_seed.astype(int)))
    return out


def judge_alt(r0, alt, kind, bias=None):
    """규칙 1~5 (T1 은 6 추가). bias = {"t1": {ft: 중앙값 |ln bias_t1|}, "in": {ft: 중앙값 |ln bias_in|}} (T1 만)"""
    rules = {
        "1": alt["fp"] <= SEL_FP_MAX + EPS,
        "2": alt["f1"] >= r0["f1"] - SEL_F1_MARGIN - EPS,
        "3": alt["early"] >= r0["early"] - SEL_EARLY_MARGIN - EPS,
        "4": alt["auprc"] >= r0["auprc"] - SEL_AUPRC_MARGIN - EPS,
        "5": all(alt["th_max_min"][ft] <= r0["th_max_min"][ft] + EPS for ft in TRACKS),
    }
    if kind == "T1":
        bias = bias or alt.get("bias")
        if not bias:
            raise SystemExit("[!] T1 의 규칙 6 계산에 편향(bias) 값이 없습니다.")
        rules["6"] = all(bias["t1"][ft] < bias["in"][ft] for ft in TRACKS)
    rules = {k: bool(v) for k, v in rules.items()}
    return rules


def pick_candidate(cands):
    """cands: {kind: {"fp": ..}} (모든 규칙 충족 대안) -> 오탐 평균 최소, 차이 0.003 이내면 코드 변경이 작은 순(T2 < T1 < T3)"""
    if not cands:
        return None
    best = min(c["fp"] for c in cands.values())
    near = [k for k, c in cands.items() if c["fp"] <= best + SEL_TIE + EPS]
    return sorted(near, key=lambda k: (CODE_CHANGE_ORDER[k], cands[k]["fp"]))[0]


def judge_select(r0, alts, candidates, t2_skip=False):
    """r0: summarize_runs 결과, alts: {kind: summarize_runs 결과 (+ bias)}, candidates: qd 의 후보 목록"""
    out_alts, passing = {}, {}
    for kind in candidates:
        if kind == "T2" and t2_skip:
            out_alts["T2"] = {"skip": "structural"}
            continue
        if kind not in alts:
            raise SystemExit(f"[!] 후보 {kind} 의 evalalt 결과(--alt-csv)가 없습니다 (T2 구조적 SKIP 이면 --t2-skip).")
        alt = alts[kind]
        rules = judge_alt(r0, alt, kind, alt.get("bias"))
        entry = {k: alt[k] for k in (*METRICS, "th_max_min")}
        if kind == "T1":
            entry["bias_abs_log_median"] = alt["bias"]["t1"]
            entry["bias_abs_log_median_in"] = alt["bias"]["in"]
        entry["rules"] = rules
        entry["candidate"] = bool(all(rules.values()))
        out_alts[kind] = entry
        if entry["candidate"]:
            passing[kind] = entry
    selected = pick_candidate(passing)
    return {"R0": {k: r0[k] for k in ("runs", *METRICS, "th_max_min")}, "alts": out_alts, "selected": selected,
            "result": "PASS" if selected else "FAIL", "criteria": SEL_CRITERIA, "exploratory": True}


def _bias_medians(run_dirs):
    t1, inn = {ft: [] for ft in TRACKS}, {ft: [] for ft in TRACKS}
    for r in run_dirs:
        alt = _load_json(os.path.join(r, "alt_T1.json"))
        for ft in TRACKS:
            t1[ft].append(abs(float(np.log(alt["tracks"][ft]["bias_t1"]))))
            inn[ft].append(abs(float(np.log(alt["tracks"][ft]["bias_in"]))))
    return {"t1": {ft: float(np.median(v)) for ft, v in t1.items()}, "in": {ft: float(np.median(v)) for ft, v in inn.items()}}


def check_select_args(a):
    check_record(a.out_json)
    for f in (a.e0_csv, a.qd, *a.alt_csv):
        if not os.path.exists(f):
            raise SystemExit(f"[!] {f} 가 없습니다.")
    qd = _load_json(a.qd)
    if "T3" in qd.get("candidates", []):
        raise SystemExit("[!] qd 의 후보에 T3 가 있으나 이번 차수에는 T3 를 구현하지 않았습니다.")
    return qd


def cmd_select(a):
    qd = check_select_args(a)
    e0 = pd.read_csv(a.e0_csv)
    r0_rows = e0[(e0.split == R0["split"]) & (e0.split_salt == R0["split_salt"]) & (e0.exclusion == R0["exclusion"])
                 & e0.data_seed.isin(ALT_DATA_SEEDS) & e0.train_seed.isin(ALT_TRAIN_SEEDS)]
    r0 = summarize_runs(r0_rows, FP_POLICY, "own")
    if len(r0["runs"]) != len(ALT_DATA_SEEDS) * len(ALT_TRAIN_SEEDS):
        raise SystemExit(f"[!] e0.csv 에서 R0 6회(데이터 {list(ALT_DATA_SEEDS)} x 난수 {list(ALT_TRAIN_SEEDS)})를 찾지 못했습니다: {r0['runs']}")
    alts = {}
    for path in a.alt_csv:
        df = pd.read_csv(path)
        kinds = set(df.th_kind)
        if kinds == {"T1"}:
            kind = "T1"
        elif kinds == {"own"}:
            kind = "T2"
        else:
            raise SystemExit(f"[!] {path} 의 th_kind 가 T1 또는 own(T2) 하나가 아닙니다: {sorted(kinds)}")
        if kind in alts:
            raise SystemExit(f"[!] {kind} 의 evalalt 결과가 두 번 주어졌습니다.")
        s = summarize_runs(df, FP_POLICY, "T1" if kind == "T1" else "own")
        if s["pairs"] != r0["pairs"]:
            raise SystemExit(f"[!] {kind} 실행의 (데이터, 난수) 짝 {s['pairs']} 이 R0 {r0['pairs']} 와 다릅니다.")
        alts[kind] = s
    if a.t2_skip and "T2" in alts:
        raise SystemExit("[!] --t2-skip 과 T2 evalalt 결과가 함께 주어졌습니다.")
    if "T1" in qd["candidates"]:
        if not a.t1_runs:
            raise SystemExit("[!] T1 이 후보이므로 규칙 6 계산용 --t1-runs(R0 6회 폴더, alt_T1.json 필요)가 필요합니다.")
        runs = expand_runs(a.t1_runs)
        for r in runs:
            if not os.path.exists(os.path.join(r, "alt_T1.json")):
                raise SystemExit(f"[!] {r} 에 alt_T1.json 이 없습니다.")
        if "T1" in alts:
            alts["T1"]["bias"] = _bias_medians(runs)
    out = judge_select(r0, alts, qd["candidates"], a.t2_skip)
    _dump_json(a.out_json, out)
    _print_select(out)
    return out


def _print_select(out):
    r0 = out["R0"]
    print(f"== select (탐색, 개발 시드) R0: 오탐 {r0['fp']:.4f} F1 {r0['f1']:.4f} 조기 {r0['early']:.4f} AUPRC {r0['auprc']:.4f} 임계치 max/min {r0['th_max_min']} ==")
    for kind, v in out["alts"].items():
        if "skip" in v:
            print(f"[{kind}] 구조적 SKIP — 후보 불가"); continue
        print(f"[{kind}] 오탐 {v['fp']:.4f} F1 {v['f1']:.4f} 조기 {v['early']:.4f} AUPRC {v['auprc']:.4f} 임계치 max/min {v['th_max_min']} 규칙 {v['rules']} -> {'후보' if v['candidate'] else '탈락'}")
    print(f"[select] selected={out['selected']} result={out['result']} (exploratory)")


# ── verify (구현·테스트만, 실제 실행은 별도 지시 후) ──
def judge_verify(r0, alt):
    """6.3: 선택 대안이 규칙 1~4 를 검증 시드에서 충족 (2~4 는 같은 실행의 R0 대비). 5·6 은 보고만"""
    rules = judge_alt(r0, alt, "T2")                    # 규칙 1~5 (T1 의 규칙 6 은 보고용으로 따로)
    must = {k: rules[k] for k in ("1", "2", "3", "4")}
    return {"rules": must, "report_only": {"5": rules["5"]}, "pass": bool(all(must.values()))}


def check_verify_args(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    if set(seeds) != set(VERIFY_SEEDS) or len(seeds) != len(VERIFY_SEEDS):
        raise SystemExit(f"[!] verify --seeds 는 {list(VERIFY_SEEDS)} 전체만 허용합니다 (받은 값: {sorted(seeds)}).")
    if not os.path.exists(a.select):
        raise SystemExit(f"[!] --select 파일 {a.select} 가 없습니다.")
    sel = _load_json(a.select)
    if sel.get("result") != "PASS" or not sel.get("selected"):
        raise SystemExit(f"[!] select 결과가 PASS 가 아니거나 선택된 대안이 없어 verify 를 실행하지 않습니다 (result={sel.get('result')}, selected={sel.get('selected')}).")
    lock_path = os.path.join(os.path.dirname(os.path.abspath(a.record_json)), EVIDENCE_LOCK)
    check_record(a.record_csv, reserved=[lock_path])
    check_record(a.record_json, reserved=[lock_path])
    if os.path.abspath(a.record_csv) == os.path.abspath(a.record_json):
        raise SystemExit("[!] --record-csv 와 --record-json 은 서로 다른 파일이어야 합니다.")
    runs = expand_runs(a.runs)
    r0_runs = [r for r in runs if run_spec(r)["val_fraction"] == 0.10]
    f20_runs = [r for r in runs if run_spec(r)["val_fraction"] == T2_FRACTION]
    check_alt_runs(r0_runs, "T1")
    if sel["selected"] == "T2":
        check_alt_runs(f20_runs, "T2")
    elif f20_runs:
        raise SystemExit("[!] T1 선택인데 --runs 에 홀드아웃 비율 0.20 실행이 섞여 있습니다.")
    if not r0_runs:
        raise SystemExit("[!] --runs 에 R0 실행이 없습니다.")
    if sel["selected"] == "T1":
        for r in r0_runs:
            if not os.path.exists(os.path.join(r, "alt_T1.json")):
                raise SystemExit(f"[!] {r} 에 alt_T1.json 이 없습니다.")
    elif {(run_spec(r)["data_seed"], run_spec(r)["train_seed"]) for r in f20_runs} != {(run_spec(r)["data_seed"], run_spec(r)["train_seed"]) for r in r0_runs}:
        raise SystemExit("[!] R0 실행과 _f20 실행의 (데이터, 난수) 짝이 다릅니다.")
    return seeds, sel, r0_runs, f20_runs, lock_path


def cmd_verify(a):
    seeds, sel, r0_runs, f20_runs, lock_path = check_verify_args(a)
    lock = cp.evidence_lock(lock_path, {"kind": "P1-5 verify", "seeds": seeds, "selected": sel["selected"], "select": a.select})   # 측정 전에 잠금
    rows = []
    for seed in seeds:
        data = cp.make_data(seed, a.nodes, a.days)
        for r in r0_runs:
            name = os.path.basename(os.path.normpath(r))
            rr = eval_run_rows(data, r, name, seed, "own")
            rows += rr
            if sel["selected"] == "T1":                   # 같은 가중치, 임계치만 교체 -> 순수 짝 비교
                alt = _load_json(os.path.join(r, "alt_T1.json"))
                rows += eval_run_rows(data, r, name, seed, "T1", {ft: alt["tracks"][ft]["th_t1"] for ft in TRACKS})
        if sel["selected"] == "T2":
            for r in f20_runs:
                rows += eval_run_rows(data, r, os.path.basename(os.path.normpath(r)), seed, "own")
        sc = cp.score_data(data, r0_runs[0])
        sim = simulate_alarms(sc, cp.POLICIES[FP_POLICY])
        for b in ("rolling_rule", "fixed_threshold", "always_alarm"):
            rows.append({"run": f"[기준] {b}", "seed": seed, "policy": "-", "th_kind": "-",
                         **cp.metrics_row(evaluate_alarms(baseline_alarms(data, sim, b), data))})
    df = pd.DataFrame(rows).reindex(columns=CSV_COLUMNS)
    real = df[~df.run.str.startswith("[기준]")]
    r0_names = [os.path.basename(os.path.normpath(r)) for r in r0_runs]
    r0_sum = summarize_runs(real[real.run.isin(r0_names)], FP_POLICY, "own")
    if sel["selected"] == "T1":
        alt_sum = summarize_runs(real[real.th_kind == "T1"], FP_POLICY, "T1")
        bias = _bias_medians(r0_runs)
    else:
        f20_names = [os.path.basename(os.path.normpath(r)) for r in f20_runs]
        alt_sum = summarize_runs(real[real.run.isin(f20_names)], FP_POLICY, "own")
        bias = None
    j = judge_verify(r0_sum, alt_sum)
    report = {"6": None if bias is None else bool(all(bias["t1"][ft] < bias["in"][ft] for ft in TRACKS))}
    base = df[df.run.str.startswith("[기준]")].groupby("run")[["auprc", "f1", "early", "fp", "lead"]].mean()
    out = {"status": "done", "seeds": seeds, "selected": sel["selected"], "R0": {k: r0_sum[k] for k in ("runs", *METRICS, "th_max_min")},
           "alt": {k: alt_sum[k] for k in ("runs", *METRICS, "th_max_min")}, "rules": j["rules"],
           "report_only": {**j["report_only"], **{k: v for k, v in report.items() if v is not None}},
           "baselines": base.round(4).to_dict(orient="index"), "result": "PASS" if j["pass"] else "FAIL",
           "criteria": SEL_CRITERIA, "exploratory": False}
    for path in (lock, a.record_json):
        _dump_json(path, out)
    df.to_csv(a.record_csv, index=False)
    print(f"[verify] 검증 시드 {seeds}: {sel['selected']} 규칙 {j['rules']} -> {out['result']}")
    return out


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
    t.add_argument("--val-fraction", type=float, default=0.10, help="포트 홀드아웃 비율 (T2 = 0.20)")
    t.add_argument("--min-val-ports", type=int, default=None, help="홀드아웃 포트가 이보다 적으면 학습 전에 거부 (T2 = 8)")
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
    q = sub.add_parser("qd")
    q.add_argument("--e0-json", required=True)
    q.add_argument("--e0-csv", required=True)
    q.add_argument("--runs", required=True)
    q.add_argument("--out-json", required=True)
    al = sub.add_parser("alt")
    al.add_argument("--kind", required=True)
    al.add_argument("--runs", required=True)
    al.add_argument("--qd", required=True)
    ev = sub.add_parser("evalalt")
    ev.add_argument("--kind", required=True)
    ev.add_argument("--runs", required=True)
    ev.add_argument("--seeds", default=",".join(map(str, DEV_SEEDS)))
    ev.add_argument("--nodes", type=int, default=6)
    ev.add_argument("--days", type=int, default=14)
    ev.add_argument("--out-csv", required=True)
    se = sub.add_parser("select")
    se.add_argument("--e0-csv", required=True)
    se.add_argument("--qd", required=True)
    se.add_argument("--alt-csv", required=True, action="append")
    se.add_argument("--t1-runs", default=None, help="규칙 6(편향) 계산용 R0 6회 폴더(alt_T1.json). T1 이 후보일 때 필요")
    se.add_argument("--t2-skip", action="store_true")
    se.add_argument("--out-json", required=True)
    vf = sub.add_parser("verify")
    vf.add_argument("--select", required=True)
    vf.add_argument("--runs", required=True)
    vf.add_argument("--seeds", default=",".join(map(str, VERIFY_SEEDS)))
    vf.add_argument("--nodes", type=int, default=6)
    vf.add_argument("--days", type=int, default=14)
    vf.add_argument("--record-csv", required=True)
    vf.add_argument("--record-json", required=True)
    a = ap.parse_args()
    {"train": cmd_train, "oracle": cmd_oracle, "e0": cmd_e0, "qd": cmd_qd, "alt": cmd_alt, "evalalt": cmd_evalalt,
     "select": cmd_select, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
