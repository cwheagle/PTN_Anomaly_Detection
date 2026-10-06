"""
재학습 의심 구간(장애 제외) 품질 검증 (P1-1 단위 D, 설계서 6.1)

시드 고정 시뮬레이터 데이터(정답 있음)에 `train_window.intervals_from_rules`(규칙 B)와
명시 구간(C, 정답 에피소드)을 적용해 측정한다. DB 불필요, 파일 저장 없음.
규칙 A(자기 알람 이력)는 학습된 모델의 알람이 필요한데 이 도구는 모델 없이 동작하므로 측정하지 않는다
(A∪B 는 모델이 있는 환경에서 `--alarms-csv` 로 알람 행을 주면 계산).

[규칙 파라미터 선택 규칙 — 데이터를 보기 전에 고정 (lessons #30)]
  후보: 아래 GRID. 개발 시드(7, 11) 두 개 모두에서 다음을 만족하는 후보만 통과.
    (1) 장애 스텝 제외율 >= 90%   (2) nuisance 스텝 제외율 <= 10%   (3) 트랙별 전체 제외율 < 20%
  통과 후보 중: nuisance 제외율(두 시드 평균) 최소 -> 동률이면 장애 제외율 최대 -> 동률이면 기본값(설계서 5장)에 가까운 쪽.
  통과 후보가 없으면 (장애 제외율 - nuisance 제외율) 평균이 최대인 후보를 고르고 "기준 미달" 로 보고.
  선택 후 검증 시드(23, 31, 47)에서 같은 지표를 확인하며, 검증 결과로 파라미터를 다시 고르지 않는다.

[지표 정의]
  장애 스텝 = labels.state > 0. 시나리오가 속한 트랙(crc_error/traffic_drop -> traffic, optical_degradation -> optical)
  의 의심 구간에 포함되면 제외된 것으로 센다 (DataCollector 가 트랙별 원본에 규칙을 적용하므로).
  nuisance 스텝(버스트/광 흔들림) 은 라벨에 트랙이 없어 두 트랙 중 하나라도 제외하면 제외로 센다 (상한 추정, 보수적).
  전체 제외율 = 트랙별 원본 행 기준 suspect_stats 의 fraction (두 트랙 중 큰 값).

사용법:
  python validation/cli/check_retrain_policy.py [--dev 7,11] [--val 23,31,47] [--nodes 6] [--days 14]
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from src.data import train_window
from src.pipeline.retrain_policy import RetrainPolicy
from validation.simulator.scenario_generator import ScenarioConfig, generate

GRID = {
    "rule_error_ge": [10.0, 20.0, 40.0],
    "rule_rx_drop_db": [2.0, 3.0, 4.0],
    "rule_traffic_ratio": [0.15, 0.3, 0.5],
    "rule_min_consecutive": [3, 4, 6],
}
DESIGN_DEFAULTS = {"rule_error_ge": 20.0, "rule_rx_drop_db": 3.0, "rule_traffic_ratio": 0.15, "rule_min_consecutive": 4}
TRAFFIC_SCENARIOS = ("crc_error", "traffic_drop")
MIN_FAULT, MAX_NUISANCE, MAX_TOTAL = 0.90, 0.10, 0.20


def make_data(seed, nodes=6, days=14):
    return generate(ScenarioConfig(seed=seed, nodes=nodes, days=days))


def rule_intervals(data, policy):
    """트랙별 원본에 규칙 B 적용 (DataCollector 와 동일)"""
    return {ft: train_window.intervals_from_rules(data[ft], policy) for ft in ("traffic", "optical")}


def explicit_intervals(data, policy):
    """정답 에피소드를 명시 구간(C)으로 변환 — 장애 시작 전 pre_steps 만큼 확장"""
    ep = data["episodes"]
    if ep is None or len(ep) == 0:
        return train_window._empty_intervals()
    iv = ep[train_window.PORT_KEYS].copy()
    iv["start_time"] = pd.to_datetime(ep["t_start"]) - policy.suspect_pre_steps * train_window.STEP
    iv["end_time"] = pd.to_datetime(ep["t_end"]) + policy.suspect_post_steps * train_window.STEP
    iv["source"] = "explicit"
    return iv


def _rate(mask, sel):
    n = int(sel.sum())
    return float(mask[sel].sum() / n) if n else float("nan")


def coverage(data, track_ivs, policy):
    """track_ivs: {'traffic': 구간 DF, 'optical': 구간 DF}. Returns 지표 dict"""
    labels = data["labels"]
    m = {ft: train_window.exclusion_mask(labels, iv) for ft, iv in track_ivs.items()}
    state = labels["state"].values
    scen = labels["scenario"].values
    fault = state > 0
    is_traffic_fault = fault & np.isin(scen, TRAFFIC_SCENARIOS)
    is_optical_fault = fault & ~np.isin(scen, TRAFFIC_SCENARIOS)
    fault_excl = (m["traffic"][is_traffic_fault].sum() + m["optical"][is_optical_fault].sum()) / max(int(fault.sum()), 1)
    nuis = labels["nuisance"].values > 0
    nuis_mask = m["traffic"] | m["optical"]
    out = {
        "fault_excl": float(fault_excl),
        "fault_excl_traffic": _rate(m["traffic"], is_traffic_fault),
        "fault_excl_optical": _rate(m["optical"], is_optical_fault),
        "nuisance_excl": _rate(nuis_mask, nuis),
        "fault_steps": int(fault.sum()), "nuisance_steps": int(nuis.sum()),
    }
    totals = {}
    for ft, iv in track_ivs.items():
        st = train_window.suspect_stats(data[ft], iv, policy.port_drop_fraction)
        totals[ft] = st["fraction"]
        out[f"dropped_ports_{ft}"] = len(st["dropped_ports"])
    out["total_excl_traffic"], out["total_excl_optical"] = totals["traffic"], totals["optical"]
    out["total_excl"] = max(totals.values())
    return out


def passes(c):
    return c["fault_excl"] >= MIN_FAULT and c["nuisance_excl"] <= MAX_NUISANCE and c["total_excl"] < MAX_TOTAL


def select_rule_params(dev_datasets, base_policy, grid=None):
    """개발 시드 전부에서 통과하는 후보 중 선택 규칙에 따라 고른다.
    Returns: (선택 파라미터 dict, 통과 여부, 후보 결과 리스트)"""
    grid = grid or GRID
    keys = list(grid)
    defaults = {k: DESIGN_DEFAULTS[k] for k in keys}
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        pol = base_policy.with_(**params)
        covs = [coverage(d, rule_intervals(d, pol), pol) for d in dev_datasets]
        results.append({
            "params": params, "covs": covs,
            "all_pass": all(passes(c) for c in covs),
            "fault": float(np.mean([c["fault_excl"] for c in covs])),
            "nuis": float(np.mean([c["nuisance_excl"] for c in covs])),
            "dist": sum(1 for k in keys if params[k] != defaults[k]),
        })
    ok = [r for r in results if r["all_pass"]]
    if ok:
        best = min(ok, key=lambda r: (r["nuis"], -r["fault"], r["dist"]))
        return best["params"], True, results
    best = max(results, key=lambda r: (r["fault"] - r["nuis"], -r["dist"]))
    return best["params"], False, results


def evaluate_methods(data, policy, alarm_rows=None):
    """한 데이터셋에서 B 단독 / 명시 구간 / (알람 행이 있으면) A, A∪B 를 측정"""
    rule = rule_intervals(data, policy)
    expl = explicit_intervals(data, policy)
    methods = {"B 규칙 단독": rule, "C 명시 구간(정답)": {ft: expl for ft in rule}}
    if alarm_rows is not None:
        a = train_window.intervals_from_alarms(alarm_rows, policy.suspect_pre_steps, policy.suspect_post_steps,
                                               policy.suspect_gap_steps)
        methods["A 알람 단독"] = {ft: a for ft in rule}
        methods["A∪B"] = {ft: train_window.merge_intervals(a, rule[ft]) for ft in rule}
    return {name: coverage(data, ivs, policy) for name, ivs in methods.items()}


def _fmt(c):
    return (f"장애 {c['fault_excl']:.1%} (traffic {c['fault_excl_traffic']:.1%} / optical {c['fault_excl_optical']:.1%})"
            f" | nuisance {c['nuisance_excl']:.1%} | 전체 {c['total_excl']:.1%}"
            f" (t {c['total_excl_traffic']:.1%}/o {c['total_excl_optical']:.1%})"
            f" | 만성포트 {c['dropped_ports_traffic']}/{c['dropped_ports_optical']}"
            f" | {'PASS' if passes(c) else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description="재학습 의심 구간 품질 검증 (P1-1 D, 설계서 6.1)")
    ap.add_argument("--dev", default="7,11")
    ap.add_argument("--val", default="23,31,47")
    ap.add_argument("--nodes", type=int, default=6)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--alarms-csv", default=None, help="모델 알람 행 CSV(occur_date, ip_addr, cid, lid) — 지정 시 A, A∪B 도 측정 (시드 단일일 때만 의미)")
    args = ap.parse_args()

    base = RetrainPolicy().with_(**DESIGN_DEFAULTS)   # 선택 규칙의 '기본값 근접' 기준은 설계서 값
    dev_seeds = [int(s) for s in args.dev.split(",")]
    val_seeds = [int(s) for s in args.val.split(",")]
    alarm_rows = pd.read_csv(args.alarms_csv) if args.alarms_csv else None

    print(f"[선택 규칙] 개발 시드 {dev_seeds} 모두에서 장애>={MIN_FAULT:.0%}, nuisance<={MAX_NUISANCE:.0%}, 전체<{MAX_TOTAL:.0%} 통과 후보 중 "
          f"nuisance 최소 -> 장애 최대 -> 기본값 근접")
    dev = [make_data(s, args.nodes, args.days) for s in dev_seeds]
    params, ok, results = select_rule_params(dev, base)
    n_pass = sum(r["all_pass"] for r in results)
    print(f"[개발] 후보 {len(results)}개 중 통과 {n_pass}개 -> 선택 {params} ({'기준 충족' if ok else '기준 미달: 차선 후보'})")
    default_params = DESIGN_DEFAULTS
    d = next((r for r in results if r["params"] == default_params), None)
    if d:
        print(f"[개발] 설계서 기본값 {default_params}: 장애 {d['fault']:.1%}, nuisance {d['nuis']:.1%}, 통과 {d['all_pass']}")

    chosen = base.with_(**params)
    rows = []
    for tag, seeds, bypass in (("개발", dev_seeds, dev), ("검증", val_seeds, None)):
        for i, s in enumerate(seeds):
            data = bypass[i] if bypass else make_data(s, args.nodes, args.days)
            for name, c in evaluate_methods(data, chosen, alarm_rows if len(seeds) == 1 else None).items():
                rows.append((tag, s, name, c))
                print(f"  [{tag} seed {s}] {name}: {_fmt(c)}")
    print("\n[요약] 방법별 평균 (검증 시드)")
    for name in sorted({r[2] for r in rows}):
        cs = [r[3] for r in rows if r[0] == "검증" and r[2] == name]
        print(f"  {name}: 장애 {np.mean([c['fault_excl'] for c in cs]):.1%} | "
              f"nuisance {np.mean([c['nuisance_excl'] for c in cs]):.1%} | 전체 {np.mean([c['total_excl'] for c in cs]):.1%} | "
              f"전 시드 PASS {all(passes(c) for c in cs)}")
    if alarm_rows is None:
        print("\n[주의] 모델/알람 이력이 없어 규칙 A, A∪B 는 측정하지 않음 (B 단독과 명시 구간만).")


if __name__ == "__main__":
    main()
