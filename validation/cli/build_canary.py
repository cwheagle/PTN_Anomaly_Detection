"""
승격 게이트 G4 용 카나리 데이터 생성 (P1-1, 설계서 2.10)

시뮬레이터로 시드 고정 데이터를 만들어 트랙별 `<ft>_canary.csv`(원본 성능 컬럼 + label)로 저장한다.
`promotion_gate._canary_auprc` 가 이 파일을 읽어 후보/활성 모델의 AUPRC 를 비교한다.

- 시드는 학습·평가·검증 시드(7, 11, 23, 31, 47)와 겹치지 않는 900 을 기본으로 한다.
- 크기 상한: 기본 4노드 x 10포트 = 40포트 x 7일 (트랙당 약 2.7만 행). 게이트가 1분 안에 끝나도록 키우지 않는다.
- label = 스텝이 장애(state > 0)이고 그 시나리오가 해당 트랙의 것일 때 1.
  (광 열화는 트래픽 모델이, 트래픽/에러 장애는 광 모델이 볼 수 없으므로 반대 트랙에서는 양성으로 세지 않는다.
   `check_retrain_policy.py` 의 트랙 매핑과 같다.)
- 한계: 시뮬레이터가 만든 장애 유형만 대표한다 (설계서 8장). 실데이터 카나리는 실 이력 확보 후 교체.

사용법:
  python validation/cli/build_canary.py [--out-dir models] [--seed 900] [--nodes 4] [--days 7]
"""
import argparse
import os
import sys

import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from validation.simulator.scenario_generator import ScenarioConfig, generate

TRACK_SCENARIOS = {
    "traffic": ("crc_error", "traffic_drop"),
    "optical": ("optical_degradation",),
}
KEY = ["ip_addr", "cid", "lid"]
MAX_PORTS, MAX_DAYS = 40, 7  # 게이트 시간 상한


def build(seed=900, nodes=4, days=7, ports_per_node=10):
    """트랙별 카나리 DataFrame 을 만든다. 양성/음성 중 하나라도 없으면 AUPRC 가 의미 없으므로 예외."""
    if nodes * ports_per_node > MAX_PORTS or days > MAX_DAYS:
        raise ValueError(f"카나리 크기 상한 초과: {nodes * ports_per_node}포트 x {days}일 (상한 {MAX_PORTS}포트 x {MAX_DAYS}일)")
    data = generate(ScenarioConfig(seed=seed, nodes=nodes, ports_per_node=ports_per_node, days=days))
    labels = data["labels"]
    out = {}
    for ft, scenarios in TRACK_SCENARIOS.items():
        lab = labels[KEY + ["occur_date"]].copy()
        lab["label"] = ((labels["state"] > 0) & labels["scenario"].isin(scenarios)).astype(int).values
        df = data[ft].merge(lab, on=KEY + ["occur_date"], how="left")
        df["label"] = df["label"].fillna(0).astype(int)
        if df["label"].nunique() < 2:
            raise ValueError(f"{ft} 카나리에 양성 또는 음성이 없습니다 (seed={seed}). 다른 시드/기간을 쓰세요.")
        out[ft] = df
    return out


def write(out_dir, **kwargs):
    """카나리를 `<out_dir>/<ft>_canary.csv` 로 저장하고 요약(dict)을 반환"""
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for ft, df in build(**kwargs).items():
        path = os.path.join(out_dir, f"{ft}_canary.csv")
        df.to_csv(path, index=False)
        summary[ft] = {"path": path, "rows": int(len(df)), "ports": int(df[KEY].drop_duplicates().shape[0]),
                       "positive_ratio": float(df["label"].mean())}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="models", help="저장 폴더 (게이트가 읽는 모델 폴더)")
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--nodes", type=int, default=4, help="노드 수 (노드당 10포트)")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    for ft, s in write(args.out_dir, seed=args.seed, nodes=args.nodes, days=args.days).items():
        print(f"[OK] {ft}: {s['rows']:,}행 / {s['ports']}포트 / 양성 {s['positive_ratio']:.1%} -> {s['path']}")


if __name__ == "__main__":
    main()
