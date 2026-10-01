"""
시나리오 생성기 데이터를 시뮬레이션 DB 에 주입 (Kafka 파이프라인 E2E 시연용)

validation/simulator/scenario_generator.py 의 시드 고정 데이터를 DB 의 시간별 테이블
(cowptn_noti_pm_YYYY_MM_DD_HH, cowptn_noti_pm_optic_power_YYYY_MM_DD_HH)에 넣는다.
정답(labels/episodes)은 --out 폴더에 함께 저장되어, 나중에 같은 데이터를 평가에 쓸 수 있다.

안전장치: 대상 DB 이름에 'test' 가 없으면 거부한다 (운영 DB 보호). 기존 데이터는 지우지 않고 추가만 한다.
대상 DB 는 환경변수로 지정: SIM_DB_HOST / SIM_DB_USER / SIM_DB_PASS / SIM_DB_NAME / SIM_DB_PORT

사용법:
  python validation/cli/seed_db.py --days 3                  # 지금까지 3일치를 DB 에 주입
  python validation/cli/seed_db.py --days 7 --seed 11 --yes
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from validation.simulator.config_sim import DB_CONFIG_SIM, SIGNAL_TYPES
from validation.simulator.db_manager import SimulatorDBManager
from validation.simulator.scenario_generator import ScenarioConfig, generate, save


def to_rows(df, columns):
    """numpy/pandas 타입을 DB 드라이버가 받는 파이썬 기본 타입으로 변환"""
    out = []
    for r in df[columns].itertuples(index=False, name=None):
        out.append(tuple(v.to_pydatetime() if hasattr(v, "to_pydatetime") else
                         (v.item() if hasattr(v, "item") else v) for v in r))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nodes", type=int, default=10, help="노드 수 (노드당 10포트)")
    ap.add_argument("--warmup-days", type=float, default=1.0, help="초반 정상 구간(일). 짧은 시연 데이터에서 장애를 빨리 보려면 줄임")
    ap.add_argument("--mean-gap-days", type=float, default=7.0, help="포트별 평균 장애 간격(일). 시연에서 장애를 자주 보려면 줄임")
    ap.add_argument("--out", default=None, help="정답/메타 저장 폴더 (기본 validation/runs/seed_db_<seed>)")
    ap.add_argument("--yes", action="store_true", help="확인 질문 없이 진행")
    args = ap.parse_args()

    db_name = DB_CONFIG_SIM["database"]
    if "test" not in db_name.lower():
        raise SystemExit(f"[!] 대상 DB '{db_name}' 에 'test' 가 없어 중단합니다 (운영 DB 보호). SIM_DB_NAME 을 확인하세요.")

    end = datetime.now().replace(second=0, microsecond=0)
    end -= timedelta(minutes=end.minute % 15)
    start = end - timedelta(days=args.days)
    cfg = ScenarioConfig(seed=args.seed, nodes=args.nodes, days=args.days, start=start.strftime("%Y-%m-%d %H:%M:%S"),
                         warmup_days=args.warmup_days, mean_gap_days=args.mean_gap_days)
    print(f"[*] 대상: {DB_CONFIG_SIM['host']}:{DB_CONFIG_SIM['port']}/{db_name} | 기간 {start} ~ {end} | "
          f"포트 {cfg.nodes * cfg.ports_per_node}개 | seed={cfg.seed}")
    if not args.yes and input("추가(INSERT)합니다. 진행? (y/n): ").lower() != "y":
        raise SystemExit("취소")

    data = generate(cfg)
    out_dir = args.out or os.path.join("validation", "runs", f"seed_db_s{cfg.seed}")
    save(data, cfg, out_dir)

    db = SimulatorDBManager()
    inserted = {"traffic": 0, "optical": 0}
    for kind, df in (("traffic", data["traffic"]), ("optical", data["optical"])):
        df = df.copy()
        df["_hour"] = df["occur_date"].dt.floor("h")
        for hour, g in df.groupby("_hour"):
            if kind == "traffic":
                table = db.ensure_traffic_table(hour)
                g = g.assign(signal_type=SIGNAL_TYPES["ETH"])
                rows = to_rows(g, ["occur_date", "ip_addr", "cid", "lid", "signal_type", "tx_packet", "rx_packet", "error_packet"])
                db.insert_traffic(table, rows)
            else:
                table = db.ensure_optical_table(hour)
                rows = to_rows(g, ["occur_date", "ip_addr", "cid", "lid", "tx_avg_power", "rx_avg_power"])
                db.insert_optical(table, rows)
            inserted[kind] += len(rows)
    print(f"[OK] inserted traffic={inserted['traffic']:,} optical={inserted['optical']:,} rows | 정답 저장: {out_dir}")


if __name__ == "__main__":
    main()
