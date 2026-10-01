"""
배치 평가 vs 스트리밍(Consumer 방식) 동등성 검사

evaluate_model.py 는 전체 시계열을 한 번에 detect() 하지만(latest_only=False), 실제 운영의
Consumer 는 포트별 16행 윈도우로 건별 호출한다(latest_only=True). 두 방식의 결과가 같은지 검증한다.

충실도를 위해 실제 PTNKafkaConsumer.process_message 와 실제 WindowStateManager 를 그대로 사용하고,
Kafka/Redis/DB/웹훅만 메모리 대역으로 교체한다 (Docker 불필요).
  - 입력 변환은 Producer 와 동일: 트래픽/광 outer 병합, 결측은 null
  - 결과는 Consumer 가 save_results 로 저장하려는 DataFrame 을 가로채서 수집

사용법:
  python validation/cli/check_stream_equivalence.py --ports 12
  python validation/cli/check_stream_equivalence.py --ports 12 --required-size 27   # 윈도우 크기 변경 실험
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

from validation.simulator.scenario_generator import load
from validation.evaluation.metrics import summarize, KEY


class FakeRedis:
    """WindowStateManager 가 쓰는 Redis 기능(lpush/ltrim/lrange/llen/delete/pipeline)만 구현한 메모리 대역"""
    def __init__(self, *_a, **_k): self.d = {}
    def ping(self): return True
    def pipeline(self): return _Pipe(self)
    def lpush(self, k, v): self.d.setdefault(k, []).insert(0, v)
    def ltrim(self, k, a, b): self.d[k] = self.d.get(k, [])[a:b + 1]
    def lrange(self, k, a, b): return list(self.d.get(k, [])) if b == -1 else self.d.get(k, [])[a:b + 1]
    def llen(self, k): return len(self.d.get(k, []))
    def delete(self, k): self.d.pop(k, None)


class _Pipe:
    def __init__(self, r): self.r, self.ops = r, []
    def lpush(self, k, v): self.ops.append(("lpush", k, v))
    def ltrim(self, k, a, b): self.ops.append(("ltrim", k, a, b))
    def execute(self):
        for op, *args in self.ops: getattr(self.r, op)(*args)
        self.ops = []


class FakeDB:
    """Consumer 가 저장하려는 결과를 수집"""
    def __init__(self): self.rows = []
    def save_results(self, df): self.rows.append(df.copy())


def build_consumer(models_dir, required_size):
    from src.config import PATHS
    if models_dir:
        for ft in ("traffic", "optical"):
            PATHS[ft] = {"model": os.path.join(models_dir, f"{ft}_ae.pth"),
                         "scaler": os.path.join(models_dir, f"{ft}_scaler.joblib")}
    import src.pipeline.window_state as ws
    import src.pipeline.kafka_consumer as kc
    from src.pipeline.inference import AnomalyDetector
    ws.redis.Redis = FakeRedis                                   # 실제 WindowStateManager + 메모리 Redis
    kc.requests.post = lambda *a, **k: None                      # 웹훅 전송 무시
    c = kc.PTNKafkaConsumer.__new__(kc.PTNKafkaConsumer)         # Kafka/Redis 접속(__init__) 우회
    c.db, c.detector, c.window_manager = FakeDB(), AnomalyDetector(), ws.WindowStateManager()
    from src.config import MODEL_CONFIG
    from src.data.data_processor import DataProcessor
    c.required_size = required_size or DataProcessor.required_rows(MODEL_CONFIG.get("window_size", 12))   # Consumer 기본값과 동일
    c._last_alarm_level = {}
    return c


def to_records(traffic, optical):
    """Producer.fetch_and_produce 와 동일한 변환: outer 병합 -> 결측 null -> occur_date 문자열"""
    df = traffic.merge(optical, on=KEY + ["occur_date"], how="outer").sort_values(KEY + ["occur_date"])
    df = df.astype(object).where(df.notna(), None)
    out = []
    for r in df.to_dict("records"):
        r["occur_date"] = str(r["occur_date"])
        out.append(r)
    return out


def ev_frame(pred, labels):
    ev = pred[KEY + ["occur_date"]].copy()
    ev["alarm"] = pred["is_anomaly"].astype(bool).values
    ev["score"] = pred["severity"].astype(float).values
    return ev.merge(labels[KEY + ["occur_date", "state", "episode_id", "scenario", "nuisance"]],
                    on=KEY + ["occur_date"], how="inner")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join("validation", "runs", "seed7_n6_d14"))
    ap.add_argument("--tag", default="active", help="비교 대상 배치 예측 (predictions_<tag>.csv)")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--ports", type=int, default=12, help="재생할 포트 수 (무작위 선택)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--required-size", type=int, default=None, help="Consumer 윈도우 크기 (기본: DataProcessor.required_rows, 12+15=27)")
    ap.add_argument("--limit-steps", type=int, default=None, help="포트당 재생할 최대 행 수 (속도 확인용)")
    args = ap.parse_args()

    data = load(args.dataset)
    batch = pd.read_csv(os.path.join(args.dataset, f"predictions_{args.tag}.csv"), parse_dates=["occur_date"])
    ports = data["labels"][KEY].drop_duplicates().reset_index(drop=True)
    pick = ports.sample(n=min(args.ports, len(ports)), random_state=args.seed)
    print(f"[*] 재생 포트 {len(pick)}개 | dataset={args.dataset} | batch tag={args.tag}")

    consumer = build_consumer(args.models_dir, args.required_size)
    print(f"[*] Consumer required_size(윈도우 행 수) = {consumer.required_size}")

    t0, n_msg = time.time(), 0
    for _, p in pick.iterrows():
        sel = lambda df: df[(df.ip_addr == p.ip_addr) & (df.cid == p.cid) & (df.lid == p.lid)]
        recs = to_records(sel(data["traffic"]), sel(data["optical"]))
        if args.limit_steps: recs = recs[:args.limit_steps]
        key = f"{p.ip_addr}:{p.cid}:{p.lid}"
        for r in recs:
            consumer.process_message(key, r)      # 실제 Consumer 로직 (Redis 윈도우 -> detect -> 저장/웹훅)
            n_msg += 1
    print(f"[*] 재생 완료: {n_msg:,}건, {time.time() - t0:.0f}s ({(time.time() - t0) / max(n_msg, 1) * 1000:.0f} ms/건)")

    if not consumer.db.rows:
        raise SystemExit("[!] Consumer 가 저장한 결과가 없음")
    stream = pd.concat(consumer.db.rows, ignore_index=True)
    stream["occur_date"] = pd.to_datetime(stream["occur_date"])
    for c in ("is_anomaly", "is_traffic_anomaly", "is_optical_anomaly"):
        stream[c] = stream[c].astype(bool)

    # ---- 배치 결과와 같은 (포트, 시각) 만 비교 ----
    b = batch.merge(pick, on=KEY)                           # 재생한 포트만
    for c in ("is_anomaly", "is_traffic_anomaly", "is_optical_anomaly"):
        b[c] = b[c].astype(bool)
    m = b.merge(stream, on=KEY + ["occur_date"], suffixes=("_b", "_s"))
    print(f"\n[비교 대상] 배치 {len(b):,}행 / 스트리밍 {len(stream):,}행 / 공통 {len(m):,}행 "
          f"(배치에만 있음 {len(b) - len(m):,}, 스트리밍에만 있음 {len(stream) - len(m):,})")

    def agree(col, label):
        x, y = m[f"{col}_b"], m[f"{col}_s"]
        both, only_b, only_s = int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum())
        print(f"  {label:<22} 일치율 {(x == y).mean():6.2%} | 둘 다 알람 {both:>5,} | 배치만 {only_b:>5,} | 스트리밍만 {only_s:>5,}")
        return {"agree": float((x == y).mean()), "both": both, "only_batch": only_b, "only_stream": only_s}

    print("\n[알람 일치도 (같은 포트·시각)]")
    res = {"final_alarm": agree("is_anomaly", "최종 알람(댐프닝 후)"),
           "traffic_track": agree("is_traffic_anomaly", "traffic 트랙 플래그"),
           "optical_track": agree("is_optical_anomaly", "optical 트랙 플래그")}
    d = (m["severity_b"] - m["severity_s"]).abs()
    corr = float(np.corrcoef(m["severity_b"], m["severity_s"])[0, 1])
    print(f"\n[심각도] 평균 절대차 {d.mean():.2f} | 중앙값 {d.median():.2f} | 95% {d.quantile(.95):.1f} | 상관계수 {corr:.3f}")
    res["severity"] = {"mae": float(d.mean()), "corr": corr}

    # ---- 같은 포트 부분집합에서 평가 지표 비교 ----
    eps = data["episodes"].merge(pick, on=KEY)
    rb, rs = summarize(ev_frame(b[KEY + ["occur_date", "is_anomaly", "severity"]], data["labels"]), eps), \
        summarize(ev_frame(stream[KEY + ["occur_date", "is_anomaly", "severity"]], data["labels"]), eps)
    print(f"\n[이벤트 지표: 같은 {len(pick)}개 포트, 에피소드 {len(eps)}건]")
    print(f"{'':<10}{'조기탐지':>9}{'탐지':>8}{'리드(분)':>10}{'오탐/포트일':>12}{'이벤트P':>9}{'이벤트F1':>10}")
    for name, r in (("배치", rb), ("스트리밍", rs)):
        e = r["event"]
        print(f"{name:<10}{e['early_detection_rate']:>9.1%}{e['episode_detection_rate']:>8.1%}"
              f"{(e['lead_time_median_min'] or 0):>10.0f}{e['false_incidents_per_port_day']:>12.3f}"
              f"{e['incident_precision']:>9.1%}{e['event_f1']:>10.3f}")
    res["event"] = {k: {kk: vv for kk, vv in v["event"].items() if kk not in ("_episodes_detail", "by_scenario")}
                    for k, v in (("batch", rb), ("stream", rs))}

    out = os.path.join(args.dataset, f"stream_equivalence_{args.tag}_w{consumer.required_size}.json")
    json.dump({"required_size": consumer.required_size, "ports": len(pick), "rows": int(len(m)), **res},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    stream.to_csv(out.replace(".json", "_stream_predictions.csv"), index=False)
    print(f"\n[*] 저장: {out}")


if __name__ == "__main__":
    main()
