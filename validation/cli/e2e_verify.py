"""
E2E 결과 검증 (실행 중인 스택 + DB + SSE 로그를 불변식으로 판정)

docs/e2e_guide.md 의 절차로 시연 데이터를 흘린 뒤 실행한다. 판정 항목:
  1. API/UI 프록시 응답
  2. 결과 행 수: 포트마다 (스텝 수 - (윈도우 - 1)) 개 (윈도우를 채운 뒤부터 추론)
  3. 알람 이벤트 수가 재알림 간격 설정과 일치 (15분이면 CRITICAL 행 수와 동일)
  4. CLEAR 가 DB 의 실제 복구 시점(CRITICAL -> 비CRITICAL)과 정확히 일치하고, 오해제가 없음
  5. 활성 구간 밖 알람(오탐) 비율 (참고용 지표)

사용법:
  python validation/cli/e2e_verify.py --sse-log validation/runs/e2e_sse.log --seed-dir validation/runs/e2e_seed
  python validation/cli/e2e_verify.py --sse-log ... --seed-dir ... --renotify 60
환경변수 DB_HOST/DB_USER/DB_PASS/DB_NAME/DB_PORT 로 DB 지정 (기본: localhost / root / root / cowptn_test)
"""
import argparse
import json
import os
import re
import sys
import urllib.request

import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

KEY = ["ip_addr", "cid", "lid"]
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def http_json(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sse-log", required=True, help="/api/stream/alarms 를 받아 저장한 파일")
    ap.add_argument("--seed-dir", required=True, help="seed_db.py 가 저장한 정답 폴더 (episodes.csv, labels.csv)")
    ap.add_argument("--ip-like", default="10.20.0.%", help="시연 데이터 IP 패턴")
    ap.add_argument("--renotify", type=int, default=15, help="스택에 설정한 ALARM_RENOTIFY_MINUTES")
    ap.add_argument("--window", type=int, default=27, help="Consumer 윈도우 행 수 (DataProcessor.required_rows)")
    ap.add_argument("--since", default=None, help="이 시각 이후 결과만 판정 (이전 실행의 잔여 행 제외, 예: '2026-10-05 10:00:00')")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--ui", default="http://localhost")
    args = ap.parse_args()

    import mysql.connector
    c = mysql.connector.connect(host=os.getenv("DB_HOST", "localhost"), user=os.getenv("DB_USER", "root"),
                                password=os.getenv("DB_PASS", "root"), database=os.getenv("DB_NAME", "cowptn_test"),
                                port=int(os.getenv("DB_PORT", 3306)))
    since_sql = " AND occur_date >= %s" if args.since else ""
    df = pd.read_sql("SELECT occur_date, ip_addr, cid, lid, alarm_label, is_anomaly FROM anomaly_detection "
                     f"WHERE ip_addr LIKE %s{since_sql} ORDER BY ip_addr, cid, lid, occur_date", c,
                     params=(args.ip_like, args.since) if args.since else (args.ip_like,))
    df["occur_date"] = pd.to_datetime(df["occur_date"])
    labels = pd.read_csv(os.path.join(args.seed_dir, "labels.csv"), parse_dates=["occur_date"])
    episodes = pd.read_csv(os.path.join(args.seed_dir, "episodes.csv"), parse_dates=["t_start", "t_fail", "t_end"])

    print("\n[1] 서비스 응답")
    try:
        check("API 루트", http_json(args.api + "/").get("status") == "online")
        d1 = http_json(args.api + "/api/anomalies?severity_min=1")
        d2 = http_json(args.ui + "/api/anomalies?severity_min=1")
        check("UI 프록시가 API 와 같은 결과를 반환", len(d1) == len(d2), f"{len(d1)} / {len(d2)}건")
        rules = http_json(args.api + "/api/rca/rules")
        check("RCA 룰 로드", rules["count"] >= 16, f"{rules['count']}개")
    except Exception as e:
        check("서비스 접속", False, str(e)[:100])

    print("\n[2] 결과 저장 (DB)")
    n_ports = df[KEY].drop_duplicates().shape[0]
    steps = labels[KEY + ["occur_date"]].drop_duplicates().groupby(KEY).size()
    expected = int(steps.min() - (args.window - 1))
    per_port = df.groupby(KEY).size()
    check("결과가 저장된 포트 수가 시연 데이터와 같음", n_ports == len(steps), f"{n_ports} / {len(steps)}")
    # 결측 행(트래픽/광 병합 시 한쪽만 있는 경우)이 있어 약간 적을 수 있으므로 5% 허용
    check(f"포트당 결과 행 수 ≈ 스텝 - (윈도우-1) = {expected}", (per_port >= expected * 0.95).all() and (per_port <= expected + 1).all(),
          f"최소 {per_port.min()}, 최대 {per_port.max()}")

    print("\n[3] 알람 이벤트 (SSE)")
    text = open(args.sse_log, encoding="utf-8").read()
    ev = [json.loads(m) for m in re.findall(r"^data: (.*)$", text, flags=re.M)]
    alarms = [e for e in ev if e["type"] == "ALARM"]
    clears = [e for e in ev if e["type"] == "CLEAR"]
    df["crit"] = df["alarm_label"].eq("CRITICAL")
    n_crit = int(df["crit"].sum())
    # 인시던트 = 포트별 연속된 CRITICAL 구간
    df["prev"] = df.groupby(KEY)["crit"].shift(1)
    df["start"] = df["crit"] & (df["prev"] != True)
    n_inc = int(df["start"].sum())
    if args.renotify <= 15:
        check("ALARM 이벤트 수 = CRITICAL 행 수 (스텝마다 알림)", len(alarms) == n_crit, f"{len(alarms)} / {n_crit}")
    elif args.renotify == 0:
        check("ALARM 이벤트 수 = 인시던트 수 (진입 시 1회)", len(alarms) == n_inc, f"{len(alarms)} / {n_inc}")
    else:
        check("ALARM 이벤트 수가 인시던트 수 이상, CRITICAL 행 수 이하 (재알림 간격 적용)",
              n_inc <= len(alarms) <= n_crit, f"인시던트 {n_inc} ≤ ALARM {len(alarms)} ≤ CRITICAL 행 {n_crit}")
    check("ALARM 이 CRITICAL 인 (포트, 시각)에서만 발생", all(
        ((df.ip_addr == a["ip_addr"]) & (df.cid == a["slot_id"]) & (df.lid == a["port_id"]) &
         (df.occur_date == pd.Timestamp(a["event_time"])) & df.crit).any() for a in alarms[:300]))

    print("\n[4] 알람 해제 (CLEAR)")
    true_rec = df[(df["prev"] == True) & (~df["crit"])]
    rec = {(r.ip_addr, int(r.cid), int(r.lid), str(r.occur_date)) for r in true_rec.itertuples()}
    matched = sum(1 for e in clears if (e["ip_addr"], e["slot_id"], e["port_id"], e["event_time"]) in rec)
    check("CLEAR 건수 = DB 의 실제 복구 건수", len(clears) == len(true_rec), f"{len(clears)} / {len(true_rec)}")
    check("모든 CLEAR 가 실제 복구 시점과 (포트, 시각) 일치", matched == len(clears), f"{matched} / {len(clears)}")
    crit_keys = {(r.ip_addr, int(r.cid), int(r.lid), str(r.occur_date)) for r in df[df.crit].itertuples()}
    wrong = [e for e in clears if (e["ip_addr"], e["slot_id"], e["port_id"], e["event_time"]) in crit_keys]
    check("오해제 없음 (다른 포트의 웹훅이 이 포트를 해제하지 않음)", not wrong, f"{len(wrong)}건")

    print("\n[5] 참고 지표")
    al = df[df.is_anomaly == 1]
    fa = 0
    for r in al.itertuples():
        inside = ((episodes.ip_addr == r.ip_addr) & (episodes.cid == r.cid) & (episodes.lid == r.lid) &
                  (episodes.t_start - pd.Timedelta(minutes=15) <= r.occur_date) & (r.occur_date <= episodes.t_end + pd.Timedelta(hours=1))).any()
        fa += 0 if inside else 1
    det = sum(1 for e in episodes.itertuples() if len(al[(al.ip_addr == e.ip_addr) & (al.cid == e.cid) & (al.lid == e.lid) &
              (al.occur_date >= e.t_start - pd.Timedelta(minutes=15)) & (al.occur_date <= e.t_end + pd.Timedelta(hours=1))]))
    print(f"  에피소드 {len(episodes)}건 중 알람과 겹친 것 {det}건 | 알람 {len(al)}건 중 활성 구간 밖 {fa}건 "
          f"(모델 성능 참고용이며 PASS/FAIL 대상이 아님)")

    print(f"\n결과: {sum(results)}/{len(results)} PASS")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
