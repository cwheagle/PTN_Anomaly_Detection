"""
P1-2 실데이터 검증 도구 (설계서 docs/design/p1_2_real_data_check.md) — U1·U2: plan, extract

읽기 전용. 접속 정보는 환경변수/CLI 인자로만 받는다 (src/config.py·.env 를 읽지 않음).
  환경변수: DB_HOST, DB_PORT, DB_NAME, RDC_DB_USER(또는 DB_USER), RDC_DB_PASS(또는 DB_PASS)

권한 검사는 fail closed 다. SELECT/USAGE 외 권한이 있는 계정(root 등)은 `--allow-write-account` 를
명령줄에 명시해야만 진행한다 (환경변수로는 켜지지 않음). 계정 생성 안내는 거부 메시지와 validation/README.md 참고.

사용법:
  python validation/cli/real_data_check.py plan --start 2026-10-05 --end 2026-10-12
  python validation/cli/real_data_check.py extract --start 2026-07-10 --end 2026-10-13 [--ports ports.csv]
  (--schema sim: 시뮬레이터 DB cowptn_test 용 — 필터 컬럼·알람 테이블이 없으면 생략)
"""
import argparse
import os
import sys
from datetime import datetime

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from validation.realdata.db_readonly import (  # noqa: E402
    DEFAULT_MAX_EXEC_MS, PrivilegeError, ReadOnlyDB, load_db_config)
from validation.realdata.extract import (  # noqa: E402
    DEFAULT_CHUNK, DEFAULT_MAX_ROWS, ExtractError, extract, load_ports_csv)
from validation.realdata.tables import DEFAULT_LOOKBACK_HOURS, KINDS, build_plan, format_plan  # noqa: E402

RUNS_DIR = os.path.join(root_dir, "validation", "runs", "p1_2")


def build_parser():
    p = argparse.ArgumentParser(description="P1-2 실데이터 검증 도구 (읽기 전용)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--host")
        sp.add_argument("--port", type=int)
        sp.add_argument("--database")
        sp.add_argument("--user")
        sp.add_argument("--schema", choices=["real", "sim"], default="real")
        sp.add_argument("--kinds", default=",".join(KINDS), help="eth,optic,alarm 중 부분집합")
        sp.add_argument("--lookback-hours", type=int, default=DEFAULT_LOOKBACK_HOURS)
        sp.add_argument("--max-exec-ms", type=int, default=DEFAULT_MAX_EXEC_MS)
        sp.add_argument("--allow-write-account", action="store_true",
                        help="쓰기 권한 계정으로 실행(테스트망 한정). 권한 검사만 건너뛰고 읽기 전용 세션·문장 가드는 유지")

    sp = sub.add_parser("plan", help="테이블 목록·예상 행 수·경계·필터 분포 (행 추출 없음)")
    common(sp)
    sp.add_argument("--start", required=True)
    sp.add_argument("--end", required=True)

    se = sub.add_parser("extract", help="읽기 전용 추출 → 스냅샷")
    common(se)
    se.add_argument("--start", required=True)
    se.add_argument("--end", required=True)
    se.add_argument("--ports", help="CSV (열 ip_addr, 선택 cid, lid)")
    se.add_argument("--run-id")
    se.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    se.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    se.add_argument("--runs-dir", default=RUNS_DIR)
    return p


def _connect(args):
    cfg = load_db_config(host=args.host, port=args.port, database=args.database, user=args.user)
    return ReadOnlyDB.connect(cfg, allow_write_account=args.allow_write_account, max_exec_ms=args.max_exec_ms)


def main(argv=None):
    args = build_parser().parse_args(argv)
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        print(f"알 수 없는 종류: {bad}", file=sys.stderr)
        return 2
    try:
        db = _connect(args)
    except PrivilegeError as e:
        print(f"[거부] {e}", file=sys.stderr)
        return 3
    try:
        if args.cmd == "plan":
            print(format_plan(build_plan(db, args.start, args.end, kinds, args.schema, args.lookback_hours)))
            return 0
        if args.cmd == "extract":
            if not args.start:
                print("--start 가 필요합니다", file=sys.stderr)
                return 2
            run_id, start = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"), args.start
            ports = load_ports_csv(args.ports) if args.ports else None
            m = extract(db, os.path.join(args.runs_dir, run_id), start, args.end, kinds, ports, args.schema,
                        args.max_rows, args.chunk, args.lookback_hours)
            print(f"run {run_id}: 완료={m['complete']} 읽은 행 {m['total_rows']:,} 회차 {len(m['rounds'])}"
                  + (f" 중단={m.get('abort_reason')}" if not m["complete"] else ""))
            for k in kinds:
                s = m["stats"].get(k, {})
                print(f"  [{k}] 읽음 {s.get('fetched', 0):,} 보관 {s.get('kept', 0):,} 격리 {s.get('quarantined', 0):,}"
                      f" ({s.get('quarantine_ratio', 0):.1%}) 필터 밖 {s.get('filtered_out', 0):,}")
            if m.get("optic", {}).get("status") == "SKIP":
                print("  [optic] 행 0 — SKIP")
            return 0 if m["complete"] else 4
    except ExtractError as e:
        print(f"[중단] {e}", file=sys.stderr)
        return 4
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
