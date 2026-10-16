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

스냅샷 기반 하위 명령 (DB 접속 없음, 결과는 <run>/out/ 에 저장):
  python validation/cli/real_data_check.py alarms --run <run_id> --summary [--range dev]
  python validation/cli/real_data_check.py alarms --run <run_id> --count-only --range dev
  python validation/cli/real_data_check.py join-check --run <run_id> [--range dev] [--manual join_manual.csv]
  python validation/cli/real_data_check.py periods --run <run_id> show|lock [--version v1]
  python validation/cli/real_data_check.py replay --run <run_id> --range dev [--purpose dev] [--versions traffic=v1] [--policy policy.json]
  python validation/cli/real_data_check.py report --run <run_id> --range dev --purpose dev
  python validation/cli/real_data_check.py report --run <run_id> --range judge --purpose judge     # 잠금 후 1회만
  python validation/cli/real_data_check.py sample --run <run_id> --range dev --purpose dev --seed 7     # 블라인드 검토 샘플 (재생 결과 필요)
  python validation/cli/real_data_check.py score-review --review out/sample_review_*.csv --key out/sample_key_*.csv [--report out/report_*.json]
  python validation/cli/real_data_check.py live --run <run_id> --range dev --purpose dev [--result-schema S]   # C5, DB 읽기 전용 접속 필요
  판정 구간과 겹치는 분석은 잠금(periods lock) 전에는 거부된다 (설계 9.7 P4). --range: dev | judge | YYYY-MM-DD..YYYY-MM-DD
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
from validation.realdata import commands as cmds  # noqa: E402
from validation.realdata import periods as pr  # noqa: E402
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

    def snap(sp):
        sp.add_argument("--run", required=True, help="extract 로 만든 run_id")
        sp.add_argument("--runs-dir", default=RUNS_DIR)
        sp.add_argument("--range", default="dev", help="dev | judge | START..END (기본 dev)")
        sp.add_argument("--alarm-names", help="알람 이름 JSON (기본 validation/runs/ref/alarm_names.json)")
        sp.add_argument("--alarm-override", help="수동 알람명 CSV (model, alarm_type, alarm_name)")
        sp.add_argument("--maintenance", help="작업 CSV (start, end, ip_addr, cid, lid, note)")

    sa = sub.add_parser("alarms", help="알람 에피소드·타겟 이벤트 요약/건수 (스냅샷 기반)")
    snap(sa)
    g = sa.add_mutually_exclusive_group(required=True)
    g.add_argument("--summary", action="store_true")
    g.add_argument("--count-only", action="store_true", help="재생 없이 판정 이벤트 수 (주별·유형별·데이터일당 r)")

    sj = sub.add_parser("join-check", help="포트 조인 안전 검사 J1~J4 (스냅샷 기반)")
    snap(sj)
    sj.add_argument("--manual", help="join_manual.csv (M1 수동 확인)")

    sp_ = sub.add_parser("periods", help="개발/판정 기간 계산과 판정 구간 잠금 (설계 9.7)")
    snap(sp_)
    sp_.add_argument("action", choices=["show", "lock"])
    sp_.add_argument("--version", help="판정에 쓸 트래픽 모델 버전 (기본: 활성 버전)")

    for name, hlp in (("replay", "AI·고정 임계치·롤링 규칙 오프라인 재생 (스냅샷 기반)"), ("report", "C1 원값·C3·C4b·탐지 지연·C6·겹침 보고")):
        sx = sub.add_parser(name, help=hlp)
        snap(sx)
        sx.add_argument("--purpose", choices=["dev", "judge"], default="dev")
        sx.add_argument("--versions", nargs="*", default=[], metavar="TRACK=VER")
        sx.add_argument("--policy", help="알람 정책 JSON (AlertPolicy 필드)")
        if name == "report":
            sx.add_argument("--diag-key-shift", type=int, default=0, help="진단 전용: 알람 cid 에 더할 값 (판정 실행에는 거부)")

    sm = sub.add_parser("sample", help="방법별 층화 무작위 블라인드 검토 샘플 (재생 결과 필요)")
    snap(sm)
    sm.add_argument("--purpose", choices=["dev", "judge"], default="dev")
    sm.add_argument("--seed", type=int, required=True, help="층화 추출 시드 (필수, 기록됨)")

    sr = sub.add_parser("score-review", help="검토 CSV 채점: 유효율 Wilson 95%%, C1·C2·C4a")
    sr.add_argument("--review", required=True)
    sr.add_argument("--key", required=True)
    sr.add_argument("--report", help="report_*.json (C1 원값 포함)")
    sr.add_argument("--out-dir", help="결과 JSON 폴더 (기본: 검토 CSV 폴더)")

    lv = sub.add_parser("live", help="C5 라이브 결과 점검 (DB 읽기 전용)")
    common(lv)
    snap(lv)
    lv.add_argument("--purpose", choices=["dev", "judge"], default="dev")
    lv.add_argument("--result-schema", help="anomaly_detection 이 있는 스키마 (기본: 접속 스키마)")
    return p


def _context(args):
    from validation.realdata.alarm_names import DEFAULT_NAMES_PATH
    return cmds.Context(os.path.join(args.runs_dir, args.run), args.runs_dir, names_path=args.alarm_names or DEFAULT_NAMES_PATH,
                        override_csv=args.alarm_override, maintenance_csv=args.maintenance)


def print_report(rep):
    h = rep["header"]
    print(f"[report] purpose={h['purpose']} range={h['range'][0]}~{h['range'][1]} join_status={h['join_status']} models={h['model_versions']}")
    print("방법별 재생 가능 슬롯:", {m: f"{v['ratio']:.1%}" for m, v in h["slots"].items()})
    for m, v in rep["c1_raw"].items():
        print(f"  C1 원값 {m:8s} 인시던트 {v['incidents']:6d} / 포트·일 {v['port_days']:6d} = {v['incidents_per_port_day']:.4f}")
    c3 = rep["c3"]["합계"]
    print(f"  C3 합계: 이벤트 {c3['events_total']} 포함 {c3['included']} 제외 {c3['excluded']} 보류={c3['hold']} {c3['hold_reasons']}")
    for m, v in c3["methods"].items():
        print(f"    {m:8s} 선행 탐지율 {v['rate']:.3f} (≥1h {v['lead_ge_1h']:.3f} ≥3h {v['lead_ge_3h']:.3f} ≥6h {v['lead_ge_6h']:.3f})")
    d = rep["delay"]
    print(f"  단절 탐지 지연: 이벤트 {d['events_total']} 포함 {d['included']} 제외 {d['excluded']}")
    print("  겹침:", rep["overlap"])
    for n in h["fixed_notes"]:
        print("  *", n)


def _run_live(args):
    try:
        ctx = _context(args)
        cfg = load_db_config(host=args.host, port=args.port, database=args.database, user=args.user)
        db = ReadOnlyDB.connect(cfg, allow_write_account=args.allow_write_account, max_exec_ms=args.max_exec_ms)
    except PrivilegeError as e:
        print(f"[거부] {e}", file=sys.stderr)
        return 3
    except (pr.PeriodError, FileNotFoundError) as e:
        print(f"[거부] {e}", file=sys.stderr)
        return 5
    try:
        res = cmds.cmd_live(ctx, db, args.range, args.purpose, args.result_schema)
        if res.get("hold"):
            print(f"C5 보류: {res['reason']}")
        else:
            print(f"C5a 처리율 {res['c5a']:.4f} ({res['matched_rows']}/{res['target_rows']}) 통과={res['c5a_pass']}, "
                  f"C5b 30분 이내 {res['c5b']['within_30min']:.3f} 통과={res['c5b']['pass']}, C5c 공백 {len(res['c5c_gaps'])}구간")
        return 0
    except pr.PeriodError as e:
        print(f"[거부] {e}", file=sys.stderr)
        return 5
    finally:
        db.close()


def _run_snapshot_command(args):
    if args.cmd == "score-review":
        res = cmds.cmd_score_review(args.out_dir or os.path.dirname(os.path.abspath(args.review)), args.review, args.key, args.report)
        ai = res["methods"]["ai"]
        print(f"AI 유효율 {ai['valid_rate']:.3f} (Wilson {ai['wilson_lo']:.3f}~{ai['wilson_hi']:.3f}) 판정 가능 {ai['judgeable']} "
              f"판단 불가 {ai['unjudgeable_ratio']:.1%} 미검토 {res['unreviewed']} C2 보류={res['c2']['hold']}")
        if "c4a" in res:
            print("C4a:", res["c4a"]["status"], " C1:", res.get("c1"))
        return 0
    try:
        ctx = _context(args)
        if args.cmd == "alarms":
            res = cmds.cmd_alarms(ctx, args.range, "summary" if args.summary else "count")
            if args.summary:
                print(res["first_line"])
                print(f"알람 {res['alarm_rows']:,}행, 에피소드 {res['episodes']}, 행 분류 {res['row_classes']}")
                print(f"슬롯만 있는 address 로 slot 재분류(6.1 보정): {res['slot_only_address_reclassified']}")
                print(f"타겟 이벤트: {res['target_events']}  작업 대용 알람: {res['work_alarms']}")
                print(f"미분류 상위: {[(r['alarm_name'], r['count']) for r in res['unclassified_top'][:10]]}")
            else:
                print(f"판정 이벤트 {res['events']} (유형별 {res['by_type']}), 데이터일 {res['data_days']}, r={res['r_per_data_day']:.3f}/데이터일")
                print(res["note"])
        elif args.cmd == "sample":
            meta = cmds.cmd_sample(ctx, args.range, args.purpose, args.seed)
            print(f"샘플 {meta['rows']}행 (방법별 선택 인시던트 {meta['per_method_incidents']}), 시드 {meta['seed']}. "
                  f"검토 CSV 만 검토자에게 전달 — 키 파일(sample_key_*)은 비공개")
        elif args.cmd == "periods":
            if args.action == "lock":
                res = cmds.cmd_lock(ctx, args.version)
                print(f"잠금 완료: {res['lock_path']}\n  J0={res['J0']} D={res['D']} 판정 데이터일 {res['judge_data_days']} join_status={res['join_status']} "
                      f"r={res['dev_event_rate_r']:.3f} C3 보류 예상={res['c3_hold_expected']}")
            else:
                p_ = ctx.periods
                print({k: str(v) for k, v in p_.items()})
                print("잠금:", "있음" if pr.load_lock(ctx.runs_dir, p_) else "없음")
        elif args.cmd in ("replay", "report"):
            versions = dict(x.split("=", 1) for x in args.versions) or None
            policy = None
            if args.policy:
                import json as _json
                from src.pipeline.alerting import policy_from_meta
                policy = policy_from_meta(_json.load(open(args.policy, encoding="utf-8")))
            if args.cmd == "replay":
                res = cmds.cmd_replay(ctx, args.range, args.purpose, versions, policy)
                print(f"재생 입력 {res['input_rows']:,}행, 평가 구간 정상 행 {res['normal_rows']:,}, 방법별 재생 가능 슬롯: "
                      + ", ".join(f"{m} {v['ratio']:.1%}" for m, v in res["slots"].items()))
            else:
                rep = cmds.cmd_report(ctx, args.range, args.purpose, versions, policy, key_shift=args.diag_key_shift)
                print_report(rep)
        else:
            res = cmds.cmd_join_check(ctx, args.range, args.manual)
            print(f"join_status = {res['join_status']}  {res['verdict']['reasons']}")
            print(f"J1 1:1 {res['j1']['one_to_one_ratio']:.3f}  J3 에피소드 {res['j3']['with_bins']} 일치 a={res['j3']['match_a']:.3f} "
                  f"c={res['j3']['match_c']:.3f} d={res['j3']['match_d']:.3f}  J4 매칭률 {res['j4']['matched_ratio']:.3f}")
            if res["verdict"].get("stop"):
                print("[멈춤] 조인 검증 실패 — 보고 후 사용자 결정 필요 (address 조인으로 자동 전환하지 않음)")
        return 0
    except (pr.PeriodError, FileNotFoundError) as e:
        print(f"[거부] {e}", file=sys.stderr)
        return 5


def _connect(args):
    cfg = load_db_config(host=args.host, port=args.port, database=args.database, user=args.user)
    return ReadOnlyDB.connect(cfg, allow_write_account=args.allow_write_account, max_exec_ms=args.max_exec_ms)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd in ("alarms", "join-check", "periods", "replay", "report", "sample", "score-review"):
        return _run_snapshot_command(args)
    if args.cmd == "live":
        return _run_live(args)
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
