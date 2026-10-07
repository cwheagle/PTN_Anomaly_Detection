"""
P1-6 승격 게이트 보정 검증 도구 (설계서 docs/design/p1_6_gate_correction.md 6장)

이 파일의 상수(시드 목록·금지 목록·고정 경로)와 판정 기준은 데이터를 열기 전에 설계서에 확정된 값이다.
결과를 본 뒤 바꾸지 않는다(설계서 6.6). 판정에 쓰는 함수(검증 손실 판정, 학습, 점수)는 src 를 그대로 호출한다 (lessons #31).
모든 수치는 시뮬레이션 기준이며 실데이터 검증이 아니다.

  이번 단위(U2-G2)는 공통(시드 가드·잠금·기록 가드)과 ② G2 만 구현한다. ① G3 (g3·g3-dev·g3-verify)는 U2-G3 에서 추가한다.

  g2-train  G2 사례 학습: 데이터 시드 {107,109,113} × 사례 {N 정상, K 누출(검증⊂학습), W 1에폭}, 각각 두 트랙.
            --train-seed 0 = 개발 사례. --train-seed 1 = 근거 사례: 개발 기록(g2_dev.json)이 있어야 하고, 학습 전에 G2 잠금을
            배타적으로 만들며, 사례 전체만 허용하고, val_loss·임계치를 화면·로그 어디에도 쓰지 않는다.
  g2        개발 사례(난수 0) 표와 판정 → validation/runs/p1_6/g2_dev.json
  g2-verify 근거 사례(난수 1) 표와 판정 (잠금 확인·1회). 학습이 끝난 잠금(status=trained)에서만, 한 번만 읽는다.

1회 보장(설계서 6.5): 잠금 파일은 **고정 경로**(`validation/runs/p1_6/locks/p1_6_g2_evidence.json`)이며 --record·모델 폴더와
무관하다. 사례 폴더를 지우고 다시 학습해도 같은 사례 집합이면 거부한다. **한계: 잠금 파일을 직접 지우는 것은 막을 수 없다 —
운영자 규율이다.**

사용법:
  python validation/cli/check_gate.py g2-train --train-seed 0
  python validation/cli/check_gate.py g2
  python validation/cli/check_gate.py g2-train --train-seed 1      # 개발 확정(g2_dev.json) 후에만
  python validation/cli/check_gate.py g2-verify
"""
import argparse
import contextlib
import io
import json
import logging
import os
import sys

import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from src.models import promotion_gate, registry
from validation.cli import check_promotion as cp

# ─────────────────────────────────────────────
# 공통: 시드 (설계서 6.0 — 이 목록이 유일한 목록)
# ─────────────────────────────────────────────
P16_SEEDS = {
    "g3_dev": (7, 11),            # G3 개발(선택·확인), 여러 번 열림
    "g3_evidence": (79, 83, 89),  # G3 근거, 미개봉, 전체 집합 1회
    "g2_data": (107, 109, 113),   # G2 학습 데이터. 학습 난수 0 = 개발 사례, 1 = 근거 사례
}
FORBIDDEN_SEEDS = (23, 31, 47,            # P1-3 근거 소진
                   53, 59, 61,            # P1-5 예약, 미개봉 유지
                   101, 102, 103, 105,    # 학습·민감도에 사용
                   211,                   # P1-5 오라클
                   900,                   # 카나리
                   33, 34, 35)            # DB 대형 시드
G2_CASES = ("N", "K", "W")                # 정상 / 누출 / 1에폭
G2_TRAIN_SEEDS = (0, 1)                   # 0 = 개발, 1 = 근거
G2_EPOCHS_W = 1
G2_MODES = {"A": "both_fail", "B": "lower_warn", "현행": "worse"}    # G2-A / G2-B(판정) / 현행

# 고정 경로 (--record·모델 폴더와 무관). 테스트는 RUN_ROOT 를 임시 폴더로 바꾼다.
RUN_ROOT = os.path.join(root_dir, "validation", "runs", "p1_6")
G2_LOCK_NAME, G2_DEV_NAME = "p1_6_g2_evidence.json", "g2_dev.json"


def lock_dir():
    return os.path.join(RUN_ROOT, "locks")


def g2_lock_path():
    return os.path.join(lock_dir(), G2_LOCK_NAME)


def g2_dev_path():
    return os.path.join(RUN_ROOT, G2_DEV_NAME)


def check_seeds(seeds, allowed):
    """시드 가드: 금지 목록·허용 집합 밖·교차 사용(다른 용도의 시드)이면 SystemExit (측정·학습 전에 거부)."""
    ss = {int(x) for x in seeds}
    bad = sorted(ss & set(FORBIDDEN_SEEDS))
    if bad:
        raise SystemExit(f"[!] 시드 {bad} 는 P1-6 에서 사용 금지입니다 (설계서 6.0).")
    outside = sorted(ss - set(allowed))
    if outside or not ss:
        others = {k: v for k, v in P16_SEEDS.items() if set(v) & ss}
        hint = f" (다른 용도의 시드: {others})" if others else ""
        raise SystemExit(f"[!] 시드 {sorted(ss)} 는 허용 집합 {tuple(allowed)} 밖입니다{hint} — 교차 사용 금지(설계서 6.0).")
    return sorted(ss)


def case_dir(train_seed, case, data_seed):
    return os.path.join(RUN_ROOT, "g2", f"rand{train_seed}", f"{case}_s{data_seed}")


# ─────────────────────────────────────────────
# G2: 사례 학습
# ─────────────────────────────────────────────
@contextlib.contextmanager
def _silent(log_path=None):
    """학습기의 표준 출력·오류를 화면에 내지 않는다. log_path 가 있으면 파일에, 없으면 버린다 (근거 사례는 버림)."""
    buf = io.StringIO()
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield
    finally:
        logging.disable(logging.NOTSET)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(buf.getvalue())


def train_cases(train_seed, data_seeds, cases, active_models_dir, nodes=4, days=21, epochs=None):
    """사례 폴더에 학습한다. 이미 있는 사례 폴더는 덮어쓰지 않고 거부. 근거(난수 1)는 출력·로그에 값을 남기지 않는다."""
    todo = [(c, s) for s in data_seeds for c in cases]
    existing = [case_dir(train_seed, c, s) for c, s in todo if os.path.exists(case_dir(train_seed, c, s))]
    if existing:
        raise SystemExit(f"[!] 사례 폴더가 이미 있습니다 — 덮어쓰지 않습니다: {existing[0]} 외 {len(existing) - 1}개")
    for c, s in todo:
        out = case_dir(train_seed, c, s)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        kwargs = {}
        if c == "K":
            kwargs["leak_val_into_train"] = True
        ep = G2_EPOCHS_W if c == "W" else epochs
        log = None if train_seed == 1 else os.path.join(os.path.dirname(out), "train.log")
        with _silent(log):
            cp.train_operational(out, data_seed=s, train_seed=train_seed, nodes=nodes, days=days, epochs=ep,
                                 active_models_dir=active_models_dir, **kwargs)
        print(f"[OK] rand{train_seed} {c} 데이터 시드 {s} 학습 완료")


def cmd_g2_train(a):
    ts = int(a.train_seed)
    if ts not in G2_TRAIN_SEEDS:
        raise SystemExit(f"[!] --train-seed 는 {G2_TRAIN_SEEDS} 중 하나여야 합니다.")
    data_seeds = check_seeds(a.data_seeds.split(","), P16_SEEDS["g2_data"])
    cases = [c for c in a.cases.split(",") if c]
    if not cases or set(cases) - set(G2_CASES):
        raise SystemExit(f"[!] --cases 는 {G2_CASES} 의 부분집합이어야 합니다: {cases}")
    if ts == 1:
        if set(data_seeds) != set(P16_SEEDS["g2_data"]) or set(cases) != set(G2_CASES):
            raise SystemExit("[!] G2 근거 사례(난수 1)는 데이터 시드·사례 전체만 허용합니다 (부분집합 거부).")
        if not os.path.isfile(g2_dev_path()):
            raise SystemExit(f"[!] 개발 기록 {g2_dev_path()} 가 없습니다 — 난수 0 개발(g2)을 먼저 끝내야 난수 1 을 학습할 수 있습니다.")
        os.makedirs(lock_dir(), exist_ok=True)        # 모든 가드를 통과한 뒤에만 고정 잠금 폴더를 만든다
        cp.evidence_lock(g2_lock_path(), {"kind": "P1-6 G2", "data_seeds": data_seeds, "train_seed": 1, "cases": list(G2_CASES)})
    train_cases(ts, data_seeds, cases, a.active_models, a.nodes, a.days, a.epochs)
    if ts == 1:
        _lock_update(g2_lock_path(), {"status": "trained"})


def _lock_update(path, fields):
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    rec.update(fields)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, default=str)


# ─────────────────────────────────────────────
# G2: 표와 판정
# ─────────────────────────────────────────────
def folder_entry(folder, ft):
    """모델 폴더의 활성 버전 레지스트리 엔트리 (threshold·final_val_loss 포함)"""
    reg = registry.load(folder, ft)
    entry = registry.find(reg, reg.get("active_version"))
    if entry is None:
        raise SystemExit(f"[!] {folder} 에 {ft} 활성 모델이 없습니다.")
    return entry


def g2_status(blocking, notes):
    return "FAIL" if blocking else ("WARN" if notes else "PASS")


def g2_row(kind, ft, act_entry, cand_entry, **extra):
    """한 사례(활성 대 후보)의 세 방식 판정. 항목별(임계치/검증 손실) 원인 태그를 따로 기록한다."""
    reg = {"active_version": "active", "versions": [{**act_entry, "version": "active"}]}
    cand = {**cand_entry, "version": "cand"}
    va, vc = act_entry.get("final_val_loss"), cand_entry.get("final_val_loss")
    row = {"kind": kind, "track": ft, "val_ratio": (vc / va) if va and vc and va > 0 else None,
           "threshold_ratio": (cand_entry["threshold"] / act_entry["threshold"]) if act_entry.get("threshold") else None,
           "val_active": va, "val_cand": vc, **extra}
    for key, mode in G2_MODES.items():
        blocking, notes = registry.compare_with_active_split(reg, cand, mode)
        row[f"{key}_status"] = g2_status(blocking, notes)
        row[f"{key}_causes"] = sorted({f.cause for f in blocking})
    return row


def judge_g2(rows):
    """G2 판정 (설계서 6.4, G2-B 기준). N: FAIL·WARN 0 / L: 모두 WARN 이상 / K: 비율 < 1/3 인 사례는 모두 WARN(없으면 판정 불가, 값을
    낮춰 맞추지 않음) / W: 비율 > 3 인 사례는 모두 FAIL 이고 원인에 검증 손실 포함(비율 ≤ 3 이면 구성 실패로 제외, 임계치 항목만으로 FAIL 은
    별도 기록). PASS = N·L·W 충족(W 면책 포함), K 는 판정 또는 면책 결과를 그대로 보고."""
    df = pd.DataFrame(rows)
    out = {"rows": int(len(df))}
    n = df[df.kind == "N"]
    out["N"] = {"rows": int(len(n)), "fail": int((n.B_status == "FAIL").sum()), "warn": int((n.B_status == "WARN").sum())}
    out["N"]["ok"] = bool(len(n)) and out["N"]["fail"] == 0 and out["N"]["warn"] == 0
    lg = df[df.kind == "L"]
    out["L"] = {"rows": int(len(lg)), "not_warn_or_fail": int((lg.B_status == "PASS").sum())}
    out["L"]["ok"] = bool(len(lg)) and out["L"]["not_warn_or_fail"] == 0
    k = df[df.kind == "K"]
    low = k[k.val_ratio < 1 / registry.VAL_LOSS_RATIO_LIMIT]
    out["K"] = {"rows": int(len(k)), "ratio_below_third": int(len(low)), "status_of_those": low.B_status.value_counts().to_dict(),
                "not_warn": int((low.B_status != "WARN").sum())}
    out["K"]["ok"] = None if not len(low) else out["K"]["not_warn"] == 0          # None = 판정 불가(이 구성에서 1/3 로 잡을 수 없음)
    w = df[df.kind == "W"]
    over = w[w.val_ratio > registry.VAL_LOSS_RATIO_LIMIT]
    out["W"] = {"rows": int(len(w)), "ratio_over_3": int(len(over)), "excluded_not_worse": int(len(w) - len(over)),
                "not_fail_or_no_val_cause": int(((over.B_status != "FAIL") | ~over.B_causes.apply(lambda c: "val_loss" in c)).sum()),
                "threshold_only_fail": int(((w.B_status == "FAIL") & ~w.B_causes.apply(lambda c: "val_loss" in c)).sum())}
    out["W"]["ok"] = None if not len(over) else out["W"]["not_fail_or_no_val_cause"] == 0     # None = 구성 실패(판정 제외)
    out["pass"] = bool(out["N"]["ok"] and out["L"]["ok"] and out["W"]["ok"] is not False and len(w) > 0)
    return out


def g2_cases_table(train_seed, active_models_dir, data_seeds=P16_SEEDS["g2_data"], g2c=True, gate_nodes=4):
    """사례 폴더로 N·L·K·W 표를 만든다. g2c=True 면 G2-C(같은 데이터 MSE 비율)를 기록만 한다."""
    cache = {}

    def scores(folder, seed, ft):
        if (folder, seed) not in cache:
            cache[(folder, seed)] = cp.score_data(cp.make_data(seed, gate_nodes, cp.GATE_DAYS + 1), folder)
        sc, _ = cache[(folder, seed)][ft]
        return sc

    def c_ratios(ft, act_folder, cand_folder, seed, vc):
        if not g2c:
            return {}
        r = promotion_gate.g2_same_data_ratios(scores(cand_folder, seed, ft), scores(act_folder, seed, ft), vc)
        return {"g2c_mse_median_ratio": r["mse_median_ratio"], "g2c_cand_mse_over_val_loss": r["cand_gate_mse_over_val_loss"]}

    rows = []
    for ft in cp.TRACKS:
        normal = {s: case_dir(train_seed, "N", s) for s in data_seeds}
        for sa in data_seeds:                                            # N: 같은 난수의 정상 후보끼리 순서쌍
            for sb in data_seeds:
                if sa != sb:
                    cand = folder_entry(normal[sb], ft)
                    rows.append(g2_row("N", ft, folder_entry(normal[sa], ft), cand, active=f"N{sa}", cand_id=f"N{sb}",
                                       **c_ratios(ft, normal[sa], normal[sb], sb, cand.get("final_val_loss"))))
        for sb in data_seeds:                                            # L: 레거시 v1(활성) 대 정상 후보
            cand = folder_entry(normal[sb], ft)
            rows.append(g2_row("L", ft, folder_entry(active_models_dir, ft), cand, active="v1", cand_id=f"N{sb}",
                               **c_ratios(ft, active_models_dir, normal[sb], sb, cand.get("final_val_loss"))))
        for kind in ("K", "W"):                                          # K·W: 활성 = 같은 데이터 시드의 정상 후보
            for s in data_seeds:
                folder = case_dir(train_seed, kind, s)
                if not os.path.isdir(folder):
                    continue
                cand = folder_entry(folder, ft)
                rows.append(g2_row(kind, ft, folder_entry(normal[s], ft), cand, active=f"N{s}", cand_id=f"{kind}{s}",
                                   **c_ratios(ft, normal[s], folder, s, cand.get("final_val_loss"))))
    return rows


def _print_g2(rows, judged, title):
    pd.set_option("display.width", 220)
    df = pd.DataFrame(rows)
    cols = ["kind", "track", "active", "cand_id", "val_active", "val_cand", "val_ratio", "threshold_ratio",
            "현행_status", "A_status", "B_status", "B_causes"] + [c for c in df.columns if c.startswith("g2c_")]
    print(f"== G2 [{title}] ==")
    print(df[cols].round(4).to_string(index=False))
    print(f"\n[N] 정상 쌍 {judged['N']['rows']}건: FAIL {judged['N']['fail']}, WARN {judged['N']['warn']}")
    print(f"[L] 레거시 활성 {judged['L']['rows']}건: PASS(경고 없음) {judged['L']['not_warn_or_fail']}건")
    print(f"[K] 누출 {judged['K']['rows']}건: 비율<1/3 {judged['K']['ratio_below_third']}건, 그 판정 {judged['K']['status_of_those']}, "
          f"WARN 아님 {judged['K']['not_warn']}건 → {'판정 불가' if judged['K']['ok'] is None else judged['K']['ok']}")
    print(f"[W] 1에폭 {judged['W']['rows']}건: 비율>3 {judged['W']['ratio_over_3']}건(구성 실패로 제외 {judged['W']['excluded_not_worse']}건), "
          f"FAIL 아님/검증 손실 원인 아님 {judged['W']['not_fail_or_no_val_cause']}건, 임계치 항목만 FAIL {judged['W']['threshold_only_fail']}건 "
          f"→ {'구성 실패(제외)' if judged['W']['ok'] is None else judged['W']['ok']}")
    print(f"G2 판정(N·L·W 충족): {'PASS' if judged['pass'] else 'FAIL'}")


def cmd_g2(a):
    out = g2_dev_path()
    cp.check_record(out)                                   # 개발 기록은 덮어쓰지 않는다
    rows = g2_cases_table(0, a.active_models, g2c=not a.no_g2c)
    judged = judge_g2(rows)
    _print_g2(rows, judged, "개발 — 학습 난수 0")
    os.makedirs(RUN_ROOT, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"phase": "dev", "train_seed": 0, "judgement": judged, "rows": rows}, f, ensure_ascii=False, indent=2, default=str)
    print(f"[OK] 개발 기록 -> {out}")


def cmd_g2_verify(a):
    path = g2_lock_path()
    if not os.path.isfile(path):
        raise SystemExit(f"[!] 잠금 {path} 가 없습니다 — 난수 1 사례를 g2-train 으로 먼저 학습하세요.")
    with open(path, "r", encoding="utf-8") as f:
        status = json.load(f).get("status")
    if status != "trained":
        raise SystemExit(f"[!] 잠금 상태가 'trained' 가 아닙니다({status!r}) — 학습이 끝나지 않았거나 이미 읽었습니다. 근거 사례는 1회만 읽습니다.")
    cp.check_record(a.record, reserved=[path])
    _lock_update(path, {"status": "verifying"})            # 읽기 시작 전에 표시: 도중에 실패해도 다시 읽지 않는다
    rows = g2_cases_table(1, a.active_models, g2c=not a.no_g2c)
    judged = judge_g2(rows)
    _print_g2(rows, judged, "근거 — 학습 난수 1 (1회)")
    rec = {"phase": "evidence", "train_seed": 1, "judgement": judged, "rows": rows}
    _lock_update(path, {"status": "verified", "judgement": judged})
    if a.record:
        with open(a.record, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
        pd.DataFrame(rows).to_csv(a.record + ".csv", index=False)


def main():
    p = argparse.ArgumentParser(description="P1-6 게이트 보정 검증 (U2-G2: 공통 + G2)")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("g2-train", help="G2 사례 학습 (난수 0 = 개발, 1 = 근거)")
    t.add_argument("--train-seed", type=int, required=True)
    t.add_argument("--data-seeds", default=",".join(map(str, P16_SEEDS["g2_data"])))
    t.add_argument("--cases", default=",".join(G2_CASES))
    t.add_argument("--active-models", default="models", help="규칙 A(자기 알람) 입력용 활성 모델 폴더")
    t.add_argument("--nodes", type=int, default=4)
    t.add_argument("--days", type=int, default=21)
    t.add_argument("--epochs", type=int, default=None)
    for name, fn, doc in (("g2", cmd_g2, "개발 사례(난수 0) 표·판정 → g2_dev.json"),
                          ("g2-verify", cmd_g2_verify, "근거 사례(난수 1) 표·판정 (1회)")):
        s = sub.add_parser(name, help=doc)
        s.add_argument("--active-models", default="models", help="L 사례의 활성(레거시 v1) 모델 폴더")
        s.add_argument("--no-g2c", action="store_true", help="G2-C(같은 데이터 MSE 비율) 기록 생략")
        if name == "g2-verify":
            s.add_argument("--record", default=None, help="결과 사본 JSON (이미 있으면 거부)")
        s.set_defaults(func=fn)
    t.set_defaults(func=cmd_g2_train)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
