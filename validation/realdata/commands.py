"""
P1-2 스냅샷 기반 하위 명령 본체 (DB 접속 없음). CLI(`validation/cli/real_data_check.py`)가 호출한다.
모든 출력은 `<run>/out/` 아래에 쓴다(validation/runs 는 git 제외 — IP·국소명이 들어갈 수 있음).
"""
import hashlib
import inspect
import json
import os

import numpy as np
import pandas as pd

from validation.realdata import alarm_episodes as ae
from validation.realdata import periods as pr
from validation.realdata import port_join as pj
from validation.realdata import replay as rp
from validation.realdata import report as rpt
from validation.realdata.alarm_names import DEFAULT_NAMES_PATH, DEFAULT_TARGET_MAP, AlarmNames, TargetMap
from validation.realdata.extract import load_snapshot


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_jsonable)


def _clip_episodes(ep, end):
    """에피소드 end 를 분석 범위 끝에서 자르고 지속 시간을 다시 계산한다 (표시용 사본)."""
    out = ep.copy()
    if len(out):
        out["end"] = out["end"].clip(upper=pd.Timestamp(end))
        out["duration_min"] = (out["end"] - out["start"]).dt.total_seconds() / 60
    return out


class Context:
    """스냅샷 한 번 로드 + 기간 계산 + 알람 준비. 판정 구간을 포함한 전체를 읽지만 출력은 resolve_range 가 허용한 범위만."""

    def __init__(self, run_dir, runs_dir, names_path=DEFAULT_NAMES_PATH, map_path=DEFAULT_TARGET_MAP, override_csv=None,
                 maintenance_csv=None):
        self.run_dir, self.runs_dir = run_dir, runs_dir
        self.target_map_path, self.override_csv, self.maintenance_csv = map_path, override_csv, maintenance_csv
        self.names_sha = hashlib.sha256(open(names_path, "rb").read()).hexdigest() if os.path.exists(names_path) else None
        with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        if not self.manifest.get("complete"):
            raise pr.PeriodError("불완전한 스냅샷입니다 — 다시 추출하세요")
        self.eth = load_snapshot(run_dir, "eth")
        self.raw_alarms = load_snapshot(run_dir, "alarm")
        self.data_days = pr.data_days_from_eth(self.eth)
        self.periods = pr.compute_periods(self.data_days, start=self.manifest.get("range_start"))
        self.names = AlarmNames.load(names_path, override_csv)
        self.target_map = TargetMap.load(map_path, known_names=self.names.all_names())
        self.maintenance = ae.load_maintenance(maintenance_csv) if maintenance_csv else None
        d_end = self.periods.get("D")
        self.alarms = ae.prepare_alarms(self.raw_alarms, self.names, self.target_map, d_end)
        port = self.alarms[self.alarms["row_class"] == "port"]
        self.episodes = ae.build_episodes(port, self.target_map, d_end) if d_end is not None else ae.build_episodes(port, self.target_map, pd.Timestamp.max)
        self.events = ae.mark_work_and_hw(ae.build_target_events(self.episodes), self.alarms, self.maintenance)
        self.out_dir = os.path.join(run_dir, "out")

    def record(self, command, start, end, purpose=None):
        """분석 장부에 기록 (가드를 통과한 뒤, 계산 전)."""
        purpose = purpose or pr.analysis_purpose(self.periods, start, end)
        return pr.append_ledger(self.runs_dir, command, purpose, start, end, self.periods, os.path.basename(self.run_dir))

    def in_range(self, start, end):
        """범위 안 알람·에피소드·이벤트. 에피소드·이벤트의 end 는 분석 범위 끝에서 잘라 표시한다(설계 9.7.2, 개발 출력의 J0 이후 정보 축소; 원 값은 스냅샷에만)."""
        a = self.alarms[(self.alarms["occur_date"] >= start) & (self.alarms["occur_date"] < end)]
        ep = _clip_episodes(self.episodes[(self.episodes["start"] >= start) & (self.episodes["start"] < end)], end)
        ev = self.events[(self.events["t0"] >= start) & (self.events["t0"] < end)].copy()
        if len(ev):
            ev["end"] = ev["end"].clip(upper=pd.Timestamp(end))
        return a, ep, ev


def cmd_alarms(ctx, range_spec, mode):
    """mode = summary | count. 판정 구간과 겹치면 잠금 전 거부(P4)."""
    start, end = pr.resolve_range(range_spec, ctx.periods, ctx.runs_dir, f"alarms --{mode}")
    ctx.record(f"alarms --{mode}", start, end)
    a, ep, ev = ctx.in_range(start, end)
    tag = range_spec.replace("..", "_").replace(" ", "")
    if mode == "summary":
        res = ae.summarize_alarms(a, ep, ev, eth=ctx.eth)
        res["merge_sensitivity_ref"] = ae.merge_sensitivity(a[a["row_class"] == "port"], ctx.target_map, ctx.periods.get("D", end))
        res["range"] = [start, end]
        write_json(os.path.join(ctx.out_dir, f"alarms_summary_{tag}.json"), res)
        un = ae.unmapped_alarm_counts(a)
        os.makedirs(ctx.out_dir, exist_ok=True)
        un.to_csv(os.path.join(ctx.out_dir, f"unmapped_alarms_{tag}.csv"), index=False, encoding="utf-8-sig")
        ep.to_csv(os.path.join(ctx.out_dir, f"episodes_{tag}.csv"), index=False, encoding="utf-8-sig")
        ev.to_csv(os.path.join(ctx.out_dir, f"target_events_{tag}.csv"), index=False, encoding="utf-8-sig")
        return res
    res = ae.count_judge_events(ctx.events, start, end, ctx.data_days)
    res["range"] = [start, end]
    need = 20
    res["judge_days_planned"] = ctx.periods.get("judge_data_days")
    if ctx.periods.get("judge_data_days"):
        res["r_needed_for_20"] = need / ctx.periods["judge_data_days"]
        res["c3_hold_expected"] = bool(res["r_per_data_day"] < res["r_needed_for_20"]) if not np.isnan(res["r_per_data_day"]) else None
    write_json(os.path.join(ctx.out_dir, f"count_only_{tag}.json"), res)
    return res


def cmd_join_check(ctx, range_spec="dev", manual_csv=None):
    start, end = pr.resolve_range(range_spec, ctx.periods, ctx.runs_dir, "join-check")
    ctx.record("join-check", start, end)
    a, ep, _ = ctx.in_range(start, end)
    eth = ctx.eth
    col = "bin_end" if "bin_end" in eth.columns else "occur_date"
    e = eth[(pd.to_datetime(eth[col]) >= start) & (pd.to_datetime(eth[col]) < end)]
    # 에피소드에 알람 address 의 addr_port(최빈값)를 붙인다 — (d) address 조인 참고용
    ap = a.assign(addr_port=a["address"].map(pj.norm_port)) if "address" in a.columns else a.assign(addr_port="")
    ap = ap[ap["addr_port"] != ""].groupby(["ip_addr", "cid", "lid"])["addr_port"].agg(lambda s: s.mode().iloc[0]).reset_index()
    cut = ep[ep["target_grades"].str.contains("트래픽단절/1차")].merge(ap, on=["ip_addr", "cid", "lid"], how="left")
    cut = cut.assign(addr_port=cut["addr_port"].fillna(""))
    j1 = pj.j1_key_consistency(e)
    j2 = pj.j2_diagnosis(a[a["row_class"].notna()], e)
    cols = ["ip_addr", "cid", "lid", "start", "end", "addr_port"]
    j3 = pj.j3_physical_match(cut[cols], e)
    # 진단(판정 아님): 알람 cid 를 -1 한 키로 본 J3 — 키 조인이 틀렸을 때 어긋남의 모양을 확인
    sh = cut[cols].copy()
    sh["cid"] = sh["cid"] - 1
    j3_shift = pj.j3_physical_match(sh[sh["cid"] > 0], e)
    j3_shift.pop("detail")
    j4 = pj.j4_coverage(a, e)
    verdict = pj.judge_join(j1, j3)
    scan = pj.cid_offset_scan(a, e)
    manual = None
    if manual_csv:
        manual = pj.check_manual(pj.load_manual(manual_csv))
        if manual["stop"]:
            verdict = {**verdict, "status": "stopped", "stop": True,
                       "reasons": verdict["reasons"] + [f"M1 수동 확인에서 키가 다른 포트 {len(manual['mismatches'])}건 — J 판정과 무관하게 멈춤"]}
    detail = j3.pop("detail")
    res = {"range": [start, end], "join_status": verdict["status"], "verdict": verdict, "j1": j1, "j2": j2, "j3": j3, "j4": j4,
           "diagnostic_j3_alarm_cid_minus_1": j3_shift, "diagnostic_cid_offset_scan": scan, "diagnostic_cid_slot_offsets": {"perf": pj.cid_slot_offsets(e), "alarm": pj.cid_slot_offsets(a[a["row_class"] == "port"])},
           "manual": manual, "note": "cid_offset_scan·cid_slot_offsets 는 진단이며 조인 키를 바꾸지 않는다 (자동 전환 없음, 설계 6.3)"}
    tag = range_spec.replace("..", "_")
    write_json(os.path.join(ctx.out_dir, f"join_check_{tag}.json"), res)
    detail.to_csv(os.path.join(ctx.out_dir, f"join_j3_detail_{tag}.csv"), index=False, encoding="utf-8-sig")
    return res


# ---- 모델 정보·잠금·재생·보고 ------------------------------------------------------------------------------
def model_info(ft="traffic", version=None):
    """레지스트리의 활성(또는 지정) 버전의 메타: version, trained_at, training_window, alert_policy."""
    from src.pipeline.inference import AnomalyDetector
    _, _, meta_path = AnomalyDetector._resolve_paths(ft, version)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if version is None:
        reg = os.path.join(os.path.dirname(meta_path), f"{ft}_registry.json")
        if os.path.exists(reg):
            with open(reg, encoding="utf-8") as f:
                version = json.load(f).get("active_version")
    model_path, scaler_path, _ = AnomalyDetector._resolve_paths(ft, version)
    return {"track": ft, "version": version, "trained_at": meta.get("trained_at"), "training_window": meta.get("training_window"),
            "alert_policy": meta.get("alert_policy"), "meta_path": meta_path,
            "model_sha": _file_sha(model_path), "scaler_sha": _file_sha(scaler_path), "meta_sha": _file_sha(meta_path)}


def _file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if path and os.path.exists(path) else None


def config_parts(ctx, join_status):
    """잠금 해시에 들어가는 설정 (매핑 CSV·작업 대용 목록·F1~F8·성능 공백 기준·격리 기준·조인 판정 결과·베이스라인 기본값)."""
    from validation.evaluation import baselines as bl
    from validation.realdata import tables as tb
    return {
        "target_map_sha": _file_sha(ctx.target_map_path), "alarm_names_sha": ctx.names_sha, "override_sha": _file_sha(ctx.override_csv),
        "maintenance_sha": _file_sha(ctx.maintenance_csv), "work_names": list(ae.WORK_NAMES), "hw_patterns": list(ae.HW_PATTERNS), "row_class": {"version": pj.ROW_CLASS_VERSION, "rules": [list(r) for r in pj.ROW_RULES]},
        "rules": {"F1": ae.FLICKER_MERGE_MIN, "F2": ae.MIN_EPISODE_MIN, "F4": ae.EVENT_MERGE_MIN, "F6": ae.WORK_WINDOW_MIN,
                  "F8": ae.HW_WINDOW_MIN, "judge": [list(ae.JUDGE_TYPES), ae.JUDGE_GRADE]},
        "perf_gap": {"c3_lookback_h": 6, "c3_min_ratio": rpt.C3_PERF_MIN_RATIO, "delay_window_h": 1, "c3_min_events": rpt.C3_MIN_EVENTS},
        "quarantine_margin": str(tb.QUARANTINE_MARGIN), "join_status": join_status,
        "baselines": {"fixed": {k: v.default for k, v in inspect.signature(bl.fixed_threshold).parameters.items() if v.default is not inspect.Parameter.empty},
                      "rolling": {k: v.default for k, v in inspect.signature(bl.rolling_rule).parameters.items() if v.default is not inspect.Parameter.empty}},
        "periods": {k: str(ctx.periods.get(k)) for k in ("S", "J0", "D")}, "alternative": ctx.periods.get("alternative"),
    }


def cmd_lock(ctx, version=None, join_status_path=None, model_infos=None):
    """판정 기간 잠금 (9.7.3-4). 조건: 기간 계산 가능, 개발 기간 join-check 결과 존재, 판정 모델 학습 구간 검사 통과(9.7.5)."""
    if not ctx.periods.get("ok"):
        raise pr.PeriodError(ctx.periods.get("reason") or "기간 계산 불가")
    conflicts = pr.ledger_conflicts(ctx.runs_dir, ctx.periods)
    if conflicts:
        raise pr.PeriodError(f"분석 장부에 판정 구간 [{ctx.periods['J0']:%Y-%m-%d}, {ctx.periods['D']:%Y-%m-%d}) 와 겹치는 기록이 {len(conflicts)}건 있어 "
                             f"잠금을 거부합니다 (이미 본 데이터로 판정할 수 없음, 9.7.4): " + "; ".join(
                                 f"{r['command']}/{r['purpose']} {r['start'][:10]}~{r['end'][:10]} (S={r['S'][:10]}, D={r['D'][:10]})" for r in conflicts[:5]))
    ledger_missing = conflicts is None
    if ledger_missing:
        import sys as _sys
        print("[경고] 분석 장부가 없습니다 — 잠금에 ledger_missing 을 기록합니다 (9.7.4)", file=_sys.stderr)
    jpath = join_status_path or os.path.join(ctx.out_dir, "join_check_dev.json")
    if not os.path.exists(jpath):
        raise pr.PeriodError(f"개발 기간 join-check 결과가 없습니다: {jpath} — 먼저 join-check 를 실행하세요 (9.7.3-3)")
    with open(jpath, encoding="utf-8") as f:
        join = json.load(f)
    model = model_infos or {"traffic": model_info("traffic", version)}
    for ft, info in model.items():
        ok, why = pr.check_training_window(info, ctx.periods["J0"])
        info["window_check"] = {"ok": ok, "reason": why}
        if not ok:
            raise pr.PeriodError(f"판정 모델 {ft} {info.get('version')} 학습 구간 검사 실패 (9.7.5, 측정 전 거부): {why}")
    dev = ae.count_judge_events(ctx.events, ctx.periods["S"], ctx.periods["J0"], ctx.data_days)
    need = 20 / ctx.periods["judge_data_days"]
    parts = config_parts(ctx, join["join_status"])
    pol = (model.get("traffic") or {}).get("alert_policy")
    payload = {"alternative": ctx.periods["alternative"], "S": ctx.periods["S"], "D": ctx.periods["D"], "J0": ctx.periods["J0"],
               "judge_data_days": ctx.periods["judge_data_days"], "data_days": ctx.periods["data_days"],
               "models": {ft: {k: v for k, v in info.items() if k != "meta_path"} for ft, info in model.items()},
               "policy_hash": pr.stable_hash(pol), "config_hash": pr.stable_hash(parts), "config_parts": parts,
               "join_status": join["join_status"], "ledger_missing": ledger_missing, "dev_event_rate_r": dev["r_per_data_day"], "r_needed_for_20": need,
               "c3_hold_expected": bool(dev["r_per_data_day"] < need), "exposure_note": rpt.EXPOSURE_NOTE,
               "locked_at": str(pd.Timestamp.now())}
    path = pr.create_lock(ctx.runs_dir, ctx.periods, payload)
    return {"lock_path": path, **{k: payload[k] for k in ("J0", "D", "judge_data_days", "join_status", "dev_event_rate_r", "c3_hold_expected", "config_hash")}}


def _tag(spec):
    return spec.replace("..", "_").replace(" ", "")


def cmd_replay(ctx, range_spec, purpose="dev", versions=None, policy=None, detector=None, save=True, command="replay", complete=True):
    """재생 (스냅샷 → AI·고정·롤링). purpose=dev: 개발 기간 안에서만. purpose=judge: 잠금·설정 해시·모델 검사·1회 표식을 통과해야 한다."""
    start, end = pr.resolve_range(range_spec, ctx.periods, ctx.runs_dir, f"{command}[{purpose}]")
    lock = None
    if purpose == "explore":
        if not ctx.periods.get("explore_only"):
            raise pr.PeriodError("purpose=explore 는 데이터일 < 28 로 기간을 정할 수 없을 때만 허용됩니다 — dev/judge 를 쓰세요.")
        ctx.record(command, start, end, "explore")
    elif ctx.periods.get("explore_only"):
        raise pr.PeriodError(ctx.periods["reason"])
    elif purpose == "dev":
        if pr.overlaps_judge(start, end, ctx.periods):
            raise pr.PeriodError("purpose=dev 는 판정 구간과 겹칠 수 없습니다 (판정 구간은 purpose=judge 1회만).")
        ctx.record(command, start, end, "dev")
    elif purpose == "judge":
        return _judge_replay(ctx, range_spec, start, end, versions, policy, detector, save, command, complete)
    else:
        raise pr.PeriodError(f"알 수 없는 purpose: {purpose}")
    if versions is None and detector is None:
        versions = {"traffic": model_info("traffic")["version"]}          # 활성 버전을 명시해 기록한다
    res = rp.replay(ctx.eth, start, end, versions=versions, policy=policy, detector=detector)
    res["purpose"], res["versions"], res["lock"] = purpose, versions, None
    if save:
        res["files"] = _save_replay(ctx, res, purpose, range_spec, start, end, versions)
    return res


def _save_replay(ctx, res, purpose, range_spec, start, end, versions):
    os.makedirs(ctx.out_dir, exist_ok=True)
    stacked = pd.concat([f.assign(method=m) for m, f in res["frames"].items()], ignore_index=True)
    f1 = os.path.join(ctx.out_dir, f"replay_{purpose}_{_tag(range_spec)}.csv.gz")
    f2 = os.path.join(ctx.out_dir, f"replay_{purpose}_{_tag(range_spec)}.json")
    stacked.to_csv(f1, index=False, compression="gzip")
    write_json(f2, {"range": [start, end], "slots": res["slots"], "versions": versions, "purpose": purpose,
                    "input_rows": res["input_rows"], "normal_rows": res["normal_rows"]})
    return [f1, f2]


def _judge_replay(ctx, range_spec, start, end, versions, policy, detector, save, command, complete):
    """판정 재생 (설계 9.7.6, 개정 4): 표식 전에 할 수 있는 일(해시 대조·모델/스케일러/정책 로드·스냅샷·출력 폴더 쓰기 확인)을 먼저 끝낸다.
    그 단계의 실패는 표식이 없으므로 그냥 다시 실행하면 된다. 이후 phase: claimed → computing → completed."""
    if policy is not None:
        raise pr.PeriodError("판정 실행에서는 --policy 를 쓸 수 없습니다 — 정책은 잠금에 동결된 모델 메타 정책만 사용 (P6)")
    lock = _verify_judge_preconditions(ctx, range_spec, start, end, versions, detector)
    versions = {ft: m["version"] for ft, m in lock["models"].items()}
    det = detector if detector is not None else rp.build_detector(versions, None)          # 모델 로드: 표식 전
    os.makedirs(ctx.out_dir, exist_ok=True)
    probe = os.path.join(ctx.out_dir, ".write_probe")
    open(probe, "w").close()
    os.remove(probe)
    ctx.record(command, start, end, "judge")
    from validation.realdata.extract import tool_version
    pr.claim_judge_run(ctx.runs_dir, ctx.periods, {"started_at": str(pd.Timestamp.now()), "config_hash": lock["config_hash"], "tool_commit": tool_version()})
    try:                                                                                   # phase=claimed: 판정 데이터는 아직 모델에 들어가지 않음
        inp = rp.traffic_frame(ctx.eth, start, end)
    except BaseException as e:                                                             # Ctrl+C 포함
        pr.abort_claimed(ctx.runs_dir, ctx.periods, repr(e))
        raise
    pr.set_phase(ctx.runs_dir, ctx.periods, "computing")                                   # 이 줄부터 실패는 자동 되돌림 없음
    res = rp.replay(ctx.eth, start, end, versions=versions, detector=det, inp=inp)
    res["purpose"], res["versions"], res["lock"] = "judge", versions, lock
    res["files"] = _save_replay(ctx, res, "judge", range_spec, start, end, versions) if save else []
    if complete:
        pr.complete_judge_run(ctx.runs_dir, ctx.periods, res["files"])
    return res


def cmd_retry_judge(ctx, reason, reviewed_by, detector=None):
    """`periods retry-judge`: 잠금의 설정·모델·정책 해시가 그대로인지 확인한 뒤 표식을 failed_1 로 돌리고 1회 재시도를 허용한다."""
    from validation.realdata.extract import tool_version
    start, end = pr.range_bounds(ctx.periods, "judge")
    _verify_judge_preconditions(ctx, "judge", start, end, None, detector)
    partial = sorted(f for f in (os.listdir(ctx.out_dir) if os.path.isdir(ctx.out_dir) else []) if "_judge_" in f)
    return pr.retry_judge(ctx.runs_dir, ctx.periods, reason, reviewed_by, tool_version(), partial)


def _verify_judge_preconditions(ctx, range_spec, start, end, versions, detector=None):
    """판정 실행 전 거부 규칙 (측정 전, 표식 생성 없음). 잠금 존재·판정 구간 정확히 일치·설정 해시 동일·모델/정책/가중치 검사."""
    if range_spec != "judge":
        raise pr.PeriodError("purpose=judge 는 --range judge 만 허용됩니다.")
    lock = pr.load_lock(ctx.runs_dir, ctx.periods)
    if lock is None:
        raise pr.PeriodError("잠금 파일이 없습니다 — periods lock 후에만 판정 실행 가능")
    parts = config_parts(ctx, lock["join_status"])
    if pr.stable_hash(parts) != lock["config_hash"]:
        raise pr.PeriodError("현재 설정 해시가 잠금 때와 다릅니다 (매핑 CSV·작업 대용 목록·규칙 등이 바뀜) — 판정 거부 (9.7.6)")
    if versions and {k: v for k, v in versions.items()} != {ft: m["version"] for ft, m in lock["models"].items()}:
        raise pr.PeriodError("--versions 가 잠금에 기록된 판정 모델 버전과 다릅니다 (동결, P6)")
    for ft, m in lock["models"].items():
        ok, why = pr.check_training_window({"trained_at": m.get("trained_at"), "training_window": m.get("training_window")}, ctx.periods["J0"])
        if not ok:
            raise pr.PeriodError(f"판정 모델 {ft} 학습 구간 검사 실패: {why}")
    # 주의: detector 를 직접 주입하면(테스트 전용) 아래 모델·정책·가중치 해시 검사는 생략된다. CLI 는 항상 detector=None 이다.
    if detector is None:                                  # 현재 레지스트리의 모델·정책·가중치가 잠금 때와 같은지 (실제 모델 로드 경로)
        for ft, m in lock["models"].items():
            cur = model_info(ft, m["version"])
            diff = [k for k in ("trained_at", "model_sha", "scaler_sha", "meta_sha") if cur.get(k) != m.get(k)]
            if pr.stable_hash(cur.get("alert_policy")) != lock["policy_hash"]:
                diff.append("alert_policy")
            if diff:
                raise pr.PeriodError(f"판정 모델 {ft} {m['version']} 가 잠금 때와 다릅니다 ({', '.join(diff)}) — 모델/정책 동결 위반, 판정 거부 (P6)")
    return lock


def cmd_report(ctx, range_spec, purpose="dev", versions=None, policy=None, detector=None, key_shift=0):
    if purpose == "judge" and key_shift != 0:
        raise pr.PeriodError("키 어긋남 보정(--diag-key-shift)은 진단 전용이며 판정 실행에는 쓸 수 없습니다 (조인 키 변경은 사용자 결정).")
    res = cmd_replay(ctx, range_spec, purpose, versions, policy, detector, command="report", complete=False)
    if purpose == "judge":
        join_status = res["lock"]["join_status"]
    else:
        jp = os.path.join(ctx.out_dir, "join_check_dev.json")
        join_status = json.load(open(jp, encoding="utf-8"))["join_status"] if os.path.exists(jp) else "not_checked"
    start, end = res["range"]
    net_days = []
    if ctx.maintenance is not None:
        for r in ctx.maintenance.itertuples():
            net_days += list(pd.date_range(pd.Timestamp(r.start).normalize(), pd.Timestamp(r.end).normalize(), freq="D"))
    a, ep, _ = ctx.in_range(start, end)
    info = {"short_episodes": int(ep["is_short"].sum()) if len(ep) else 0,
            "no_table_devices": ae.no_table_share(a)["devices_without_table"] if len(a) else 0}
    header = {"model_versions": res["versions"], "privilege_check": ctx.manifest.get("privilege_check"),
              "snapshot_complete": ctx.manifest.get("complete"), "periods": {k: str(ctx.periods.get(k)) for k in ("S", "J0", "D")}}
    # 설계 9.5(개정 4): 절단 경계 E = 분석 기간의 끝(개발=J0, 판정=D, START..END 이면 END). 개발 실행에 D 를 넘기면 J0 직전 이벤트가 미탐으로 잘못 센다.
    # 태깅·J0 이후 정보 노출 축소(9.7.2): 분석 끝 이전에 시작한 에피소드만, 끝은 분석 끝에서 자른 값을 쓴다.
    ep_view = _clip_episodes(ctx.episodes[ctx.episodes["start"] < end], end)
    rep = rpt.build_report(ctx.events, ep_view, info, res, ctx.eth, end, ctx.data_days, join_status, purpose,
                           header, key_shift, net_days)
    tag = f"{purpose}_{_tag(range_spec)}"
    ev_rows, de_rows = rep.pop("c3_events"), rep.pop("delay_events")
    write_json(os.path.join(ctx.out_dir, f"report_{tag}.json"), rep)
    ev_rows.to_csv(os.path.join(ctx.out_dir, f"report_c3_events_{tag}.csv"), index=False, encoding="utf-8-sig")
    de_rows.to_csv(os.path.join(ctx.out_dir, f"report_delay_events_{tag}.csv"), index=False, encoding="utf-8-sig")
    if purpose == "judge":                                                               # 모든 출력 작성 후 completed (출력 해시 기록)
        pr.complete_judge_run(ctx.runs_dir, ctx.periods, res.get("files", []) + [os.path.join(ctx.out_dir, f"report_{tag}.json"),
                              os.path.join(ctx.out_dir, f"report_c3_events_{tag}.csv"), os.path.join(ctx.out_dir, f"report_delay_events_{tag}.csv")])
    return rep


# ---- 샘플·채점·C5 ---------------------------------------------------------------------------------------
def load_replay_frames(ctx, purpose, range_spec):
    """cmd_replay 가 저장한 방법별 재생 프레임. 없으면 먼저 replay/report 를 실행하라고 안내(판정 구간은 여기서 재생하지 않는다)."""
    path = os.path.join(ctx.out_dir, f"replay_{purpose}_{_tag(range_spec)}.csv.gz")
    if not os.path.exists(path):
        raise pr.PeriodError(f"재생 결과가 없습니다: {path} — 먼저 `replay`(또는 `report`)를 같은 --purpose/--range 로 실행하세요")
    df = pd.read_csv(path, compression="gzip", parse_dates=["occur_date"], low_memory=False)
    df["ip_addr"] = df["ip_addr"].astype(str).str.strip()
    return {m: g.drop(columns="method").reset_index(drop=True) for m, g in df.groupby("method")}


def _ems_lookup(ctx):
    by_port = {k: g for k, g in ctx.alarms.groupby(["ip_addr", "cid", "lid"])}
    by_dev = {ip: g for ip, g in ctx.alarms[ctx.alarms["row_class"] == "device"].groupby("ip_addr")}

    def look(ip, cid, lid, lo, hi):
        names = []
        for g in (by_port.get((ip, cid, lid)), by_dev.get(ip)):
            if g is not None:
                near = g[(g["occur_date"] >= lo) & (g["occur_date"] <= hi)]
                names += list(near["alarm_name"])
        uniq = sorted(set(names))
        return ";".join(uniq[:10]) + (f";…(+{len(uniq) - 10})" if len(uniq) > 10 else "")
    return look


def _work_lookup(ctx):
    work = ctx.alarms[ctx.alarms["alarm_name"].isin(ae.WORK_NAMES) & ctx.alarms["name_status"].isin(["ok", "override"])]
    by_ip = {ip: g for ip, g in work.groupby("ip_addr")}

    def look(ip, cid, lid, lo, hi):
        g = by_ip.get(ip)
        if g is None:
            return False
        near = g[(g["occur_date"] >= lo) & (g["occur_date"] <= hi)]
        if near.empty:
            return False
        scope = np.where(near["row_class"].isin(["device", "unit"]), True, np.where(near["row_class"] == "slot", near["cid"] == cid,
                                                                      (near["cid"] == cid) & (near["lid"] == lid)))
        return bool(np.asarray(scope).any())
    return look


def cmd_sample(ctx, range_spec, purpose, seed):
    from validation.realdata import review as rv
    start, end = pr.resolve_range(range_spec, ctx.periods, ctx.runs_dir, f"sample[{purpose}]")
    if purpose == "explore" and not ctx.periods.get("explore_only"):
        raise pr.PeriodError("purpose=explore 는 데이터일 < 28 로 기간을 정할 수 없을 때만 허용됩니다.")
    if purpose != "explore" and ctx.periods.get("explore_only"):
        raise pr.PeriodError(ctx.periods["reason"])
    if purpose == "dev" and pr.overlaps_judge(start, end, ctx.periods):
        raise pr.PeriodError("purpose=dev 샘플은 판정 구간과 겹칠 수 없습니다.")
    if purpose == "judge" and range_spec != "judge":
        raise pr.PeriodError("purpose=judge 샘플은 --range judge 만 허용됩니다.")
    frames = load_replay_frames(ctx, purpose, range_spec)
    ctx.record("sample", start, end, purpose)
    tag = f"{purpose}_{_tag(range_spec)}"
    rpath = os.path.join(ctx.out_dir, f"sample_review_{tag}.csv")
    if os.path.exists(rpath):
        raise pr.PeriodError(f"검토 샘플이 이미 있습니다(라벨 보호를 위해 덮어쓰지 않음): {rpath}")
    review, key, meta = rv.build_sample(frames, purpose, seed, _ems_lookup(ctx), _work_lookup(ctx))
    meta["join_note"] = "EMS 알람 열은 (ip_addr,cid,lid) 키 조인 기준 — join_status 가 verified 가 아니면 참고만"
    review.to_csv(rpath, index=False, encoding="utf-8-sig")
    key.to_csv(os.path.join(ctx.out_dir, f"sample_key_{tag}.csv"), index=False, encoding="utf-8-sig")      # 비공개 키: 검토자에게 주지 않는다
    write_json(os.path.join(ctx.out_dir, f"sample_meta_{tag}.json"), meta)
    return meta


def cmd_score_review(ctx_or_out, review_csv, key_csv, report_json=None):
    from validation.realdata import review as rv
    review = pd.read_csv(review_csv, encoding="utf-8-sig", dtype={"ip_addr": str})
    key = pd.read_csv(key_csv, encoding="utf-8-sig")
    c1 = None
    if report_json:
        with open(report_json, encoding="utf-8") as f:
            c1 = json.load(f)["c1_raw"]
    res = rv.score_review(review, key, c1)
    out_dir = ctx_or_out if isinstance(ctx_or_out, str) else ctx_or_out.out_dir
    write_json(os.path.join(out_dir, "score_review_" + os.path.splitext(os.path.basename(review_csv))[0].replace("sample_review_", "") + ".json"), res)
    return res


def cmd_live(ctx, db, range_spec, purpose, result_schema=None):
    from validation.realdata import live as lv
    start, end = pr.resolve_range(range_spec, ctx.periods, ctx.runs_dir, f"live[{purpose}]")
    frames = load_replay_frames(ctx, purpose, range_spec)
    stored = lv.fetch_results(db, start, end, result_schema)
    res = lv.compute_c5(stored, frames["ai"])
    res["range"] = [start, end]
    write_json(os.path.join(ctx.out_dir, f"live_c5_{purpose}_{_tag(range_spec)}.json"), res)
    return res
