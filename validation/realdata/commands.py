"""
P1-2 스냅샷 기반 하위 명령 본체 (DB 접속 없음). CLI(`validation/cli/real_data_check.py`)가 호출한다.
모든 출력은 `<run>/out/` 아래에 쓴다(validation/runs 는 git 제외 — IP·국소명이 들어갈 수 있음).
"""
import json
import os

import numpy as np
import pandas as pd

from validation.realdata import alarm_episodes as ae
from validation.realdata import periods as pr
from validation.realdata import port_join as pj
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


class Context:
    """스냅샷 한 번 로드 + 기간 계산 + 알람 준비. 판정 구간을 포함한 전체를 읽지만 출력은 resolve_range 가 허용한 범위만."""

    def __init__(self, run_dir, runs_dir, names_path=DEFAULT_NAMES_PATH, map_path=DEFAULT_TARGET_MAP, override_csv=None,
                 maintenance_csv=None):
        self.run_dir, self.runs_dir = run_dir, runs_dir
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

    def in_range(self, start, end):
        a = self.alarms[(self.alarms["occur_date"] >= start) & (self.alarms["occur_date"] < end)]
        ep = self.episodes[(self.episodes["start"] >= start) & (self.episodes["start"] < end)]
        ev = self.events[(self.events["t0"] >= start) & (self.events["t0"] < end)]
        return a, ep, ev


def cmd_alarms(ctx, range_spec, mode):
    """mode = summary | count. 판정 구간과 겹치면 잠금 전 거부(P4)."""
    start, end = pr.resolve_range(range_spec, ctx.periods, ctx.runs_dir, f"alarms --{mode}")
    a, ep, ev = ctx.in_range(start, end)
    tag = range_spec.replace("..", "_").replace(" ", "")
    if mode == "summary":
        res = ae.summarize_alarms(a, ep, ev)
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
