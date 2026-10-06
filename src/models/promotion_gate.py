"""
승격 게이트 (P1-1): 후보 모델을 활성화하기 전의 사전 검증

학습 직후 1회 실행해 결과를 레지스트리 엔트리의 `gate` 에 저장하고, 승격 API 는 저장된 결과를 사용한다
(결과 재현성 · 응답 지연 없음). `POST /api/model/gate` 로 재실행할 수 있다.

| ID | 검사 | 실패 시 |
|----|------|---------|
| G1 | 아티팩트: 파일 존재, threshold/val_loss 가 유한한 양수, input_dim 이 현재 DataProcessor 와 일치 | FAIL (force 로도 승격 불가) |
| G2 | 임계치·검증 손실이 활성 모델 대비 3배 초과/미만 (registry.compare_with_active) | FAIL |
| G3 | 홀드아웃 포트 알람 비율: 후보 ≤ max(활성×1.5, 0.01) 이고 ≤ 0.05 인시던트/포트·일 | FAIL |
| G4 | 카나리 구분력: AUPRC(후보) ≥ AUPRC(활성) − 0.05 (카나리 파일이 없으면 SKIP) | FAIL |
| G5 | 임계치 상승 추세: 최근 활성 버전들 + 후보의 임계치가 단조 증가하고 누적 2배 이상 | WARN |

종합: FAIL 이 하나라도 있으면 FAIL, 없고 WARN 이 있으면 WARN, 나머지 PASS (SKIP 은 무시). 실행 자체가 실패하면 ERROR.

G3 의 '알람 비율'은 정답 없는 대리 지표다 — 후보의 알람이 적은 것이 오탐 감소인지 둔감화인지 G3 만으로는 구분할 수 없어
G4/G5 와 사람의 검토가 필요하다. 알람 계산은 운영과 같은 코드(alerting.alarms_from_track_scores)와 현재 운영 정책을 쓴다.
"""
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Callable, List

import numpy as np
import pandas as pd

from src.data import train_window
from src.models import registry
from src.pipeline import alerting

PASS, FAIL, WARN, SKIP, ERROR = "PASS", "FAIL", "WARN", "SKIP", "ERROR"
KEY = alerting.KEY


@dataclass
class GateCheck:
    id: str
    status: str            # PASS | FAIL | WARN | SKIP
    value: Any = None
    limit: Any = None
    message: str = ""
    detail: Any = None     # 검사별 보조 수치 (G3: 후보/활성의 인시던트 수·포트·일)


@dataclass
class GateResult:
    status: str                                   # PASS | FAIL | WARN | ERROR
    checks: List[GateCheck] = field(default_factory=list)
    evaluated_at: str = ""
    data: dict = field(default_factory=dict)      # 기간, 포트 수, 비교 대상 버전, 정책 등

    def to_dict(self):
        return asdict(self)


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def overall_status(checks) -> str:
    """FAIL > WARN > PASS. G3(알람 부하) 를 판정하지 못한 SKIP 은 WARN 으로 올린다 — auto 모드가 PASS 에서만
    승격하므로, 알람 부하를 확인하지 못한 후보는 자동 승격되지 않는다. 다른 검사의 SKIP(예: 카나리 없음)은 무시."""
    statuses = {c.status for c in checks}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses or any(c.id == "G3" and c.status == SKIP for c in checks):
        return WARN
    return PASS


def _finite_positive(x):
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


# ─────────────────────────────────────────────
# 개별 검사 (순수)
# ─────────────────────────────────────────────
def check_g1(cand_meta, artifact_issues) -> GateCheck:
    issues = list(artifact_issues or [])
    if not _finite_positive(cand_meta.get("threshold")):
        issues.append(f"threshold 가 유한한 양수가 아님: {cand_meta.get('threshold')}")
    val = cand_meta.get("final_val_loss")
    if val is not None and not _finite_positive(val):
        issues.append(f"검증 손실이 유한한 양수가 아님: {val}")
    if issues:
        return GateCheck("G1", FAIL, None, None, "; ".join(issues))
    return GateCheck("G1", PASS, None, None, "아티팩트 정상")


def check_g2(cand_meta, active_meta) -> GateCheck:
    if not active_meta or active_meta.get("version") == cand_meta.get("version"):
        return GateCheck("G2", SKIP, None, None, "비교할 활성 모델 없음")
    reg = {"active_version": active_meta.get("version"), "versions": [active_meta]}
    warnings = registry.compare_with_active(reg, cand_meta)
    ta, tc = active_meta.get("threshold"), cand_meta.get("threshold")
    ratio = tc / ta if ta and tc else None
    if warnings:
        return GateCheck("G2", FAIL, ratio, registry.THRESHOLD_RATIO_LIMIT, "; ".join(warnings))
    return GateCheck("G2", PASS, ratio, registry.THRESHOLD_RATIO_LIMIT, "임계치·검증 손실이 활성 모델과 비슷함")


def _port_days(stats, limits):
    return stats.get("port_days", stats["ports"] * limits.gate_days)


def _g3_detail(cand_stats, active_stats, limits):
    def one(st):
        return None if not st or not st.get("ports") else {"incidents": st.get("incidents"), "port_days": _port_days(st, limits)}
    return {"cand": one(cand_stats), "active": one(active_stats)}


def check_g3(cand_stats, active_stats, limits) -> GateCheck:
    if not cand_stats or not cand_stats.get("ports"):
        return GateCheck("G3", SKIP, None, None, "게이트 데이터(홀드아웃 포트)가 없음")
    detail = _g3_detail(cand_stats, active_stats, limits)
    port_days, min_pd = _port_days(cand_stats, limits), limits.gate_min_port_days
    if port_days < min_pd:
        # 인시던트 1건의 비율이 하한·상한보다 커서 학습 난수만으로 PASS/FAIL 이 갈리므로 판정하지 않는다
        return GateCheck("G3", SKIP, port_days, min_pd,
                         f"포트·일 {port_days:g}<{min_pd}, 판정 불가 (홀드아웃 {cand_stats['ports']}포트, "
                         f"인시던트 {cand_stats.get('incidents')}건) — 사람이 검토하세요", detail)
    cand = cand_stats["incidents_per_port_day"]
    abs_limit = limits.gate_max_incidents_per_port_day
    limit = abs_limit
    if active_stats and active_stats.get("ports"):
        limit = min(abs_limit, max(active_stats["incidents_per_port_day"] * limits.gate_max_alarm_ratio,
                                   limits.gate_alarm_floor))
    count = f"인시던트 {cand_stats.get('incidents')}건/{port_days:g}포트·일"
    if cand <= limit:
        return GateCheck("G3", PASS, cand, limit,
                         f"홀드아웃 {cand_stats['ports']}포트 알람 {cand:.3f}건/포트·일 ≤ {limit:.3f} ({count})", detail)
    why = "활성 모델 대비 과다" if limit < abs_limit else "절대 상한 초과"
    return GateCheck("G3", FAIL, cand, limit, f"홀드아웃 {cand_stats['ports']}포트 알람 {cand:.3f}건/포트·일 > 기준 {limit:.3f} "
                     f"({registry.format_ratio(cand / limit if limit > 0 else None)}, {why}; {count})", detail)


def check_g4(canary, limits) -> GateCheck:
    if not canary:
        return GateCheck("G4", SKIP, None, None, "카나리 데이터 없음")
    cand, active = canary["cand_auprc"], canary.get("active_auprc")
    if active is None:
        return GateCheck("G4", SKIP, cand, None, "비교할 활성 모델 없음")
    limit = active - limits.gate_canary_auprc_drop
    if cand >= limit:
        return GateCheck("G4", PASS, cand, limit, f"카나리 AUPRC {cand:.3f} ≥ {limit:.3f} (활성 {active:.3f})")
    return GateCheck("G4", FAIL, cand, limit, f"카나리 AUPRC {cand:.3f} < {limit:.3f} (활성 {active:.3f}) — 구분력 저하")


def check_g5(cand_meta, history_thresholds, limits) -> GateCheck:
    n = limits.gate_threshold_trend_versions
    hist = [t for t in (history_thresholds or []) if _finite_positive(t)][-n:]
    cand = cand_meta.get("threshold")
    if len(hist) < n or not _finite_positive(cand):
        return GateCheck("G5", SKIP, None, None, f"임계치 추세를 볼 활성 이력이 {n}개 미만")
    seq = hist + [cand]
    rising = all(b > a for a, b in zip(seq, seq[1:]))
    factor = cand / hist[0]
    if rising and factor >= limits.gate_threshold_trend_factor:
        return GateCheck("G5", WARN, factor, limits.gate_threshold_trend_factor,
                         f"임계치가 {len(seq)}개 버전 연속 상승 (누적 {factor:.1f}배) — 장애를 정상으로 흡수하는 둔감화 의심")
    return GateCheck("G5", PASS, factor, limits.gate_threshold_trend_factor, "임계치 상승 추세 없음")


def evaluate(cand_meta, active_meta, cand_stats, active_stats, canary, history, limits,
             artifact_issues=None, data=None) -> GateResult:
    """검사 결과를 종합 (순수 함수). history: 최근 활성 버전들의 임계치 (오래된 것 -> 최신)"""
    checks = [check_g1(cand_meta, artifact_issues), check_g2(cand_meta, active_meta),
              check_g3(cand_stats, active_stats, limits), check_g4(canary, limits),
              check_g5(cand_meta, history, limits)]
    return GateResult(status=overall_status(checks), checks=checks, evaluated_at=_now(), data=data or {})


# ─────────────────────────────────────────────
# 추론을 포함한 실행
# ─────────────────────────────────────────────
def artifact_issues(model_dir, ft, entry) -> list:
    """파일 존재와 입력 차원(현재 DataProcessor 의 파생 변수 구성)이 맞는지"""
    issues = []
    try:
        registry._check_files(model_dir, entry)
    except registry.RegistryError as e:
        return [str(e)]
    try:
        import json
        from src.data.data_processor import DataProcessor
        with open(os.path.join(model_dir, entry["config_path"]), "r", encoding="utf-8") as f:
            cfg = json.load(f).get("config", {})
        expected = len(DataProcessor(ft, config=cfg or None).extended_feature_cols)
        if cfg.get("input_dim") not in (None, expected):
            issues.append(f"input_dim {cfg.get('input_dim')} != 현재 파생 변수 구성 {expected}")
    except Exception as e:
        issues.append(f"메타 파일을 읽을 수 없음: {e}")
    return issues


def _sample_ports(raw: pd.DataFrame, policy) -> pd.DataFrame:
    """검증(홀드아웃) 포트만, 최대 gate_max_ports 개 (결정적 샘플링)"""
    ports = raw[KEY].drop_duplicates()
    val = [tuple(r) for r in ports.itertuples(index=False)
           if train_window.is_val_port(r.ip_addr, r.cid, r.lid, policy.val_port_fraction, policy.split_salt)]
    val.sort()
    if len(val) > policy.gate_max_ports:
        step = len(val) / policy.gate_max_ports
        val = [val[int(i * step)] for i in range(policy.gate_max_ports)]
    return raw[pd.MultiIndex.from_frame(raw[KEY]).isin(val)]


def _model_policy(detector, ft, fallback):
    """그 모델이 쓸 알람 정책 (모델 메타 정책 > 운영 기본 정책). 가짜 탐지기(테스트)는 fallback."""
    getter = getattr(detector, "policy_for", None)
    return getter(ft) if getter else fallback


def _track_scores(detector, raw, ft, since):
    scores, th = detector.track_scores(raw, ft)
    if scores is None:
        return None, th
    scores = scores[pd.to_datetime(scores["occur_date"]) >= since]
    return scores, th


def _alarm_stats(scores, th, alert_policy, days, ft):
    """(통계 dict, 알람 프레임). 점수가 없으면 (None, None)"""
    if scores is None or scores.empty:
        return None, None
    alarms = alerting.alarms_from_track_scores({ft: (scores, th)}, alert_policy)
    ports = int(scores[KEY].drop_duplicates().shape[0])
    n_inc = alerting.count_incidents(alarms)
    return {"ports": ports, "incidents": n_inc, "alarm_rows": int(alarms["alarm"].sum()), "port_days": ports * days,
            "incidents_per_port_day": n_inc / (ports * days)}, alarms


def run_gate(model_dir: str, ft: str, version: str, window, policy, fetch_raw: Callable,
             alert_policy, detector_factory=None) -> GateResult:
    """후보 `version` 을 검증해 GateResult 를 반환 (레지스트리에는 쓰지 않음 — 호출자가 set_gate).

    fetch_raw(ft, start, end) -> 원본 성능 DataFrame (게이트 구간 + MA 이력용 1일 앞). 어떤 단계든 예외가 나면
    status=ERROR 로 반환한다 (게이트가 학습/API 를 중단시키지 않도록, 사람이 재실행하거나 force 로 판단).
    detector_factory(version|None) -> AnomalyDetector : 테스트에서 가짜 주입용.
    """
    reg = registry.load(model_dir, ft)
    entry = registry.find(reg, version)
    if entry is None:
        raise registry.VersionNotFound(f"버전 '{version}' 이(가) 없습니다.")
    active_ver = reg.get("active_version")
    active = registry.find(reg, active_ver) if active_ver and active_ver != version else None
    data = {"version": version, "against": active["version"] if active else None,
            "gate_start": window.gate_start.strftime("%Y-%m-%d %H:%M:%S"),
            "gate_end": window.gate_end.strftime("%Y-%m-%d %H:%M:%S"), "track": ft}
    try:
        issues = artifact_issues(model_dir, ft, entry)
        cand_stats = active_stats = canary = None
        if not issues:
            if detector_factory is None:
                from src.pipeline.inference import AnomalyDetector

                def detector_factory(v):
                    return AnomalyDetector(policy=alert_policy, versions={ft: v} if v else None, tracks=(ft,))
            cand_det = detector_factory(version)
            act_det = detector_factory(None) if active else None

            raw = fetch_raw(ft, window.gate_start - timedelta(days=1), window.gate_end)
            if raw is not None and len(raw):
                raw = _sample_ports(raw, policy)
            cand_scores = act_scores = cand_th = act_th = None
            if raw is not None and len(raw):
                cand_scores, cand_th = _track_scores(cand_det, raw, ft, window.gate_start)
                if act_det is not None:
                    act_scores, act_th = _track_scores(act_det, raw, ft, window.gate_start)
            days = max(policy.gate_days, 1)
            # 후보는 후보의 정책으로, 활성은 활성의 정책으로 계산한다 (운영에서 실제로 날 알람의 비교, P1-3 4.2-6)
            cand_policy = _model_policy(cand_det, ft, alert_policy)
            act_policy = _model_policy(act_det, ft, alert_policy) if act_det is not None else None
            data["alert_policy"] = {"candidate": alerting.policy_to_meta(cand_policy),
                                    "active": alerting.policy_to_meta(act_policy) if act_policy is not None else None}
            cand_stats, cand_alarms = _alarm_stats(cand_scores, cand_th, cand_policy, days, ft)
            active_stats, act_alarms = _alarm_stats(act_scores, act_th, act_policy, days, ft)
            if cand_stats and act_alarms is not None and len(cand_alarms):
                ca = cand_alarms[cand_alarms["alarm"]][KEY + ["occur_date"]]
                aa = act_alarms[act_alarms["alarm"]][KEY + ["occur_date"]]
                cand_stats["overlap_with_active"] = (len(ca.merge(aa, on=KEY + ["occur_date"])) / len(ca)) if len(ca) else None
            data["ports"] = cand_stats["ports"] if cand_stats else 0

            canary = _canary_auprc(model_dir, ft, cand_det, act_det)

        history = _activation_thresholds(reg, exclude=version, n=policy.gate_threshold_trend_versions)
        result = evaluate(entry, active, cand_stats, active_stats, canary, history, policy,
                          artifact_issues=issues, data=data)
        result.data.update({"candidate_stats": cand_stats, "active_stats": active_stats, "canary": canary})
        return result
    except Exception as e:
        return GateResult(status=ERROR, checks=[GateCheck("ERROR", ERROR, None, None, f"게이트 실행 실패: {e}")],
                          evaluated_at=_now(), data=data)


def _activation_thresholds(reg, exclude, n):
    """활성화 이력 순서대로 최근 n 개 서로 다른 버전의 임계치 (오래된 것 -> 최신)"""
    order = []
    for h in reg.get("history", []):
        v = h.get("version")
        if v == exclude:
            continue
        if v in order:
            order.remove(v)
        order.append(v)
    out = []
    for v in order[-n:]:
        e = registry.find(reg, v)
        out.append(e.get("threshold") if e else None)
    return out


def canary_path(model_dir, ft):
    return os.path.join(model_dir, f"{ft}_canary.csv")


def _canary_auprc(model_dir, ft, cand_det, act_det):
    """<ft>_canary.csv(원본 성능 컬럼 + label)가 있으면 후보/활성의 AUPRC 를 계산. 없으면 None."""
    path = canary_path(model_dir, ft)
    if not os.path.exists(path):
        return None
    from sklearn.metrics import average_precision_score
    df = pd.read_csv(path)
    labels = df[KEY + ["occur_date", "label"]].copy()
    labels["occur_date"] = pd.to_datetime(labels["occur_date"]).dt.round("15min")

    def auprc(det):
        scores, _ = det.track_scores(df.drop(columns=["label"]), ft)
        if scores is None:
            return None
        m = scores.assign(occur_date=pd.to_datetime(scores["occur_date"]).dt.round("15min")).merge(
            labels, on=KEY + ["occur_date"])
        if m["label"].nunique() < 2:
            return None
        return float(average_precision_score(m["label"].astype(int), m["mse"]))

    cand = auprc(cand_det)
    if cand is None:
        return None
    return {"cand_auprc": cand, "active_auprc": auprc(act_det) if act_det is not None else None, "rows": int(len(df))}
