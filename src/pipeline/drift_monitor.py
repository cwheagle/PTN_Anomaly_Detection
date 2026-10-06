"""
Data Drift 감지 (P1-1)

신호: 전체 평균 score 1개가 아니라 **포트별 평균 score 의 분포**를 학습 시 저장한 기준 분포와 비교한다.
  - 소수 포트의 급등 = 장애 의심(localized) → 재학습 대상 아님
  - 다수 포트의 이동 = 정상 패턴 변화(widespread) → 재학습 후보

판정(`evaluate_track`)은 DB 없이 테스트할 수 있는 순수 함수이고, `DriftMonitor.check_drift()` 는 조회만 한다.
"""
import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.db_connector import DBConnector
from src.models import registry as model_registry
from src.pipeline.retrain_policy import RetrainPolicy, load_retrain_policy
from src.config import PATHS

TRACKS = ("traffic", "optical")
EVALUATED = ("normal", "drifted", "localized")        # 판정이 수행된 상태 (그 외는 판정 불가)


def _parse_ts(value):
    try:
        ts = pd.to_datetime(value)
        return None if pd.isna(ts) else ts.to_pydatetime()
    except Exception:
        return None


def evaluate_track(port_stats: pd.DataFrame, baseline: dict, policy: RetrainPolicy) -> dict:
    """포트별 통계(ip_addr, cid, lid, mean_score, n)를 기준 분포와 비교해 한 트랙의 드리프트 판정을 만든다.

    baseline: {port_median, port_p90, baseline_mse, legacy}
      legacy=True 는 포트 분포 기준값이 없는 구버전 모델 — baseline_mse 로 근사하며 재학습 대상이 아니다.
    """
    legacy = bool(baseline.get("legacy"))
    base_median = baseline.get("port_median") or baseline.get("baseline_mse")
    result = {"status": "normal", "kind": "none", "mean_mse": 0.0, "baseline_mse": float(base_median or 0.0),
              "drift_ratio": 0.0, "median_ratio": 0.0, "drifted_port_fraction": 0.0, "ports": 0,
              "localized_ports": 0, "top_ports": [], "baseline_legacy": legacy}
    if not base_median or base_median <= 0:
        result.update(status="no_baseline", message="기준값이 없어 판정할 수 없습니다.")
        return result

    stats = port_stats if port_stats is not None else pd.DataFrame(columns=["mean_score", "n"])
    stats = stats[(stats["n"] >= policy.drift_min_port_rows) & (stats["mean_score"] > 0)]
    if stats.empty:
        result.update(status="insufficient_data", message="판정에 쓸 수 있는 포트 데이터가 부족합니다.")
        return result

    # 구버전 기준값은 p90 이 없으므로 median × 배율로 근사
    base_p90 = baseline.get("port_p90") or base_median * policy.drift_median_factor
    scores = stats["mean_score"].astype(float)
    median_ratio = float(scores.median() / base_median)
    frac = float((scores > base_p90).mean())
    top = stats.sort_values("mean_score", ascending=False).head(10)
    localized_ports = int((scores > base_p90 * policy.localized_factor).sum())

    if median_ratio > policy.drift_median_factor or frac > policy.drift_port_fraction:
        kind, status = "widespread", "drifted"
    elif localized_ports > 0:
        kind, status = "localized", "localized"
    else:
        kind, status = "none", "normal"

    mean_mse = float(np.average(scores, weights=stats["n"].astype(float)))
    result.update(
        status=status, kind=kind, mean_mse=mean_mse, drift_ratio=median_ratio, median_ratio=median_ratio,
        drifted_port_fraction=frac, ports=int(len(stats)), localized_ports=localized_ports,
        top_ports=[{"ip_addr": r.ip_addr, "cid": int(r.cid), "lid": int(r.lid),
                    "mean_score": float(r.mean_score), "n": int(r.n)} for r in top.itertuples()],
    )
    return result


class DriftMonitor:
    def __init__(self, drift_factor=None, policy: RetrainPolicy = None):
        self.db = DBConnector()
        self._policy = policy
        if drift_factor is not None and policy is None:        # 구 인자 호환: 중앙값 배율
            self._policy = load_retrain_policy().with_(drift_median_factor=drift_factor)

    @property
    def policy(self) -> RetrainPolicy:
        return self._policy or load_retrain_policy()

    def _get_active_meta_path(self, ft: str):
        p = PATHS.get(ft, {})
        if not p: return ""
        base_model = p.get('model', '')
        if not base_model: return ""
        
        model_dir = os.path.dirname(base_model)
        registry_path = os.path.join(model_dir, f"{ft}_registry.json")
        actual_meta = base_model.replace('.pth', '.json')
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    registry = json.load(f)
                active_ver = registry.get("active_version")
                if active_ver:
                    for v in registry.get("versions", []):
                        if v["version"] == active_ver and "config_path" in v:
                            actual_meta = os.path.join(model_dir, v["config_path"])
                            break
            except: pass
        return actual_meta

    def _get_baseline_mse(self, feature_type):
        """저장된 메타데이터에서 기준 MSE(val_loss 또는 threshold/10)를 가져옴"""
        return self._get_single_baseline(feature_type)

    def _get_single_baseline(self, ft):
        meta_path = self._get_active_meta_path(ft)
        if not meta_path or not os.path.exists(meta_path):
            return None
            
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        # 추론 score와 같은 지표(마지막 시점 MSE 평균)로 저장된 기준값 우선 사용
        baseline = meta.get("baseline_mse")
        if baseline is not None and baseline > 0:
            return baseline

        # (구버전 모델 호환) baseline_mse가 없으면 val_loss 사용 — 지표가 달라 정확도가 낮음, 재학습 시 해소됨
        val_loss = meta.get("final_val_loss")
        if val_loss is not None and val_loss > 0:
            return val_loss
        
        # val_loss가 없을 경우(단순 테스트 등) 임계치의 특정 비율을 baseline으로 간주 (Heuristic)
        return max(meta.get("threshold", 0.1) / 5.0, 1e-6)

    def _get_baseline(self, ft):
        """활성 모델의 드리프트 기준값. 포트 분포(baseline_port_median/p90)가 있으면 정식, 없으면 legacy(감지만)."""
        meta_path = self._get_active_meta_path(ft)
        if not meta_path or not os.path.exists(meta_path):
            return None
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        median, p90 = meta.get("baseline_port_median"), meta.get("baseline_port_p90")
        if median and median > 0 and p90 and p90 > 0:
            return {"port_median": float(median), "port_p90": float(p90), "baseline_mse": meta.get("baseline_mse"),
                    "legacy": False}
        legacy = self._get_single_baseline(ft)
        return {"baseline_mse": legacy, "legacy": True} if legacy else None

    def _active_activated_at(self, ft):
        """활성 버전이 활성화된 시각 (모델 교체 직후 이전 모델의 score 가 섞이지 않게 하기 위함)"""
        p = PATHS.get(ft, {}).get('model')
        if not p:
            return None
        reg = model_registry.load(os.path.dirname(p), ft)
        active = reg.get("active_version")
        for h in reversed(reg.get("history", [])):
            if h.get("version") == active:
                return _parse_ts(h.get("activated_at"))
        return None

    def check_drift(self, now=None):
        """트랙별로 최근 구간의 포트별 평균 score 분포를 기준 분포와 비교"""
        now = now or datetime.now()
        policy = self.policy
        result = {
            "drift_detected": False,
            "drifted_tracks": [],
            "timestamp": now.strftime('%Y-%m-%d %H:%M:%S'),
        }
        try:
            for ft in TRACKS:
                baseline = self._get_baseline(ft)
                if baseline is None:
                    result[ft] = {"status": "no_baseline", "kind": "none", "mean_mse": 0.0, "baseline_mse": 0.0,
                                  "drift_ratio": 0.0, "message": "활성 모델이 없어 판정할 수 없습니다."}
                    continue
                activated = self._active_activated_at(ft)
                since = now - timedelta(hours=policy.drift_window_hours)
                if activated:
                    age_h = (now - activated).total_seconds() / 3600
                    if age_h < policy.min_hours_since_activation:
                        result[ft] = {"status": "warming_up", "kind": "none", "mean_mse": 0.0,
                                      "baseline_mse": float(baseline.get("port_median") or baseline.get("baseline_mse") or 0.0),
                                      "drift_ratio": 0.0, "baseline_legacy": bool(baseline.get("legacy")),
                                      "message": f"활성화 후 {age_h:.0f}h — {policy.min_hours_since_activation}h 이후부터 판정"}
                        continue
                    since = max(since, activated)
                stats = self.db.fetch_port_score_stats(ft, since)
                result[ft] = evaluate_track(stats, baseline, policy)
                if result[ft]["kind"] == "widespread":
                    result["drift_detected"] = True
                    result["drifted_tracks"].append(ft)

            if all(result[ft].get("status") in ("insufficient_data", "no_baseline") for ft in TRACKS):
                result["message"] = "No recent data available for drift check."
            return result
        except Exception as e:
            print(f"[!] Drift Check Error: {e}")
            return {"status": "error", "message": str(e)}
