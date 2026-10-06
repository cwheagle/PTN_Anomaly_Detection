"""
승격 게이트 검증 (P1-1): 개별 검사 경계값, 종합 규칙, 임계치 추세, 소형 모델 2개로 end-to-end
"""
from datetime import datetime

import pandas as pd
import pytest

from src.models import promotion_gate as pg
from src.models import registry
from src.pipeline.alerting import AlertPolicy
from src.pipeline.retrain_policy import RetrainPolicy

L = RetrainPolicy()


def _meta(version="v2", threshold=0.3, val=0.02):
    return {"version": version, "threshold": threshold, "final_val_loss": val}


def _stats(rate, ports=100):
    return {"ports": ports, "incidents": int(rate * ports * 3), "incidents_per_port_day": rate}


# T-G1: 경계값
def test_g1_artifact_and_finite_values():
    assert pg.check_g1(_meta(), []).status == pg.PASS
    assert pg.check_g1(_meta(), ["model_path 파일이 없습니다"]).status == pg.FAIL
    for bad in (float("nan"), float("inf"), 0, -1, None):
        assert pg.check_g1(_meta(threshold=bad), []).status == pg.FAIL, bad
    assert pg.check_g1(_meta(val=float("nan")), []).status == pg.FAIL
    assert pg.check_g1(_meta(val=None), []).status == pg.PASS                   # 검증 손실은 없을 수 있음


@pytest.mark.parametrize("cand_th,expected", [(0.3, pg.PASS), (0.89, pg.PASS), (0.91, pg.FAIL), (0.101, pg.PASS), (0.099, pg.FAIL)])
def test_g2_threshold_ratio_limits_are_three_x_each_way(cand_th, expected):
    c = pg.check_g2(_meta("v2", cand_th), _meta("v1", 0.3))
    assert c.status == expected


def test_g2_val_loss_blowup_and_no_active():
    assert pg.check_g2(_meta("v2", 0.3, 0.07), _meta("v1", 0.3, 0.02)).status == pg.FAIL
    assert pg.check_g2(_meta("v2"), None).status == pg.SKIP


@pytest.mark.parametrize("cand,active,expected", [
    (0.010, 0.010, pg.PASS),          # 같음
    (0.015, 0.010, pg.PASS),          # 활성 × 1.5 = 0.015 (경계 포함)
    (0.016, 0.010, pg.FAIL),          # 활성 × 1.5 초과, floor(0.01) 보다도 큼
    (0.010, 0.000, pg.PASS),          # 활성이 0 이어도 floor 0.01 까지 허용
    (0.011, 0.000, pg.FAIL),
    (0.049, 0.200, pg.PASS),          # 활성이 많이 울려도 절대 상한(0.05) 이내면 통과
    (0.051, 0.200, pg.FAIL),          # 절대 상한 초과는 활성 대비와 무관하게 FAIL
])
def test_g3_alarm_rate_rule(cand, active, expected):
    assert pg.check_g3(_stats(cand), _stats(active), L).status == expected


def test_g3_skip_without_gate_data_and_absolute_limit_only_without_active():
    assert pg.check_g3(None, _stats(0.01), L).status == pg.SKIP
    assert pg.check_g3({"ports": 0, "incidents_per_port_day": 0.0}, None, L).status == pg.SKIP
    assert pg.check_g3(_stats(0.05), None, L).status == pg.PASS
    assert pg.check_g3(_stats(0.06), None, L).status == pg.FAIL


@pytest.mark.parametrize("cand,expected", [(0.80, pg.PASS), (0.75, pg.PASS), (0.749, pg.FAIL)])
def test_g4_canary_auprc_drop_limit(cand, expected):
    assert pg.check_g4({"cand_auprc": cand, "active_auprc": 0.80}, L).status == expected


def test_g4_skip_cases():
    assert pg.check_g4(None, L).status == pg.SKIP
    assert pg.check_g4({"cand_auprc": 0.5, "active_auprc": None}, L).status == pg.SKIP


# T-G3: 임계치 추세
@pytest.mark.parametrize("hist,cand,expected", [
    ([0.1, 0.15, 0.2], 0.25, pg.WARN),        # 단조 증가, 누적 2.5배
    ([0.1, 0.15, 0.2], 0.19, pg.PASS),        # 마지막이 내려감
    ([0.1, 0.11, 0.12], 0.13, pg.PASS),       # 단조 증가지만 누적 1.3배
    ([0.1, 0.1, 0.2], 0.3, pg.PASS),          # 같은 값이 있으면 단조 증가 아님
    ([0.1, 0.2], 0.4, pg.SKIP),               # 이력 3개 미만
    ([0.05, 0.1, 0.15, 0.2], 0.3, pg.WARN),   # 최근 3개만 본다 (0.1,0.15,0.2,0.3 -> 3배)
])
def test_g5_threshold_trend(hist, cand, expected):
    assert pg.check_g5(_meta(threshold=cand), hist, L).status == expected


# T-G2: 종합 규칙
def _c(status):
    return pg.GateCheck("X", status)


@pytest.mark.parametrize("statuses,expected", [
    ([pg.PASS, pg.PASS], pg.PASS), ([pg.PASS, pg.SKIP], pg.PASS), ([pg.SKIP, pg.SKIP], pg.PASS),
    ([pg.PASS, pg.WARN], pg.WARN), ([pg.WARN, pg.FAIL, pg.PASS], pg.FAIL), ([pg.FAIL, pg.SKIP], pg.FAIL),
])
def test_overall_status_precedence(statuses, expected):
    assert pg.overall_status([_c(s) for s in statuses]) == expected


def test_evaluate_combines_all_checks_in_order_and_serializes():
    res = pg.evaluate(_meta("v2", 0.3), _meta("v1", 0.3), _stats(0.01), _stats(0.01), None, [0.1, 0.1, 0.1], L,
                      data={"against": "v1"})
    assert [c.id for c in res.checks] == ["G1", "G2", "G3", "G4", "G5"]
    assert res.status == pg.PASS
    d = res.to_dict()
    assert d["status"] == "PASS" and d["checks"][3]["status"] == "SKIP" and d["data"] == {"against": "v1"}
    import json
    json.dumps(d)                                                              # 레지스트리에 저장 가능


def test_evaluate_one_failing_check_fails_overall():
    res = pg.evaluate(_meta("v2", 0.3), _meta("v1", 0.3), _stats(0.2), _stats(0.01), None, [], L)
    assert res.status == pg.FAIL and [c.id for c in res.checks if c.status == pg.FAIL] == ["G3"]


# T-G4: 소형 모델 2개로 끝까지 (DB 대신 fetch_raw 주입)
WINDOW_END = datetime(2026, 3, 5, 0, 0)       # tiny_scenario: 2026-03-01 ~ 03-05


def _window():
    from src.data.train_window import plan_window
    return plan_window(WINDOW_END, L.with_(gate_days=3))


def _fetch(scenario, calls=None):
    def fetch(ft, start, end):
        if calls is not None:
            calls.append((ft, start, end))
        df = scenario[ft]
        return df[(df.occur_date >= start) & (df.occur_date <= end)]
    return fetch


def _gate_policy():
    return L.with_(val_port_fraction=1.0, gate_days=3, gate_min_port_days=1)   # 모든 포트를 홀드아웃으로 사용, 최소 포트·일 가드는 끔


SENSITIVE = AlertPolicy(threshold_scale=0.02, dampening_steps={1: 1, 2: 1, 3: 1})    # 1에폭 소형 모델도 알람이 나도록


def test_run_gate_end_to_end_with_two_tiny_models(tiny_env, tiny_scenario):
    calls = []
    res = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), _gate_policy(), _fetch(tiny_scenario, calls), SENSITIVE)
    assert res.status in (pg.PASS, pg.FAIL, pg.WARN), res.checks
    by = {c.id: c for c in res.checks}
    assert by["G1"].status == pg.PASS
    assert by["G2"].status in (pg.PASS, pg.FAIL) and by["G2"].value > 0          # 활성 v1 과 비교됨
    assert by["G3"].status in (pg.PASS, pg.FAIL) and by["G3"].value is not None  # 게이트 데이터로 계산됨
    assert by["G4"].status == pg.SKIP                                             # 카나리 파일 없음
    assert by["G5"].status == pg.SKIP                                             # 활성 이력 부족
    assert res.data["against"] == "v1" and res.data["ports"] == 6
    cs, as_ = res.data["candidate_stats"], res.data["active_stats"]
    assert cs["ports"] == as_["ports"] == 6 and cs["incidents"] > 0 and as_["incidents"] > 0
    assert 0 <= cs["overlap_with_active"] <= 1                                    # 후보 알람 중 활성 모델도 울린 비율(정보용)
    ft, start, end = calls[0]                                                     # MA 이력용으로 게이트 구간보다 1일 앞에서 조회
    assert ft == "traffic" and start == _window().gate_start - pd.Timedelta(days=1) and end == _window().gate_end
    assert registry.load(str(tiny_env), "traffic")["active_version"] == "v1"      # run_gate 는 레지스트리를 바꾸지 않음


def test_run_gate_uses_the_given_operating_alert_policy(tiny_env, tiny_scenario):
    """게이트는 현재 운영 정책으로 두 모델을 비교한다 — 정책이 바뀌면 같은 모델의 알람 비율도 바뀜"""
    quiet = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), _gate_policy(), _fetch(tiny_scenario), AlertPolicy())
    loud = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), _gate_policy(), _fetch(tiny_scenario), SENSITIVE)
    assert loud.data["candidate_stats"]["incidents_per_port_day"] > (quiet.data["candidate_stats"] or {"incidents_per_port_day": 0})["incidents_per_port_day"]


def test_run_gate_works_for_optical_track_too(tiny_env, tiny_scenario):
    res = pg.run_gate(str(tiny_env), "optical", "v2", _window(), _gate_policy(), _fetch(tiny_scenario), AlertPolicy())
    assert res.status != pg.ERROR and {c.id for c in res.checks} == {"G1", "G2", "G3", "G4", "G5"}


def test_run_gate_only_scores_holdout_ports(tiny_env, tiny_scenario):
    """val_port_fraction 으로 검증 포트가 아닌 포트는 게이트 데이터에서 제외"""
    pol = L.with_(val_port_fraction=0.0, gate_days=3)
    res = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), pol, _fetch(tiny_scenario), AlertPolicy())
    assert by_id(res)["G3"].status == pg.SKIP and res.data["ports"] == 0


def by_id(res):
    return {c.id: c for c in res.checks}


def test_run_gate_caps_ports_at_gate_max_ports(tiny_env, tiny_scenario):
    res = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), _gate_policy().with_(gate_max_ports=2),
                      _fetch(tiny_scenario), AlertPolicy())
    assert res.data["ports"] == 2


def test_run_gate_missing_artifact_fails_g1_without_inference(tiny_env, tiny_scenario):
    (tiny_env / "traffic_ae_v2.pth").unlink()
    calls = []
    res = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), _gate_policy(), _fetch(tiny_scenario, calls), AlertPolicy())
    assert res.status == pg.FAIL and by_id(res)["G1"].status == pg.FAIL and calls == []


def test_run_gate_data_failure_is_reported_as_error_not_raised(tiny_env):
    def boom(ft, start, end):
        raise RuntimeError("db down")
    res = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), _gate_policy(), boom, AlertPolicy())
    assert res.status == pg.ERROR and "db down" in res.checks[0].message


def test_run_gate_unknown_version_raises(tiny_env, tiny_scenario):
    with pytest.raises(registry.VersionNotFound):
        pg.run_gate(str(tiny_env), "traffic", "v9", _window(), _gate_policy(), _fetch(tiny_scenario), AlertPolicy())


def test_run_gate_with_canary_file_compares_auprc(tiny_env, tiny_scenario):
    lab = tiny_scenario["labels"][["ip_addr", "cid", "lid", "occur_date", "state"]]
    canary = tiny_scenario["traffic"].merge(lab, on=["ip_addr", "cid", "lid", "occur_date"])
    canary["label"] = (canary.pop("state") > 0).astype(int)
    assert canary["label"].nunique() == 2
    canary.to_csv(tiny_env / "traffic_canary.csv", index=False)
    res = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), _gate_policy(), _fetch(tiny_scenario), AlertPolicy())
    g4 = by_id(res)["G4"]
    assert g4.status in (pg.PASS, pg.FAIL)
    assert 0 <= res.data["canary"]["cand_auprc"] <= 1 and 0 <= res.data["canary"]["active_auprc"] <= 1


def test_gate_result_roundtrips_through_registry_and_drives_promotion(tiny_env, tiny_scenario):
    res = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), _gate_policy(), _fetch(tiny_scenario), AlertPolicy())
    registry.set_gate(str(tiny_env), "traffic", "v2", res.to_dict())
    item = {v["version"]: v for v in registry.list_versions(str(tiny_env), "traffic")["versions"]}["v2"]
    assert item["gate_status"] == res.status
    if res.status in (pg.PASS, pg.WARN):
        assert registry.promote(str(tiny_env), "traffic", "v2")["active"] == "v2"
    else:
        with pytest.raises(registry.PromotionWarning):
            registry.promote(str(tiny_env), "traffic", "v2")


def test_g3_fail_message_uses_shared_ratio_format():
    c = pg.check_g3(_stats(0.083), _stats(0.0), L)
    assert c.status == pg.FAIL and "×8.3" in c.message and "0.083" in c.message and "기준 0.010" in c.message


def test_g2_fail_message_uses_inverse_format_for_decrease():
    c = pg.check_g2(_meta("v2", 0.4399), _meta("v1", 16.48))
    assert c.status == pg.FAIL and "1/37.5" in c.message and "기준 1/3 ~ ×3" in c.message


# T-G5: G3 최소 포트·일 가드 (발견 B)
def _pd_stats(rate, port_days, incidents=1, ports=50):
    return {"ports": ports, "incidents": incidents, "port_days": port_days, "incidents_per_port_day": rate}


def test_g3_skips_below_min_port_days_and_records_detail():
    c = pg.check_g3(_pd_stats(0.083, 149, incidents=12), _pd_stats(0.0, 149, incidents=0), L)
    assert c.status == pg.SKIP and "포트·일 149<150" in c.message and "판정 불가" in c.message
    assert c.value == 149 and c.limit == 150
    assert c.detail == {"cand": {"incidents": 12, "port_days": 149}, "active": {"incidents": 0, "port_days": 149}}


@pytest.mark.parametrize("rate,expected", [(0.0067, pg.PASS), (0.011, pg.FAIL)])
def test_g3_judges_normally_at_exactly_min_port_days(rate, expected):
    c = pg.check_g3(_pd_stats(rate, 150, incidents=1), _pd_stats(0.0, 150, incidents=0), L)
    assert c.status == expected and c.detail["cand"]["port_days"] == 150
    assert "인시던트 1건/150포트·일" in c.message


def test_g3_port_days_default_to_ports_times_gate_days_when_missing():
    assert pg.check_g3({"ports": 49, "incidents": 0, "incidents_per_port_day": 0.0}, None, L).status == pg.SKIP   # 147
    assert pg.check_g3({"ports": 50, "incidents": 0, "incidents_per_port_day": 0.0}, None, L).status == pg.PASS    # 150


def test_g3_skip_makes_overall_warn_even_if_everything_else_passes():
    res = pg.evaluate(_meta("v2", 0.3), _meta("v1", 0.3), _pd_stats(0.0, 120, incidents=0), _pd_stats(0.0, 120, 0),
                      None, [0.1, 0.1, 0.1], L)
    by = {c.id: c.status for c in res.checks}
    assert by["G3"] == pg.SKIP and by["G4"] == pg.SKIP and res.status == pg.WARN


def test_g3_fail_still_wins_over_warn_and_other_skips_do_not_warn():
    assert pg.overall_status([pg.GateCheck("G3", pg.SKIP), pg.GateCheck("G1", pg.FAIL)]) == pg.FAIL
    assert pg.overall_status([pg.GateCheck("G4", pg.SKIP), pg.GateCheck("G3", pg.PASS)]) == pg.PASS


def test_run_gate_default_policy_skips_g3_for_small_holdout_and_records_port_days(tiny_env, tiny_scenario):
    pol = L.with_(val_port_fraction=1.0, gate_days=3)                      # 6포트 x 3일 = 18 < 150
    res = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), pol, _fetch(tiny_scenario), SENSITIVE)
    assert by_id(res)["G3"].status == pg.SKIP and res.status == pg.WARN
    assert res.data["candidate_stats"]["port_days"] == 18
    import json
    json.dumps(res.to_dict())


def test_default_policy_min_port_days_is_150():
    from src.pipeline.retrain_policy import RetrainPolicy
    assert RetrainPolicy().gate_min_port_days == 150


def test_gate_uses_each_models_own_policy_and_records_both(tiny_env, tiny_scenario):
    """T-P3-G2: 후보는 후보 정책(precision), 활성은 활성 정책(메타 없음 -> 운영 기본 정책)으로 알람을 계산하고 data 에 기록"""
    from src.pipeline.alerting import get_preset, policy_to_meta
    same_weights = registry.create_policy_version(str(tiny_env), "traffic", "v2", policy_to_meta(get_preset("precision"), "precision"))
    pol = _gate_policy()
    plain = pg.run_gate(str(tiny_env), "traffic", "v2", _window(), pol, _fetch(tiny_scenario), SENSITIVE)       # v2: 메타 없음 -> SENSITIVE
    derived = pg.run_gate(str(tiny_env), "traffic", same_weights["version"], _window(), pol, _fetch(tiny_scenario), SENSITIVE)
    assert derived.data["alert_policy"]["candidate"]["preset"] == "precision"
    assert derived.data["alert_policy"]["active"] == policy_to_meta(SENSITIVE)                               # 활성 v1 은 메타 없음
    assert plain.data["alert_policy"]["candidate"] == policy_to_meta(SENSITIVE)
    # 같은 가중치인데 정책만 달라 후보 알람이 줄어듦 (SENSITIVE: 임계치 x0.02·즉시 알람 / precision: 임계치 x3·길게 확인)
    assert derived.data["candidate_stats"]["incidents"] < plain.data["candidate_stats"]["incidents"]
    assert derived.data["active_stats"] == plain.data["active_stats"]                                         # 활성 쪽은 그대로
