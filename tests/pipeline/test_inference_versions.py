"""
버전 지정 로드(F5): 게이트가 비활성 후보를 추론할 수 있어야 하고, 지정하지 않으면 기존 동작(활성 버전) 그대로
소형 학습 모델(tests/conftest.py)로 검증하므로 models/ 가 없어도 실행된다.
"""
import json

import pytest

from src.pipeline.inference import AnomalyDetector


def _threshold(model_dir, version, ft="traffic"):
    return json.load(open(model_dir / f"{ft}_ae_{version}.json"))["threshold"]


def test_default_load_is_active_version(tiny_env):
    det = AnomalyDetector()
    assert set(det.tracks) == {"traffic", "optical"}
    assert det.tracks["traffic"]["th"] == pytest.approx(_threshold(tiny_env, "v1"))


def test_version_argument_loads_candidate_without_touching_registry(tiny_env):
    before = (tiny_env / "traffic_registry.json").read_text()
    det = AnomalyDetector(versions={"traffic": "v2"}, tracks=("traffic",))
    assert set(det.tracks) == {"traffic"}
    assert det.tracks["traffic"]["th"] == pytest.approx(_threshold(tiny_env, "v2"))
    assert (tiny_env / "traffic_registry.json").read_text() == before            # 활성 버전 불변


def test_candidate_and_active_instances_coexist_and_score_differently(tiny_env, tiny_scenario):
    active = AnomalyDetector(tracks=("traffic",))
    cand = AnomalyDetector(versions={"traffic": "v2"}, tracks=("traffic",))
    s_a, th_a = active.track_scores(tiny_scenario["traffic"], "traffic")
    s_c, th_c = cand.track_scores(tiny_scenario["traffic"], "traffic")
    assert len(s_a) == len(s_c) > 0 and th_a != th_c
    assert not (s_a["mse"].values == s_c["mse"].values).all()


def test_unknown_version_is_an_explicit_error(tiny_env):
    with pytest.raises(KeyError):
        AnomalyDetector(versions={"traffic": "v99"}, tracks=("traffic",))


def test_reload_model_without_version_follows_registry_active(tiny_env):
    from src.models import registry
    det = AnomalyDetector()
    registry.promote(str(tiny_env), "traffic", "v2", force=True)
    det.reload_model("traffic")
    assert det.tracks["traffic"]["th"] == pytest.approx(_threshold(tiny_env, "v2"))


# ──────────────────────────────────────────────
# T-P3-I2/I3: 모델별 알람 정책 (P1-3)
# ──────────────────────────────────────────────
def _precision_version(tiny_env, ft="traffic", src="v1"):
    from src.models import registry
    from src.pipeline.alerting import get_preset, policy_to_meta
    return registry.create_policy_version(str(tiny_env), ft, src, policy_to_meta(get_preset("precision"), "precision"))["version"]


def test_policy_is_applied_per_track(tiny_env):
    """T-P3-I2: traffic 만 precision 메타 -> traffic 은 precision, optical 은 (메타 없음) 기본 정책"""
    from src.models import registry
    from src.pipeline.alerting import AlertPolicy, get_preset
    ver = _precision_version(tiny_env)
    registry.promote(str(tiny_env), "traffic", ver, force=True)
    det = AnomalyDetector()
    assert det.policy_for("traffic") == get_preset("precision")
    assert det.policy_for("optical") == AlertPolicy()
    assert set(det.policies) == {"traffic"}


def test_reload_to_version_without_policy_restores_default_and_priority(tiny_env, monkeypatch):
    """T-P3-I3: 롤백(정책 있는 버전 -> 없는 버전)이면 정책도 복귀. 우선순위: 모델 메타 > 기본(config) 정책"""
    from src.models import registry
    from src.pipeline import inference
    from src.pipeline.alerting import AlertPolicy, get_preset
    configured = AlertPolicy(sigma_k=2.5)                                  # config.ALERT_POLICY 가 있는 환경을 흉내
    monkeypatch.setattr(inference, "load_policy", lambda: configured)
    ver = _precision_version(tiny_env)
    det = AnomalyDetector(tracks=("traffic",))
    assert det.policy_for("traffic") == configured                         # 메타 없음 -> config 정책
    registry.promote(str(tiny_env), "traffic", ver, force=True)
    det.reload_model("traffic")
    assert det.policy_for("traffic") == get_preset("precision")            # 메타가 config 보다 우선
    registry.rollback(str(tiny_env), "traffic")
    det.reload_model("traffic")
    assert det.policy_for("traffic") == configured and "traffic" not in det.policies   # 롤백 -> 정책도 복귀
