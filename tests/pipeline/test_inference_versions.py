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
