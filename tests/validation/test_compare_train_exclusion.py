"""validation/cli/compare_train_exclusion.py 의 제외 방식 비교 로직 테스트 (P1-1 D, 6.2)"""
import pytest

from src.pipeline.retrain_policy import RetrainPolicy
from validation.cli import compare_train_exclusion as cte
from validation.simulator.scenario_generator import ScenarioConfig, generate


@pytest.fixture(scope="module")
def data():
    return generate(ScenarioConfig(seed=101, nodes=2, days=10))


def test_none_keeps_all_rows_and_others_remove_some(data):
    pol = RetrainPolicy()
    kept = {v: cte.exclude(v, data, pol)[1] for v in ("none", "truth", "auto")}
    for ft in cte.TRACKS:
        assert kept["none"][ft]["kept"] == kept["none"][ft]["rows"] == len(data[ft])
        assert kept["truth"][ft]["kept"] < kept["none"][ft]["kept"]
        assert kept["auto"][ft]["kept"] <= kept["none"][ft]["kept"]


def test_unknown_variant_rejected(data):
    with pytest.raises(ValueError):
        cte.exclude("bogus", data, RetrainPolicy())
