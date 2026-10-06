"""validation/cli/build_canary.py 테스트 (P1-1 D, 설계서 2.10)"""
import pandas as pd
import pytest

from src.models import promotion_gate
from validation.cli import build_canary

KEY = ["ip_addr", "cid", "lid"]


@pytest.fixture(scope="module")
def canary():
    return build_canary.build(seed=900, nodes=1, days=7)


def test_columns_match_gate_format(canary):
    # promotion_gate._canary_auprc 는 KEY + occur_date + label 과 원본 성능 컬럼을 요구한다
    assert set(canary["traffic"].columns) == {"occur_date", *KEY, "tx_packet", "rx_packet", "error_packet", "label"}
    assert set(canary["optical"].columns) == {"occur_date", *KEY, "tx_avg_power", "rx_avg_power", "label"}


def test_labels_are_binary_with_both_classes(canary):
    for df in canary.values():
        assert set(df["label"].unique()) == {0, 1}


def test_deterministic_for_same_seed(canary):
    again = build_canary.build(seed=900, nodes=1, days=7)
    for ft in canary:
        pd.testing.assert_frame_equal(canary[ft], again[ft])


def test_labels_are_track_specific():
    # 트랙에 속하지 않는 시나리오 구간은 양성이 아니다 (광 열화 구간은 traffic 라벨 0)
    from validation.simulator.scenario_generator import ScenarioConfig, generate
    data = generate(ScenarioConfig(seed=900, nodes=1, days=7))
    labels = data["labels"]
    optical_only = labels[(labels["state"] > 0) & (labels["scenario"] == "optical_degradation")]
    assert len(optical_only) > 0
    traffic = build_canary.build(seed=900, nodes=1, days=7)["traffic"]
    m = traffic.merge(optical_only[KEY + ["occur_date"]], on=KEY + ["occur_date"])
    assert len(m) > 0 and (m["label"] == 0).all()


def test_size_cap_rejected():
    with pytest.raises(ValueError):
        build_canary.build(nodes=5)           # 50포트
    with pytest.raises(ValueError):
        build_canary.build(days=8)


def test_default_size_within_design_limit():
    # 설계서: 40포트 x 7일 (트랙당 약 2.7만 행)
    assert build_canary.MAX_PORTS == 40 and build_canary.MAX_DAYS == 7


def test_write_and_gate_path(tmp_path, canary):
    s = build_canary.write(str(tmp_path), seed=900, nodes=1, days=7)
    for ft in ("traffic", "optical"):
        assert s[ft]["path"] == promotion_gate.canary_path(str(tmp_path), ft)
        df = pd.read_csv(s[ft]["path"])
        assert len(df) == s[ft]["rows"] > 0
        assert s[ft]["ports"] == 10


def test_build_fails_when_a_track_has_no_positive_or_negative(monkeypatch):
    """양성(또는 음성)이 한쪽이라도 없으면 AUPRC 가 무의미하므로 생성 실패"""
    from validation.simulator.scenario_generator import ScenarioConfig, generate
    real = generate(ScenarioConfig(seed=900, nodes=1, days=7))

    no_fault = {**real, "labels": real["labels"].assign(state=0)}
    monkeypatch.setattr(build_canary, "generate", lambda cfg: no_fault)
    with pytest.raises(ValueError, match="양성 또는 음성이 없습니다"):
        build_canary.build(seed=900, nodes=1, days=7)

    all_fault = {**real, "labels": real["labels"].assign(state=1, scenario="crc_error")}
    monkeypatch.setattr(build_canary, "generate", lambda cfg: all_fault)
    with pytest.raises(ValueError):                       # traffic 은 전부 양성, optical 은 전부 음성
        build_canary.build(seed=900, nodes=1, days=7)


def test_failure_does_not_write_any_file(tmp_path, monkeypatch):
    from validation.simulator.scenario_generator import ScenarioConfig, generate
    real = generate(ScenarioConfig(seed=900, nodes=1, days=7))
    monkeypatch.setattr(build_canary, "generate", lambda cfg: {**real, "labels": real["labels"].assign(state=0)})
    with pytest.raises(ValueError):
        build_canary.write(str(tmp_path), seed=900, nodes=1, days=7)
    assert list(tmp_path.iterdir()) == []
