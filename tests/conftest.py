"""
공용 픽스처: 소형 학습 모델 (models/ 가 없는 환경에서도 추론·게이트 경로를 검증하기 위함)

`tiny_model_dir` 는 시뮬레이터 데이터로 traffic/optical 각각 v1(활성), v2(드리프트 후보)를 1에폭 학습해 만든
읽기 전용 원본 폴더다 (세션당 1회 학습). 테스트는 `tiny_env` 로 복사본을 받아 PATHS 를 그 폴더로 돌려 쓴다.
"""
import shutil

import pytest

TRACKS = ("traffic", "optical")


def _scenario(seed, ports=8, days=5):
    from validation.simulator.scenario_generator import ScenarioConfig, generate
    return generate(ScenarioConfig(seed=seed, nodes=1, ports_per_node=ports, days=days, mean_gap_days=1.0,
                                   warmup_days=0.5, start="2026-03-01 00:00:00"))


@pytest.fixture(scope="session")
def tiny_scenario():
    """검증용 시나리오 (학습 시드와 분리): {'traffic', 'optical', 'labels', 'episodes'}"""
    return _scenario(seed=21, ports=6, days=4)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from src.config import PATHS
    from src.models.trainer import Trainer

    root = tmp_path_factory.mktemp("tiny_models")
    mp = pytest.MonkeyPatch()
    try:
        for ft in TRACKS:
            mp.setitem(PATHS, ft, {"model": str(root / f"{ft}_ae.pth"), "scaler": str(root / f"{ft}_scaler.joblib")})
        train, val = _scenario(seed=5), _scenario(seed=6, ports=3, days=3)
        for ft in TRACKS:
            t_csv, v_csv = root / f"{ft}_train.csv", root / f"{ft}_val.csv"
            train[ft].to_csv(t_csv, index=False)
            val[ft].to_csv(v_csv, index=False)
            for trigger, activate in (("manual", True), ("drift", False)):
                tr = Trainer(ft, config_override={"epochs": 1, "batch_size": 64}, activate=activate, trigger=trigger)
                assert tr.train(train_path=str(t_csv), val_path=str(v_csv))
    finally:
        mp.undo()
    return root


@pytest.fixture
def tiny_env(tiny_model_dir, tmp_path, monkeypatch):
    """모델 폴더 복사본 + PATHS 교체. 반환: 복사본 경로 (레지스트리/게이트 기록을 마음껏 바꿔도 원본은 불변)"""
    from src.config import PATHS
    dest = tmp_path / "models"
    shutil.copytree(tiny_model_dir, dest)
    for ft in TRACKS:
        monkeypatch.setitem(PATHS, ft, {"model": str(dest / f"{ft}_ae.pth"), "scaler": str(dest / f"{ft}_scaler.joblib")})
    return dest
