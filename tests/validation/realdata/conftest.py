import pytest

from validation.realdata import commands as cmds


@pytest.fixture(autouse=True)
def _isolated_service_json(tmp_path, monkeypatch):
    """추적 파일 service_main_inclusion.json 의 실측값에 테스트가 의존하거나 덮어쓰지 않도록 기본 경로를 tmp 의 (없는) 파일로 돌린다."""
    monkeypatch.setattr(cmds, "DEFAULT_SERVICE_JSON", str(tmp_path / "default_service_main_inclusion.json"))


@pytest.fixture(autouse=True)
def _allow_model_infos_in_tests(monkeypatch):
    """cmd_lock(model_infos=...) 는 테스트 전용 경로다 (운영 코드는 거부)."""
    monkeypatch.setattr(cmds, "_ALLOW_MODEL_INFOS_FOR_TESTS", True)
