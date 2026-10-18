import pytest

from validation.realdata import commands as cmds


@pytest.fixture(autouse=True)
def _isolated_service_json(tmp_path, monkeypatch):
    """추적 파일 service_main_inclusion.json 의 실측값에 테스트가 의존하거나 덮어쓰지 않도록 기본 경로를 tmp 의 (없는) 파일로 돌린다."""
    monkeypatch.setattr(cmds, "DEFAULT_SERVICE_JSON", str(tmp_path / "default_service_main_inclusion.json"))
