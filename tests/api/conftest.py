"""
API 테스트 공용 픽스처: src.api.main 은 import 시 MySQL/Redis 객체를 만든다.
DB 서버 없이도 앱을 불러올 수 있게 커넥션 풀을 MagicMock 으로 대체한다 (실제 쿼리는 각 테스트가 대체).
"""
import sys
from unittest import mock

import pytest


@pytest.fixture(scope="session")
def api_main():
    if "src.api.main" in sys.modules:
        return sys.modules["src.api.main"]
    with mock.patch("mysql.connector.pooling.MySQLConnectionPool", mock.MagicMock()):
        try:
            import src.api.main as main
        except Exception as e:
            pytest.skip(f"API 앱을 불러올 수 없음: {str(e)[:80]}")
    return main
