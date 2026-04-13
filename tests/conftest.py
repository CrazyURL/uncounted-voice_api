import os

import pytest

# 테스트 환경임을 표시 — app/main.py의 lifespan이 이 값을 보고 모델 로딩을 스킵한다.
# 이미 실행 중인 voice-api@dev 서비스와 GPU 점유 충돌을 피하기 위함.
os.environ.setdefault("TESTING", "1")

from app.core.job_store import job_store  # noqa: E402


@pytest.fixture
def client():
    """FastAPI test client.

    TESTING=1 환경변수로 lifespan이 모델 로딩을 스킵하므로 GPU/WhisperX가
    없는 환경(또는 이미 실행 중인 서버가 있는 환경)에서도 동작한다.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_job_store():
    """Clear job store between tests."""
    job_store._tasks.clear()
    yield
    job_store._tasks.clear()
