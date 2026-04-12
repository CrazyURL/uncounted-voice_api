import pytest

from app.core.job_store import job_store


@pytest.fixture
def client():
    """FastAPI test client (no model loading).

    `app.main` is imported lazily so unit tests that don't use this fixture
    can run in environments without the full WhisperX/torch stack installed.
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
