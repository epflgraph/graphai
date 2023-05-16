import pytest
from fastapi.testclient import TestClient

from graphai.api.main import app


@pytest.fixture(scope='module')
def fixture_app():
    client = TestClient(app)
    yield client
