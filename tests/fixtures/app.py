import pytest
from fastapi.testclient import TestClient

from graphai.api.auth.router import (
    get_current_active_user,
    get_active_user_dummy
)
from graphai.api.main.main import app


@pytest.fixture(scope='module')
def fixture_app():
    # We override the authentication dependency before launching tests to avoid having to obtain a token before every
    # integration test.
    app.dependency_overrides[get_current_active_user] = get_active_user_dummy
    client = TestClient(app)
    # This is where the tests happen
    yield client
    # After the tests are done, we disable all overrides.
    app.dependency_overrides = {}
