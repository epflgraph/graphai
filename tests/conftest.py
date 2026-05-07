import os

import pytest

from fixtures.app import *

from fixtures.text import *

from fixtures.translation import *

from fixtures.embedding import *

from fixtures.video import *

from fixtures.scraping import *

# from fixtures.ontology import *


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services, models, or network access.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require external services, model downloads, or network access",
    )


def pytest_collection_modifyitems(config, items):
    run_integration = config.getoption("--run-integration") or os.getenv("RUN_INTEGRATION_TESTS") == "1"
    if run_integration:
        return

    skip_marker = pytest.mark.skip(
        reason="integration test skipped (set RUN_INTEGRATION_TESTS=1 or pass --run-integration)"
    )
    for item in items:
        if "celery" in item.keywords or "integration" in item.keywords:
            item.add_marker(skip_marker)

@pytest.fixture
def celery_worker_parameters():
    return {
        "concurrency": 2,
        "pool": "solo",
    }