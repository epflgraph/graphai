import pytest


@pytest.fixture
def test_url():
    return "https://www.epfl.ch/labs/chili"


@pytest.fixture
def test_url_2():
    return "https://www.epfl.ch"
