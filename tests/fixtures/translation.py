import pytest


@pytest.fixture
def en_to_fr_text():
    return "Hi guys, how's it going?"


@pytest.fixture
def fr_to_en_text():
    return "Mesdames et messieurs, bienvenue."
