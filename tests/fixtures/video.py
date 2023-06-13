import pytest


@pytest.fixture
def test_video_url():
    return "https://api.cast.switch.ch/p/113/sp/11300/playManifest/entryId/0_jc1ok6j7/v/2/ev/5/flavorId/" \
           "0_ju4by4lc/format/applehttp/protocol/https/a.m3u8"


@pytest.fixture
def test_video_token():
    return "test_token_vid.mp4"
