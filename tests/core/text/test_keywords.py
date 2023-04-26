import pytest

from graphai.core.text.keywords import get_keywords


@pytest.mark.usefixtures('sultans')
def test_get_keywords(sultans):
    keywords_list = get_keywords(sultans)

    # Check if output is a list
    assert isinstance(keywords_list, list)

    # Check if list is not empty
    assert len(keywords_list) > 0

    # Check that list contains strings
    for keywords in keywords_list:
        assert isinstance(keywords, str)
