import pytest
import pandas as pd

from graphai.core.text.keywords import get_keywords


@pytest.mark.usefixtures('sultans')
def test_get_keywords(sultans):
    keywords = get_keywords(sultans)

    # Check if output is a pd.DataFrame
    assert isinstance(keywords, pd.DataFrame)

    # Check that number of columns is correct
    assert len(keywords.columns) == 1

    # Check that column name is correct
    assert 'Keywords' in keywords.columns

    # Check that there is at least one keyword
    assert len(keywords) > 0

