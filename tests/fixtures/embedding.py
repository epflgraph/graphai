import pytest


@pytest.fixture
def example_word():
    return "sparrow"


@pytest.fixture
def very_long_text():
    return """
    That means, the position embedding layer of the transformers has 512 weights, but the sentence transformer will 
    only use and was trained with the first 256 of them. Therefore, you should be careful with increasing the
    value above 256. It will work from a technical perspective, but the position embedding weights (>256) are not
    properly trained and can therefore mess up your results. That means, the position embedding layer of the 
    transformers has 512 weights, but the sentence transformer will 
    only use and was trained with the first 256 of them. Therefore, you should be careful with increasing the
    value above 256. It will work from a technical perspective, but the position embedding weights (>256) are not
    properly trained and can therefore mess up your results. That means, the position embedding layer of the 
    transformers has 512 weights, but the sentence transformer will 
    only use and was trained with the first 256 of them. Therefore, you should be careful with increasing the
    value above 256. It will work from a technical perspective, but the position embedding weights (>256) are not
    properly trained and can therefore mess up your results.
    """
