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


@pytest.fixture
def example_word_list():
    return ["car", "vehicle", "truck", "van", "minivan", "bus", "minibus", "sedan", "SUV", "trailer",
            "Early electric vehicles first came into existence in the late 19th century, "
            "when the Second Industrial Revolution brought forth electrification. Using "
            "electricity was among the preferred methods for motor vehicle propulsion as "
            "it provides a level of quietness, comfort and ease of operation that could not "
            "be achieved by the gasoline engine cars of the time, but range anxiety due to "
            "the limited energy storage offered by contemporary battery technologies hindered "
            "any mass adoption of private electric vehicles throughout the 20th century. Internal "
            "combustion engines (both gasoline and diesel engines) were the dominant propulsion "
            "mechanisms for cars and trucks for about 100 years, "
            "but electricity-powered locomotion remained commonplace.",
            "motorcycle", "pickup truck", "flatbed"]
