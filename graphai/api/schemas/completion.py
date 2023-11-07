from pydantic import BaseModel, Field
from typing import Union, Literal, Dict, List

from graphai.api.schemas.common import TaskStatusResponse


class SummaryFingerprintRequest(BaseModel):
    text: Union[str, Dict[str, str]] = Field(
        title="Text",
        description="Text to summarize. Can be one string or a string to string dictionary."
    )

    completion_type: Literal['title', 'summary'] = Field(
        title="Summary type",
        description="Whether the summarization to be performed is title or summary generation",
        default='summary'
    )

    text_type: Literal['person', 'unit', 'concept', 'course', 'lecture', 'MOOC', 'text'] = Field(
        title="Text type",
        description="What the text being summarized describes/comes from. Defaults to 'text', which results in "
                    "generic summarization behavior.",
        default="text"
    )

    len_class: Literal['vshort', 'short', 'normal'] = Field(
        title="Length class",
        description="Whether the summary is to be one sentence (vshort), two sentences (short), "
                    "or without a sentence count limit (normal). Default is 'normal'.",
        default="normal"
    )

    tone: Literal['info', 'promo'] = Field(
        title="Tone of the summary",
        description="What tone to use in the summarization. Defaults to 'info', which is an informative tone. "
                    "'promo' results in a marketing tone.",
        default="info"
    )

    force: bool = Field(
        title="Force recomputation",
        default=False
    )


class SummaryFingerprintTaskResponse(BaseModel):
    result: Union[str, None] = Field(
        title="Fingerprint",
        description="Fingerprint of the provided text."
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    closest_token: Union[str, None] = Field(
        title="Closest token",
        description="The token of the most similar existing text that the fingerprint lookup was able to find. Equal "
                    "to original token if the most similar existing text did not satisfy the minimum similarity "
                    "threshold."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class SummaryFingerprintResponse(TaskStatusResponse):
    task_result: Union[SummaryFingerprintTaskResponse, None] = Field(
        title="Text fingerprinting response",
        description="A dict containing the resulting text fingerprint and a freshness flag."
    )


class SummarizationRequest(BaseModel):
    text: Union[str, Dict[str, str]] = Field(
        title="Text",
        description="Text to summarize. Can be one string or a string to string dictionary."
    )

    text_type: Literal['person', 'unit', 'concept', 'course', 'lecture', 'publication', 'MOOC', 'text'] = Field(
        title="Text type",
        description="What the text being summarized describes/comes from. Defaults to 'text', which results in "
                    "generic summarization behavior.",
        default="text"
    )

    len_class: Literal['vshort', 'short', 'normal'] = Field(
        title="Length class",
        description="Whether the summary is to be one sentence (vshort), two sentences (short), "
                    "or without a sentence count limit (normal). Default is 'normal'.",
        default="normal"
    )

    use_keywords: bool = Field(
        title="Use keywords",
        description="Whether to use keywords for summarization or to "
                    "use the raw text, default true (keywords are used).",
        default=False
    )

    tone: Literal['info', 'promo'] = Field(
        title="Tone of the summary",
        description="What tone to use in the summarization. Defaults to 'info', which is an informative tone. "
                    "'promo' results in a marketing tone.",
        default="info"
    )

    force: bool = Field(
        title="Force recomputation",
        default=False
    )

    debug: bool = Field(
        title="Debug",
        description="Whether to return the system message sent to ChatGPT",
        default=False
    )


class CleanupRequest(BaseModel):
    text: Union[str, Dict[str, str]] = Field(
        title="Text",
        description="Text to summarize. Can be one string or a string to string dictionary."
    )

    text_type: str = Field(
        title="Text type",
        description="The source of the text to be cleaned up. Defaults to 'slide', meaning that the text is extracted "
                    "from a slide using OCR.",
        default="slide"
    )

    force: bool = Field(
        title="Force recomputation",
        default=False
    )

    debug: bool = Field(
        title="Debug",
        description="Whether to return the system message sent to ChatGPT",
        default=False
    )

    simulate: bool = Field(
        title="Simulate",
        description="Simulation flag. If true, only simulates the request and estimates the # of tokens and "
                    "total cost of the request.",
        default=False
    )


class CleanupResponseDict(BaseModel):
    subject: str = Field(
        title="Subject matter",
        description="Subject matter of the input, which is the name of a Wikipedia page"
    )

    text: str = Field(
        title="Cleaned up text"
    )

    for_wikify: str = Field(
        title="Results for /text/wikify endpoint",
        description="The subject matter and cleaned up text combined into one for better concept detection performance"
    )


class CompletionTaskResponseBase(BaseModel):
    result_type: Union[Literal['title', 'summary', 'cleanup'], None] = Field(
        title="Summary type",
        description="Whether the result is a title, a summary, or cleaned-up text",
        default='summary'
    )

    text_too_large: bool = Field(
        title="Text too large",
        description="This boolean flag is true if the text provided for summarization is too long (over 16K tokens)."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether or not the summarization was successful"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether or not the result is fresh"
    )

    tokens: Union[Dict[str, int], None] = Field(
        title="Number of input/output tokens",
        description="A dictionary containing the total number of input and output tokens of the full request, "
                    "plus the total number of ChatGPT API requests that were made."
    )

    approx_cost: Union[float, None] = Field(
        title="Cost approximation",
        description="Approximate cost of all the requests made, taking into account the different costs of "
                    "input and output tokens."
    )

    debug_message: Union[str, None] = Field(
        title="Message",
        description="System message sent to ChatGPT"
    )


class SummaryTaskResponse(CompletionTaskResponseBase):
    result: Union[str, None] = Field(
        title="Summary results",
        description="Summarized text"
    )


class CleanupTaskResponse(CompletionTaskResponseBase):
    result: Union[CleanupResponseDict, None] = Field(
        title="Cleanup results",
        description="Cleaned-up text"
    )


class SummaryResponse(TaskStatusResponse):
    task_result: Union[SummaryTaskResponse, None] = Field(
        title="Summarization response",
        description="A dict containing the resulting summarized text and a success flag."
    )


class CleanupResponse(TaskStatusResponse):
    task_result: Union[CleanupTaskResponse, None] = Field(
        title="Summarization response",
        description="A dict containing the resulting summarized text and a success flag."
    )


class SlideConceptsMap(BaseModel):
    number: int = Field(
        title="Slide Number"
    )

    concepts: List[str] = Field(
        title="Concepts",
        description="List of concepts for this slide"
    )


class SlideSubsetRequest(BaseModel):
    slides: List[SlideConceptsMap] = Field(
        title="Slides list",
        description="A list of dictionaries with two fields: an integer 'number' that indicates the slide number, "
                    "and 'concepts', the list containing the slide's concepts."
    )

    coverage: float = Field(
        title="Coverage",
        description="What proportion of the concepts to cover.",
        default=1.0
    )

    min_freq: int = Field(
        title="Minimum frequency",
        description="Minimum number of occurrences a concept must have in all the slides combined in order not to "
                    "be removed as noise.",
        default=2
    )


class SlideSubsetResponse(BaseModel):
    subset: List[int] = Field(
        title="Optimal subset",
        description="The Slide Numbers of the slides that were chosen as part of the "
                    "optimal set cover for the concepts."
    )
