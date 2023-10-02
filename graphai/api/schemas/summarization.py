from pydantic import BaseModel, Field
from typing import Union, Literal, Dict

from graphai.api.schemas.common import TaskStatusResponse


class SummaryFingerprintRequest(BaseModel):
    text: Union[str, Dict[str, str]] = Field(
        title="Text",
        description="Text to summarize. Can be one string or a string to string dictionary."
    )

    summary_type: Literal['title', 'summary'] = Field(
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


class SummarizationTaskResponse(BaseModel):
    summary: Union[str, None] = Field(
        title="Summarization results",
        description="Summarized text"
    )

    summary_type: Union[Literal['title', 'summary'], None] = Field(
        title="Summary type",
        description="Whether the result is a title or a summary",
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


class SummarizationDebugTaskResponse(SummarizationTaskResponse):
    debug_message: Union[str, None] = Field(
        title="Message",
        description="System message sent to ChatGPT"
    )


class SummarizationResponse(TaskStatusResponse):
    task_result: Union[SummarizationTaskResponse, None] = Field(
        title="Summarization response",
        description="A dict containing the resulting summarized text and a success flag."
    )


class SummarizationDebugResponse(TaskStatusResponse):
    task_result: Union[SummarizationDebugTaskResponse, None] = Field(
        title="Summarization response",
        description="A dict containing the resulting summarized text and a success flag."
    )
