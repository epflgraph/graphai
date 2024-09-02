from pydantic import BaseModel, Field
from typing import Union, Literal, List

from graphai.api.common.schemas import TaskStatusResponse


class TextFingerprintTaskResponse(BaseModel):
    result: Union[str, None] = Field(
        None,
        title="Fingerprint",
        description="Fingerprint of the provided text."
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class TextFingerprintResponse(TaskStatusResponse):
    task_result: Union[TextFingerprintTaskResponse, None] = Field(
        title="Text fingerprinting response",
        description="A dict containing the resulting text fingerprint and a freshness flag."
    )


class TranslationRequest(BaseModel):
    text: Union[str, List[str]] = Field(
        title="Text",
        description="Text to translate, can be one string or a list of strings."
    )

    source: Literal['en', 'fr', 'de', 'it'] = Field(
        title="Source language",
        description="Language of the provided text",
        default='fr'
    )

    target: Literal['en', 'fr', 'de', 'it'] = Field(
        title="Target language",
        description="Language to translate the text into",
        default='en'
    )

    force: bool = Field(
        title="Force recomputation",
        default=False
    )

    skip_segmentation: bool = Field(
        title="Skip sentence segmentation",
        description="If set to true, passes the entire text directly to the translation model without "
                    "sentence segmentation (i.e. breaking the text up on full stops). Increases the risk of "
                    "'text too large' errors.",
        default=False
    )


class TranslationTaskResponse(BaseModel):
    result: Union[str, List[str], None] = Field(
        None, title="Translation results",
        description="Translated text"
    )

    text_too_large: bool = Field(
        title="Text too large",
        description="This boolean flag is true if the text provided for translation has an overly long "
                    "unpunctuated segment (over 512 tokens)."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether or not the translation was successful"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether or not the result is fresh"
    )

    device: Union[str, None] = Field(
        None, title="Translation device",
        description="The device used ('cuda' or 'cpu') for the translation. `None` in case of cache hit or failure."
    )


class TranslationResponse(TaskStatusResponse):
    task_result: Union[TranslationTaskResponse, None] = Field(
        title="Translation response",
        description="A dict containing the resulting translated text and a success flag."
    )


class TextDetectLanguageRequest(BaseModel):
    text: Union[str, List[str]] = Field(
        title="Text",
        description="Text to detect the language of"
    )


class TextDetectLanguageTaskResponse(BaseModel):
    language: Union[str, None] = Field(
        None, title="Language detection results",
        description="Detected language"
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether or not the translation was successful"
    )


class TextDetectLanguageResponse(TaskStatusResponse):
    task_result: Union[TextDetectLanguageTaskResponse, None] = Field(
        title="Language detection response",
        description="A dict containing the detected language of the text and a success flag."
    )
