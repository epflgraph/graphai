from pydantic import BaseModel, Field
from typing import List, Union, Any, Literal
from .common import TaskStatusResponse

class TranslationRequest(BaseModel):
    text: str = Field(
        title="Text",
        description="Text to translate"
    )

    source: Literal['en', 'fr'] = Field(
        title="Source language",
        description="Language of the provided text",
        default='fr'
    )

    target: Literal['en', 'fr'] = Field(
        title="Target language",
        description="Language to translate the text into",
        default='en'
    )


class TranslationTaskResponse(BaseModel):
    result: Union[str, None] = Field(
        title="Translation results",
        description="Translated text"
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether or not the translation was successful"
    )


class TranslationResponse(TaskStatusResponse):
    task_result: Union[TranslationTaskResponse, None] = Field(
        title="Translation response",
        description="A dict containing the resulting translated text and a success flag."
    )


class TextDetectLanguageRequest(BaseModel):
    text: str = Field(
        title="Text",
        description="Text to detect the language of"
    )


class TextDetectLanguageTaskResponse(BaseModel):
    language: Union[str, None] = Field(
        title="Language detection results",
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