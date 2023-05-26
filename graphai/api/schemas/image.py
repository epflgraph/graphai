from pydantic import BaseModel, Field
from typing import Union, List, Literal

from graphai.api.schemas.common import TaskStatusResponse


class ImageFingerprintRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )


class ImageFingerprintTaskResponse(BaseModel):
    result: Union[str, None] = Field(
        title="Fingerprint",
        description="Fingerprint of the image file."
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    closest_token: Union[str, None] = Field(
        title="Closest token",
        description="The token of the most similar existing image that the fingerprint lookup was able to find. Equal "
                    "to original token if the most similar existing image did not satisfy the minimum similarity "
                    "threshold."
    )

    closest_token_origin: Union[str, None] = Field(
        title="Original token of the closest token",
        description="The token of video that the closest slide token originated from."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class ImageFingerprintResponse(TaskStatusResponse):
    task_result: Union[ImageFingerprintTaskResponse, None] = Field(
        title="Image fingerprinting response",
        description="A dict containing the resulting image fingerprint and a freshness flag."
    )


class ExtractTextRequest(BaseModel):
    token: str = Field(
        title="Token",
        description="The token that identifies the requested file"
    )

    method: Literal['google', 'tesseract'] = Field(
        title="Method",
        description="OCR method. Available methods are 'google' (default) and 'tesseract' (not recommended)",
        default="google"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )


class IndividualOCRResult(BaseModel):
    method: str = Field(
        title="OCR method",
        description="OCR method of the result"
    )
    token: str = Field(
        title="OCR token",
        description="Token of the file where the OCR results have been stored"
    )
    text: str = Field(
        title="OCR text",
        description="Textual results of the OCR"
    )


class ExtractTextTaskResponse(BaseModel):
    result: Union[List[IndividualOCRResult], None] = Field(
        title="OCR results",
        description="List of OCR results"
    )

    language: Union[str, None] = Field(
        title="Language",
        description="Language of the detected text"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class ExtractTextResponse(TaskStatusResponse):
    task_result: Union[ExtractTextTaskResponse, None] = Field(
        title="Extract text response",
        description="A dict containing the OCR results, plus the freshness and success flags."
    )


class DetectOCRLanguageTaskResponse(BaseModel):
    language: Union[str, None] = Field(
        title="Language",
        description="Language of the detected text"
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether the computation was successful."
    )


class DetectOCRLanguageResponse(TaskStatusResponse):
    task_result: Union[DetectOCRLanguageTaskResponse, None] = Field(
        title="Detect OCR language response",
        description="A dict containing the language of the text found in the image, "
                    "plus the freshness and success flags."
    )
