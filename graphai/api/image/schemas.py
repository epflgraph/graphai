from pydantic import BaseModel, Field
from typing import Union, List, Literal

from graphai.api.common.schemas import (
    TaskStatusResponse,
    TokenStatus
)


class RetrieveImageURLRequest(BaseModel):
    url: str = Field(
        title="URL",
        description="The URL that the image is to be retrieved from"
    )

    force: bool = Field(
        title="Force redownload",
        default=False
    )


class RetrieveImageURLResponseInner(BaseModel):
    token: Union[str, None] = Field(
        None, title="Token",
        description="Result token, null if task has failed"
    )

    token_status: Union[TokenStatus, None] = Field(
        title="Token status",
        description="Status of the returned token",
        default=None
    )

    token_size: Union[int, None] = Field(
        title="Token size",
        description="Size of the returned token",
        default=None
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="False if the URI had already been retrieved before, True if not and if retrieval was successful"
    )

    successful: bool = Field(
        title="Success flag",
        description="True if task successful, False otherwise"
    )


class RetrieveImageURLResponse(TaskStatusResponse):
    task_result: Union[RetrieveImageURLResponseInner, None] = Field(
        title="File retrieval response",
        description="A dict containing a flag for whether the retrieval was successful, plus a token "
                    "that refers to the now-retrieved file if so (and null if not)."
    )


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
        None, title="Fingerprint",
        description="Fingerprint of the image file."
    )

    fresh: bool = Field(
        title="Freshness flag",
        description="Whether the result was computed freshly or an existing cached result was returned."
    )

    closest_token: Union[str, None] = Field(
        None, title="Closest token",
        description="The token of the most similar existing image that the fingerprint lookup was able to find. Equal "
                    "to original token if the most similar existing image did not satisfy the minimum similarity "
                    "threshold."
    )

    closest_token_origin: Union[str, None] = Field(
        None, title="Original token of the closest token",
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

    google_api_token: str = Field(
        title="Google API token",
        description="Token that authenticates the user on the Google OCR API."
                    "Without a valid token, Google OCR will fail. Not required for Tesseract.",
        default=None
    )


class IndividualOCRResult(BaseModel):
    method: str = Field(
        title="OCR method",
        description="OCR method of the result"
    )

    text: str = Field(
        title="OCR text",
        description="Textual results of the OCR"
    )


class ExtractTextTaskResponse(BaseModel):
    result: Union[List[IndividualOCRResult], None] = Field(
        None, title="OCR results",
        description="List of OCR results"
    )

    language: Union[str, None] = Field(
        None, title="Language",
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
        None, title="Language",
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
