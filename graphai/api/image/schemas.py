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

    no_cache: bool = Field(
        title="No caching",
        default=False
    )


class UploadedFileOriginInfo(BaseModel):
    id: str = Field(
        title="File ID"
    )

    name: str = Field(
        title="File name"
    )


class UploadImageRequest(BaseModel):
    contents: str = Field(
        title="Contents",
        description="The contents of the file to be uploaded. Must be a base64 encoded string. Given a file handle f, "
                    "you could generate it with the following line using the 'base64' library: "
                    "base64.b64encode(f.read()).decode('utf-8')"
    )

    file_extension: Literal['bmp', 'png', 'jpg', 'jpeg', 'pdf'] = Field(
        title="File extension",
        description="The extension of the file you are uploading"
    )

    origin: Literal['gdrive'] = Field(
        title="File origin",
        description="Original location from which the file was retrieved. Currently only 'gdrive' is allowed."
    )

    origin_info: UploadedFileOriginInfo = Field(
        title="File origin info"
    )

    force: bool = Field(
        title="Force reupload",
        default=False
    )

    no_cache: bool = Field(
        title="No caching",
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
        description="False if the URI/file had already been retrieved/uploaded before, "
                    "True if not and if it was successful"
    )

    successful: bool = Field(
        title="Success flag",
        description="True if task successful, False otherwise"
    )

    error: Union[str, None] = Field(
        title="Error",
        description="Contains whatever exception made the retrieval/upload unsuccessful, if any.",
        default=None
    )


class RetrieveImageURLResponse(TaskStatusResponse):
    task_result: Union[RetrieveImageURLResponseInner, None] = Field(
        title="File retrieval response",
        description="A dict containing a flag for whether the retrieval/upload was successful, plus a token "
                    "that refers to the now-retrieved/uploaded file if so (and null if not)."
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

    method: Literal['google', 'tesseract', 'openai'] = Field(
        title="Method",
        description="OCR method. Available methods are 'google' (default), 'openai', 'gemini',"
                    "and 'tesseract' (not recommended)",
        default="google"
    )

    force: bool = Field(
        title="Force",
        description="Whether to force a recomputation",
        default=False
    )

    no_cache: bool = Field(
        title="No caching",
        description="Skips the entire cache: does not do a lookup, and does not write results to cache.",
        default=False
    )

    google_api_token: str = Field(
        title="Google API token",
        description="Token that authenticates the user on the Google OCR API."
                    "Without a valid token, Google OCR will fail. Not required for Tesseract, OpenAI, or Gemini.",
        default=None
    )

    openai_api_token: str = Field(
        title="OpenAI API token",
        description="Token that authenticates the user on the OpenAI API."
                    "Without a valid token, OpenAI OCR will fail. Not required for Tesseract, Google, or Gemini.",
        default=None
    )

    gemini_api_token: str = Field(
        title="Gemini API token",
        description="Token that authenticates the user on the Gemini API."
                    "Without a valid token, Gemini OCR will fail. Not required for Tesseract, Google, or OpenAI.",
        default=None
    )

    pdf_in_pages: bool = Field(
        title="PDF in pages",
        description="Whether to return the results of PDF OCR in separate pages (JSON format) or one joined string. "
                    "Flag only used when the file being OCR'ed is a PDF.",
        default=True
    )

    model_type: str = Field(
        title="Model type",
        description="For OpenAI and Gemini options, allows the user to specify the model that they want to use. "
                    "Do not specify this option unless you know exactly what you are doing.",
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
