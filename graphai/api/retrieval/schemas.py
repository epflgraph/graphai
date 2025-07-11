from pydantic import BaseModel, Field
from typing import Union, List, Dict, Literal


class RetrievalRequest(BaseModel):
    text: str = Field(
        title="Text",
        description="Text to search for"
    )

    index: str = Field(
        title="Index",
        description="Index to search in. Call GET `/rag/retrieve/info` to get list of available indexes.",
    )

    filters: Union[Dict[str, str], None] = Field(
        title="Filters",
        description="A dictionary of filters. "
                    "Call GET `/rag/retrieve/info` to get list of available filters for each index.",
        default=None
    )

    limit: int = Field(
        title="Limit",
        description="Number of search results to return",
        default=10
    )

    return_scores: bool = Field(
        title="Return scores",
        default=False
    )

    filter_by_date: bool = Field(
        title="Filter by current date",
        description="If True, if the requested index has 'from' and 'until' fields, only returns documents "
                    "that are available at the current date and time based on those two fields. Basically "
                    "a smart custom filter that doesn't require the user to manually provide the current "
                    "datetime and ask for 'from' to be before it and for 'until' to be after it. "
                    "If the index does not have 'from' and 'until' fields, this results in an empty response.",
        default=False
    )


class RetrievalResponse(BaseModel):
    n_results: int = Field(
        title="Number of results"
    )

    result: Union[List[dict], None] = Field(
        title="Retrieval results",
        description="Retrieval results"
    )

    successful: bool = Field(
        title="Success flag",
        description="Whether or not the retrieval was successful"
    )


class RetrievalIndexType(BaseModel):
    index: str = Field(
        title="Index",
        description="Name of index"
    )

    filters: Dict[str, List[Union[str, None]]] = Field(
        title="Filters",
        description="List of applicable filters for the index with their allowed values"
    )


class RetrievalInfoResponse(BaseModel):
    indexes: List[RetrievalIndexType] = Field(
        title="List of indexes"
    )


class ChunkRequest(BaseModel):
    text: Union[str, Dict[int, str]] = Field(
        title="Text",
        description="Text to chunk, either a pure string or an int to string dictionary (page number to content)."
    )

    chunk_size: int = Field(
        title="Chunk size (tokens)",
        default=400
    )

    chunk_overlap: int = Field(
        title="Chunk overlap (tokens)",
        default=100
    )

    one_chunk_per_page: bool = Field(
        title="One chunk per page (for PDFs)",
        default=False
    )

    one_chunk_per_doc: bool = Field(
        title="One chunk for the whole document. Overrides one_chunk_per_page if True.",
        default=False
    )


class ChunkResponse(BaseModel):
    split: List[str] = Field(
        title="Chunked text"
    )

    full: str = Field(
        title="Full text"
    )


class AnonymizeRequest(BaseModel):
    text: str = Field(
        title="Text",
        description="Text to anonymize"
    )

    lang: Literal['en', 'fr'] = Field(
        title="Language",
        default="en"
    )


class AnonymizeResponse(BaseModel):
    result: str = Field(
        title="Result",
        description="Anonymized text"
    )

    successful: bool = Field(
        title="Successful"
    )
