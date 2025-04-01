from pydantic import BaseModel, Field
from typing import Union, List, Dict


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


class ChunkResponse(BaseModel):
    split: List[str] = Field(
        title="Chunked text"
    )

    full: str = Field(
        title="Full text"
    )
