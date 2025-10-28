from pydantic import BaseModel, Field
from typing import Literal


class SearchRequest(BaseModel):
    """Search request model"""

    query: str = Field(..., description="Search query text")
    top_n: int = Field(10, ge=1, le=10000, description="Number of top results to return")
    method: Literal["cosine", "L2", "dot_product"] = Field(
        "cosine", description="Distance metric method"
    )


class SearchResult(BaseModel):
    """Single search result"""

    table_name: str = Field(..., description="Name of the table")
    score: float = Field(..., description="Similarity score")


class SearchResponse(BaseModel):
    """Search response model"""

    results: list[SearchResult] = Field(..., description="List of search results")
    total: int = Field(..., description="Total number of results")


class RerankRequest(BaseModel):
    """Rerank search request model"""

    query: str = Field(..., description="Search query text")
    prompt: str = Field(
        "Find the most relevant database tables for this query",
        description="Context prompt for LLM reranking"
    )
    top_n: int = Field(
        50,
        ge=1,
        le=10000,
        description='Number of candidates for reranking (1-10000)'
    )
    method: Literal["cosine", "L2", "dot_product"] = Field(
        "cosine", description="Distance metric method"
    )


class RerankResult(BaseModel):
    """Single rerank result with LLM score"""

    table_name: str = Field(..., description="Name of the table")
    score: float = Field(..., description="LLM relevance score (0-1)")


class RerankResponse(BaseModel):
    """Rerank response model"""

    results: list[RerankResult] = Field(..., description="List of reranked results")
    total: int = Field(..., description="Total number of results")


class HybridSearchRequest(BaseModel):
    """Hybrid search request model (RRF: vector + tags keyword search)"""

    query: str = Field(..., description="Search query text")
    top_n: int = Field(10, ge=1, le=100, description="Number of final results to return")
    vector_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for vector search (0.0-1.0)")
    tags_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for tags keyword search (0.0-1.0)")
    method: Literal["cosine", "L2", "dot_product"] = Field(
        "cosine", description="Distance metric method for vector search"
    )
    rrf_k: int = Field(60, ge=1, le=100, description="RRF constant (default 60)")


class HybridSearchResult(BaseModel):
    """Single hybrid search result"""

    table_name: str = Field(..., description="Name of the table")
    rrf_score: float = Field(..., description="Combined RRF score")
    vector_rank: int | None = Field(None, description="Rank in vector search (None if not found)")
    tags_rank: int | None = Field(None, description="Rank in tags search (None if not found)")


class HybridSearchResponse(BaseModel):
    """Hybrid search response model"""

    results: list[HybridSearchResult] = Field(..., description="List of hybrid search results")
    total: int = Field(..., description="Total number of results")
