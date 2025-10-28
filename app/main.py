from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.config import settings
from app.database import db
from app.embedding import embedding_service
from app.search import vector_search
from app.rerank import rerank_service
from app.hybrid_search import hybrid_search_service
from app.hybrid_search_bm25 import hybrid_search_bm25_service
from app.models import (
    SearchRequest,
    SearchResponse,
    RerankRequest,
    RerankResponse,
    HybridSearchRequest,
    HybridSearchResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events"""
    # Startup
    await db.connect()
    print("Vector Search API started")
    yield
    # Shutdown
    await db.disconnect()
    await embedding_service.close()
    print("Vector Search API stopped")


app = FastAPI(
    title="Vector Search API",
    description="Microservice for vector similarity search in PostgreSQL",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vector Search API",
        "version": "1.0.0",
        "endpoints": ["/search", "/search_rerank", "/search_hybrid", "/search_hybrid_bm25", "/health"],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with db.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_tables(request: SearchRequest):
    """
    Search for similar tables using vector similarity

    Args:
        request: SearchRequest with query, top_n, and method

    Returns:
        SearchResponse with list of similar tables and scores
    """
    try:
        results = await vector_search.search(
            query=request.query, top_n=request.top_n, method=request.method
        )

        return SearchResponse(results=results, total=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search_rerank", response_model=RerankResponse)
async def search_with_rerank(request: RerankRequest):
    """
    Search for similar tables using vector similarity + LLM reranking

    Args:
        request: RerankRequest with query, prompt, top_n, and method

    Returns:
        RerankResponse with list of reranked tables and LLM scores (0-1)
    """
    try:
        results = await rerank_service.rerank(
            query=request.query,
            context=request.prompt,
            top_n=request.top_n,
            method=request.method
        )

        return RerankResponse(results=results, total=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")


@app.post("/search_hybrid", response_model=HybridSearchResponse)
async def search_hybrid(request: HybridSearchRequest):
    """
    Search for similar tables using hybrid search (RRF: vector + tags embedding)

    This endpoint combines:
    - Vector similarity search on table descriptions
    - Tags-based vector search on LLM-generated keywords embeddings
    - Reciprocal Rank Fusion (RRF) to merge results

    Args:
        request: HybridSearchRequest with query, top_n, weights, method, and rrf_k

    Returns:
        HybridSearchResponse with list of tables ranked by RRF score
    """
    try:
        results = await hybrid_search_service.search(
            query=request.query,
            top_n=request.top_n,
            method=request.method,
            vector_weight=request.vector_weight,
            tags_weight=request.tags_weight,
            rrf_k=request.rrf_k,
        )

        return HybridSearchResponse(results=results, total=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@app.post("/search_hybrid_bm25", response_model=HybridSearchResponse)
async def search_hybrid_bm25(request: HybridSearchRequest):
    """
    Search for similar tables using hybrid search (RRF: vector + BM25)

    This endpoint combines:
    - Vector similarity search on table descriptions
    - BM25 keyword search on tags field (PostgreSQL full-text search)
    - Reciprocal Rank Fusion (RRF) to merge results

    Args:
        request: HybridSearchRequest with query, top_n, weights, method, and rrf_k

    Returns:
        HybridSearchResponse with list of tables ranked by RRF score
    """
    try:
        results = await hybrid_search_bm25_service.search(
            query=request.query,
            top_n=request.top_n,
            method=request.method,
            vector_weight=request.vector_weight,
            bm25_weight=request.tags_weight,  # Используем tags_weight для BM25
            rrf_k=request.rrf_k,
        )

        return HybridSearchResponse(results=results, total=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid BM25 search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
