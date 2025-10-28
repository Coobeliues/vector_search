"""
Hybrid Search Service with BM25: Combines vector search and BM25 keyword search using RRF
"""
import asyncio
from typing import Dict, List
from app.database import db
from app.embedding import embedding_service
from app.models import HybridSearchResult

# Map method names to pgvector operators
OPERATORS = {
    "cosine": "<=>",  # Cosine distance
    "L2": "<->",  # Euclidean distance (L2)
    "dot_product": "<#>",  # Negative inner product
}


class HybridSearchBM25Service:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF)
    Combines vector similarity search with BM25 keyword search on tags
    """

    async def vector_search(
        self, query: str, top_n: int = 100, method: str = "cosine"
    ) -> List[Dict]:
        """
        Perform vector similarity search

        Args:
            query: Search query
            top_n: Number of results
            method: Distance metric (cosine, L2, dot_product)

        Returns:
            List of dicts with table_name and rank
        """
        # Generate query embedding
        query_embedding = await embedding_service.get_embedding(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Get operator for distance metric
        operator = OPERATORS.get(method, "<=>")

        # SQL query for vector search
        sql = f"""
            SELECT table_name,
                   description_embedding {operator} $1::vector AS distance
            FROM tables_metadata
            WHERE description_embedding IS NOT NULL
            ORDER BY distance
            LIMIT $2
        """

        async with db.acquire() as conn:
            rows = await conn.fetch(sql, embedding_str, top_n)

        # Return with rank (1-indexed)
        return [
            {"table_name": row["table_name"], "rank": idx + 1}
            for idx, row in enumerate(rows)
        ]

    async def bm25_search(self, query: str, top_n: int = 100) -> List[Dict]:
        """
        Perform BM25 keyword search on tags field using PostgreSQL full-text search

        Args:
            query: Search query
            top_n: Number of results

        Returns:
            List of dicts with table_name and rank
        """
        # Преобразовать query в tsquery (для поиска)
        # Поддержка русского и английского языков

        # SQL query for BM25 search using PostgreSQL full-text search
        # tags это JSONB поле вида: {"keywords": "crimes, location, date, ..."}
        # Извлекаем keywords и делаем полнотекстовый поиск
        sql = """
            SELECT
                table_name,
                ts_rank_cd(
                    to_tsvector('russian', COALESCE(tags->>'keywords', '')),
                    plainto_tsquery('russian', $1)
                ) as rank_score
            FROM tables_metadata
            WHERE
                tags IS NOT NULL
                AND tags->>'keywords' IS NOT NULL
                AND to_tsvector('russian', tags->>'keywords') @@ plainto_tsquery('russian', $1)
            ORDER BY rank_score DESC
            LIMIT $2
        """

        async with db.acquire() as conn:
            rows = await conn.fetch(sql, query, top_n)

        # Return with rank (1-indexed)
        return [
            {"table_name": row["table_name"], "rank": idx + 1, "bm25_score": float(row["rank_score"])}
            for idx, row in enumerate(rows)
        ]

    def calculate_rrf(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[HybridSearchResult]:
        """
        Calculate Reciprocal Rank Fusion scores

        RRF formula: score = Σ weight_i / (k + rank_i)

        Args:
            vector_results: Results from vector search with ranks
            bm25_results: Results from BM25 search with ranks
            k: RRF constant (default 60)
            vector_weight: Weight for vector search
            bm25_weight: Weight for BM25 search

        Returns:
            List of HybridSearchResult sorted by RRF score
        """
        # Build rank maps
        vector_ranks = {r["table_name"]: r["rank"] for r in vector_results}
        bm25_ranks = {r["table_name"]: r["rank"] for r in bm25_results}

        # Get all unique table names
        all_tables = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        # Calculate RRF scores
        rrf_scores = []
        for table_name in all_tables:
            score = 0.0
            vector_rank = vector_ranks.get(table_name)
            bm25_rank = bm25_ranks.get(table_name)

            # Add vector score
            if vector_rank is not None:
                score += vector_weight / (k + vector_rank)

            # Add BM25 score
            if bm25_rank is not None:
                score += bm25_weight / (k + bm25_rank)

            rrf_scores.append(
                HybridSearchResult(
                    table_name=table_name,
                    rrf_score=score,
                    vector_rank=vector_rank,
                    tags_rank=bm25_rank,  # Используем поле tags_rank для BM25 rank
                )
            )

        # Sort by RRF score (descending)
        rrf_scores.sort(key=lambda x: x.rrf_score, reverse=True)

        return rrf_scores

    async def search(
        self,
        query: str,
        top_n: int = 10,
        method: str = "cosine",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search using RRF (Vector + BM25)

        Args:
            query: Search query
            top_n: Number of final results to return
            method: Distance metric for vector search
            vector_weight: Weight for vector search (0.0-1.0)
            bm25_weight: Weight for BM25 search (0.0-1.0)
            rrf_k: RRF constant

        Returns:
            List of HybridSearchResult sorted by RRF score
        """
        # Run both searches in parallel (fetch more candidates for fusion)
        candidate_count = max(100, top_n * 10)

        vector_results, bm25_results = await asyncio.gather(
            self.vector_search(query, candidate_count, method),
            self.bm25_search(query, candidate_count),
        )

        # Calculate RRF scores
        rrf_results = self.calculate_rrf(
            vector_results, bm25_results, rrf_k, vector_weight, bm25_weight
        )

        # Return top_n results
        return rrf_results[:top_n]


# Global hybrid search BM25 service instance
hybrid_search_bm25_service = HybridSearchBM25Service()
