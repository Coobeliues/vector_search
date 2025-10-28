"""
Hybrid Search Service: Combines vector search and tags-based keyword search using RRF
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


class HybridSearchService:
    """
    Hybrid search using Reciprocal Rank Fusion (RRF)
    Combines vector similarity search with tags keyword matching
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

    async def tags_search(self, query: str, top_n: int = 100) -> List[Dict]:
        """
        Perform tags-based keyword search using vector similarity on tags_embedding

        Args:
            query: Search query
            top_n: Number of results

        Returns:
            List of dicts with table_name and rank
        """
        # Generate query embedding for tags
        query_embedding = await embedding_service.get_embedding(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # SQL query for tags vector search (using cosine distance)
        sql = """
            SELECT table_name,
                   tags_embedding <=> $1::vector AS distance
            FROM tables_metadata
            WHERE tags_embedding IS NOT NULL
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

    def calculate_rrf(
        self,
        vector_results: List[Dict],
        tags_results: List[Dict],
        k: int = 60,
        vector_weight: float = 0.5,
        tags_weight: float = 0.5,
    ) -> List[HybridSearchResult]:
        """
        Calculate Reciprocal Rank Fusion scores

        RRF formula: score = Σ weight_i / (k + rank_i)

        Args:
            vector_results: Results from vector search with ranks
            tags_results: Results from tags search with ranks
            k: RRF constant (default 60)
            vector_weight: Weight for vector search
            tags_weight: Weight for tags search

        Returns:
            List of HybridSearchResult sorted by RRF score
        """
        # Build rank maps
        vector_ranks = {r["table_name"]: r["rank"] for r in vector_results}
        tags_ranks = {r["table_name"]: r["rank"] for r in tags_results}

        # Get all unique table names
        all_tables = set(vector_ranks.keys()) | set(tags_ranks.keys())

        # Calculate RRF scores
        rrf_scores = []
        for table_name in all_tables:
            score = 0.0
            vector_rank = vector_ranks.get(table_name)
            tags_rank = tags_ranks.get(table_name)

            # Add vector score
            if vector_rank is not None:
                score += vector_weight / (k + vector_rank)

            # Add tags score
            if tags_rank is not None:
                score += tags_weight / (k + tags_rank)

            rrf_scores.append(
                HybridSearchResult(
                    table_name=table_name,
                    rrf_score=score,
                    vector_rank=vector_rank,
                    tags_rank=tags_rank,
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
        tags_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search using RRF

        Args:
            query: Search query
            top_n: Number of final results to return
            method: Distance metric for vector search
            vector_weight: Weight for vector search (0.0-1.0)
            tags_weight: Weight for tags search (0.0-1.0)
            rrf_k: RRF constant

        Returns:
            List of HybridSearchResult sorted by RRF score
        """
        # Run both searches in parallel (fetch more candidates for fusion)
        candidate_count = max(100, top_n * 10)

        vector_results, tags_results = await asyncio.gather(
            self.vector_search(query, candidate_count, method),
            self.tags_search(query, candidate_count),
        )

        # Calculate RRF scores
        rrf_results = self.calculate_rrf(
            vector_results, tags_results, rrf_k, vector_weight, tags_weight
        )

        # Return top_n results
        return rrf_results[:top_n]


# Global hybrid search service instance
hybrid_search_service = HybridSearchService()
