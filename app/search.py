from app.database import db
from app.embedding import embedding_service
from app.models import SearchResult


class VectorSearch:
    """Vector similarity search service"""

    # Map method names to pgvector operators
    OPERATORS = {
        "cosine": "<=>",  # Cosine distance
        "L2": "<->",  # Euclidean distance (L2)
        "dot_product": "<#>",  # Negative inner product
    }

    async def search(
        self, query: str, top_n: int = 10, method: str = "cosine"
    ) -> list[SearchResult]:
        """
        Search for similar tables using vector similarity

        Args:
            query: Search query text
            top_n: Number of top results to return
            method: Distance metric ('cosine', 'L2', 'dot_product')

        Returns:
            List of SearchResult objects
        """
        # Generate embedding for the query
        query_embedding = await embedding_service.get_embedding(query)

        # Convert embedding list to pgvector format string
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Get the operator for the selected method
        operator = self.OPERATORS.get(method, "<=>")

        # Build SQL query with vector similarity search
        sql = f"""
            SELECT
                table_name,
                description_embedding {operator} $1::vector AS distance
            FROM tables_metadata
            WHERE description_embedding IS NOT NULL
            ORDER BY description_embedding {operator} $1::vector
            LIMIT $2
        """

        async with db.acquire() as conn:
            rows = await conn.fetch(sql, embedding_str, top_n)

        # Convert distances to similarity scores (inverse)
        results = []
        for row in rows:
            distance = float(row["distance"])

            # Convert distance to similarity score (0-1 range)
            # For cosine: similarity = 1 - distance
            # For L2: we use inverse (smaller distance = higher score)
            # For dot product: already negative, so negate it
            if method == "cosine":
                score = 1 - distance
            elif method == "L2":
                score = 1 / (1 + distance)
            else:  # dot_product
                score = -distance

            results.append(
                SearchResult(table_name=row["table_name"], score=round(score, 4))
            )

        return results


# Global search service instance
vector_search = VectorSearch()
