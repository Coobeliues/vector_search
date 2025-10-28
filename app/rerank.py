import json
import re
import asyncio
from ollama import Client
from app.config import settings
from app.database import db
from app.search import vector_search
from app.models import RerankResult


class RerankService:
    """LLM-based reranking service using Gemma3"""

    def __init__(self):
        self.client = Client(host=settings.OLLAMA_RERANK_HOST)
        self.model = settings.RERANK_MODEL

    def _build_prompt(self, query: str, context: str, candidates: list) -> str:
        """
        Build prompt for LLM reranking

        Args:
            query: User search query
            context: Additional context from user
            candidates: List of table candidates with descriptions

        Returns:
            Formatted prompt string
        """
        candidates_text = ""
        for i, candidate in enumerate(candidates, 1):
            candidates_text += f"{i}. Table: {candidate['table_name']}\n"
            candidates_text += f"   Description: {candidate['description']}\n\n"

        prompt = f"""You are a database relevance scoring assistant. Given a user query and context, evaluate how relevant each database table is to the query.

Task: For each table, provide a relevance score from 0 to 1, where:
- 0.0-0.3: Not relevant (table has no relation to the query)
- 0.4-0.6: Somewhat relevant (table has indirect relation)
- 0.7-0.8: Relevant (table contains related information)
- 0.9-1.0: Highly relevant (table directly answers the query)

Return ONLY a valid JSON array with this exact format (no additional text):
[
  {{"table_name": "exact_table_name", "score": 0.8}},
  {{"table_name": "exact_table_name", "score": 0.6}}
]

Important:
- Use exact table names from the list above
- Scores must be floats from 0.0 to 1.0
- Return ALL tables (even with low scores)

Database Tables to Evaluate:
{candidates_text}

User Query: {query}
Context: {context}
"""
        # - Sort by score (highest first)
        return prompt

    def _parse_llm_response(self, response: str) -> list[dict]:
        """
        Parse LLM JSON response

        Args:
            response: Raw LLM response text

        Returns:
            List of dictionaries with table_name and score
        """
        try:
            # Try to find JSON array in response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
            else:
                # Fallback: try parsing entire response
                return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response}")
            return []

    def _rerank_batch(self, query: str, context: str, batch_candidates: list) -> list[dict]:
        """
        Rerank a single batch of candidates using LLM

        Args:
            query: User search query
            context: Additional context
            batch_candidates: List of candidates to rerank (max 10)

        Returns:
            List of dicts with table_name and score
        """
        prompt = self._build_prompt(query, context, batch_candidates)

        try:
            llm_response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1}
            )

            response_text = llm_response.get("response", "")
            parsed_scores = self._parse_llm_response(response_text)
            return parsed_scores
        except Exception as e:
            print(f"LLM batch reranking failed: {e}")
            return []

    async def rerank(
        self, query: str, context: str, top_n, method: str = "cosine"
    ) -> list[RerankResult]:
        """
        Rerank search results using LLM with batch processing

        Args:
            query: Search query
            context: Additional context for LLM
            top_n: Number of candidates to retrieve (int or "full" for all)
            method: Vector search method

        Returns:
            List of reranked results with LLM scores
        """
        # Step 1: Get top_n candidates using vector search
        initial_results = await vector_search.search(
            query=query, top_n=top_n, method=method
        )

        if not initial_results:
            return []

        # Step 2: Fetch table descriptions from database
        table_names = [r.table_name for r in initial_results]
        placeholders = ",".join([f"${i+1}" for i in range(len(table_names))])

        sql = f"""
            SELECT table_name, table_description
            FROM tables_metadata
            WHERE table_name IN ({placeholders})
        """

        async with db.acquire() as conn:
            rows = await conn.fetch(sql, *table_names)

        # Build candidates list with descriptions
        candidates = []
        for row in rows:
            candidates.append({
                "table_name": row["table_name"],
                "description": row["table_description"] or "No description available"
            })

        if not candidates:
            return []

        # Step 3: Process candidates in batches of 10
        BATCH_SIZE = 25
        all_results = []

        try:
            for i in range(0, len(candidates), BATCH_SIZE):
                batch = candidates[i:i + BATCH_SIZE]

                # Rerank this batch
                batch_scores = self._rerank_batch(query, context, batch)

                # Build score map for this batch
                score_map = {item["table_name"]: item["score"] for item in batch_scores}

                # Add results, ensuring all batch candidates are included
                for candidate in batch:
                    table_name = candidate["table_name"]
                    llm_score = score_map.get(table_name, 0.0)  # Default 0.0 if LLM didn't score it
                    all_results.append(
                        RerankResult(
                            table_name=table_name,
                            score=float(llm_score)
                        )
                    )

            # Step 4: Sort all results by score (descending)
            all_results.sort(key=lambda x: x.score, reverse=True)

            return all_results

        except Exception as e:
            print(f"LLM reranking failed: {e}")
            # Fallback: return original results without reranking
            return [
                RerankResult(
                    table_name=c["table_name"],
                    score=0.0
                )
                for c in candidates
            ]


# Global rerank service instance
rerank_service = RerankService()
