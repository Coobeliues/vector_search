import httpx
from app.config import settings


class EmbeddingService:
    """Service for generating text embeddings using Ollama"""

    def __init__(self):
        self.model = settings.EMBEDDING_MODEL
        self.base_url = settings.OLLAMA_HOST
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding vector for given text

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except httpx.HTTPError as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Global embedding service instance
embedding_service = EmbeddingService()
