import asyncpg
from contextlib import asynccontextmanager
from app.config import settings


class Database:
    """PostgreSQL database connection manager"""

    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def connect(self):
        """Create database connection pool"""
        self.pool = await asyncpg.create_pool(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            database=settings.DB_NAME,
            min_size=2,
            max_size=10,
        )
        print(f"Connected to PostgreSQL at {settings.DB_HOST}:{settings.DB_PORT}")

    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            print("Disconnected from PostgreSQL")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        async with self.pool.acquire() as connection:
            yield connection


# Global database instance
db = Database()
