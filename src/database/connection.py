"""
Database Connection Manager

Handles MySQL connection using SQLAlchemy.
"""

import os
import sys
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.logging_config import get_logger

logger = get_logger(__name__)

# Add config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import settings


class DatabaseManager:
    """Manages database connections and sessions"""

    def __init__(self):
        self._engine = None
        self._session_factory = None

    @property
    def connection_string(self) -> str:
        """Build MySQL connection string"""
        return (
            f"mysql+pymysql://{settings.db_user}:{settings.db_password}"
            f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
            "?charset=utf8mb4"
        )

    @property
    def engine(self):
        """Get or create database engine"""
        if self._engine is None:
            self._engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Check connection health
                echo=False,  # Set True for SQL debugging
            )
        return self._engine

    @property
    def session_factory(self):
        """Get or create session factory"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup"""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test database connectivity.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as e:
            logger.error("Database connection failed: %s", e)
            return False

    def create_database(self) -> bool:
        """Create database if it doesn't exist.

        Returns:
            True if database creation is successful, False otherwise.
        """
        try:
            # Connect without database name first
            temp_url = (
                f"mysql+pymysql://{settings.db_user}:{settings.db_password}"
                f"@{settings.db_host}:{settings.db_port}"
            )
            temp_engine = create_engine(temp_url)

            with temp_engine.connect() as conn:
                conn.execute(
                    text(
                        f"CREATE DATABASE IF NOT EXISTS {settings.db_name} "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                )
                conn.commit()

            temp_engine.dispose()
            return True
        except SQLAlchemyError as e:
            logger.error("Failed to create database: %s", e)
            return False

    def init_schema(self) -> None:
        """Initialize database schema from models."""
        from src.database.models import Base

        Base.metadata.create_all(self.engine)
        logger.info("Database schema initialized successfully")

    def drop_all(self) -> None:
        """Drop all tables (use with caution!)."""
        from src.database.models import Base

        Base.metadata.drop_all(self.engine)
        logger.info("All tables dropped")

    def close(self):
        """Close all connections"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None


# Global database manager instance
db = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session"""
    with db.get_session() as session:
        yield session
