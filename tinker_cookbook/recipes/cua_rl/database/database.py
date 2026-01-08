"""
Database management and initialization for CUA RL training.

This module provides PostgreSQL database connection, initialization, and session management.
"""

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from tinker_cookbook.recipes.cua_rl.database.database_models import Base

logger = logging.getLogger(__name__)

# Global session factory
_session_factory: Optional[sessionmaker] = None
_db_url: Optional[str] = None


def init_database(db_url: Optional[str] = None, echo: bool = False) -> None:
    """
    Initialize the PostgreSQL database connection and create all tables.
    
    Args:
        db_url: PostgreSQL connection URL (e.g., postgresql://user:pass@host:port/db)
                If None, uses DATABASE_URL environment variable or constructs from POSTGRES_* env vars
        echo: If True, log all SQL statements
    """
    global _session_factory, _db_url
    
    # Determine database URL
    if db_url:
        if not db_url.startswith(("postgresql://", "postgres://")):
            raise ValueError(f"Invalid database URL. Must start with 'postgresql://' or 'postgres://', got: {db_url}")
        database_url = db_url
    else:
        # Try DATABASE_URL environment variable first
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            # Construct from individual environment variables
            postgres_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
            postgres_port = os.getenv("POSTGRES_PORT", "5433")  # Use 5433 to avoid conflict with Cursor
            postgres_db = os.getenv("POSTGRES_DB", "training_db")
            postgres_user = os.getenv("POSTGRES_USER", "training_user")
            postgres_password = os.getenv("POSTGRES_PASSWORD", "training_password")
            
            database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
            logger.info(f"Constructed database URL from environment variables: {postgres_host}:{postgres_port}/{postgres_db}")
    
    if not database_url:
        raise RuntimeError(
            "Database URL not provided. Set DATABASE_URL environment variable or "
            "POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD"
        )
    
    # Create PostgreSQL engine
    engine = create_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=10,  # Connection pool size
        max_overflow=20,  # Max overflow connections
        pool_recycle=3600,  # Recycle connections after 1 hour
    )
    
    # Run Alembic migrations first to ensure schema is up to date
    # This is done BEFORE create_all to avoid conflicts
    migration_success = False
    try:
        from alembic import command
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from alembic.runtime.migration import MigrationContext
        
        # Get alembic.ini path (should be in the parent directory, not in database/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # alembic.ini is in tinker_cookbook/recipes/cua_rl/, not in database/
        alembic_dir = os.path.dirname(current_dir)  # Go up from database/ to cua_rl/
        alembic_ini_path = os.path.join(alembic_dir, "alembic.ini")
        
        if os.path.exists(alembic_ini_path):
            alembic_cfg = Config(alembic_ini_path)
            # Set database URL in config
            alembic_cfg.set_main_option("sqlalchemy.url", database_url)
            
            # Check if database needs initialization
            with engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()
                
                if current_rev is None:
                    # Database is uninitialized, create base schema first
                    logger.info("Database is uninitialized, creating initial schema...")
                    Base.metadata.create_all(engine)
                    # Stamp with initial revision
                    command.stamp(alembic_cfg, "head")
                    logger.info("Database initialized with current schema")
                else:
                    # Database exists, run migrations
                    script = ScriptDirectory.from_config(alembic_cfg)
                    head_rev = script.get_current_head()
                    
                    if current_rev != head_rev:
                        logger.info(f"Running migrations from {current_rev} to {head_rev}...")
                        command.upgrade(alembic_cfg, "head")
                        logger.info("Migrations applied successfully")
                    else:
                        logger.info(f"Database schema is up to date (revision: {current_rev})")
            
            migration_success = True
        else:
            logger.error(f"Alembic config not found at {alembic_ini_path}")
            raise RuntimeError(f"Cannot initialize database: alembic.ini not found at {alembic_ini_path}")
    except Exception as e:
        logger.error(f"Failed to run database migrations: {e}")
        logger.error("Please run: cd tinker_cookbook/recipes/cua_rl && uv run python migrate_database.py")
        raise RuntimeError(f"Database migration failed: {e}. Cannot continue.") from e
    
    if not migration_success:
        raise RuntimeError("Database migration did not complete successfully")
    
    # Create session factory
    _session_factory = sessionmaker(bind=engine)
    _db_url = database_url
    
    logger.info(f"PostgreSQL database initialized: {database_url.split('@')[-1] if '@' in database_url else database_url}")


def get_db_url() -> Optional[str]:
    """Get the current database URL."""
    return _db_url


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Get a database session context manager.
    
    Usage:
        with get_session() as session:
            # Use session here
            pass
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_direct() -> Session:
    """
    Get a database session directly (without context manager).
    Caller is responsible for committing and closing.
    
    Usage:
        session = get_session_direct()
        try:
            # Use session
            session.commit()
        finally:
            session.close()
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    return _session_factory()


def json_serialize(obj: Any) -> str:
    """Serialize an object to JSON string."""
    if obj is None:
        return None
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, default=str)
    return str(obj)


def json_deserialize(json_str: Optional[str]) -> Any:
    """Deserialize a JSON string to object."""
    if json_str is None:
        return None
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return json_str


def commit_with_retry(session: Session, max_retries: int = 3, initial_delay: float = 0.1) -> None:
    """
    Commit a session with retry logic for transient database errors.
    
    Args:
        session: SQLAlchemy session
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (exponential backoff)
    """
    import time
    from sqlalchemy.exc import OperationalError, DisconnectionError
    
    for attempt in range(max_retries):
        try:
            session.commit()
            return
        except (OperationalError, DisconnectionError) as e:
            # Retry on connection errors
            if attempt < max_retries - 1:
                wait_time = initial_delay * (2 ** attempt)
                logger.warning(
                    f"Database error during commit, retrying in {wait_time:.2f}s "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(wait_time)
                continue
            # Re-raise if out of retries
            raise
        except Exception:
            # Re-raise non-transient errors immediately
            raise
    
    # If we get here, all retries failed
    raise RuntimeError(f"Failed to commit after {max_retries} retries")

