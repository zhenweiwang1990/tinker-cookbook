#!/usr/bin/env python3
"""
Script to rebuild the database from scratch.

This script will:
1. Drop all existing tables
2. Create all tables from the current models
3. Create an initial Alembic migration

Usage:
    python -m tinker_cookbook.recipes.cua_rl.rebuild_database
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from tinker_cookbook.recipes.cua_rl.database.database_models import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_database_url():
    """Get database URL from environment variables."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5432")
        postgres_db = os.getenv("POSTGRES_DB", "training_db")
        postgres_user = os.getenv("POSTGRES_USER", "training_user")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "training_password")
        database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    return database_url


def drop_all_tables(engine):
    """Drop all tables in the database."""
    logger.info("Dropping all existing tables...")
    
    # Get all table names
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
        """))
        table_names = [row[0] for row in result]
    
    if not table_names:
        logger.info("No tables to drop.")
        return
    
    logger.info(f"Found {len(table_names)} tables: {', '.join(table_names)}")
    
    # Drop all tables
    with engine.connect() as conn:
        # Disable foreign key checks temporarily
        conn.execute(text("SET session_replication_role = 'replica';"))
        
        for table_name in table_names:
            try:
                conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;'))
                logger.info(f"Dropped table: {table_name}")
            except Exception as e:
                logger.warning(f"Failed to drop table {table_name}: {e}")
        
        conn.commit()
        conn.execute(text("SET session_replication_role = 'origin';"))
    
    logger.info("All tables dropped.")


def create_all_tables(engine):
    """Create all tables from models."""
    logger.info("Creating all tables from models...")
    Base.metadata.create_all(engine)
    logger.info("All tables created.")


def main():
    """Main function to rebuild database."""
    database_url = get_database_url()
    logger.info(f"Connecting to database: {database_url.split('@')[-1] if '@' in database_url else database_url}")
    
    engine = create_engine(database_url, echo=False)
    
    try:
        # Drop all existing tables
        drop_all_tables(engine)
        
        # Create all tables from models
        create_all_tables(engine)
        
        logger.info("✓ Database rebuilt successfully!")
        logger.info("You can now create an initial Alembic migration with:")
        logger.info("  cd tinker_cookbook/recipes/cua_rl && uv run alembic revision --autogenerate -m 'Initial migration'")
        logger.info("  uv run alembic stamp head")
        
    except Exception as e:
        logger.error(f"Error rebuilding database: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

