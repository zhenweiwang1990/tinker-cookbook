#!/usr/bin/env python3
"""
Initialize the PostgreSQL database for CUA RL training monitor.

This script creates all necessary tables in the database.
Run this once before starting the training monitor.

Usage:
    python init_db.py
    
Environment variables:
    DATABASE_URL: PostgreSQL connection URL (e.g., postgresql://user:pass@host:port/db)
    Or:
    POSTGRES_HOST: PostgreSQL host (default: 127.0.0.1)
    POSTGRES_PORT: PostgreSQL port (default: 5433)
    POSTGRES_DB: Database name (default: training_db)
    POSTGRES_USER: Database user (default: training_user)
    POSTGRES_PASSWORD: Database password (default: training_password)
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path to import database module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tinker_cookbook.recipes.cua_rl.database.database import init_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Initialize the database."""
    try:
        logger.info("Initializing database...")
        init_database(echo=True)
        logger.info("✅ Database initialized successfully!")
        logger.info("All tables have been created.")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

