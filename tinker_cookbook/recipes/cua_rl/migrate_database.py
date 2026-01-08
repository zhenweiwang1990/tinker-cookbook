#!/usr/bin/env python3
"""
Database migration helper script.

This script runs Alembic migrations to update the database schema.
It's useful when:
- You pull new code with schema changes
- Training/benchmark scripts fail with "column does not exist" errors
- You want to manually ensure the database is up to date

Usage:
    # Run migrations
    uv run python migrate_database.py

    # Check current migration status
    uv run python migrate_database.py --status

    # Rebuild database (WARNING: deletes all data!)
    uv run python migrate_database.py --rebuild
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import from tinker_cookbook
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def run_migrations(db_url: str = None) -> None:
    """Run Alembic migrations to upgrade database to latest version."""
    from alembic import command
    from alembic.config import Config

    # Get alembic.ini path
    current_dir = Path(__file__).parent
    alembic_ini_path = current_dir / "alembic.ini"

    if not alembic_ini_path.exists():
        print(f"ERROR: alembic.ini not found at {alembic_ini_path}")
        sys.exit(1)

    # Create Alembic config
    alembic_cfg = Config(str(alembic_ini_path))

    # Set database URL
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    else:
        # Use same logic as database.py
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            postgres_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
            postgres_port = os.getenv("POSTGRES_PORT", "5433")
            postgres_db = os.getenv("POSTGRES_DB", "training_db")
            postgres_user = os.getenv("POSTGRES_USER", "training_user")
            postgres_password = os.getenv("POSTGRES_PASSWORD", "training_password")
            db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        print(f"Database: {db_url.split('@')[-1] if '@' in db_url else db_url}")

    print("Running migrations...")
    try:
        command.upgrade(alembic_cfg, "head")
        print("✓ Migrations completed successfully!")
    except Exception as e:
        print(f"ERROR: Migration failed: {e}")
        sys.exit(1)


def check_status(db_url: str = None) -> None:
    """Check current migration status."""
    from alembic import command
    from alembic.config import Config

    current_dir = Path(__file__).parent
    alembic_ini_path = current_dir / "alembic.ini"

    if not alembic_ini_path.exists():
        print(f"ERROR: alembic.ini not found at {alembic_ini_path}")
        sys.exit(1)

    alembic_cfg = Config(str(alembic_ini_path))

    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    else:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            postgres_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
            postgres_port = os.getenv("POSTGRES_PORT", "5433")
            postgres_db = os.getenv("POSTGRES_DB", "training_db")
            postgres_user = os.getenv("POSTGRES_USER", "training_user")
            postgres_password = os.getenv("POSTGRES_PASSWORD", "training_password")
            db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        print(f"Database: {db_url.split('@')[-1] if '@' in db_url else db_url}")

    print("\nCurrent migration status:")
    command.current(alembic_cfg)

    print("\nMigration history:")
    command.history(alembic_cfg)


def rebuild_database(db_url: str = None) -> None:
    """Rebuild database from scratch (WARNING: deletes all data!)."""
    print("WARNING: This will delete all data in the database!")
    response = input("Are you sure you want to continue? (yes/no): ")

    if response.lower() != "yes":
        print("Aborted.")
        return

    from tinker_cookbook.recipes.cua_rl.database.rebuild_database import main as rebuild_main

    print("\nRebuilding database...")
    rebuild_main()
    print("✓ Database rebuilt successfully!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Database migration helper for CUA RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run migrations to update schema
    uv run python migrate_database.py

    # Check current migration status
    uv run python migrate_database.py --status

    # Rebuild database (deletes all data!)
    uv run python migrate_database.py --rebuild

Environment Variables:
    DATABASE_URL          - Full PostgreSQL connection URL
    POSTGRES_HOST         - Database host (default: 127.0.0.1)
    POSTGRES_PORT         - Database port (default: 5433)
    POSTGRES_DB           - Database name (default: training_db)
    POSTGRES_USER         - Database user (default: training_user)
    POSTGRES_PASSWORD     - Database password (default: training_password)
        """,
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current migration status",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild database from scratch (WARNING: deletes all data!)",
    )

    parser.add_argument(
        "--db-url",
        type=str,
        help="Database URL (overrides environment variables)",
    )

    args = parser.parse_args()

    if args.rebuild:
        rebuild_database(args.db_url)
    elif args.status:
        check_status(args.db_url)
    else:
        run_migrations(args.db_url)


if __name__ == "__main__":
    main()

