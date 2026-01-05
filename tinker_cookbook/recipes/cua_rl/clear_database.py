#!/usr/bin/env python3
"""
Clear all data from the database.

This script will delete all records from all tables in the correct order
to respect foreign key constraints.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tinker_cookbook.recipes.cua_rl.database import init_database, get_session_direct
from sqlalchemy import text

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

def clear_database():
    """Clear all data from the database."""
    # Get database URL from environment
    db_url = get_database_url()
    
    # Initialize database connection
    init_database(db_url=db_url)
    session = get_session_direct()
    
    try:
        print("⚠️  WARNING: This will delete ALL data from the database!")
        print("Tables to be cleared:")
        
        # List of tables in reverse dependency order (children first, parents last)
        # This ensures foreign key constraints are respected
        tables = [
            'status_history',  # No dependencies
            'validation',      # Depends on rollout, validator
            'obs',            # Depends on turn
            'action',         # Depends on turn
            'turn',           # Depends on rollout
            'rollout',        # Depends on task, environment, step/eval/baseline, group
            'environment',    # No dependencies (referenced by rollout)
            'group',          # Depends on step/eval/baseline
            'step',           # Depends on training
            'eval',           # Depends on training
            'baseline',       # Depends on training
            'training',       # Top level
            'validator',      # Depends on task
            'task',           # Top level
        ]
        
        for table in tables:
            print(f"  - {table}")
        
        # Get confirmation
        response = input("\nAre you sure you want to proceed? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return
        
        print("\nStarting database cleanup...")
        
        # Disable foreign key checks temporarily (PostgreSQL doesn't have this, but we'll delete in order)
        # For PostgreSQL, we need to use CASCADE or delete in correct order
        
        # Delete in order (children first)
        deleted_counts = {}
        for table in tables:
            try:
                # Use TRUNCATE CASCADE to handle foreign keys automatically
                result = session.execute(text(f'TRUNCATE TABLE "{table}" CASCADE'))
                session.commit()
                print(f"✓ Cleared table: {table}")
                deleted_counts[table] = "all"
            except Exception as e:
                # If CASCADE doesn't work, try DELETE
                try:
                    result = session.execute(text(f'DELETE FROM "{table}"'))
                    count = result.rowcount
                    session.commit()
                    print(f"✓ Deleted {count} rows from: {table}")
                    deleted_counts[table] = count
                except Exception as e2:
                    print(f"✗ Error clearing {table}: {e2}")
                    session.rollback()
        
        # Reset sequences (for auto-increment IDs)
        print("\nResetting sequences...")
        sequence_tables = {
            'training': 'training_id_seq',
            'task': 'task_id_seq',
            'validator': 'validator_id_seq',
            'step': 'step_id_seq',
            'eval': 'eval_id_seq',
            'baseline': 'baseline_id_seq',
            'group': 'group_id_seq',
            'rollout': 'rollout_id_seq',
            'turn': 'turn_id_seq',
            'action': 'action_id_seq',
            'obs': 'obs_id_seq',
            'validation': 'validation_id_seq',
            'environment': 'environment_id_seq',
        }
        
        for table_name, seq_name in sequence_tables.items():
            try:
                session.execute(text(f"SELECT setval('{seq_name}', 1, false);"))
                session.commit()
                print(f"✓ Reset sequence: {seq_name}")
            except Exception as e:
                # Sequence might not exist, ignore
                pass
        
        print("\n✅ Database cleared successfully!")
        print("\nDeleted counts:")
        for table, count in deleted_counts.items():
            print(f"  {table}: {count}")
        
    except Exception as e:
        print(f"\n❌ Error clearing database: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    clear_database()

