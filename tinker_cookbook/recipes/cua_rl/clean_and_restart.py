#!/usr/bin/env python3
"""
Clean all training data and logs for a fresh restart.

This script will:
1. Clear all data from the PostgreSQL database (training sessions, rollouts, turns, etc.)
2. Clean the logs/ directory
3. Clean training-monitor screenshot storage

Usage:
    # Interactive mode (asks for confirmation)
    python -m tinker_cookbook.recipes.cua_rl.clean_and_restart
    
    # Non-interactive mode (no confirmation, use with caution!)
    python -m tinker_cookbook.recipes.cua_rl.clean_and_restart --yes
    
    # Only clean database (keep logs)
    python -m tinker_cookbook.recipes.cua_rl.clean_and_restart --only-database
    
    # Only clean logs (keep database)
    python -m tinker_cookbook.recipes.cua_rl.clean_and_restart --only-logs
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from sqlalchemy import text

from tinker_cookbook.recipes.cua_rl.database.database import init_database, get_session

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"\n{message} [y/N]: ").strip().lower()
    return response in ['y', 'yes']


def clean_database(session, skip_confirmation: bool = False):
    """
    Clean all data from the database.
    
    This will delete all records from all tables in the correct order
    to avoid foreign key constraint violations.
    """
    logger.info("=" * 80)
    logger.info("DATABASE CLEANUP")
    logger.info("=" * 80)
    
    if not skip_confirmation:
        logger.warning("\n⚠️  WARNING: This will DELETE ALL training data from the database!")
        logger.warning("   - All training sessions")
        logger.warning("   - All baselines and evaluations")
        logger.warning("   - All rollouts, turns, actions, observations")
        logger.warning("   - All tasks and validators")
        logger.warning("   - All status history")
        
        if not confirm_action("Are you ABSOLUTELY SURE you want to continue?"):
            logger.info("❌ Database cleanup cancelled.")
            return False
    
    logger.info("\n🗑️  Deleting all records from database...")
    
    try:
        # Order matters due to foreign key constraints
        # Delete from child tables first, then parent tables
        tables_to_clean = [
            "status_history",
            "validation",
            "obs",  # observations
            "action",
            "turn",
            "rollout",
            "group",
            "step",
            "eval",
            "baseline",
            "validator",
            "task",
            "environment",
            "training",
        ]
        
        for table in tables_to_clean:
            result = session.execute(text(f"DELETE FROM {table}"))
            count = result.rowcount
            logger.info(f"  ✓ Deleted {count:,} records from '{table}'")
        
        session.commit()
        
        logger.info("\n✅ Database cleaned successfully!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Error cleaning database: {e}")
        session.rollback()
        return False


def clean_logs_directory(logs_dir: str = "logs", skip_confirmation: bool = False):
    """
    Clean the logs directory.
    
    Args:
        logs_dir: Path to logs directory (relative to project root)
        skip_confirmation: If True, skip confirmation prompt
    """
    logger.info("\n" + "=" * 80)
    logger.info("LOGS CLEANUP")
    logger.info("=" * 80)
    
    # Get project root (5 levels up from this file)
    project_root = Path(__file__).parent.parent.parent.parent.parent
    logs_path = project_root / logs_dir
    
    if not logs_path.exists():
        logger.info(f"\nℹ️  Logs directory does not exist: {logs_path}")
        return True
    
    # Count files and directories
    file_count = sum(1 for _ in logs_path.rglob('*') if _.is_file())
    dir_count = sum(1 for _ in logs_path.rglob('*') if _.is_dir())
    
    if file_count == 0 and dir_count == 0:
        logger.info(f"\nℹ️  Logs directory is already empty: {logs_path}")
        return True
    
    if not skip_confirmation:
        logger.warning(f"\n⚠️  WARNING: This will DELETE the entire logs directory!")
        logger.warning(f"   Path: {logs_path}")
        logger.warning(f"   Files: {file_count:,}")
        logger.warning(f"   Directories: {dir_count:,}")
        
        if not confirm_action("Are you sure you want to delete all logs?"):
            logger.info("❌ Logs cleanup cancelled.")
            return False
    
    logger.info(f"\n🗑️  Deleting logs directory: {logs_path}")
    
    try:
        shutil.rmtree(logs_path)
        logger.info("✅ Logs directory deleted successfully!")
        
        # Recreate empty logs directory
        logs_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created fresh logs directory: {logs_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error cleaning logs: {e}")
        return False


def clean_screenshots(skip_confirmation: bool = False):
    """
    Clean training-monitor screenshot storage.
    """
    logger.info("\n" + "=" * 80)
    logger.info("SCREENSHOTS CLEANUP")
    logger.info("=" * 80)
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    screenshots_path = project_root / "training-monitor" / "public" / "screenshots"
    
    if not screenshots_path.exists():
        logger.info(f"\nℹ️  Screenshots directory does not exist: {screenshots_path}")
        return True
    
    # Count files
    file_count = sum(1 for _ in screenshots_path.rglob('*') if _.is_file())
    
    if file_count == 0:
        logger.info(f"\nℹ️  Screenshots directory is already empty: {screenshots_path}")
        return True
    
    if not skip_confirmation:
        logger.warning(f"\n⚠️  WARNING: This will DELETE all screenshots!")
        logger.warning(f"   Path: {screenshots_path}")
        logger.warning(f"   Files: {file_count:,}")
        
        if not confirm_action("Are you sure you want to delete all screenshots?"):
            logger.info("❌ Screenshots cleanup cancelled.")
            return False
    
    logger.info(f"\n🗑️  Deleting screenshots: {screenshots_path}")
    
    try:
        # Delete all rollout_* subdirectories
        for rollout_dir in screenshots_path.glob("rollout_*"):
            if rollout_dir.is_dir():
                shutil.rmtree(rollout_dir)
        
        logger.info("✅ Screenshots deleted successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error cleaning screenshots: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clean all training data and logs for a fresh restart",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (asks for confirmation)
  python -m tinker_cookbook.recipes.cua_rl.clean_and_restart
  
  # Non-interactive mode (use with caution!)
  python -m tinker_cookbook.recipes.cua_rl.clean_and_restart --yes
  
  # Only clean database
  python -m tinker_cookbook.recipes.cua_rl.clean_and_restart --only-database --yes
  
  # Only clean logs
  python -m tinker_cookbook.recipes.cua_rl.clean_and_restart --only-logs --yes
"""
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts (use with caution!)"
    )
    parser.add_argument(
        "--only-database",
        action="store_true",
        help="Only clean database (keep logs and screenshots)"
    )
    parser.add_argument(
        "--only-logs",
        action="store_true",
        help="Only clean logs and screenshots (keep database)"
    )
    
    args = parser.parse_args()
    
    # Show banner
    logger.info("\n" + "=" * 80)
    logger.info("CUA RL TRAINING - CLEAN AND RESTART")
    logger.info("=" * 80)
    
    if args.yes:
        logger.warning("\n⚠️  Running in NON-INTERACTIVE mode (--yes flag)")
        logger.warning("   All confirmations will be skipped!")
    
    # Determine what to clean
    clean_db = not args.only_logs
    clean_logs_and_screenshots = not args.only_database
    
    success = True
    
    # Clean database
    if clean_db:
        try:
            init_database()
            with get_session() as session:
                if not clean_database(session, skip_confirmation=args.yes):
                    success = False
        except Exception as e:
            logger.error(f"\n❌ Failed to clean database: {e}")
            success = False
    
    # Clean logs and screenshots
    if clean_logs_and_screenshots:
        if not clean_logs_directory(skip_confirmation=args.yes):
            success = False
        
        if not clean_screenshots(skip_confirmation=args.yes):
            success = False
    
    # Final summary
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ CLEANUP COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("\nYou can now start fresh training with:")
        logger.info("  python -m tinker_cookbook.recipes.cua_rl.train \\")
        logger.info("    --model_name Qwen/Qwen3-VL-30B-A3B-Instruct \\")
        logger.info("    --tasks '{\"source_type\": \"demo_training\"}' \\")
        logger.info("    --group_size 4 \\")
        logger.info("    --groups_per_batch 2")
    else:
        logger.error("❌ CLEANUP FAILED!")
        logger.error("=" * 80)
        logger.error("\nPlease check the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

