"""
Fix Turn-Rollout-Task mismatches in the database.

This script identifies and reports Turn records where the model_response
does not match the associated Task, which can happen due to concurrent
session sharing issues.

Usage:
    python -m tinker_cookbook.recipes.cua_rl.database.fix_turn_mismatch [--dry-run]
"""

import argparse
import logging
from typing import List, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from tinker_cookbook.recipes.cua_rl.database.database import init_database, get_session
from tinker_cookbook.recipes.cua_rl.database.database_models import Turn, Rollout, Task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_mismatched_turns(session: Session) -> List[Tuple[Turn, Rollout, Task]]:
    """
    Find Turn records where model_response content doesn't match the task.
    
    This is a heuristic check - we look for keywords in model_response
    and check if they're related to the task description.
    
    Returns:
        List of (Turn, Rollout, Task) tuples that are potentially mismatched
    """
    mismatched = []
    
    # Query all Turns with their associated Rollouts and Tasks
    query = (
        session.query(Turn)
        .join(Rollout, Turn.rollout_id == Rollout.id)
        .join(Task, Rollout.task_id == Task.id)
        .options(
            joinedload(Turn.rollout).joinedload(Rollout.task)
        )
        .filter(Turn.model_response.isnot(None))
        .order_by(Rollout.id, Turn.turn)
    )
    
    total_turns = query.count()
    logger.info(f"Scanning {total_turns} turns for potential mismatches...")
    
    for turn in query:
        rollout = turn.rollout
        task = rollout.task
        
        if not turn.model_response or not task.description:
            continue
        
        # Simple heuristic: check if model_response contains keywords from task
        # This is not perfect but can catch obvious mismatches
        model_resp_lower = turn.model_response.lower()
        task_desc_lower = task.description.lower()
        
        # Extract key phrases from task description (words longer than 4 chars)
        task_keywords = set([
            word for word in task_desc_lower.split()
            if len(word) > 4 and word.isalpha()
        ])
        
        # Check for common mismatch patterns
        # Example: Task is about "cancel booking" but response is about "search accommodation"
        mismatch_patterns = [
            ("cancel", "search"),
            ("search", "cancel"),
            ("delete", "create"),
            ("create", "delete"),
            ("book", "cancel"),
            ("cancel", "book"),
        ]
        
        for pattern_a, pattern_b in mismatch_patterns:
            if pattern_a in task_desc_lower and pattern_b in model_resp_lower and pattern_a not in model_resp_lower:
                logger.warning(
                    f"Potential mismatch found:\n"
                    f"  Turn ID: {turn.id}\n"
                    f"  Rollout ID: {rollout.rollout_id}\n"
                    f"  Task ID: {task.task_id}\n"
                    f"  Task desc: {task.description[:100]}...\n"
                    f"  Model response: {turn.model_response[:100]}...\n"
                    f"  Pattern: Task has '{pattern_a}' but response has '{pattern_b}'\n"
                )
                mismatched.append((turn, rollout, task))
                break
    
    return mismatched


def report_mismatches(session: Session, dry_run: bool = True):
    """
    Generate a report of Turn-Task mismatches.
    
    Args:
        session: Database session
        dry_run: If True, only report issues without fixing them
    """
    logger.info("=" * 80)
    logger.info("Turn-Task Mismatch Detection Report")
    logger.info("=" * 80)
    
    mismatched = find_mismatched_turns(session)
    
    if not mismatched:
        logger.info("✓ No obvious mismatches found!")
        return
    
    logger.info(f"\n⚠ Found {len(mismatched)} potential mismatches:\n")
    
    for i, (turn, rollout, task) in enumerate(mismatched, 1):
        logger.info(f"\n[{i}] Mismatch Details:")
        logger.info(f"  Turn DB ID: {turn.id}")
        logger.info(f"  Turn Number: {turn.turn}")
        logger.info(f"  Rollout UUID: {rollout.rollout_id}")
        logger.info(f"  Task ID: {task.task_id}")
        logger.info(f"  Task Description: {task.description[:150]}")
        logger.info(f"  Model Response Preview: {turn.model_response[:150]}")
        logger.info(f"  Training Monitor URL: http://localhost:3001/{rollout.step_id or rollout.eval_id or rollout.baseline_id}/{rollout.source_type}/{rollout.group_id}?rollout={rollout.id}&turn={turn.turn}")
    
    if dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("Dry run mode - no changes made to database")
        logger.info("To investigate these issues, please:")
        logger.info("1. Review the URLs above in the Training Monitor")
        logger.info("2. Check the logs for these rollouts to understand what happened")
        logger.info("3. Consider whether the fix in rollout.py will prevent future issues")
        logger.info("=" * 80)
    else:
        logger.warning("\n⚠ Fix mode is not implemented yet.")
        logger.warning("Please manually investigate the mismatches above.")


def main():
    parser = argparse.ArgumentParser(description="Find and report Turn-Task mismatches")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Only report issues without fixing them (default: True)")
    args = parser.parse_args()
    
    # Initialize database
    init_database()
    
    # Run report
    with get_session() as session:
        report_mismatches(session, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

