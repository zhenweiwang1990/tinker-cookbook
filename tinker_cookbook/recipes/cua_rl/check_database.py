#!/usr/bin/env python3
"""
Check database for Turn-Task mismatches in current training data.

This script queries the database and checks if Turn records have
model_response that matches their associated Task.
"""

import logging
from datetime import datetime

from sqlalchemy.orm import joinedload

from tinker_cookbook.recipes.cua_rl.database.database import init_database, get_session
from tinker_cookbook.recipes.cua_rl.database.database_models import Turn, Rollout, Task, Training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_database():
    """Check database for Turn-Task consistency."""
    
    logger.info("=" * 100)
    logger.info("DATABASE CONSISTENCY CHECK")
    logger.info("=" * 100)
    
    with get_session() as session:
        # Get latest training session
        latest_training = (
            session.query(Training)
            .order_by(Training.created_at.desc())
            .first()
        )
        
        if not latest_training:
            logger.info("\n❌ No training sessions found in database.")
            return
        
        logger.info(f"\n📊 Latest Training Session:")
        logger.info(f"   ID: {latest_training.id}")
        logger.info(f"   Run Name: {latest_training.run_name}")
        logger.info(f"   Model: {latest_training.model_name}")
        logger.info(f"   Status: {latest_training.status}")
        logger.info(f"   Created: {latest_training.created_at}")
        
        # Get all rollouts for this training
        rollouts = (
            session.query(Rollout)
            .filter(
                (Rollout.step_id.in_(
                    session.query(Training.id).filter(Training.id == latest_training.id)
                )) |
                (Rollout.eval_id.in_(
                    session.query(Training.id).filter(Training.id == latest_training.id)
                )) |
                (Rollout.baseline_id.in_(
                    session.query(Training.id).filter(Training.id == latest_training.id)
                ))
            )
            .options(joinedload(Rollout.task))
            .all()
        )
        
        if not rollouts:
            logger.info("\n❌ No rollouts found for this training session.")
            return
        
        logger.info(f"\n📦 Total Rollouts: {len(rollouts)}")
        
        # Check each rollout's turns
        logger.info("\n" + "=" * 100)
        logger.info("CHECKING ROLLOUT-TASK-TURN CONSISTENCY")
        logger.info("=" * 100)
        
        issues_found = 0
        
        for rollout in rollouts[:10]:  # Check first 10 rollouts
            task = rollout.task
            
            # Get turns for this rollout
            turns = (
                session.query(Turn)
                .filter(Turn.rollout_id == rollout.id)
                .order_by(Turn.turn)
                .all()
            )
            
            if not turns:
                continue
            
            logger.info(f"\n{'─' * 100}")
            logger.info(f"🔍 Rollout ID: {rollout.rollout_id} (DB ID: {rollout.id})")
            logger.info(f"   Task ID: {task.task_id}")
            logger.info(f"   Task Description: {task.description[:100]}...")
            logger.info(f"   Number of Turns: {len(turns)}")
            
            for turn in turns:
                if not turn.model_response:
                    continue
                
                # Check if model_response contains keywords that seem inconsistent with task
                task_lower = task.description.lower()
                response_lower = turn.model_response.lower()
                
                # Look for obvious mismatches
                mismatch = False
                mismatch_reason = ""
                
                # Common mismatch patterns
                patterns = [
                    ("cancel", "search", "Task mentions 'cancel' but response mentions 'search'"),
                    ("search", "cancel", "Task mentions 'search' but response mentions 'cancel'"),
                    ("delete", "add", "Task mentions 'delete' but response mentions 'add'"),
                    ("book", "cancel", "Task mentions 'book' but response mentions 'cancel'"),
                    ("double-book", "search", "Task mentions 'double-book' but response mentions 'search'"),
                ]
                
                for task_keyword, response_keyword, reason in patterns:
                    if task_keyword in task_lower and response_keyword in response_lower and task_keyword not in response_lower:
                        mismatch = True
                        mismatch_reason = reason
                        break
                
                if mismatch:
                    issues_found += 1
                    logger.warning(f"\n   ⚠️  POTENTIAL MISMATCH in Turn {turn.turn}:")
                    logger.warning(f"      Reason: {mismatch_reason}")
                    logger.warning(f"      Model Response: {turn.model_response[:150]}...")
                    logger.warning(f"      Monitor URL: http://localhost:3001/{rollout.step_id or rollout.eval_id or rollout.baseline_id}/{rollout.source_type}/{rollout.group_id}?rollout={rollout.id}&turn={turn.turn}")
                else:
                    logger.info(f"   ✅ Turn {turn.turn}: Response matches task (first 100 chars: {turn.model_response[:100]}...)")
        
        logger.info(f"\n{'=' * 100}")
        if issues_found == 0:
            logger.info("✅ NO MISMATCHES FOUND! All Turns appear to match their Tasks.")
        else:
            logger.warning(f"⚠️  FOUND {issues_found} POTENTIAL MISMATCHES")
            logger.warning("   Please review the URLs above to verify the issues.")
        logger.info("=" * 100)


if __name__ == "__main__":
    init_database()
    check_database()

