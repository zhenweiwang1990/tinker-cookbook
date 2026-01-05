#!/usr/bin/env python3
"""
Diagnostic script to check rollout data integrity.
Usage: python -m tinker_cookbook.recipes.cua_rl.check_rollout_data <rollout_id>
"""

import sys
from tinker_cookbook.recipes.cua_rl.database import get_session
from tinker_cookbook.recipes.cua_rl.database_models import Rollout, Turn, Task

def check_rollout(rollout_id: int):
    """Check rollout data integrity."""
    session = get_session()
    try:
        # Get rollout
        rollout = session.query(Rollout).filter(Rollout.id == rollout_id).first()
        if not rollout:
            print(f"Rollout {rollout_id} not found")
            return
        
        print(f"Rollout {rollout_id}:")
        print(f"  rollout_id (UUID): {rollout.rollout_id}")
        print(f"  task_id: {rollout.task_id}")
        
        # Get task
        task = session.query(Task).filter(Task.id == rollout.task_id).first()
        if task:
            print(f"  Task: id={task.id}, task_id='{task.task_id}', description='{task.description[:100] if task.description else None}...'")
        else:
            print(f"  Task {rollout.task_id} not found!")
        
        # Get turns
        turns = session.query(Turn).filter(Turn.rollout_id == rollout.id).order_by(Turn.turn).all()
        print(f"  Turns: {len(turns)}")
        
        for turn in turns[:3]:  # Show first 3 turns
            print(f"    Turn {turn.turn}: id={turn.id}, rollout_id={turn.rollout_id}")
            if turn.model_response:
                preview = turn.model_response[:150].replace('\n', ' ')
                print(f"      Model response preview: {preview}...")
        
        # Check if any turns belong to wrong rollout
        for turn in turns:
            if turn.rollout_id != rollout.id:
                print(f"  ERROR: Turn {turn.id} has rollout_id={turn.rollout_id}, but should be {rollout.id}!")
        
    finally:
        session.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tinker_cookbook.recipes.cua_rl.check_rollout_data <rollout_id>")
        sys.exit(1)
    
    rollout_id = int(sys.argv[1])
    check_rollout(rollout_id)

