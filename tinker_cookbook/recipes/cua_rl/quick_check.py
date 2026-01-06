#!/usr/bin/env python3
"""
Quick check of recent rollouts and turns to see if mismatches still exist.
"""

from sqlalchemy import desc, func

from tinker_cookbook.recipes.cua_rl.database.database import init_database, get_session
from tinker_cookbook.recipes.cua_rl.database.database_models import Turn, Rollout, Task

def quick_check():
    print("\n" + "=" * 100)
    print("QUICK DATABASE CHECK - Recent Rollouts & Turns")
    print("=" * 100)
    
    init_database()
    
    with get_session() as session:
        # Get recent rollouts with their tasks
        recent_rollouts = (
            session.query(Rollout, Task)
            .join(Task, Rollout.task_id == Task.id)
            .order_by(desc(Rollout.created_at))
            .limit(10)
            .all()
        )
        
        if not recent_rollouts:
            print("\n❌ No rollouts found in database.")
            return
        
        print(f"\n📊 Found {len(recent_rollouts)} recent rollouts")
        
        issues_found = 0
        
        for rollout, task in recent_rollouts:
            # Get turns for this rollout
            turns = (
                session.query(Turn)
                .filter(Turn.rollout_id == rollout.id)
                .order_by(Turn.turn)
                .all()
            )
            
            if not turns:
                continue
            
            print(f"\n{'─' * 100}")
            print(f"Rollout UUID: {rollout.rollout_id}")
            print(f"Task ID: {task.task_id}")
            print(f"Task: {task.description[:80]}...")
            print(f"Turns: {len(turns)}")
            
            for turn in turns[:3]:  # Check first 3 turns
                if not turn.model_response:
                    print(f"  Turn {turn.turn}: (no model_response)")
                    continue
                
                response_preview = turn.model_response[:100].replace('\n', ' ')
                print(f"  Turn {turn.turn}: {response_preview}...")
                
                # Simple keyword check
                task_lower = task.description.lower()
                response_lower = turn.model_response.lower()
                
                # Check for obvious mismatches
                if "cancel" in task_lower and "search" in response_lower and "cancel" not in response_lower:
                    print(f"    ⚠️  MISMATCH: Task has 'cancel' but response has 'search'")
                    issues_found += 1
                elif "search" in task_lower and "cancel" in response_lower and "search" not in response_lower:
                    print(f"    ⚠️  MISMATCH: Task has 'search' but response has 'cancel'")
                    issues_found += 1
                elif "double-book" in task_lower and "search" in response_lower:
                    print(f"    ⚠️  MISMATCH: Task has 'double-book' but response has 'search'")
                    issues_found += 1
        
        print(f"\n{'=' * 100}")
        if issues_found == 0:
            print("✅ NO OBVIOUS MISMATCHES FOUND")
        else:
            print(f"⚠️  FOUND {issues_found} POTENTIAL MISMATCHES")
        print("=" * 100)

if __name__ == "__main__":
    quick_check()

