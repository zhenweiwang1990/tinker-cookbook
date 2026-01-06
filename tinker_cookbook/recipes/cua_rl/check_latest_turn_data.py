#!/usr/bin/env python3
"""
Check the latest turn data to see if metrics_json and action data are being recorded correctly.
"""

import json
from sqlalchemy import desc
from tinker_cookbook.recipes.cua_rl.database.database import init_database
from tinker_cookbook.recipes.cua_rl.database.database_dao import get_session
from tinker_cookbook.recipes.cua_rl.database.database_models import Turn, Action, Rollout

def main():
    # Initialize database
    init_database()
    
    with get_session() as session:
        # Get the most recent rollout
        latest_rollout = session.query(Rollout).order_by(desc(Rollout.created_at)).first()
        
        if not latest_rollout:
            print("No rollouts found in database")
            return
        
        print(f"Latest Rollout ID: {latest_rollout.id}")
        print(f"Created at: {latest_rollout.created_at}")
        print(f"Status: {latest_rollout.status}")
        print()
        
        # Get all turns for this rollout
        turns = session.query(Turn).filter(Turn.rollout_id == latest_rollout.id).order_by(Turn.id).all()
        
        print(f"Found {len(turns)} turns for this rollout")
        print()
        
        for turn in turns[:3]:  # Check first 3 turns
            print(f"=" * 80)
            print(f"Turn ID: {turn.id}")
            print(f"-" * 80)
            
            # Check metrics_json
            if turn.metrics_json:
                if isinstance(turn.metrics_json, str):
                    metrics = json.loads(turn.metrics_json)
                else:
                    metrics = turn.metrics_json
                
                print("✓ metrics_json exists")
                if 'stage_timings' in metrics:
                    print("  ✓ stage_timings found:")
                    stage_timings = metrics['stage_timings']
                    for key, value in stage_timings.items():
                        print(f"    - {key}: {value}s")
                else:
                    print("  ✗ stage_timings NOT found in metrics_json")
                    print(f"  Available keys: {list(metrics.keys())}")
            else:
                print("✗ metrics_json is NULL")
            
            print()
            
            # Check action
            action = session.query(Action).filter(Action.turn_id == turn.id).first()
            if action:
                print("✓ Action exists")
                print(f"  - action_type: {action.action_type}")
                print(f"  - num_tokens: {action.num_tokens}")
                print(f"  - coordinates: {action.coordinates}")
            else:
                print("✗ No action found for this turn")
            
            print()

if __name__ == "__main__":
    main()

