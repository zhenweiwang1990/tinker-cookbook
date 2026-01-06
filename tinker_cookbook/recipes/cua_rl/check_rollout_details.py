#!/usr/bin/env python3
"""Quick check of rollout details in database"""

import sys
from tinker_cookbook.recipes.cua_rl.database.database import init_database, get_session
from tinker_cookbook.recipes.cua_rl.database.database_models import (
    Rollout, Turn, Action, Observation, Task
)

def check_rollout_details():
    """Check if rollout details are being saved correctly"""
    init_database()
    
    with get_session() as session:
        # Get the most recent rollout
        recent_rollout = session.query(Rollout).order_by(Rollout.created_at.desc()).first()
        
        if not recent_rollout:
            print("❌ No rollouts found in database")
            return
        
        print(f"\n{'='*80}")
        print(f"Most Recent Rollout: {recent_rollout.rollout_id}")
        print(f"{'='*80}")
        
        # Get task
        task = session.query(Task).filter(Task.id == recent_rollout.task_id).first()
        if task:
            print(f"\n✓ Task found:")
            print(f"  ID: {task.task_id}")
            print(f"  Description: {task.description[:100]}...")
        else:
            print(f"\n❌ Task not found (task_id={recent_rollout.task_id})")
        
        # Get turns
        turns = session.query(Turn).filter(
            Turn.rollout_id == recent_rollout.id
        ).order_by(Turn.turn).all()
        
        print(f"\n{'='*80}")
        print(f"Turns: {len(turns)}")
        print(f"{'='*80}")
        
        for turn in turns:
            print(f"\n--- Turn {turn.turn} (DB ID: {turn.id}) ---")
            print(f"  Reward: {turn.reward}")
            print(f"  Episode done: {turn.episode_done}")
            
            # Model response
            if turn.model_response:
                print(f"  ✓ Model response: {turn.model_response[:100]}...")
            else:
                print(f"  ❌ No model response")
            
            # Check actions
            actions = session.query(Action).filter(
                Action.turn_id == turn.id
            ).all()
            print(f"  Actions: {len(actions)}")
            for action in actions:
                print(f"    - Type: {action.action_type}, Target: {action.target_description[:50] if action.target_description else 'None'}...")
                if action.screenshot_before_path:
                    print(f"      Screenshot before: {action.screenshot_before_path}")
                else:
                    print(f"      ❌ No screenshot_before_path")
            
            # Check observations
            observations = session.query(Observation).filter(
                Observation.turn_id == turn.id
            ).all()
            print(f"  Observations: {len(observations)}")
            for obs in observations:
                print(f"    - Obs {obs.id} | type={obs.obs_type}")
                if obs.screenshot_uri:
                    print(f"      Screenshot URI: {obs.screenshot_uri[:120]}...")
                else:
                    print(f"      ❌ No screenshot_uri")
                if obs.text_content:
                    print(f"      Text content: {obs.text_content[:120]}...")
                else:
                    print(f"      (no text_content)")
                if obs.model_input_json:
                    print(f"      ✓ model_input_json present (len={len(obs.model_input_json)})")
                else:
                    print(f"      ❌ No model_input_json")
        
        print(f"\n{'='*80}")
        print("\nChecking all recent rollouts (last 5):")
        print(f"{'='*80}")
        
        recent_rollouts = session.query(Rollout).order_by(
            Rollout.created_at.desc()
        ).limit(5).all()
        
        for rollout in recent_rollouts:
            turn_count = session.query(Turn).filter(
                Turn.rollout_id == rollout.id
            ).count()
            action_count = session.query(Action).join(Turn).filter(
                Turn.rollout_id == rollout.id
            ).count()
            obs_count = session.query(Observation).join(Turn).filter(
                Turn.rollout_id == rollout.id
            ).count()
            
            print(f"\nRollout: {rollout.rollout_id}")
            print(f"  Created: {rollout.created_at}")
            print(f"  Turns: {turn_count}, Actions: {action_count}, Observations: {obs_count}")

if __name__ == "__main__":
    check_rollout_details()

