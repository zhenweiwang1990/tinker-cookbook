#!/usr/bin/env python3
"""Direct database query to check rollout 94 data consistency."""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tinker_cookbook.recipes.cua_rl.database import get_session_direct, init_database
import json
import os

def check_rollout_94():
    """Check rollout 94 data directly from database."""
    # Initialize database connection
    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # Construct from individual environment variables
        postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port = os.getenv("POSTGRES_PORT", "5432")
        postgres_db = os.getenv("POSTGRES_DB", "training_db")
        postgres_user = os.getenv("POSTGRES_USER", "training_user")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "")
        db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    
    init_database(db_url=db_url)
    session = get_session_direct()
    
    try:
        from tinker_cookbook.recipes.cua_rl.database_models import Rollout, Turn, Task, Observation
        
        # Get rollout 94 by DB ID
        rollout = session.query(Rollout).filter(Rollout.id == 94).first()
        
        if not rollout:
            print(f"❌ Rollout with DB ID=94 not found!")
            return
        
        print(f"📊 Rollout Details:")
        print(f"   DB ID: {rollout.id}")
        print(f"   Rollout ID (UUID): {rollout.rollout_id}")
        print(f"   Status: {rollout.status}")
        print(f"   Task ID (FK): {rollout.task_id}")
        print(f"   Source Type: {rollout.source_type}")
        print(f"   Baseline ID: {rollout.baseline_id}")
        
        # Get task
        task = session.query(Task).filter(Task.id == rollout.task_id).first()
        
        if task:
            print(f"\n📝 Task Details:")
            print(f"   DB ID: {task.id}")
            print(f"   Task ID: {task.task_id}")
            print(f"   Task Description (first 500 chars):\n{task.description[:500]}")
            print(f"\n   Full Task Description:\n{task.description}")
        else:
            print(f"\n❌ Task with ID {rollout.task_id} not found!")
            return
        
        # Get turns
        turns = session.query(Turn).filter(Turn.rollout_id == rollout.id).order_by(Turn.turn).all()
        print(f"\n🔄 Turns ({len(turns)} total):")
        
        for idx, turn in enumerate(turns[:3], 1):  # Show first 3 turns
            print(f"\n   {'='*80}")
            print(f"   Turn {turn.turn} (DB ID: {turn.id}, rollout_id FK: {turn.rollout_id}):")
            print(f"      Start Time: {turn.start_time}")
            print(f"      End Time: {turn.end_time}")
            
            # Verify turn belongs to correct rollout
            if turn.rollout_id != rollout.id:
                print(f"      ❌ ERROR: Turn has rollout_id={turn.rollout_id}, but should be {rollout.id}!")
            
            # Check model response
            if turn.model_response:
                response_preview = turn.model_response[:500].replace('\n', ' ')
                print(f"\n      Model Response (first 500 chars):\n      {response_preview}")
                
                # Check for contradictions
                task_desc_lower = task.description.lower()
                response_lower = turn.model_response.lower()
                
                contradictions = []
                if "没有预算限制" in task.description or "no budget limit" in task_desc_lower or "无预算限制" in task.description:
                    if "最便宜" in response_lower or "cheapest" in response_lower:
                        contradictions.append("Task says 'no budget limit' but response says 'cheapest'")
                if "预算" in task.description and ("不限" in task.description or "no limit" in task_desc_lower):
                    if "便宜" in response_lower or "cheap" in response_lower:
                        contradictions.append("Task says unlimited budget but response mentions cheap/cheapest")
                
                if contradictions:
                    print(f"\n      ⚠️  CONTRADICTIONS FOUND:")
                    for c in contradictions:
                        print(f"         - {c}")
            
            # Get observations for this turn
            observations = session.query(Observation).filter(Observation.turn_id == turn.id).all()
            print(f"\n      Observations ({len(observations)} total):")
            for obs in observations:
                print(f"         - Type: {obs.obs_type}, Screenshot: {'Yes' if obs.screenshot_uri else 'No'}, Model Input: {'Yes' if obs.model_input_json else 'No'}")
                
                # Try to parse model_input_json to check if it contains task description
                if obs.model_input_json:
                    try:
                        model_input = json.loads(obs.model_input_json) if isinstance(obs.model_input_json, str) else obs.model_input_json
                        model_input_str = json.dumps(model_input, ensure_ascii=False)
                        if task.task_id in model_input_str:
                            print(f"           ✓ Model input contains task_id '{task.task_id}'")
                        else:
                            print(f"           ⚠️  Model input does NOT contain task_id '{task.task_id}'")
                        # Check if model input mentions budget/cheapest
                        model_input_lower = model_input_str.lower()
                        if "cheapest" in model_input_lower or "最便宜" in model_input_str:
                            print(f"           ⚠️  Model input mentions 'cheapest/最便宜'")
                        if "no budget" in model_input_lower or "无预算" in model_input_str or "没有预算限制" in model_input_str:
                            print(f"           ✓ Model input mentions 'no budget/无预算'")
                    except Exception as e:
                        print(f"           ⚠️  Failed to parse model_input_json: {e}")
        
        print(f"\n" + "="*80)
        print(f"SUMMARY:")
        print(f"  Rollout {rollout.id} (rollout_id={rollout.rollout_id})")
        print(f"  Task: {task.task_id if task else 'NOT FOUND'} (DB ID: {rollout.task_id})")
        print(f"  Task Description: {task.description[:200] if task else 'N/A'}...")
        if turns:
            first_turn = turns[0]
            print(f"  First Turn Response: {first_turn.model_response[:200] if first_turn.model_response else 'N/A'}...")
            
            # Final check
            if task and first_turn.model_response:
                task_has_no_budget = "没有预算限制" in task.description or "no budget limit" in task.description.lower()
                response_has_cheapest = "最便宜" in first_turn.model_response or "cheapest" in first_turn.model_response.lower()
                if task_has_no_budget and response_has_cheapest:
                    print(f"\n  ❌ DATA MISMATCH CONFIRMED:")
                    print(f"     Task says NO BUDGET LIMIT, but first turn response mentions CHEAPEST")
                    print(f"     This indicates Turn may belong to wrong Task/Rollout!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    check_rollout_94()

