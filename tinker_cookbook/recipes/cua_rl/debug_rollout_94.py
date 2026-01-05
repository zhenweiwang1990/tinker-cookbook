#!/usr/bin/env python3
"""Debug script to check rollout 94 data consistency."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tinker_cookbook.recipes.cua_rl.database import get_session
from tinker_cookbook.recipes.cua_rl.database_dao import get_rollout_by_rollout_id
from sqlalchemy.orm import Session
import json

def debug_rollout(session: Session, rollout_id: str):
    """Debug a specific rollout by rollout_id."""
    rollout = get_rollout_by_rollout_id(session, rollout_id)
    
    if not rollout:
        print(f"❌ Rollout with rollout_id='{rollout_id}' not found!")
        return
    
    print(f"📊 Rollout Details:")
    print(f"   DB ID: {rollout.id}")
    print(f"   Rollout ID (UUID): {rollout.rollout_id}")
    print(f"   Status: {rollout.status}")
    print(f"   Task ID (FK): {rollout.task_id}")
    print(f"   Source Type: {rollout.source_type}")
    
    # Get task
    from tinker_cookbook.recipes.cua_rl.database_models import Task
    task = session.query(Task).filter(Task.id == rollout.task_id).first()
    
    if task:
        print(f"\n📝 Task Details:")
        print(f"   DB ID: {task.id}")
        print(f"   Task ID: {task.task_id}")
        print(f"   Task Description (first 200 chars): {task.description[:200]}...")
    else:
        print(f"\n❌ Task with ID {rollout.task_id} not found!")
    
    # Get turns
    print(f"\n🔄 Turns ({len(rollout.turns)} total):")
    for idx, turn in enumerate(sorted(rollout.turns, key=lambda t: t.turn), 1):
        print(f"\n   Turn {turn.turn} (DB ID: {turn.id}):")
        print(f"      Start Time: {turn.start_time}")
        
        # Get observations
        screenshot_obs = [obs for obs in turn.observations if obs.obs_type == 'screenshot_before']
        if screenshot_obs:
            print(f"      Screenshot Before: {len(screenshot_obs)} observation(s)")
            for obs in screenshot_obs:
                if obs.model_input_json:
                    try:
                        model_input = json.loads(obs.model_input_json) if isinstance(obs.model_input_json, str) else obs.model_input_json
                        print(f"         Model Input present: Yes (type: {type(model_input).__name__})")
                    except:
                        print(f"         Model Input present: Yes (raw string, {len(str(obs.model_input_json))} chars)")
                else:
                    print(f"         Model Input present: No")
        
        # Check model response
        if turn.model_response:
            response_preview = turn.model_response[:200].replace('\n', ' ')
            print(f"      Model Response (first 200 chars): {response_preview}...")
            
            # Check if response matches task description
            if task and task.description:
                task_lower = task.description.lower()
                response_lower = turn.model_response.lower()
                
                # Check for contradictions
                if "预算" in task.description or "budget" in task_lower:
                    if "便宜" in response_lower or "cheap" in response_lower or "cheapest" in response_lower:
                        if "无预算" in task.description or "no budget" in task_lower or "不限预算" in task.description:
                            print(f"      ⚠️  WARNING: Task says no budget limit, but response mentions cheapest/cheap!")
                if "没有预算限制" in task.description or "no budget limit" in task_lower:
                    if "最便宜" in response_lower or "cheapest" in response_lower:
                        print(f"      ❌ ERROR: Task says 'no budget limit' but response says 'cheapest'!")
        
        # Get actions
        if turn.actions:
            print(f"      Actions: {len(turn.actions)}")
            for action in turn.actions[:3]:  # Show first 3
                print(f"         - {action.action_type}: {action.tool_name}({action.tool_args[:100] if action.tool_args else 'N/A'}...)")
    
    print(f"\n" + "="*80)
    print(f"SUMMARY:")
    print(f"  Rollout {rollout.id} (rollout_id={rollout.rollout_id})")
    print(f"  Task: {task.task_id if task else 'NOT FOUND'} (DB ID: {rollout.task_id})")
    print(f"  Turns: {len(rollout.turns)}")
    if task:
        print(f"  Task Description Preview: {task.description[:100]}...")
        print(f"  First Turn Response Preview: {rollout.turns[0].model_response[:100] if rollout.turns and rollout.turns[0].model_response else 'N/A'}...")

if __name__ == "__main__":
    # Check rollout 94 - first try as integer, then as UUID string
    with get_session() as session:
        try:
            # Try to find rollout with rollout_id = '94' or ID = 94
            from tinker_cookbook.recipes.cua_rl.database_models import Rollout
            rollout_by_id = session.query(Rollout).filter(Rollout.id == 94).first()
            if rollout_by_id:
                print(f"Found rollout by DB ID=94, rollout_id={rollout_by_id.rollout_id}")
                debug_rollout(session, rollout_by_id.rollout_id)
            else:
                # Try as UUID string
                debug_rollout(session, '94')
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

