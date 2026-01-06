#!/usr/bin/env python3
"""
Script to migrate tinker_cua_agent.py from db_session to rollout_recorder.

This script automates the tedious manual replacement of database calls.
"""

import re
import sys

def migrate_agent_file(input_file: str, output_file: str):
    """Migrate agent file from db_session to rollout_recorder."""
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Step 1: Replace record_turn_start blocks
    # This is complex because we need to replace the entire try-except block
    pattern_turn_start = r'''                # Record turn start in database
                if self\.rollout_recorder is not None:
                    try:
                        turn_id = self\.rollout_recorder\.start_turn\(turn\)
                        if not turn_id:
                            logger\.error\(f"\[Turn \{turn\}\] Failed to start turn in database"\)
                    except Exception as e:
                        logger\.warning\(f"\[Turn \{turn\}\] Failed to record turn start in database: \{e\}", exc_info=True\)'''
    
    # Step 2: Replace all remaining db_session checks with rollout_recorder checks
    # Pattern: if self.db_session is not None and self.rollout_id is not None:
    # Replace with: if self.rollout_recorder is not None:
    content = re.sub(
        r'if self\.db_session is not None and self\.rollout_id is not None:',
        r'if self.rollout_recorder is not None:  # Simplified DB check',
        content
    )
    
    # Pattern: if self.db_session is not None:
    # Replace with: if self.rollout_recorder is not None:
    content = re.sub(
        r'if self\.db_session is not None:',
        r'if self.rollout_recorder is not None:  # Simplified DB check',
        content
    )
    
    # Step 3: Comment out all direct database recording calls
    # These are:
    # - record_turn()
    # - record_action()
    # - record_observation()
    # - record_rollout_status()
    
    database_recording_funcs = [
        'record_turn',
        'record_action',
        'record_observation',
        'record_rollout_status',
    ]
    
    for func in database_recording_funcs:
        # Find all lines that call these functions and comment them out
        # Pattern: Any line containing func(...) but not in a comment already
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if f'{func}(' in line and not line.strip().startswith('#'):
                # Check if it's an import statement
                if 'import' in line and func in line:
                    # Keep import but comment it
                    new_lines.append(f"                        # DEPRECATED: {line.strip()}")
                elif func == 'record_turn' and 'end_turn' not in line:
                    # This is a record_turn() call - comment it out
                    # Add a TODO comment
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + f"# TODO: Migrate to rollout_recorder.end_turn()")
                    new_lines.append(' ' * indent + f"# {line.strip()}")
                elif func in ['record_action', 'record_observation']:
                    # These are detail recordings - comment them out
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(' ' * indent + f"# DETAIL: Action/Observation recording (optional)")
                    new_lines.append(' ' * indent + f"# {line.strip()}")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)
    
    # Step 4: Replace self.db_session.commit() with pass
    content = re.sub(
        r'(\s+)self\.db_session\.commit\(\)',
        r'\1pass  # DB commit handled by rollout_recorder',
        content
    )
    
    # Step 5: Replace self.db_session.rollback() with pass
    content = re.sub(
        r'(\s+)self\.db_session\.rollback\(\)',
        r'\1pass  # DB rollback handled by rollout_recorder',
        content
    )
    
    # Write output
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Migrated {input_file} -> {output_file}")
    print(f"  - Replaced db_session checks with rollout_recorder checks")
    print(f"  - Commented out direct database recording calls")
    print(f"  - Simplified commit/rollback logic")
    print()
    print("⚠️  IMPORTANT: Manual review required!")
    print("    - Check all 'TODO:' comments")
    print("    - Verify turn recording logic")
    print("    - Test the changes")

if __name__ == '__main__':
    input_file = '/Users/zhenwei/workspace/tinker-cookbook/tinker_cookbook/recipes/cua_rl/agent/tinker_cua_agent.py'
    output_file = input_file  # Overwrite in place
    
    if '--dry-run' in sys.argv:
        output_file = input_file + '.migrated'
        print(f"DRY RUN: Output will be written to {output_file}")
    
    migrate_agent_file(input_file, output_file)

