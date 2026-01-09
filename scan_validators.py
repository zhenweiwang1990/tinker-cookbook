#!/usr/bin/env python3
"""Scan all validator files to extract shell commands they use."""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_shell_commands(file_path: Path) -> list[str]:
    """Extract shell commands from a validator file."""
    commands = []
    try:
        content = file_path.read_text()
        
        # Pattern 1: adb_client._run("shell", "command", ...)
        pattern1 = r'adb_client\._run\(["\']shell["\']\s*,\s*["\']([^"\']+)["\']'
        commands.extend(re.findall(pattern1, content))
        
        # Pattern 2: Direct string assignments that look like shell commands
        # query = "settings get ..."
        pattern2 = r'query\s*=\s*["\']([^"\']+)["\']'
        commands.extend(re.findall(pattern2, content))
        
        # Pattern 3: command = "..."
        pattern3 = r'command\s*=\s*["\']([^"\']+)["\']'
        for cmd in re.findall(pattern3, content):
            if any(keyword in cmd for keyword in ['settings', 'dumpsys', 'getprop', 'service', 'pm', 'am']):
                commands.append(cmd)
        
        # Pattern 4: Direct calls like .run("shell", f"...")
        pattern4 = r'\.run\(["\']shell["\']\s*,\s*f?["\']([^"\']+)["\']'
        commands.extend(re.findall(pattern4, content))
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return commands

def main():
    tasks_dir = Path("/Users/zhenwei/workspace/tinker-cookbook/tinker_cookbook/recipes/cua_rl/tasks")
    
    # Find all validator.py files
    validator_files = list(tasks_dir.rglob("**/validator.py"))
    
    print(f"Found {len(validator_files)} validator files")
    print("=" * 80)
    print()
    
    # Group by command type
    command_types = defaultdict(list)
    all_commands = defaultdict(list)
    
    for validator_file in sorted(validator_files):
        commands = extract_shell_commands(validator_file)
        if commands:
            rel_path = validator_file.relative_to(tasks_dir)
            task_name = rel_path.parent
            
            for cmd in commands:
                # Categorize command
                if cmd.startswith('settings ') or cmd.startswith('/system/bin/settings'):
                    cmd_type = "settings"
                elif cmd.startswith('dumpsys ') or cmd.startswith('/system/bin/dumpsys'):
                    cmd_type = "dumpsys"
                elif cmd.startswith('getprop') or 'getprop' in cmd:
                    cmd_type = "getprop"
                elif cmd.startswith('service '):
                    cmd_type = "service"
                elif cmd.startswith('pm '):
                    cmd_type = "pm (package manager)"
                elif cmd.startswith('am '):
                    cmd_type = "am (activity manager)"
                elif 'ls ' in cmd or 'cat ' in cmd or 'grep ' in cmd:
                    cmd_type = "file system"
                else:
                    cmd_type = "other"
                
                command_types[cmd_type].append((task_name, cmd))
                all_commands[str(task_name)].append(cmd)
    
    # Print by command type
    print("\n" + "=" * 80)
    print("COMMANDS BY TYPE")
    print("=" * 80)
    
    # Test results from our successful run
    working_commands = {
        'settings': '✓ WORKS',
        'dumpsys': '✓ WORKS', 
        'getprop': '✓ WORKS',
        'service': '✓ WORKS',
        'pm (package manager)': '❓ UNTESTED',
        'am (activity manager)': '❓ UNTESTED',
        'file system': '✓ WORKS (ls)',
        'other': '❓ UNKNOWN'
    }
    
    for cmd_type in sorted(command_types.keys()):
        status = working_commands.get(cmd_type, '❓ UNKNOWN')
        print(f"\n{status} {cmd_type.upper()}")
        print("-" * 80)
        tasks = command_types[cmd_type]
        for task_name, cmd in sorted(set(tasks)):
            print(f"  {task_name}")
            print(f"    → {cmd}")
    
    # Print summary by task
    print("\n" + "=" * 80)
    print("COMMANDS BY TASK")
    print("=" * 80)
    
    for task_name in sorted(all_commands.keys()):
        commands = all_commands[task_name]
        print(f"\n{task_name} ({len(commands)} commands)")
        for cmd in commands:
            print(f"  → {cmd}")
    
    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Based on test results (all 19 commands succeeded):")
    print()
    print("✅ WORKING COMMANDS:")
    print("  • settings get/list/put - All settings commands work")
    print("  • dumpsys battery/power - All dumpsys commands work")
    print("  • getprop - Property queries work")
    print("  • service list/check - Service commands work")
    print("  • File system commands (ls, which) work")
    print()
    print("🔧 FIXES NEEDED:")
    print("  • Remove /system/bin/ prefix from commands (not needed)")
    print("  • Simplify validators to use direct commands")
    print("  • Add proper exception handling for all validators")
    print()
    print("📝 KEY FINDING for Battery Saver:")
    print("  • Use: settings get global low_power")
    print("  • Returns: 0 (off) or 1 (on)")
    print("  • This is the simplest and most reliable method!")

if __name__ == "__main__":
    main()

