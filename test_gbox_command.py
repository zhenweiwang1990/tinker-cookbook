#!/usr/bin/env python3
"""
Test script to debug battery saver validation in GBox Android environment.

This script creates a GBox, enables battery saver via UI simulation,
then tests various shell commands to find the correct validation method.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tinker_cookbook.recipes.cua_rl.gbox.client import CuaGBoxClient


async def test_battery_saver_commands():
    """Test various commands to check battery saver status."""
    
    # Get API key from environment
    api_key = os.environ.get("GBOX_API_KEY")
    if not api_key:
        print("ERROR: GBOX_API_KEY not set")
        sys.exit(1)
    
    print("=" * 80)
    print("GBox Battery Saver Validation Test")
    print("=" * 80)
    print()
    
    # Create GBox client
    print("[1/5] Creating GBox client...")
    client = CuaGBoxClient(api_key=api_key)
    
    try:
        # Create box
        print("[2/5] Creating Android box (this may take a while)...")
        await client.create_box(box_type="android")
        print(f"✓ Box created: {client.box_id}")
        print()
        
        # Wait for box to be ready
        print("[3/5] Waiting for box to be ready...")
        await asyncio.sleep(3)
        
        # Test available commands
        print("[4/5] Testing available commands...")
        print("-" * 80)
        
        commands_to_test = [
            # Package manager commands
            ("pm list packages", "List all installed packages"),
            ("pm list packages | grep -c com", "Count packages"),
            ("pm list packages | grep com.android.settings", "Check if Settings app exists"),
            
            # File system test commands  
            ("test -d /storage/emulated/0 && echo YES || echo NO", "Test if storage dir exists"),
            ("test -d /storage/emulated/0/Download && echo YES || echo NO", "Test if Download dir exists"),
            ("test -d /storage/emulated/0/Documents && echo YES || echo NO", "Test if Documents dir exists"),
            ("test -e /system/bin/settings && echo YES || echo NO", "Test if settings file exists"),
            
            # Create and test a folder
            ("mkdir -p /storage/emulated/0/Download/test_folder", "Create test folder"),
            ("test -d /storage/emulated/0/Download/test_folder && echo YES || echo NO", "Verify test folder created"),
            ("rmdir /storage/emulated/0/Download/test_folder", "Remove test folder"),
            ("test -d /storage/emulated/0/Download/test_folder && echo YES || echo NO", "Verify test folder removed"),
        ]
        
        print(f"\nTesting {len(commands_to_test)} commands...\n")
        
        successful_commands = []
        
        for i, (cmd, description) in enumerate(commands_to_test, 1):
            print(f"[{i}/{len(commands_to_test)}] {description}")
            print(f"    Command: {cmd}")
            
            try:
                # Use GBox box.command() API directly
                result = client._get_box().command(command=cmd, timeout="10s")
                
                # Extract output
                stdout = ""
                stderr = ""
                exit_code = 0
                
                if hasattr(result, 'stdout'):
                    stdout = result.stdout or ""
                    stderr = result.stderr or ""
                    exit_code = result.exitCode if hasattr(result, 'exitCode') else 0
                elif isinstance(result, dict):
                    stdout = result.get('stdout', '')
                    stderr = result.get('stderr', '')
                    exit_code = result.get('exitCode', 0)
                else:
                    stdout = str(result)
                
                if exit_code == 0 and stdout:
                    print(f"    ✓ Success (exit code: {exit_code})")
                    output_preview = stdout.strip()[:200]
                    if len(output_preview) == 200:
                        output_preview += "..."
                    print(f"    Output: {output_preview}")
                    successful_commands.append((cmd, stdout))
                elif exit_code == 0:
                    print(f"    ⚠ Success but no output (exit code: {exit_code})")
                else:
                    print(f"    ✗ Failed (exit code: {exit_code})")
                    if stderr:
                        print(f"    Error: {stderr.strip()[:100]}")
                
            except Exception as e:
                print(f"    ✗ Exception: {str(e)[:100]}")
            
            print()
        
        # Summary
        print("=" * 80)
        print(f"SUMMARY: {len(successful_commands)}/{len(commands_to_test)} commands succeeded")
        print("=" * 80)
        print()
        
        if successful_commands:
            print("Successful commands:")
            for i, (cmd, output) in enumerate(successful_commands, 1):
                print(f"{i}. {cmd}")
                if "power" in output.lower() or "battery" in output.lower():
                    print(f"   (Contains power/battery keywords)")
            print()
        
        # Suggest next steps
        print("[5/5] Recommendations:")
        print()
        
        if not successful_commands:
            print("⚠ No commands succeeded. This suggests:")
            print("  1. The shell environment is very restricted")
            print("  2. We may need to use alternative validation methods")
            print("  3. Consider UI-based validation (screenshot analysis)")
        else:
            print("✓ Some commands succeeded. Review the output above to determine:")
            print("  1. Which command provides the most reliable battery saver status")
            print("  2. What the output looks like when battery saver is enabled/disabled")
            print("  3. Update the validator to use the working command")
        
        print()
        print("To test battery saver enabled state:")
        print("  1. Manually enable battery saver via Android UI")
        print("  2. Run the successful commands again")
        print("  3. Compare outputs to identify the status indicator")
        
    finally:
        # Cleanup
        print()
        print("Cleaning up...")
        await client.close()
        print("✓ Box terminated")
        print()


if __name__ == "__main__":
    asyncio.run(test_battery_saver_commands())

