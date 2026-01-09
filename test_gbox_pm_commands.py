#!/usr/bin/env python3
"""Test pm and test -e commands in GBox Android environment."""

import asyncio
import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinker_cookbook.recipes.cua_rl.gbox.client import CuaGBoxClient

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


async def test_pm_and_file_commands():
    """Test package manager and file system test commands."""
    print("=" * 80)
    print("GBox PM and File Test Commands Validation")
    print("=" * 80)
    print()
    
    api_key = os.environ.get("GBOX_API_KEY")
    if not api_key:
        print("ERROR: GBOX_API_KEY not set")
        return
    
    print("[1/4] Creating GBox client...")
    client = CuaGBoxClient(api_key=api_key)
    
    try:
        print("[2/4] Creating Android box...")
        await client.create_box(box_type="android")
        print(f"✓ Box created: {client.box_id}\n")
        
        print("[3/4] Waiting for box to be ready...")
        await asyncio.sleep(3)
        
        print("[4/4] Testing commands...")
        print("-" * 80)
        print()
        
        commands_to_test = [
            # Package manager commands
            ("pm list packages | head -5", "List first 5 packages"),
            ("pm list packages | wc -l", "Count total packages"),
            ("pm list packages | grep com.android.settings", "Check Settings app"),
            ("pm list packages | grep com.instagram.android", "Check Instagram app"),
            
            # File system test commands  
            ("test -d /storage/emulated/0 && echo YES || echo NO", "Test storage dir"),
            ("test -d /storage/emulated/0/Download && echo YES || echo NO", "Test Download dir"),
            ("test -d /storage/emulated/0/Documents && echo YES || echo NO", "Test Documents dir"),
            ("test -e /system/bin/settings && echo YES || echo NO", "Test settings binary"),
            ("test -e /data/nonexistent && echo YES || echo NO", "Test nonexistent file"),
            
            # Create, test, and remove a folder
            ("mkdir -p /storage/emulated/0/Download/test_gbox", "Create test folder"),
            ("test -d /storage/emulated/0/Download/test_gbox && echo YES || echo NO", "Verify folder created"),
            ("rmdir /storage/emulated/0/Download/test_gbox", "Remove test folder"),
            ("test -d /storage/emulated/0/Download/test_gbox && echo YES || echo NO", "Verify folder removed"),
        ]
        
        successful = 0
        failed = 0
        
        for i, (cmd, description) in enumerate(commands_to_test, 1):
            print(f"[{i}/{len(commands_to_test)}] {description}")
            print(f"    Command: {cmd}")
            
            try:
                result = client._get_box().command(command=cmd, timeout="10s")
                
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
                
                if exit_code == 0:
                    print(f"    ✓ Success (exit: {exit_code})")
                    output = stdout.strip()[:200]
                    if output:
                        print(f"    Output: {output}")
                    successful += 1
                else:
                    print(f"    ✗ Failed (exit: {exit_code})")
                    if stderr:
                        print(f"    Error: {stderr.strip()[:100]}")
                    failed += 1
                    
            except Exception as e:
                print(f"    ✗ Exception: {str(e)[:100]}")
                failed += 1
            
            print()
        
        print("=" * 80)
        print(f"SUMMARY: {successful}/{len(commands_to_test)} commands succeeded")
        print("=" * 80)
        print()
        
        if successful > 0:
            print("✓ Commands work! Key findings:")
            print("  • pm (package manager) commands - check results above")
            print("  • test -e/-d (file tests) commands - check results above")
            print("  • File operations (mkdir, rmdir) - check results above")
        else:
            print("⚠ No commands succeeded")
            print("  This may indicate restricted shell access")
            
    finally:
        print("\nCleaning up...")
        await client.close()
        print("✓ Box terminated")


if __name__ == "__main__":
    asyncio.run(test_pm_and_file_commands())

