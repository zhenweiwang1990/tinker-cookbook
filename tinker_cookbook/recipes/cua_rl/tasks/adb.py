from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any

from . import config


@dataclass
class Device:
    device_id: str
    model: str | None
    product: str | None


class AdbError(RuntimeError):
    pass


class AdbClient:
    def __init__(
        self, 
        preselected_device_id: Optional[str] = None,
        gbox_box: Optional[Any] = None,
        gbox_client: Optional[Any] = None,
        enable_command_history: bool = False,
    ) -> None:
        """Initialize AdbClient.
        
        Args:
            preselected_device_id: Optional device ID (for local ADB mode)
            gbox_box: Optional GBox box object (for GBox command mode)
            gbox_client: Optional GBox client (alternative to gbox_box)
            enable_command_history: If True, record all executed commands for later retrieval
        """
        self.gbox_box = gbox_box
        self.gbox_client = gbox_client
        
        self.use_gbox = True
        # Get box from client if needed
        if gbox_box is None and gbox_client is not None:
            self.gbox_box = gbox_client._get_box() if hasattr(gbox_client, '_get_box') else None
        self.selected_device_id = "gbox"  # Placeholder for GBox mode
        
        # Command history tracking
        self.enable_command_history = enable_command_history
        self.command_history: List[str] = []

    # ---- device selection ----
    def _list_devices(self) -> List[Device]:
        result = self._run_raw(["adb", "devices", "-l"], capture_output=True)
        lines = result.strip().splitlines()
        devices: List[Device] = []
        for line in lines[1:]:  # skip header
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2 or parts[1] != "device":
                continue
            device_id = parts[0]
            # parse attributes like model:Pixel_7 product:sdk_gphone64... etc
            attrs = {kv.split(":", 1)[0]: kv.split(":", 1)[1] for kv in parts[2:] if ":" in kv}
            devices.append(Device(device_id=device_id, model=attrs.get("model"), product=attrs.get("product")))
        return devices

    def _select_device_interactive(self) -> str:
        devices = self._list_devices()
        if not devices:
            # Provide a helpful hint for starting an emulator
            raise AdbError(
                "No ADB devices found. Start an emulator or connect a device.\n"
                "Hint: run `make android` to launch an Android emulator."
            )
        if len(devices) == 1:
            return devices[0].device_id
        # interactive selection in terminal
        print("Multiple devices detected:")
        for idx, d in enumerate(devices, start=1):
            label = d.model or d.product or d.device_id
            print(f"  {idx}. {label} ({d.device_id})")
        while True:
            try:
                choice = input("Select device [1-{}]: ".format(len(devices)))
                sel = int(choice)
                if 1 <= sel <= len(devices):
                    return devices[sel - 1].device_id
            except Exception:
                pass
            print("Invalid selection. Try again.")

    # ---- command helpers ----
    def _run_raw(self, args: List[str], capture_output: bool = False) -> str:
        """Run raw command (local ADB mode only)."""
        if self.use_gbox:
            raise AdbError("_run_raw should not be called in GBox mode")
        try:
            proc = subprocess.run(args, check=True, capture_output=capture_output, text=True)
            return proc.stdout if capture_output else ""
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            stdout = e.stdout or ""
            details = ""
            if stdout.strip():
                details += f"\nSTDOUT:\n{stdout.strip()}"
            if stderr.strip():
                details += f"\nSTDERR:\n{stderr.strip()}"
            raise AdbError(f"Command failed: {' '.join(shlex.quote(a) for a in args)}{details}")

    def _run_gbox_command(self, command: str, timeout: Optional[str] = None) -> str:
        """Run command via GBox SDK."""
        if not self.use_gbox or self.gbox_box is None:
            raise AdbError("GBox box not available")
        
        # Record command if history is enabled
        if self.enable_command_history:
            self.command_history.append(command)
        
        try:
            result = self.gbox_box.command(command=command, timeout=timeout)
            # GBox command returns an object with stdout, stderr, exitCode
            if hasattr(result, 'stdout'):
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                exit_code = result.exitCode if hasattr(result, 'exitCode') else 0
                
                if exit_code != 0:
                    details = ""
                    if stdout.strip():
                        details += f"\nSTDOUT:\n{stdout.strip()}"
                    if stderr.strip():
                        details += f"\nSTDERR:\n{stderr.strip()}"
                    raise AdbError(f"Command failed with exit code {exit_code}: {command}{details}")
                
                return stdout
            else:
                # Fallback: try to get as dict
                if isinstance(result, dict):
                    stdout = result.get('stdout', '')
                    stderr = result.get('stderr', '')
                    exit_code = result.get('exitCode', 0)
                    if exit_code != 0:
                        details = ""
                        if stdout.strip():
                            details += f"\nSTDOUT:\n{stdout.strip()}"
                        if stderr.strip():
                            details += f"\nSTDERR:\n{stderr.strip()}"
                        raise AdbError(f"Command failed with exit code {exit_code}: {command}{details}")
                    return stdout
                return str(result)
        except Exception as e:
            raise AdbError(f"GBox command failed: {command}\nError: {str(e)}")

    def _run(self, *adb_args: str, capture_output: bool = False) -> str:
        """Run ADB command (local or via GBox)."""
        if self.use_gbox:
            # Convert adb args to shell command
            # For example: adb -s device shell pm list packages -> pm list packages
            # Skip 'adb', '-s', device_id, 'shell' if present
            args_list = list(adb_args)
            if args_list and args_list[0] == "shell":
                args_list = args_list[1:]
            
            # Handle special cases
            if args_list and args_list[0] == "pull":
                # For pull command, we need to read the file content
                if len(args_list) < 3:
                    raise AdbError("pull command requires source and destination")
                source = args_list[1]
                dest = args_list[2]
                # For binary files, use base64 encoding
                try:
                    # Try base64 first (works for binary files)
                    import base64
                    b64_content = self._run_gbox_command(f"base64 {shlex.quote(source)}")
                    # Remove any whitespace/newlines
                    b64_content = "".join(b64_content.split())
                    # Decode and write
                    content = base64.b64decode(b64_content)
                    with open(dest, "wb") as f:
                        f.write(content)
                    return ""
                except Exception:
                    # Fallback to cat for text files
                    try:
                        content = self._run_gbox_command(f"cat {shlex.quote(source)}")
                        with open(dest, "w", encoding="utf-8") as f:
                            f.write(content)
                        return ""
                    except Exception as e:
                        raise AdbError(f"Failed to pull file {source}: {e}")
            
            # Build command string for other commands
            command = " ".join(shlex.quote(str(arg)) for arg in args_list)
            return self._run_gbox_command(command)
        else:
            # Local ADB mode
            if not self.selected_device_id:
                raise AdbError("Device not selected")
            args = ["adb", "-s", self.selected_device_id, *adb_args]
            # Record command if history is enabled
            if self.enable_command_history:
                full_command = " ".join(shlex.quote(str(arg)) for arg in args)
                self.command_history.append(full_command)
            return self._run_raw(args, capture_output=capture_output)
    
    def get_command_history(self) -> List[str]:
        """Get the history of all executed commands.
        
        Returns:
            List of command strings that were executed
        """
        return self.command_history.copy()
    
    def clear_command_history(self) -> None:
        """Clear the command history."""
        self.command_history.clear()

    # ---- app lifecycle ----
    def uninstall(self, package_name: str) -> None:
        self._run("uninstall", package_name, capture_output=True)

    def install(self, apk_path: str) -> None:
        self._run("install", "-r", apk_path, capture_output=True)

    def is_installed(self, package_name: str) -> bool:
        out = self._run("shell", "pm", "list", "packages", package_name, capture_output=True)
        return package_name in out

    def start_activity(self, package_name: str, activity: str) -> None:
        # activity can be fully-qualified or relative (e.g., .MainActivity)
        comp = f"{package_name}/{activity}"
        self._run("shell", "am", "start", "-n", comp, capture_output=True)

    def launch(self, package_name: str, activity: Optional[str] = None) -> None:
        if not self.is_installed(package_name):
            raise AdbError(f"Package not installed: {package_name}")
        if activity:
            try:
                self.start_activity(package_name, activity)
                return
            except AdbError as err:
                # fall back to generic methods below
                last_err = err
        # Prefer Activity Manager start with MAIN/LAUNCHER for the package
        try:
            self._run(
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.MAIN",
                "-c",
                "android.intent.category.LAUNCHER",
                "-p",
                package_name,
                capture_output=True,
            )
            return
        except AdbError as am_err:
            # Fallback to monkey if am start fails
            try:
                self._run(
                    "shell",
                    "monkey",
                    "-p",
                    package_name,
                    "-c",
                    "android.intent.category.LAUNCHER",
                    "1",
                    capture_output=True,
                )
                return
            except AdbError as monkey_err:
                raise AdbError(
                    f"Failed to launch {package_name}.\n"
                    f"am -n error: {str(locals().get('last_err', 'N/A'))}\n"
                    f"am MAIN/LAUNCHER error: {am_err}\nmonkey error: {monkey_err}"
                )

    def screenshot_to_file(self, dest_path: str | None = None, *, prefix: str = "shot") -> str:
        """Take screenshot and save to file.
        """

        tmp_on_device = "/sdcard/ui_runner_screencap.png"
        self._run("shell", "screencap", "-p", tmp_on_device)
        path = dest_path
        # In GBox mode, we can't easily pull files, so we'll just return the path
        # The actual screenshot should be handled by gbox_client
        # Clean up the temp file
        try:
            self._run("shell", "rm", "-f", tmp_on_device)
        except Exception:
            pass
        return path

    # ---- data access ----
    def run_sqlite_query(self, package_name: str, db_relative_path: str, sql: str) -> str:
        # Quote SQL so adb shell doesn't mis-parse parentheses/semicolons.
        # In GBox mode, _run will quote it again, so we don't pre-quote here.
        # In local ADB mode, we need to quote it for subprocess.
        if self.use_gbox:
            sql_arg = sql
        else:
            sql_arg = shlex.quote(sql)
        return self._run(
            "shell",
            "run-as",
            package_name,
            "sqlite3",
            db_relative_path,
            sql_arg,
            capture_output=True,
        )

def list_connected_devices() -> List[Device]:
    """
    Non-interactive helper to list connected ADB devices.
    Mirrors parsing logic from AdbClient._list_devices but without requiring an AdbClient instance.
    """
    try:
        proc = subprocess.run(["adb", "devices", "-l"], check=True, capture_output=True, text=True)
        output = proc.stdout
    except subprocess.CalledProcessError as e:
        raise AdbError(f"Command failed: {' '.join(shlex.quote(a) for a in ['adb','devices','-l'])}\n{e}")
    lines = output.strip().splitlines()
    devices: List[Device] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2 or parts[1] != "device":
            continue
        device_id = parts[0]
        attrs = {kv.split(":", 1)[0]: kv.split(":", 1)[1] for kv in parts[2:] if ":" in kv}
        devices.append(Device(device_id=device_id, model=attrs.get("model"), product=attrs.get("product")))
    return devices
