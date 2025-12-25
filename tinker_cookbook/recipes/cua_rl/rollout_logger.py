"""
Rollout Logger for buffering and organizing logs per rollout.

This module provides a logger that buffers logs for each rollout and outputs
them all at once to avoid log interleaving in parallel rollouts.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import base64

logger = logging.getLogger(__name__)

# Create a simple logger for rollout logs without package name prefix
_rollout_logger = logging.getLogger("rollout")
_rollout_logger.setLevel(logging.INFO)
# Remove existing handlers to avoid duplication
_rollout_logger.handlers.clear()
# Add a simple handler with minimal format
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(message)s")
_handler.setFormatter(_formatter)
_rollout_logger.addHandler(_handler)
_rollout_logger.propagate = False  # Don't propagate to root logger


class RolloutLogger:
    """Logger that buffers logs for a single rollout and outputs them at once."""
    
    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    
    def __init__(
        self, 
        rollout_id: str, 
        output_dir: Optional[Path] = None,
        step: Optional[int] = None,
        batch: Optional[int] = None,
        group: Optional[int] = None,
        rollout_index: Optional[int] = None,
    ):
        """
        Initialize rollout logger.
        
        Args:
            rollout_id: Unique identifier for this rollout
            output_dir: Optional directory to save trajectory data
            step: Training step number
            batch: Batch number
            group: Group index
            rollout_index: Index of rollout within the group
        """
        self.rollout_id = rollout_id
        self.output_dir = output_dir
        self.step = step
        self.batch = batch
        self.group = group
        self.rollout_index = rollout_index
        self.log_buffer: List[str] = []
        self.trajectory_data: Dict[str, Any] = {
            "rollout_id": rollout_id,
            "turns": [],
            "screenshots": [],
            "summary": {},
        }
        self.current_turn: Optional[Dict[str, Any]] = None
        self.start_time = time.time()
        
        # Check if colors should be enabled (disable if NO_COLOR is set)
        self._enable_color = os.environ.get("NO_COLOR", "").strip() == ""
        
    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color to text if colors are enabled."""
        if not self._enable_color:
            return text
        color_code = getattr(self, color.upper(), "")
        return f"{color_code}{text}{self.RESET}" if color_code else text
    
    def _wrap_long_line(self, message: str, max_message_width: int = 150) -> List[str]:
        """
        Wrap a long log message into multiple lines, breaking at logical points.
        
        Args:
            message: The message to wrap (without timestamp)
            max_message_width: Maximum width for message part (timestamp is ~26 chars, so total ~176)
            
        Returns:
            List of wrapped lines
        """
        # If message is short enough, return as-is
        if len(message) <= max_message_width:
            return [message]
        
        lines = []
        # Try to break at " | " first to preserve structure
        parts = message.split(" | ")
        
        if len(parts) > 1:
            # Has delimiters, try to keep parts together when possible
            for i, part in enumerate(parts):
                part_with_delim = (" | " + part if i > 0 else part)
                
                # If this part alone would exceed width, wrap it by words
                if len(part) > max_message_width:
                    # Wrap the long part
                    words = part.split()
                    part_lines = []
                    current_part_line = ""
                    prefix_len = len(" | ") if i > 0 else 0
                    
                    for word in words:
                        # Available width for first line of part (with prefix) vs continuation
                        available = max_message_width - prefix_len if not part_lines else max_message_width - 2
                        
                        if len(current_part_line) + len(word) + (1 if current_part_line else 0) <= available:
                            current_part_line += (" " + word if current_part_line else word)
                        else:
                            if current_part_line:
                                if part_lines:
                                    part_lines.append("  " + current_part_line)  # Indent continuation
                                else:
                                    part_lines.append((" | " if i > 0 else "") + current_part_line)
                            current_part_line = word
                    
                    if current_part_line:
                        if part_lines:
                            part_lines.append("  " + current_part_line)
                        else:
                            part_lines.append((" | " if i > 0 else "") + current_part_line)
                    
                    lines.extend(part_lines)
                else:
                    # Part fits, try to add it to last line or start new line
                    if lines and len(lines[-1]) + len(part_with_delim) <= max_message_width:
                        # Add to last line
                        lines[-1] += part_with_delim
                    else:
                        # Start new line
                        lines.append(part_with_delim if i > 0 else part)
        else:
            # No delimiters, wrap by words
            words = message.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_message_width:
                    current_line += (" " + word if current_line else word)
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
        
        return lines if lines else [message]
    
    def log(self, message: str, level: str = "INFO", color: Optional[str] = None):
        """Add a log message to the buffer. Long messages will be wrapped automatically."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # Auto-detect error messages and apply red color if not explicitly specified
        # Check for common error indicators: ⚠, ✗, or "error" (case-insensitive)
        if color is None and ("⚠" in message or "✗" in message or " error" in message.lower() or message.lower().startswith("error")):
            color = "RED"
        
        # Wrap long messages (max ~150 chars for message part, timestamp is ~26 chars)
        # Wrap before applying color to avoid counting color codes in length
        message_lines = self._wrap_long_line(message, max_message_width=150)
        
        # Apply color to each line if specified
        if color:
            message_lines = [self._color(line, color) for line in message_lines]
        
        # Store each line with timestamp (first line has timestamp, continuation lines have indent)
        for i, line in enumerate(message_lines):
            if i == 0:
                log_entry = f"[{timestamp}] {line}"
            else:
                # Indent continuation lines (timestamp width + 1 space = ~27 chars)
                log_entry = " " * 28 + line
            self.log_buffer.append(log_entry)
        
        # Also add to trajectory data (store original message, not wrapped)
        if "logs" not in self.trajectory_data:
            self.trajectory_data["logs"] = []
        original_message = self._color(message, color) if color else message
        self.trajectory_data["logs"].append({
            "timestamp": timestamp,
            "level": level,
            "message": original_message,
        })
    
    def start_turn(self, turn_num: int, max_turns: int):
        """Start logging a new turn."""
        self.current_turn = {
            "turn_num": turn_num,
            "max_turns": max_turns,
            "start_time": time.time(),
            "screenshot_uri": None,
            "user_input": None,
            "model_response": None,
            "parse_success": None,
            "tool_calls": [],
            "action_results": [],
            "turn_header_logged": False,  # Track if turn header has been logged
            "total_tool_coord_time": 0.0,  # Total time for getting coordinates
            "total_tool_exec_time": 0.0,  # Total time for tool execution
            "total_tool_time": 0.0,  # Total time for all tools (non-perform_action)
        }
        # Don't log turn header here, will be logged together with screenshot/model inference
        self.log("")
        self.log("┌" + "─" * 118 + "┐")
    
    def log_screenshot(self, screenshot_uri: str, time_taken: float):
        """Log screenshot capture (stored, will be combined with model inference in one line)."""
        if self.current_turn:
            self.current_turn["screenshot_uri"] = screenshot_uri
            self.current_turn["screenshot_time"] = time_taken
        
        # Extract base64 data if it's a data URI
        screenshot_data = None
        if screenshot_uri.startswith("data:image"):
            try:
                header, encoded = screenshot_uri.split(",", 1)
                screenshot_data = encoded
            except:
                pass
        
        self.trajectory_data["screenshots"].append({
            "turn": self.current_turn["turn_num"] if self.current_turn else None,
            "uri": screenshot_uri[:100] + "..." if len(screenshot_uri) > 100 else screenshot_uri,
            "data": screenshot_data,
            "time_taken": time_taken,
        })
        
        # Don't log here, will be combined with model inference
    
    def log_model_inference(
        self,
        turn_num: int,
        response_text: str,
        parse_success: bool,
        inference_time: float,
    ):
        """Log model inference result."""
        if self.current_turn:
            self.current_turn["model_response"] = response_text
            self.current_turn["parse_success"] = parse_success
            self.current_turn["model_inference_time"] = inference_time
            # Don't log turn header here - will be logged in end_turn() with all timing info
        else:
            self.log(
                f"[Turn {turn_num}] Model inference: {len(response_text)} chars, "
                f"parse={'✓' if parse_success else '✗'}, time={inference_time:.3f}s"
            )
            self.log(response_text)
    
    def log_action(
        self,
        action_type: str,
        target_desc: Optional[str] = None,
        start_target: Optional[str] = None,
        end_target: Optional[str] = None,
        coordinates: Optional[Dict[str, Any]] = None,
        coord_time: Optional[float] = None,
        exec_time: Optional[float] = None,
        total_time: Optional[float] = None,
    ):
        """Log action execution (combined format)."""
        action_info = {
            "action_type": action_type,
            "target": target_desc,
            "start_target": start_target,
            "end_target": end_target,
            "coordinates": coordinates,
            "coord_time": coord_time,
            "exec_time": exec_time,
            "total_time": total_time,
        }
        
        if self.current_turn:
            self.current_turn["action_results"].append(action_info)
            # Accumulate tool execution times
            if coord_time is not None:
                self.current_turn["total_tool_coord_time"] += coord_time
            if exec_time is not None:
                self.current_turn["total_tool_exec_time"] += exec_time
            # Don't log here - will be logged in end_turn() after Model Response
        else:
            # Fallback for when not in a turn - log immediately
            colored_action_type = self._color(action_type, "CYAN")
            if action_type == "swipe":
                line1 = (
                    f"Action: {colored_action_type} | "
                    f"start={start_target or 'N/A'} | "
                    f"end={end_target or 'N/A'}"
                )
                if coordinates:
                    start_coords = coordinates.get("start", {})
                    end_coords = coordinates.get("end", {})
                    coord_time_str = f"{coord_time:.3f}s" if coord_time is not None else "N/A"
                    exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
                    total_time_str = f"{total_time:.3f}s" if total_time is not None else "N/A"
                    colored_start_coords = self._color(f"({start_coords.get('x', 0)}, {start_coords.get('y', 0)})", "YELLOW")
                    colored_end_coords = self._color(f"({end_coords.get('x', 0)}, {end_coords.get('y', 0)})", "YELLOW")
                    line2 = (
                        f"  ↳ Coords: start={colored_start_coords} | "
                        f"end={colored_end_coords} | "
                        f"coord_time={coord_time_str} | exec_time={exec_time_str} | total={total_time_str}"
                    )
                else:
                    coord_time_str = f"{coord_time:.3f}s" if coord_time is not None else "N/A"
                    exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
                    total_time_str = f"{total_time:.3f}s" if total_time is not None else "N/A"
                    line2 = f"  ↳ coord_time={coord_time_str} | exec_time={exec_time_str} | total={total_time_str}"
            else:
                line1 = f"Action: {colored_action_type}"
                if target_desc:
                    line1 += f" | target={target_desc}"
                if coordinates:
                    coord_time_str = f"{coord_time:.3f}s" if coord_time is not None else "N/A"
                    exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
                    total_time_str = f"{total_time:.3f}s" if total_time is not None else "N/A"
                    colored_coords = self._color(f"({coordinates.get('x', 0)}, {coordinates.get('y', 0)})", "YELLOW")
                    line2 = (
                        f"  ↳ Coords: {colored_coords} | "
                        f"coord_time={coord_time_str} | exec_time={exec_time_str} | total={total_time_str}"
                    )
                else:
                    coord_time_str = f"{coord_time:.3f}s" if coord_time is not None else "N/A"
                    exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
                    total_time_str = f"{total_time:.3f}s" if total_time is not None else "N/A"
                    line2 = f"  ↳ coord_time={coord_time_str} | exec_time={exec_time_str} | total={total_time_str}"
            self.log(line1)
            self.log(line2)
    
    def log_tool_calls(
        self,
        turn_num: int,
        tool_calls: List[Dict[str, Any]],
        parse_time: float,
        parser_type: str = "renderer",
    ):
        """Log tool call parsing result (combined format)."""
        num_tools = len(tool_calls)
        if self.current_turn:
            self.current_turn["tool_calls"] = tool_calls
            self.current_turn["tool_call_parse_time"] = parse_time
            self.current_turn["tool_call_parser_type"] = parser_type
            # Don't log here - will be logged in end_turn() after Model Response
        else:
            # Fallback for when not in a turn
            self.log(
                f"[Turn {turn_num}] Tool calls: {num_tools} found via {parser_type}, parse_time={parse_time:.3f}s"
            )
    
    def log_tool_execution(
        self,
        turn_num: int,
        tool_idx: int,
        total_tools: int,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
        exec_time: Optional[float] = None,
        error: Optional[str] = None,
    ):
        """Log tool execution (combined format: 2 lines - summary and details).
        
        Note: For perform_action, details are logged separately via log_action in cua_tools.py,
        so this method only logs the summary line for perform_action.
        """
        # Accumulate tool execution time for non-perform_action tools
        if self.current_turn and tool_name != "perform_action" and exec_time is not None:
            self.current_turn["total_tool_time"] += exec_time
        
        # Store tool execution info for later logging
        if self.current_turn:
            if "tool_executions" not in self.current_turn:
                self.current_turn["tool_executions"] = []
            self.current_turn["tool_executions"].append({
                "tool_idx": tool_idx,
                "total_tools": total_tools,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "result": result,
                "exec_time": exec_time,
                "error": error,
            })
            # Don't log here - will be logged in end_turn() after Model Response
        else:
            # Fallback for when not in a turn - log immediately
            status = "✓" if error is None else "✗"
            exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
            summary = f"[Turn {turn_num}] Tool {tool_idx}/{total_tools}: {tool_name} {status} in {exec_time_str}"
            self.log(summary)
            
            # For perform_action, details are already logged via log_action
            if tool_name == "perform_action":
                return
            
            # For other tools, show details on second line
            args_summary = json.dumps(tool_args, separators=(',', ':'))[:100]
            if len(args_summary) >= 100:
                args_summary += "..."
            details = f"  ↳ args={args_summary}"
            if result:
                result_summary = json.dumps(result, default=str, separators=(',', ':'))[:100]
                if len(result_summary) >= 100:
                    result_summary += "..."
                details += f" | result={result_summary}"
            if error:
                details += f" | error={error}"
            self.log(details)
    
    def end_turn(self, turn_num: int):
        """End logging for current turn."""
        if self.current_turn:
            self.current_turn["end_time"] = time.time()
            self.current_turn["duration"] = (
                self.current_turn["end_time"] - self.current_turn["start_time"]
            )
            
            # Calculate total tool execution time (coord_time + exec_time for actions + exec_time for other tools)
            total_tool_exec_time = (
                self.current_turn.get("total_tool_coord_time", 0.0) +
                self.current_turn.get("total_tool_exec_time", 0.0) +
                self.current_turn.get("total_tool_time", 0.0)
            )
            
            # Build and log the complete turn summary line with all timing info
            turn_info = f"Turn {self.current_turn['turn_num']}/{self.current_turn['max_turns']}"
            
            # Turn total time
            turn_total_time = self.current_turn["duration"]
            turn_total_info = f" | Total: {turn_total_time:.3f}s"
            
            # Screenshot time
            screenshot_info = ""
            if self.current_turn.get("screenshot_time") is not None:
                screenshot_time = self.current_turn["screenshot_time"]
                screenshot_info = f" | Screenshot: {screenshot_time:.3f}s"
            
            # Model inference time
            model_info = ""
            if self.current_turn.get("model_inference_time") is not None:
                inference_time = self.current_turn["model_inference_time"]
                parse_success = self.current_turn.get("parse_success", False)
                response_text = self.current_turn.get("model_response", "")
                model_info = f" | Model inference: {len(response_text)} chars, parse={'✓' if parse_success else '✗'}, time={inference_time:.3f}s"
            
            # Tool execution time (with coord_time and exec_time breakdown)
            tool_info = ""
            if total_tool_exec_time > 0:
                coord_time = self.current_turn.get("total_tool_coord_time", 0.0)
                exec_time = self.current_turn.get("total_tool_exec_time", 0.0) + self.current_turn.get("total_tool_time", 0.0)
                tool_info = f" | Tool execution: coord={coord_time:.3f}s, exec={exec_time:.3f}s, total={total_tool_exec_time:.3f}s"
            
            msg = f"{turn_info}{turn_total_info}{screenshot_info}{model_info}{tool_info}"
            padding = max(0, 116 - len(msg))
            self.log(f"│ {msg}" + " " * padding + "│")
            
            # Close the turn block
            self.log("└" + "─" * 118 + "┘")
            
            # Log Model Response in a new box
            if self.current_turn.get("model_response"):
                response_text = self.current_turn["model_response"]
                # Create a new box for response text
                self.log("")
                self.log("┌" + "─" * 118 + "┐")
                self.log("│ Model Response:" + " " * 102 + "│")
                self.log("├" + "─" * 118 + "┤")
                # Process response text line by line
                max_line_length = 116  # Account for "│ " prefix and " │" suffix (2 + 2 = 4, 118 - 4 = 114, but we use 116 for consistency)
                lines = response_text.split("\n")
                for line in lines:
                    # Handle empty lines
                    if not line:
                        self.log("│" + " " * 116 + "│")
                    else:
                        # If line is too long, wrap it
                        while len(line) > max_line_length:
                            chunk = line[:max_line_length]
                            padding = max(0, 116 - len(chunk))
                            self.log(f"│ {chunk}" + " " * padding + "│")
                            line = line[max_line_length:]
                        # Log the remaining part (or the whole line if it fits)
                        padding = max(0, 116 - len(line))
                        self.log(f"│ {line}" + " " * padding + "│")
                self.log("└" + "─" * 118 + "┘")
            
            # Log tool parsing and execution details
            if self.current_turn.get("tool_calls") or self.current_turn.get("action_results") or self.current_turn.get("tool_executions"):
                self.log("")
                self.log("┌" + "─" * 118 + "┐")
                
                # Log tool call parsing info
                if self.current_turn.get("tool_calls"):
                    tool_calls = self.current_turn["tool_calls"]
                    parse_time = self.current_turn.get("tool_call_parse_time", 0.0)
                    parser_type = self.current_turn.get("tool_call_parser_type", "renderer")
                    num_tools = len(tool_calls)
                    msg = f"[Turn {turn_num}] Tool calls: {num_tools} found via {parser_type}, parse_time={parse_time:.3f}s"
                    padding = max(0, 116 - len(msg))
                    self.log(f"│ {msg}" + " " * padding + "│")
                    self.log("├" + "─" * 118 + "┤")
                
                # Log action execution details
                if self.current_turn.get("action_results"):
                    for action_info in self.current_turn["action_results"]:
                        action_type = action_info["action_type"]
                        target_desc = action_info.get("target")
                        start_target = action_info.get("start_target")
                        end_target = action_info.get("end_target")
                        coordinates = action_info.get("coordinates")
                        coord_time = action_info.get("coord_time")
                        exec_time = action_info.get("exec_time")
                        total_time = action_info.get("total_time")
                        
                        colored_action_type = self._color(action_type, "CYAN")
                        if action_type == "swipe":
                            line1 = (
                                f"Action: {colored_action_type} | "
                                f"start={start_target or 'N/A'} | "
                                f"end={end_target or 'N/A'}"
                            )
                            if coordinates:
                                start_coords = coordinates.get("start", {})
                                end_coords = coordinates.get("end", {})
                                coord_time_str = f"{coord_time:.3f}s" if coord_time is not None else "N/A"
                                exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
                                total_time_str = f"{total_time:.3f}s" if total_time is not None else "N/A"
                                colored_start_coords = self._color(f"({start_coords.get('x', 0)}, {start_coords.get('y', 0)})", "YELLOW")
                                colored_end_coords = self._color(f"({end_coords.get('x', 0)}, {end_coords.get('y', 0)})", "YELLOW")
                                line2 = (
                                    f"  ↳ Coords: start={colored_start_coords} | "
                                    f"end={colored_end_coords} | "
                                    f"coord_time={coord_time_str} | exec_time={exec_time_str} | total={total_time_str}"
                                )
                            else:
                                coord_time_str = f"{coord_time:.3f}s" if coord_time is not None else "N/A"
                                exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
                                total_time_str = f"{total_time:.3f}s" if total_time is not None else "N/A"
                                line2 = f"  ↳ coord_time={coord_time_str} | exec_time={exec_time_str} | total={total_time_str}"
                        else:
                            line1 = f"Action: {colored_action_type}"
                            if target_desc:
                                line1 += f" | target={target_desc}"
                            if coordinates:
                                coord_time_str = f"{coord_time:.3f}s" if coord_time is not None else "N/A"
                                exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
                                total_time_str = f"{total_time:.3f}s" if total_time is not None else "N/A"
                                colored_coords = self._color(f"({coordinates.get('x', 0)}, {coordinates.get('y', 0)})", "YELLOW")
                                line2 = (
                                    f"  ↳ Coords: {colored_coords} | "
                                    f"coord_time={coord_time_str} | exec_time={exec_time_str} | total={total_time_str}"
                                )
                            else:
                                coord_time_str = f"{coord_time:.3f}s" if coord_time is not None else "N/A"
                                exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
                                total_time_str = f"{total_time:.3f}s" if total_time is not None else "N/A"
                                line2 = f"  ↳ coord_time={coord_time_str} | exec_time={exec_time_str} | total={total_time_str}"
                        
                        # Calculate padding (accounting for ANSI color codes)
                        line1_stripped = re.sub(r'\033\[[0-9;]*m', '', line1)
                        padding1 = max(0, 116 - len(line1_stripped))
                        self.log(f"│ {line1}" + " " * padding1 + "│")
                        
                        line2_stripped = re.sub(r'\033\[[0-9;]*m', '', line2)
                        padding2 = max(0, 116 - len(line2_stripped))
                        self.log(f"│ {line2}" + " " * padding2 + "│")
                
                # Log tool execution details (non-perform_action tools)
                if self.current_turn.get("tool_executions"):
                    for tool_exec in self.current_turn["tool_executions"]:
                        tool_idx = tool_exec["tool_idx"]
                        total_tools = tool_exec["total_tools"]
                        tool_name = tool_exec["tool_name"]
                        tool_args = tool_exec["tool_args"]
                        result = tool_exec.get("result")
                        exec_time = tool_exec.get("exec_time")
                        error = tool_exec.get("error")
                        
                        # Skip perform_action as it's already logged above
                        if tool_name == "perform_action":
                            continue
                        
                        status = "✓" if error is None else "✗"
                        exec_time_str = f"{exec_time:.3f}s" if exec_time is not None else "N/A"
                        summary = f"[Turn {turn_num}] Tool {tool_idx}/{total_tools}: {tool_name} {status} in {exec_time_str}"
                        padding = max(0, 116 - len(summary))
                        self.log(f"│ {summary}" + " " * padding + "│")
                        
                        # Show details on second line
                        args_summary = json.dumps(tool_args, separators=(',', ':'))[:100]
                        if len(args_summary) >= 100:
                            args_summary += "..."
                        details = f"  ↳ args={args_summary}"
                        if result:
                            result_summary = json.dumps(result, default=str, separators=(',', ':'))[:100]
                            if len(result_summary) >= 100:
                                result_summary += "..."
                            details += f" | result={result_summary}"
                        if error:
                            details += f" | error={error}"
                        padding = max(0, 116 - len(details))
                        self.log(f"│ {details}" + " " * padding + "│")
                
                self.log("└" + "─" * 118 + "┘")
            
            self.trajectory_data["turns"].append(self.current_turn)
            self.current_turn = None
    
    def set_summary(self, summary: Dict[str, Any]):
        """Set rollout summary."""
        self.trajectory_data["summary"] = summary
    
    def flush(self):
        """Output all buffered logs using a simple logger without package name prefix."""
        # Output all logs at once using the simple logger
        for log_entry in self.log_buffer:
            _rollout_logger.info(log_entry)
        
        # Clear buffer
        self.log_buffer.clear()
    
    def save_trajectory(
        self,
        base_dir: Path,
        step: Optional[int] = None,
        batch: Optional[int] = None,
        group: Optional[int] = None,
        is_eval: bool = False,
    ):
        """Save trajectory data to disk."""
        if not base_dir:
            return
        
        # Build directory structure: {base_dir}/trajectories/{phase}_step{step}_{timestamp}/{batch}/{group}/{rollout_id}/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        phase = "eval" if is_eval else "train"
        
        # Build top-level directory name with phase and step info
        if step is not None:
            top_dir_name = f"{phase}_step{step}_{timestamp}"
        else:
            top_dir_name = f"{phase}_{timestamp}"
        
        parts = [base_dir, "trajectories", top_dir_name]
        if batch is not None:
            parts.append(f"batch_{batch}")
        if group is not None:
            parts.append(f"group_{group}")
        parts.append(f"{phase}_{self.rollout_id}")
        
        trajectory_dir = Path(*parts)
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        # Save logs
        log_file = trajectory_dir / "logs.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(self.log_buffer))
        
        # Save trajectory data as JSON
        trajectory_file = trajectory_dir / "trajectory.json"
        # Convert to JSON-serializable format
        trajectory_json = json.loads(json.dumps(self.trajectory_data, default=str))
        with open(trajectory_file, "w", encoding="utf-8") as f:
            json.dump(trajectory_json, f, indent=2, ensure_ascii=False)
        
        # Save screenshots
        screenshots_dir = trajectory_dir / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, screenshot_info in enumerate(self.trajectory_data.get("screenshots", [])):
            if screenshot_info.get("data"):
                screenshot_path = screenshots_dir / f"turn_{screenshot_info.get('turn', idx)}.png"
                try:
                    image_data = base64.b64decode(screenshot_info["data"])
                    with open(screenshot_path, "wb") as f:
                        f.write(image_data)
                except Exception as e:
                    logger.warning(f"Failed to save screenshot {idx}: {e}")
        
        # Save summary
        summary_file = trajectory_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.trajectory_data.get("summary", {}), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Trajectory saved to: {trajectory_dir}")
        return trajectory_dir

