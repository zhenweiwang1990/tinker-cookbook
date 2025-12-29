"""Reward functions for CUA Agent GRPO training.

This module provides reward functions for CUA Agent training, ranging from
simple binary rewards to comprehensive reward shaping inspired by Link Search Agent.

**Quick Start:**
- `simple_reward_function`: Binary 0/1 reward (baseline)
- `completion_reward_function`: Partial credit for attempts
- `efficiency_reward_function`: Considers steps and errors
- `comprehensive_reward_function`: Full reward shaping with strategy rewards

**Using Comprehensive Rewards:**

The `comprehensive_reward_function` provides detailed reward shaping based on:
- Base score (0-1.5 for success)
- Strategy rewards (good behavior patterns)
- Penalties (inefficiency, errors)
- Efficiency bonus (fewer turns)
- Perfect execution bonus (3.0)

To use it effectively, you need to collect extended metrics during rollout:
- Tool call quality (repeated/invalid actions)
- Error classification (parse/tool name/tool arg/runtime errors)
- Strategy quality (navigation, backtracking, exploration)

See `CUARolloutResult` for all available metrics. Use `create_rollout_result_from_dict`
to convert rollout results, then pass to `comprehensive_reward_function`.

**Example:**
    result = create_rollout_result_from_dict(rollout_dict, task_id="task_1")
    reward = comprehensive_reward_function(result)
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from tinker_cookbook.recipes.cua_rl.demo_tasks import CUATask


logger = logging.getLogger(__name__)


@dataclass
class CUARolloutResult:
    """Result of a CUA rollout.
    
    This class tracks comprehensive metrics for reward calculation,
    similar to LinkSearchRubric in the Link Search Agent.
    """
    
    # Core results
    task_id: str
    task_completed: bool
    task_success: bool
    num_turns: int
    max_turns: int
    errors: List[str]
    
    # Validation results
    validation_passed: bool = False
    validation_details: Optional[Dict[str, Any]] = None
    
    # Timing
    total_time: float = 0.0
    
    # Tool call quality (to be collected from rollout)
    num_total_actions: int = 0  # Total tool calls made
    consecutive_repeated_actions: int = 0  # Number of consecutive repeated actions (same tool + same args)
    
    # Error classification (to be collected from rollout)
    parse_errors: int = 0  # Tool call parsing errors (missing name/args)
    tool_name_errors: int = 0  # Invalid tool names
    tool_arg_errors: int = 0  # Invalid tool arguments
    runtime_errors: int = 0  # Runtime errors during tool execution
    
    # Turn tracking
    ran_out_of_turns: bool = False  # Did we run out of turns?
    attempted_completion: bool = False  # Did agent call finish tool?
    
    # Timing milestones
    turn_first_success: int = -1  # Turn when first successful action occurred
    turn_task_completed: int = -1  # Turn when task was completed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "num_turns": self.num_turns,
            "max_turns": self.max_turns,
            "errors": self.errors,
            "validation_passed": self.validation_passed,
            "validation_details": self.validation_details,
            "total_time": self.total_time,
            # Extended metrics
            "num_total_actions": self.num_total_actions,
            "consecutive_repeated_actions": self.consecutive_repeated_actions,
            "parse_errors": self.parse_errors,
            "tool_name_errors": self.tool_name_errors,
            "tool_arg_errors": self.tool_arg_errors,
            "runtime_errors": self.runtime_errors,
            "ran_out_of_turns": self.ran_out_of_turns,
            "attempted_completion": self.attempted_completion,
            "turn_first_success": self.turn_first_success,
            "turn_task_completed": self.turn_task_completed,
        }


def simple_reward_function(
    result: CUARolloutResult,
    task: Optional[CUATask] = None,
) -> float:
    """Simple binary reward function.
    
    Returns 1.0 for successful completion, 0.0 otherwise.
    
    Args:
        result: Rollout result
        task: Original task (optional, for additional context)
        
    Returns:
        Reward value (0.0 or 1.0)
    """
    if result.task_success:
        return 1.0
    return 0.0


def completion_reward_function(
    result: CUARolloutResult,
    task: Optional[CUATask] = None,
) -> float:
    """Reward function with partial credit for completion attempt.
    
    - 1.0 for successful completion
    - 0.1 for attempted completion (marked as complete but failed)
    - 0.0 for timeout/incomplete
    
    Args:
        result: Rollout result
        task: Original task (optional)
        
    Returns:
        Reward value
    """
    if result.task_success:
        return 1.0
    elif result.task_completed:
        return 0.1  # Attempted but failed
    return 0.0


def efficiency_reward_function(
    result: CUARolloutResult,
    task: Optional[CUATask] = None,
    step_penalty: float = -0.02,
    error_penalty: float = -0.1,
    timeout_penalty: float = -0.5,
) -> float:
    """Reward function that considers efficiency.
    
    Rewards:
    - Base reward for success: 1.0
    - Step penalty: -0.02 per step
    - Error penalty: -0.1 per error
    - Timeout penalty: -0.5 if not completed
    
    Args:
        result: Rollout result
        task: Original task
        step_penalty: Penalty per step
        error_penalty: Penalty per error
        timeout_penalty: Penalty for timeout
        
    Returns:
        Reward value (can be negative)
    """
    reward = 0.0
    
    # Base reward for outcome
    if result.task_success:
        reward = 1.0
    elif result.task_completed:
        reward = 0.1
    else:
        reward = timeout_penalty
    
    # Step efficiency penalty
    reward += step_penalty * result.num_turns
    
    # Error penalty
    reward += error_penalty * len(result.errors)
    
    return reward


def shaped_reward_function(
    result: CUARolloutResult,
    task: CUATask,
    validation_weight: float = 0.5,
) -> float:
    """Shaped reward function with validation bonus.
    
    Combines completion reward with validation state check.
    
    Args:
        result: Rollout result
        task: Original task
        validation_weight: Weight for validation bonus
        
    Returns:
        Reward value
    """
    base_reward = completion_reward_function(result, task)
    
    # Add validation bonus
    if result.validation_passed:
        base_reward += validation_weight
    
    # Normalize to [0, 1]
    return min(1.0, base_reward)


def create_rollout_result_from_dict(
    rollout_dict: Dict[str, Any],
    task_id: str = "unknown",
) -> CUARolloutResult:
    """Create CUARolloutResult from rollout dictionary.
    
    This is a helper function to convert rollout results to CUARolloutResult.
    Currently, most extended metrics default to 0/False since they need to be
    collected during rollout. To use comprehensive_reward_function effectively,
    you should extend the rollout code to collect these metrics.
    
    Args:
        rollout_dict: Dictionary from rollout (e.g., from TinkerCuaAgent.run_task)
        task_id: Task identifier
        
    Returns:
        CUARolloutResult instance
    """
    # Extract basic fields
    task_completed = rollout_dict.get("task_completed", False)
    task_success = rollout_dict.get("task_success", False)
    num_turns = rollout_dict.get("num_turns", 0)
    max_turns = rollout_dict.get("max_turns", 15)
    errors = rollout_dict.get("errors", [])
    total_time = rollout_dict.get("total_time", 0.0)
    
    # Check if ran out of turns
    ran_out_of_turns = not task_completed and num_turns >= max_turns
    
    # Extract extended metrics if available (default to 0/False)
    return CUARolloutResult(
        task_id=task_id,
        task_completed=task_completed,
        task_success=task_success,
        num_turns=num_turns,
        max_turns=max_turns,
        errors=errors,
        validation_passed=rollout_dict.get("validation_passed", False),
        validation_details=rollout_dict.get("validation_details"),
        total_time=total_time,
        # Extended metrics (to be collected during rollout)
        num_total_actions=rollout_dict.get("num_total_actions", 0),
        consecutive_repeated_actions=rollout_dict.get("consecutive_repeated_actions", 0),
        parse_errors=rollout_dict.get("parse_errors", 0),
        tool_name_errors=rollout_dict.get("tool_name_errors", 0),
        tool_arg_errors=rollout_dict.get("tool_arg_errors", 0),
        runtime_errors=rollout_dict.get("runtime_errors", 0),
        ran_out_of_turns=ran_out_of_turns,
        attempted_completion=rollout_dict.get("attempted_completion", task_completed),
        turn_first_success=rollout_dict.get("turn_first_success", -1),
        turn_task_completed=rollout_dict.get("turn_task_completed", num_turns if task_completed else -1),
    )


def comprehensive_reward_function(
    result: CUARolloutResult,
    task: Optional[CUATask] = None,
    max_turns: Optional[int] = None,
) -> float:
    """Comprehensive reward function inspired by Link Search Agent.
    
    This reward function provides detailed reward shaping based on:
    - Base score for task success (0-1.5)
    - Early success bonus (earlier completion is better)
    - Penalties for consecutive repeated actions and runtime errors
    - Efficiency bonus for fewer turns
    - Perfect execution bonus (3.0)
    
    The reward range is [-2.0, +3.0] to provide strong learning signals.
    
    Args:
        result: Rollout result with comprehensive metrics
        task: Original task (optional, for additional context)
        max_turns: Maximum turns (uses result.max_turns if not provided)
        
    Returns:
        Reward value between -2.0 and +3.0
    """
    max_turns = max_turns or result.max_turns
    
    # ========== BASE SCORE ==========
    if result.task_success:
        base_reward = 1.5
    elif result.task_completed:
        base_reward = 0.3  # Attempted but failed
    elif result.ran_out_of_turns:
        base_reward = 0.0
    else:
        base_reward = -0.5  # Gave up early
    
    # ========== PARTIAL CREDIT ==========
    partial_rewards = 0.0
    
    # Early success reward (earlier is better)
    if result.turn_first_success > 0:
        timing_bonus = 0.15 * (1 - result.turn_first_success / max_turns)
        partial_rewards += max(0.05, timing_bonus)
    
    # ========== PENALTIES ==========
    penalties = 0.0
    
    # Consecutive repeated actions are wasteful (same tool + same args)
    if result.consecutive_repeated_actions > 0:
        # Penalty increases with number of consecutive repeats
        penalties += 0.15 * min(result.consecutive_repeated_actions, 5)  # Cap at 5
    
    # Runtime errors
    if result.runtime_errors > 0:
        penalties += 0.08 * min(result.runtime_errors, 5)  # Cap at 5
    
    # ========== SEVERE ERRORS ==========
    # Parse errors (most severe - can't even parse tool calls)
    if result.parse_errors > 0:
        return -2.0 + partial_rewards - penalties
    
    # Tool name errors (severe - calling non-existent tools)
    if result.tool_name_errors > 0:
        return -1.8 + partial_rewards - penalties
    
    # Tool argument errors (moderate - wrong arguments)
    if result.tool_arg_errors > 0:
        return -1.5 + partial_rewards - penalties
    
    # ========== NO COMPLETION CASE ==========
    if not result.task_completed:
        if result.ran_out_of_turns:
            # Ran out of turns but made effort
            effort_bonus = 0.0
            if result.num_total_actions >= 3:
                effort_bonus += 0.10
            return -0.5 + partial_rewards + effort_bonus - penalties
        else:
            # Gave up early without completing
            return -1.0 + partial_rewards - penalties
    
    # ========== COMPLETED CASE ==========
    # Perfect execution bonus
    # Perfect execution requires:
    # - Task succeeded
    # - No consecutive repeated actions (wasteful)
    # - No errors of any kind (parse, tool_name, tool_arg, runtime)
    # - Efficient: used <= 20% of max_turns (e.g., <= 3 turns if max_turns=15, <= 4 if max_turns=20)
    is_perfect = (
        result.task_success and
        result.consecutive_repeated_actions == 0 and
        result.runtime_errors == 0 and
        result.parse_errors == 0 and
        result.tool_name_errors == 0 and
        result.tool_arg_errors == 0 and
        result.num_turns <= max_turns * 0.2  # Efficient: used <= 20% of max turns
    )
    
    if is_perfect:
        logger.info(
            f"Perfect execution: success={result.task_success}, "
            f"turns={result.num_turns}, task_id={result.task_id}"
        )
        return 3.0
    
    # Normal case: combine everything
    reward = base_reward + partial_rewards - penalties
    
    # Efficiency bonus (fewer turns is better)
    if result.num_turns < max_turns:
        efficiency = 0.20 * (1 - result.num_turns / max_turns)
        reward += efficiency
    
    # Validation bonus (if validation passed)
    if result.validation_passed:
        reward += 0.3
    
    # Cap at 2.8 (perfect is 3.0)
    return min(max(reward, -2.0), 2.8)


class RewardTracker:
    """Track reward statistics during training."""
    
    def __init__(self):
        self.rewards: List[float] = []
        self.successes: List[bool] = []
        self.num_turns: List[int] = []
        
    def add(self, reward: float, success: bool, turns: int):
        """Add a rollout result."""
        self.rewards.append(reward)
        self.successes.append(success)
        self.num_turns.append(turns)
    
    def reset(self):
        """Reset tracker."""
        self.rewards.clear()
        self.successes.clear()
        self.num_turns.clear()
    
    @property
    def mean_reward(self) -> float:
        """Get mean reward."""
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
    
    @property
    def success_rate(self) -> float:
        """Get success rate."""
        return sum(self.successes) / len(self.successes) if self.successes else 0.0
    
    @property
    def mean_turns(self) -> float:
        """Get mean turns."""
        return sum(self.num_turns) / len(self.num_turns) if self.num_turns else 0.0
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "mean_reward": self.mean_reward,
            "success_rate": self.success_rate,
            "mean_turns": self.mean_turns,
            "total_rollouts": len(self.rewards),
        }

async def validate_task_completion(
    task: CUATask,
    gbox_client,
    result_message: Optional[str] = None,
) -> bool:
    """Validate task completion using ADB/shell commands via GBox.

    We avoid using gbox-handy-1 for validation, and instead:
      1. Execute Android shell commands inside the box (via GBox Command API).
      2. Parse the output to determine whether the target system state matches
         the expected result (e.g., WiFi on/off, current app, screen timeout).
    
    Args:
        task: The task to validate
        gbox_client: GBox client for executing shell commands
        result_message: Optional result message from rollout (for result_message_contains validation)
    """

    async def _run_shell(cmd: str) -> str:
        """Run a shell command inside the box via GBox SDK and return stdout."""
        sdk = getattr(gbox_client, "_sdk", None)
        box_id = getattr(gbox_client, "box_id", None)
        if sdk is None or not box_id:
            raise RuntimeError("gbox_client must provide `_sdk` and `box_id` for command execution")

        try:
            # Use SDK's get() method to get the box object, then use box.command()
            box = sdk.get(box_id)
            result = box.command(command=cmd)
            # result should have stdout, stderr, and exitCode attributes
            if hasattr(result, 'stdout'):
                return str(result.stdout) if result.stdout else ""
            # Fallback for dict-like responses
            if isinstance(result, dict):
                return str(result.get("stdout") or result.get("output") or "")
            return str(result)
        except Exception as e:
            logger.error(f"Failed to run shell command '{cmd}': {e}")
            return ""

    try:
        q = (task.validation_query or "").lower()
        expected = task.expected_result

        # 1) 当前前台 App (Settings / Chrome 等)
        if q == "current_app":
            out = await _run_shell(
                "dumpsys window | grep mCurrentFocus || "
                "dumpsys activity | grep mResumedActivity"
            )
            if not out:
                return False
            if isinstance(expected, str) and expected:
                return expected in out
            return "mCurrentFocus" in out or "mResumedActivity" in out

        # 2) WiFi 开关
        if q == "wifi_enabled":
            out = await _run_shell("settings get global wifi_on")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return bool(val) == bool(expected)

        # 3) 飞行模式
        if q == "airplane_mode":
            out = await _run_shell("settings get global airplane_mode_on")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return bool(val) == bool(expected)

        # 4) 亮度
        if q == "brightness_level":
            out = await _run_shell("settings get system screen_brightness")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return val == int(expected) if isinstance(expected, int) else False

        # 5) 当前 Activity（比如电池页）
        if q == "current_activity":
            out = await _run_shell(
                "dumpsys window | grep mCurrentFocus || "
                "dumpsys activity | grep mResumedActivity"
            )
            if not out:
                return False
            if isinstance(expected, str) and expected:
                return expected.lower() in out.lower()
            return False

        # 6) 是否在桌面
        if q == "is_home_screen":
            out = await _run_shell(
                "dumpsys window | grep mCurrentFocus || "
                "dumpsys activity | grep mResumedActivity"
            )
            if not out:
                return False
            lowered = out.lower()
            is_home = (
                "launcher" in lowered
                or "home" in lowered
                or "com.android.launcher" in lowered
                or "launcher3" in lowered
            )
            return is_home == bool(expected)

        # 7) 屏幕熄屏时间 (ms)
        if q == "screen_timeout":
            out = await _run_shell("settings get system screen_off_timeout")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return val == int(expected) if isinstance(expected, int) else False

        # 8) 勿扰模式 (DND / zen_mode)
        if q == "dnd_enabled":
            out = await _run_shell("settings get global zen_mode")
            try:
                val = int(out.strip())
            except ValueError:
                return False
            is_enabled = val > 0
            return is_enabled == bool(expected)

        # 9) 蓝牙开关（配合 train_08_enable_bluetooth）
        if q == "bluetooth_enabled":
            out = await _run_shell(
                "settings get global bluetooth_on || "
                "settings get secure bluetooth_on"
            )
            try:
                val = int(out.strip())
            except ValueError:
                return False
            return bool(val) == bool(expected)

        # 10) Result message contains check
        if q == "result_message_contains":
            if result_message is None:
                logger.warning(f"result_message_contains validation requires result_message parameter (task {task.id})")
                return False
            if isinstance(expected, str):
                return expected in result_message
            return str(expected) in result_message

        # 11) Media volume
        if q == "media_volume":
            out = await _run_shell("dumpsys audio | grep -A 10 'STREAM_MUSIC' | grep 'volume:' | head -1 || settings get system volume_music")
            try:
                import re
                volume_match = re.search(r'volume:\s*(\d+)/(\d+)', out)
                if volume_match:
                    current_vol = int(volume_match.group(1))
                    max_vol = int(volume_match.group(2))
                    val = int((current_vol / max_vol) * 100) if max_vol > 0 else 0
                else:
                    val = int(out.strip())
                return val == int(expected) if isinstance(expected, (int, float)) else False
            except (ValueError, AttributeError):
                return False

        # 12) Notification count
        if q == "notification_count":
            out = await _run_shell("dumpsys notification | grep -c 'NotificationRecord' || echo '0'")
            try:
                val = int(out.strip())
                return val == int(expected) if isinstance(expected, (int, float)) else False
            except ValueError:
                return False

        # 13) Auto time enabled
        if q == "auto_time_enabled":
            out = await _run_shell("settings get global auto_time")
            try:
                val = int(out.strip())
                return bool(val) == bool(expected)
            except ValueError:
                return False

        # 14) Battery saver enabled
        if q == "battery_saver_enabled":
            out = await _run_shell("settings get global low_power")
            try:
                val = int(out.strip())
                return bool(val) == bool(expected)
            except ValueError:
                return False

        # 15) Chrome cache cleared
        if q == "chrome_cache_cleared":
            out = await _run_shell("dumpsys package com.android.chrome | grep -A 5 'cacheSize' || pm clear com.android.chrome --dry-run")
            try:
                import re
                cache_match = re.search(r'cacheSize[=:]\s*(\d+)', out)
                if cache_match:
                    cache_size = int(cache_match.group(1))
                    return (cache_size < 1048576) == bool(expected)
                # If we can't find cache size, we can't confirm cache was cleared
                # Return False to be conservative (can't verify task completion)
                return False
            except (ValueError, AttributeError):
                return False

        # 16) Facebook cache cleared
        if q == "facebook_cache_cleared":
            out = await _run_shell("dumpsys package com.facebook.katana | grep -A 5 'cacheSize' || pm clear com.facebook.katana --dry-run")
            try:
                import re
                cache_match = re.search(r'cacheSize[=:]\s*(\d+)', out)
                if cache_match:
                    cache_size = int(cache_match.group(1))
                    return (cache_size < 1048576) == bool(expected)
                # If we can't find cache size, we can't confirm cache was cleared
                # Return False to be conservative (can't verify task completion)
                return False
            except (ValueError, AttributeError):
                return False

        # 17) Instagram cache cleared
        if q == "instagram_cache_cleared":
            out = await _run_shell("dumpsys package com.instagram.android | grep -A 5 'cacheSize' || pm clear com.instagram.android --dry-run")
            try:
                import re
                cache_match = re.search(r'cacheSize[=:]\s*(\d+)', out)
                if cache_match:
                    cache_size = int(cache_match.group(1))
                    return (cache_size < 1048576) == bool(expected)
                # If we can't find cache size, we can't confirm cache was cleared
                # Return False to be conservative (can't verify task completion)
                return False
            except (ValueError, AttributeError):
                return False

        # 18) System language
        if q == "system_language":
            out = await _run_shell("settings get system system_locales || getprop ro.product.locale")
            import re
            lang_match = re.search(r'([a-z]{2})[-_]', out.lower())
            if lang_match:
                actual_lang = lang_match.group(1)
            else:
                actual_lang = out.strip()[:2].lower()
            if isinstance(expected, str):
                return expected.lower() in actual_lang or actual_lang in expected.lower()
            return False

        # 19) Dark theme enabled
        if q == "dark_theme_enabled":
            out = await _run_shell("settings get secure ui_night_mode || settings get system dark_theme")
            try:
                val = int(out.strip())
                is_dark = val > 0
                return is_dark == bool(expected)
            except ValueError:
                return False

        # 20) App installed
        if q == "app_installed":
            if not isinstance(expected, dict) or "package" not in expected or "installed" not in expected:
                logger.warning(f"app_installed validation requires expected_result with 'package' and 'installed' keys (task {task.id})")
                return False
            package = expected["package"]
            expected_installed = expected["installed"]
            out = await _run_shell(f"pm list packages | grep -q '^{package}$' && echo '1' || echo '0'")
            try:
                is_installed = int(out.strip()) == 1
                return is_installed == bool(expected_installed)
            except ValueError:
                return False

        # 21) App permission
        if q == "app_permission":
            if not isinstance(expected, dict) or "package" not in expected or "permission" not in expected or "granted" not in expected:
                logger.warning(f"app_permission validation requires expected_result with 'package', 'permission', and 'granted' keys (task {task.id})")
                return False
            package = expected["package"]
            permission = expected["permission"]
            expected_granted = expected["granted"]
            out = await _run_shell(f"dumpsys package {package} | grep -A 2 '{permission}' | grep 'granted=true' && echo '1' || echo '0'")
            try:
                is_granted = int(out.strip()) == 1
                return is_granted == bool(expected_granted)
            except ValueError:
                return False

        # 22) File exists
        if q == "file_exists":
            if not isinstance(expected, dict) or "path" not in expected or "exists" not in expected:
                logger.warning(f"file_exists validation requires expected_result with 'path' and 'exists' keys (task {task.id})")
                return False
            file_path = expected["path"]
            expected_exists = expected["exists"]
            out = await _run_shell(f"test -e '{file_path}' && echo '1' || echo '0'")
            try:
                file_exists = int(out.strip()) == 1
                return file_exists == bool(expected_exists)
            except ValueError:
                return False

        # 23) Finish message contains
        if q == "finish_message_contains":
            if result_message is None:
                logger.warning(f"finish_message_contains validation requires result_message parameter (task {task.id})")
                return False
            if isinstance(expected, str):
                return expected in result_message
            return str(expected) in result_message

        # 24) Finish message contains size
        if q == "finish_message_contains_size":
            if result_message is None:
                logger.warning(f"finish_message_contains_size validation requires result_message parameter (task {task.id})")
                return False
            import re
            size_pattern = r'\d+\.?\d*\s*(KB|MB|GB|TB|bytes?)'
            has_size = bool(re.search(size_pattern, result_message, re.IGNORECASE))
            if isinstance(expected, str) and expected == "storage_size_reported":
                return has_size
            return str(expected) in result_message if expected else has_size

        logger.warning(f"No shell-based validator implemented for query '{q}' (task {task.id})")
        return False

    except Exception as e:
        logger.error(f"Validation failed for task {task.id}: {e}", exc_info=True)
        return False


@dataclass
class ADBValidationResult:
    """Result of ADB validation with detailed information."""
    command: str  # ADB/shell command executed
    expected_result: Any  # Expected result
    actual_result: str  # Actual output from command
    success: bool  # Whether validation passed
    execution_time: float  # Time taken to execute (seconds)
    validation_query: str  # Type of validation query (e.g., "wifi_enabled")


async def validate_task_completion_with_details(
    task: CUATask,
    gbox_client,
    result_message: Optional[str] = None,
) -> Optional[ADBValidationResult]:
    """Validate task completion using ADB/shell commands via GBox, returning detailed information.
    
    Args:
        task: The task to validate
        gbox_client: GBox client for executing shell commands
        result_message: Optional result message from rollout (for result_message_contains validation)
    
    Returns:
        ADBValidationResult with command, expected result, actual result, success, and execution time.
        Returns None if validation_query is not set or not supported.
    """
    import time
    
    if not task.validation_query:
        return None
    
    async def _run_shell(cmd: str) -> str:
        """Run a shell command inside the box via GBox SDK and return stdout."""
        sdk = getattr(gbox_client, "_sdk", None)
        box_id = getattr(gbox_client, "box_id", None)
        if sdk is None or not box_id:
            raise RuntimeError("gbox_client must provide `_sdk` and `box_id` for command execution")

        try:
            # Use SDK's get() method to get the box object, then use box.command()
            box = sdk.get(box_id)
            result = box.command(command=cmd)
            # result should have stdout, stderr, and exitCode attributes
            if hasattr(result, 'stdout'):
                return str(result.stdout) if result.stdout else ""
            # Fallback for dict-like responses
            if isinstance(result, dict):
                return str(result.get("stdout") or result.get("output") or "")
            return str(result)
        except Exception as e:
            logger.error(f"Failed to run shell command '{cmd}': {e}")
            return ""

    try:
        q = (task.validation_query or "").lower()
        expected = task.expected_result
        start_time = time.time()
        
        # Determine command based on query type
        command = ""
        actual_output = ""
        success = False
        
        # 1) 当前前台 App (Settings / Chrome 等)
        if q == "current_app":
            command = "dumpsys window | grep mCurrentFocus || dumpsys activity | grep mResumedActivity"
            actual_output = await _run_shell(command)
            if not actual_output:
                success = False
            elif isinstance(expected, str) and expected:
                success = expected in actual_output
            else:
                success = "mCurrentFocus" in actual_output or "mResumedActivity" in actual_output

        # 2) WiFi 开关
        elif q == "wifi_enabled":
            command = "settings get global wifi_on"
            actual_output = await _run_shell(command)
            try:
                val = int(actual_output.strip())
                success = bool(val) == bool(expected)
            except ValueError:
                success = False

        # 3) 飞行模式
        elif q == "airplane_mode":
            command = "settings get global airplane_mode_on"
            actual_output = await _run_shell(command)
            try:
                val = int(actual_output.strip())
                success = bool(val) == bool(expected)
            except ValueError:
                success = False

        # 4) 亮度
        elif q == "brightness_level":
            command = "settings get system screen_brightness"
            actual_output = await _run_shell(command)
            try:
                val = int(actual_output.strip())
                success = val == int(expected) if isinstance(expected, int) else False
            except ValueError:
                success = False

        # 7) 屏幕熄屏时间 (ms)
        elif q == "screen_timeout":
            command = "settings get system screen_off_timeout"
            actual_output = await _run_shell(command)
            try:
                val = int(actual_output.strip())
                success = val == int(expected) if isinstance(expected, int) else False
            except ValueError:
                success = False

        # 8) 勿扰模式 (DND / zen_mode)
        elif q == "dnd_enabled":
            command = "settings get global zen_mode"
            actual_output = await _run_shell(command)
            try:
                val = int(actual_output.strip())
                is_enabled = val > 0
                success = is_enabled == bool(expected)
            except ValueError:
                success = False

        # 12) Notification count
        elif q == "notification_count":
            command = "dumpsys notification | grep -c 'NotificationRecord' || echo '0'"
            actual_output = await _run_shell(command)
            try:
                val = int(actual_output.strip())
                success = val == int(expected) if isinstance(expected, (int, float)) else False
            except ValueError:
                success = False

        # 13) Auto time enabled
        elif q == "auto_time_enabled":
            command = "settings get global auto_time"
            actual_output = await _run_shell(command)
            try:
                val = int(actual_output.strip())
                success = bool(val) == bool(expected)
            except ValueError:
                success = False

        # 14) Battery saver enabled
        elif q == "battery_saver_enabled":
            command = "settings get global low_power"
            actual_output = await _run_shell(command)
            try:
                val = int(actual_output.strip())
                success = bool(val) == bool(expected)
            except ValueError:
                success = False

        # 20) App installed
        elif q == "app_installed":
            if not isinstance(expected, dict) or "package" not in expected or "installed" not in expected:
                logger.warning(f"app_installed validation requires expected_result with 'package' and 'installed' keys (task {task.id})")
                return None
            package = expected["package"]
            expected_installed = expected["installed"]
            command = f"pm list packages | grep -q '^package:{package}$' && echo '1' || echo '0'"
            actual_output = await _run_shell(command)
            try:
                is_installed = int(actual_output.strip()) == 1
                success = is_installed == bool(expected_installed)
            except ValueError:
                success = False

        # 22) File exists
        elif q == "file_exists":
            if not isinstance(expected, dict) or "path" not in expected or "exists" not in expected:
                logger.warning(f"file_exists validation requires expected_result with 'path' and 'exists' keys (task {task.id})")
                return None
            file_path = expected["path"]
            expected_exists = expected["exists"]
            command = f"test -e '{file_path}' && echo '1' || echo '0'"
            actual_output = await _run_shell(command)
            try:
                file_exists = int(actual_output.strip()) == 1
                success = file_exists == bool(expected_exists)
            except ValueError:
                success = False

        # 23) Finish message contains (same as result_message_contains)
        elif q == "finish_message_contains":
            if result_message is None:
                logger.warning(f"finish_message_contains validation requires result_message parameter (task {task.id})")
                return None
            command = "finish_message_validation"
            actual_output = result_message
            if isinstance(expected, str):
                success = expected in result_message
            else:
                success = str(expected) in result_message

        # 24) Finish message contains size (checks if storage size is reported)
        elif q == "finish_message_contains_size":
            if result_message is None:
                logger.warning(f"finish_message_contains_size validation requires result_message parameter (task {task.id})")
                return None
            command = "finish_message_size_validation"
            actual_output = result_message
            # Check if message contains size pattern (e.g., "20MB", "1.5GB", "500KB")
            import re
            size_pattern = r'\d+\.?\d*\s*(KB|MB|GB|TB|bytes?)'
            has_size = bool(re.search(size_pattern, result_message, re.IGNORECASE))
            if isinstance(expected, str) and expected == "storage_size_reported":
                success = has_size
            else:
                # Fallback: check if expected string is in message
                success = str(expected) in result_message if expected else has_size

        else:
            logger.warning(f"No shell-based validator implemented for query '{q}' (task {task.id})")
            return None
        
        execution_time = time.time() - start_time
        
        return ADBValidationResult(
            command=command,
            expected_result=expected,
            actual_result=actual_output.strip() if actual_output else "",
            success=success,
            execution_time=execution_time,
            validation_query=q,
        )

    except Exception as e:
        logger.error(f"Validation failed for task {task.id}: {e}", exc_info=True)
        return None
        
def calculate_grpo_advantages(
    rewards: List[float],
    min_std: float = 0.05,
) -> List[float]:
    """Calculate GRPO advantages for a group of rewards.
    
    GRPO uses group-relative normalization:
    advantage_i = (reward_i - mean(rewards)) / std(rewards)
    
    Args:
        rewards: List of rewards for a group
        min_std: Minimum standard deviation threshold
        
    Returns:
        List of advantages (or zeros if std < min_std)
    """
    import numpy as np
    
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Skip low-variance groups
    if std_reward < min_std:
        return [0.0] * len(rewards)
    
    # Normalize advantages
    advantages = [
        (r - mean_reward) / (std_reward + 1e-8)
        for r in rewards
    ]
    
    return advantages


__all__ = [
    "CUARolloutResult",
    "simple_reward_function",
    "completion_reward_function",
    "efficiency_reward_function",
    "shaped_reward_function",
    "comprehensive_reward_function",
    "create_rollout_result_from_dict",
    "RewardTracker",
    "validate_task_completion",
    "validate_task_completion_with_details",
    "ADBValidationResult",
    "calculate_grpo_advantages",
    "cua_reward_fn",
]


def cua_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    """CUA 奖励函数 - 直接使用 workflow 传入的终局奖励。
    
    这个函数用于 AReaL 训练框架，直接从 kwargs 中获取终局奖励。
    
    Args:
        prompt: 输入提示
        completions: 模型输出
        prompt_ids: 提示的 token IDs
        completion_ids: 输出的 token IDs
        answer: 答案（未使用）
        **kwargs: 包含 reward, task_success, task_completed 等字段
        
    Returns:
        奖励值（float）
    """
    if isinstance(completions, list):
        completion = completions[0] if completions else ""
    else:
        completion = completions
    
    # 直接使用 workflow 传入的 reward
    reward = kwargs.get("reward")
    if reward is None:
        task_success = kwargs.get("task_success", False)
        task_completed = kwargs.get("task_completed", False)
        reward = 1.0 if (task_success or task_completed) else 0.0
    
    try:
        reward = float(reward)
    except Exception:
        reward = 0.0
    
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        task_id = kwargs.get("task_id", "N/A")
        num_turns = kwargs.get("num_turns", 0)
        max_turns = kwargs.get("max_turns", 15)
        task_success = kwargs.get("task_success", False)
        errors = kwargs.get("errors", [])
        
        logger.info("=" * 80)
        logger.info(f"[CUA Rollout Complete] Task ID: {task_id}")
        logger.info(f"  Turns: {num_turns}/{max_turns}")
        logger.info(f"  Task Success: {task_success}")
        logger.info(f"  Reward: {reward:.4f}")
        if errors:
            logger.info(f"  Errors ({len(errors)}): {errors[:3]}")
        logger.info("=" * 80)
    
    return reward

