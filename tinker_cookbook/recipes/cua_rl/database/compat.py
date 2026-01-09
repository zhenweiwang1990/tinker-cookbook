"""
Backward-compatible database recording functions for agent.

This module provides compatibility wrappers that allow old database recording
code to work with the new RolloutRecorder system.
"""

import logging
import contextvars
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Per-async-task rollout recorder instance (set by agent).
# This is CRITICAL: baseline/eval runs many rollouts concurrently; using a plain
# global causes cross-rollout contamination (turns/actions/obs recorded into the
# wrong rollout).
_recorder_var: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "cua_rollout_recorder",
    default=None,
)

def set_recorder(recorder):
    """Set the current rollout recorder for this asyncio task/context."""
    _recorder_var.set(recorder)

def get_recorder():
    """Get the current rollout recorder."""
    return _recorder_var.get()


def record_turn(
    session,  # Ignored - kept for compatibility
    rollout_id,  # Ignored - kept for compatibility
    turn: int,
    model_response: str,
    reward: float = 0.0,
    episode_done: bool = False,
    metrics: Optional[Dict[str, Any]] = None,
    turn_time: Optional[float] = None,
    **kwargs
) -> Optional[int]:
    """
    Backward-compatible turn recording function.
    
    Routes calls to RolloutRecorder if available.
    """
    logger.info(f"[compat] record_turn called: turn={turn}, rollout_id={rollout_id}, model_response_len={len(model_response) if model_response else 0}")
    recorder = get_recorder()
    if recorder is None:
        logger.warning(f"[compat] No rollout recorder available for turn {turn} (rollout_id={rollout_id})")
        return None
    
    logger.info(f"[compat] Found recorder, calling record_turn_wrapper for turn {turn}")
    try:
        result = recorder.record_turn_wrapper(
            turn_num=turn,
            model_response=model_response,
            reward=reward,
            episode_done=episode_done,
            metrics=metrics,
            turn_time=turn_time,
        )
        logger.info(f"[compat] record_turn_wrapper returned: {result}")
        return result
    except Exception as e:
        logger.error(f"[compat] Failed to record turn {turn}: {e}", exc_info=True)
        return None

def record_action(
    session,
    turn_id,  # Ignored - we get turn from recorder context
    action_type: str = None,
    **kwargs
) -> Optional[int]:
    """
    Backward-compatible action recording function.
    
    Routes calls to RolloutRecorder if available.
    """
    logger.info(f"[compat] record_action called: turn_id={turn_id}, action_type={action_type}")
    recorder = get_recorder()
    if recorder is None:
        logger.warning(f"[compat] No rollout recorder available for action recording")
        return None
    
    # Extract turn number from kwargs if present (fallback)
    turn_num = kwargs.pop('turn', None)
    if turn_num is None:
        # Try to infer from current recorder state
        logger.warning(f"[compat] No turn number provided for action recording")
        return None
    
    logger.info(f"[compat] Found recorder, calling record_action for turn {turn_num}")
    try:
        result = recorder.record_action(
            turn_num=turn_num,
            action_type=action_type,
            **kwargs
        )
        logger.info(f"[compat] record_action returned: {result}")
        return result
    except Exception as e:
        logger.error(f"[compat] Failed to record action: {e}", exc_info=True)
        return None

def record_observation(
    session,
    turn_id,  # Ignored - we get turn from recorder context
    **kwargs
) -> Optional[int]:
    """
    Backward-compatible observation recording function.
    
    Routes calls to RolloutRecorder if available.
    """
    logger.info(f"[compat] record_observation called: turn_id={turn_id}, kwargs_keys={list(kwargs.keys())}")
    recorder = get_recorder()
    if recorder is None:
        logger.warning(f"[compat] No rollout recorder available for observation recording")
        return None
    
    # Extract turn number from kwargs if present
    turn_num = kwargs.pop('turn', None)
    if turn_num is None:
        logger.warning(f"[compat] No turn number provided for observation recording")
        return None
    
    logger.info(f"[compat] Found recorder, calling record_observation for turn {turn_num}")
    try:
        result = recorder.record_observation(
            turn_num=turn_num,
            **kwargs
        )
        logger.info(f"[compat] record_observation returned: {result}")
        return result
    except Exception as e:
        logger.error(f"[compat] Failed to record observation: {e}", exc_info=True)
        return None

def record_rollout_status(
    session,
    rollout_id,
    status: str,
    **kwargs
):
    """Backward-compatible rollout status recording."""
    recorder = get_recorder()
    if recorder is None:
        return
    
    try:
        recorder.update_status(status=status, **kwargs)
    except Exception as e:
        logger.error(f"[compat] Failed to update rollout status: {e}")

