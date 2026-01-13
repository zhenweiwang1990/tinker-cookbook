from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np
from PIL import Image

from tinker_cookbook.recipes.cua_rl.database.rollout_recorder import RolloutRecorder

logger = logging.getLogger(__name__)


def _is_nonempty_uint8_image(arr: Any) -> bool:
    if not isinstance(arr, np.ndarray):
        return False
    if arr.size == 0:
        return False
    # Expect HxWxC (RGB/RGBA). Some genv code uses np.array([]) sentinel.
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return False
    return True


def ndarray_to_png_data_uri(arr: np.ndarray) -> str:
    """
    Convert a HxWxC numpy array into a PNG data URI.
    """
    if arr.size == 0:
        return ""

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported screenshot array shape: {arr.shape}")

    img = Image.fromarray(arr.astype("uint8"))
    buf = BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@dataclass
class GenvRolloutRecorder:
    """
    Thin wrapper over RolloutRecorder that knows how to record genv observations.
    """

    recorder: RolloutRecorder
    rollout_uuid: str

    def record_screenshot(self, *, turn: int, obs_type: str, screenshot: Any) -> None:
        # genv screenshots are np.ndarray; treat empty arrays as missing.
        if not _is_nonempty_uint8_image(screenshot):
            return
        data_uri = ndarray_to_png_data_uri(screenshot)
        if not data_uri:
            return
        self.recorder.record_observation(
            turn_num=turn,
            obs_type=obs_type,
            screenshot_uri=data_uri,
            rollout_id=self.rollout_uuid,
        )

    def record_json_obs(self, *, turn: int, obs_type: str, payload: Any) -> None:
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = str(payload)
        self.recorder.record_observation(
            turn_num=turn,
            obs_type=obs_type,
            text_content=text,
            rollout_id=self.rollout_uuid,
        )

    def record_action(
        self,
        *,
        turn: int,
        action_type: str,
        tool_name: Optional[str],
        tool_args: Optional[dict[str, Any]],
        tokens: Optional[list[int]] = None,
        logprobs: Optional[list[float]] = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if tool_name is not None:
            kwargs["tool_name"] = tool_name
        if tool_args is not None:
            kwargs["tool_args"] = tool_args
        if tokens is not None:
            kwargs["tokens"] = tokens
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        self.recorder.record_action(turn_num=turn, action_type=action_type, **kwargs)

    def record_turn_end(
        self,
        *,
        turn: int,
        reward: float,
        episode_done: bool,
        model_response_text: str,
        metrics: Optional[dict[str, Any]] = None,
        turn_time: Optional[float] = None,
    ) -> None:
        self.recorder.record_turn_wrapper(
            turn_num=turn,
            model_response=model_response_text,
            reward=reward,
            episode_done=episode_done,
            metrics=metrics,
            turn_time=turn_time,
        )

