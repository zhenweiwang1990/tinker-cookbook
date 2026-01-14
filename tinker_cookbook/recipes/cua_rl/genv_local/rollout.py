from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.recipes.cua_rl.database.rollout_recorder import RolloutRecorder
from tinker_cookbook.recipes.cua_rl.genv_local.action_adapter import tool_call_to_genv_action
from tinker_cookbook.recipes.cua_rl.genv_local.env_manager import close_env, prepare_local_env
from tinker_cookbook.recipes.cua_rl.genv_local.recorder import GenvRolloutRecorder
from tinker_cookbook.recipes.cua_rl.utils.cua_prompts import create_system_prompt
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.utils.json_repair import extract_first_json

logger = logging.getLogger(__name__)


def _is_nonempty_ndarray_image(x: Any) -> bool:
    try:
        import numpy as np

        return isinstance(x, np.ndarray) and x.size > 0 and x.ndim == 3 and x.shape[2] in (3, 4)
    except Exception:
        return False


def _pick_step_current_screenshot(step_obs: dict[str, Any], fallback: Any) -> Any:
    """
    genv step() observations follow StepObservation:
      - screenshot_before (np.ndarray or empty np.array([]))
      - screenshot_after  (np.ndarray or empty np.array([]))
      - optional screenshot_before_annotated (np.ndarray)
    """
    after = step_obs.get("screenshot_after")
    if _is_nonempty_ndarray_image(after):
        return after
    before = step_obs.get("screenshot_before")
    if _is_nonempty_ndarray_image(before):
        return before
    return fallback


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(float(x)))))


def _coords_from_genv_action(genv_action: dict[str, Any], *, screen_w: int, screen_h: int) -> dict[str, Any] | None:
    """
    Convert a genv action dict into pixel coordinates used by training-monitor overlays.

    training-monitor expects pixel coords in action_results[*].coordinates:
      - tap/long_press: {x, y}
      - swipe: {start: {x,y}, end: {x,y}}
    """
    a_type = genv_action.get("type")
    if isinstance(a_type, int):
        action_type_map = {0: "tap", 1: "swipe", 2: "type", 3: "key", 4: "long_press"}
        a_type = action_type_map.get(a_type, "tap")
    if not isinstance(a_type, str):
        return None

    if a_type in {"tap", "long_press"}:
        x_norm = genv_action.get("x_norm")
        y_norm = genv_action.get("y_norm")
        if x_norm is None or y_norm is None:
            return None
        return {
            "x": _clamp_int(float(x_norm) * screen_w, 0, max(screen_w - 1, 0)),
            "y": _clamp_int(float(y_norm) * screen_h, 0, max(screen_h - 1, 0)),
        }

    if a_type == "swipe":
        x1 = genv_action.get("x_norm")
        y1 = genv_action.get("y_norm")
        x2 = genv_action.get("x2_norm")
        y2 = genv_action.get("y2_norm")
        if None in (x1, y1, x2, y2):
            return None
        return {
            "start": {
                "x": _clamp_int(float(x1) * screen_w, 0, max(screen_w - 1, 0)),
                "y": _clamp_int(float(y1) * screen_h, 0, max(screen_h - 1, 0)),
            },
            "end": {
                "x": _clamp_int(float(x2) * screen_w, 0, max(screen_w - 1, 0)),
                "y": _clamp_int(float(y2) * screen_h, 0, max(screen_h - 1, 0)),
            },
        }

    return None


def _image_wh_from_ndarray(arr: Any) -> tuple[int, int] | None:
    """
    Extract (width,height) from a screenshot ndarray (HxWxC).
    """
    try:
        import numpy as np

        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return None
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return None
        h = int(arr.shape[0])
        w = int(arr.shape[1])
        if w <= 0 or h <= 0:
            return None
        return (w, h)
    except Exception:
        return None


def _extract_raw_xy_from_tool_args(tool_args: dict[str, Any]) -> dict[str, Any] | None:
    """
    Best-effort extraction of the *raw* coordinates the model emitted (often 0-1000 scaled)
    so we can store them alongside pixel coordinates for debugging.
    """
    try:
        target = tool_args.get("target")
        if isinstance(target, dict):
            coords = target.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                return {"x": float(coords[0]), "y": float(coords[1])}
            if "x" in target and "y" in target:
                return {"x": float(target["x"]), "y": float(target["y"])}
    except Exception:
        return None
    return None


def _action_type_str_from_genv_action(genv_action: dict[str, Any]) -> str:
    a_type = genv_action.get("type")
    if isinstance(a_type, int):
        return {0: "tap", 1: "swipe", 2: "type", 3: "key", 4: "long_press"}.get(a_type, "unknown")
    if isinstance(a_type, str):
        return a_type
    return "unknown"


def _update_rollout_execution_details(
    rollout_recorder: GenvRolloutRecorder | None,
    *,
    execution_details_turns: list[dict[str, Any]],
    env_build: dict[str, Any] | None = None,
) -> None:
    """
    Persist minimal execution_details needed by training-monitor to show:
      - parse_success/parse_error
      - action_results (with pixel coordinates)

    training-monitor reads these from rollout.trajectory_data_json, so we need to write them
    even before the rollout completes.
    """
    if rollout_recorder is None:
        return
    try:
        ok = rollout_recorder.recorder.update(
            trajectory_data_json=json.dumps(
                {
                    "execution_details": {
                        "turns": execution_details_turns,
                        **({"env_build": env_build} if env_build is not None else {}),
                    }
                },
                ensure_ascii=False,
                default=str,
            )
        )
        if not ok:
            logger.warning("[genv_local] failed to update rollout.trajectory_data_json with execution_details")
    except Exception as e:
        logger.warning("[genv_local] failed to write execution_details to DB: %s", e)


def _require_genv_task_loader() -> Any:
    try:
        from genv.sdk.tasks import TaskLoader  # type: ignore

        return TaskLoader
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "genv is required for env_mode='genv_local'. "
            "Install it with: pip install -e /Users/zhenwei/workspace/genv-umetrip/rl"
        ) from e


def _parse_tool_calls_from_text(response_text: str) -> list[dict[str, Any]]:
    """
    Best-effort tool call extraction.

    Supported:
      - <tool_call>{\"name\":...,\"args\":...}</tool_call>
      - <function_calls>[...]</function_calls>
      - fallback: first JSON object with name+args
    """
    tool_calls: list[dict[str, Any]] = []

    # Pattern 1: <tool_call>...</tool_call>
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL):
        content = match.group(1).strip()
        content = re.sub(r"\}\s*,\s*\{\s*(\"[a-zA-Z0-9_]+\"\s*:)", r"}, \1", content)
        obj = extract_first_json(content, start_chars=("{",))
        if isinstance(obj, dict) and "name" in obj and "args" in obj:
            tool_calls.append(obj)

    # Pattern 2: <function_calls>...</function_calls>
    for match in re.finditer(r"<function_calls>(.*?)</function_calls>", response_text, re.DOTALL):
        content = match.group(1).strip()
        arr = extract_first_json(content, start_chars=("[",))
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, dict) and "name" in item and "args" in item:
                    tool_calls.append(item)

    # Pattern 3: first JSON tool call in raw text
    if not tool_calls:
        obj = extract_first_json(response_text, start_chars=("{",))
        if isinstance(obj, dict) and "name" in obj and "args" in obj:
            tool_calls.append(obj)

    return tool_calls


def _tool_calls_from_renderer(parsed_message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = []
    for tc in parsed_message.get("tool_calls", []) or []:
        if hasattr(tc, "function") and hasattr(tc.function, "name"):
            try:
                tool_calls.append({"name": tc.function.name, "args": json.loads(tc.function.arguments)})
            except Exception:
                continue
        elif isinstance(tc, dict) and "name" in tc and "args" in tc:
            tool_calls.append(tc)
    return tool_calls


def _compute_step_reward_from_checks(checks: dict[str, Any]) -> float:
    total = 0.0
    for _check_id, cr in (checks or {}).items():
        try:
            if int(cr.get("status", 2)) == 1:
                total += float(cr.get("score", 0.0))
        except Exception:
            continue
    return float(total)


def _get_model_path_from_policy(policy: TokenCompleter, model_path: Optional[str]) -> str:
    if model_path is not None:
        if not isinstance(model_path, str):
            raise ValueError(f"Invalid model_path type: {type(model_path)}")
        return model_path
    if isinstance(policy, TinkerTokenCompleter):
        sampling_client = policy.sampling_client
        mp = getattr(sampling_client, "model_path", None)
        if isinstance(mp, str) and mp:
            return mp
        holder = getattr(sampling_client, "holder", None)
        if holder is not None:
            mp = getattr(holder, "model_path", None) or getattr(holder, "_model_path", None)
            if isinstance(mp, str) and mp:
                return mp
    raise ValueError("model_path not provided and could not be inferred from policy")


@dataclass
class _EpisodeResult:
    trajectory: Trajectory
    total_reward: float
    metrics: dict[str, Any]
    summary: dict[str, Any]


async def _run_single_episode(
    *,
    task: Any,
    env: Any,
    policy: TokenCompleter,
    renderer: Any,
    model_path: str,
    max_turns: int,
    max_recent_turns: int,
    rollout_recorder: Optional[GenvRolloutRecorder],
    rollout_id: str,
    source_type: str,
) -> _EpisodeResult:
    """
    Run one genv episode (reset + step loop) and return a Trajectory.
    """
    episode_start_time = time.time()
    # Load task config (env + evaluation) from genv tasks dir.
    TaskLoader = _require_genv_task_loader()
    tasks_dir = getattr(task, "genv_tasks_dir", None)
    identifier = getattr(task, "genv_identifier", None)
    if not isinstance(tasks_dir, str) or not isinstance(identifier, str):
        raise ValueError("CUATask missing genv_tasks_dir/genv_identifier metadata")

    loader = TaskLoader(tasks_dir=tasks_dir)
    resolved = loader.find_task(identifier) or identifier
    task_obj = loader.load(resolved)
    task_data: dict[str, Any] = task_obj.data

    meta = task_data.get("meta") or {}
    task_id_for_context = str(meta.get("id") or getattr(task, "id"))

    evaluation = task_data.get("evaluation") or {}
    graphql_config = evaluation.get("graphql") or {}
    checks = evaluation.get("checks") or []

    # env_config is already embedded (merged with defaults) by TaskLoader.
    env_config = task_data.get("env") or {}
    reset_db_cfg = (env_config.get("reset") or {}).get("db") or {}
    deep_link_cfg = (env_config.get("reset") or {}).get("app") or {}
    deep_link_cfg = deep_link_cfg.get("deepLink") if isinstance(deep_link_cfg, dict) else None

    # IMPORTANT:
    # genv's reset/database seeding logic needs to know where the task definitions live (tasks_dir)
    # for tasks that use GraphQL seeds / external env.yaml files. If not provided, genv falls back
    # to cwd-relative defaults like "./tasks", which breaks when benchmark is run from a different
    # repo (e.g. tinker-cookbook).
    if isinstance(reset_db_cfg, dict):
        reset_db_cfg = {**reset_db_cfg, "tasksDir": tasks_dir}

    # genv reset needs db_seed + deep_link in options.
    reset_obs, reset_info = env.reset(
        options={
            "reset_db": True,
            "db_seed": reset_db_cfg,
            "deep_link": deep_link_cfg,
            "checks": checks,
            "graphql_config": graphql_config,
            "context_vars": {"taskId": task_id_for_context},
        }
    )
    execution_id = str(reset_info.get("execution_id") or "")

    # Persist reset metadata in Env Build tab (but do NOT create a synthetic "turn 0").
    # training-monitor reads this from rollout.trajectory_data_json.execution_details.env_build.
    #
    # Also persist the reset screenshot as a dedicated observation on turn 1, so the monitor
    # can show it without embedding large base64 blobs in trajectory_data_json.
    if rollout_recorder is not None:
        try:
            rollout_recorder.record_screenshot(turn=1, obs_type="env_reset_screenshot", screenshot=reset_obs.get("screenshot"))  # type: ignore[arg-type]
        except Exception:
            pass
    env_build: dict[str, Any] = {
        "status": "success",
        "total_time": None,
        "stages": [
            {
                "name": "env_reset",
                "status": "success",
                "duration": None,
                "details": {
                    "execution_id": execution_id,
                    "taskId": task_id_for_context,
                    "deep_link": deep_link_cfg,
                    # Avoid huge blobs; keep the structured reset metadata.
                    "reset_info": reset_info,
                },
            }
        ],
        # Reference the reset screenshot without embedding it in trajectory_data_json.
        "reset_screenshot": {"turn": 1, "obs_type": "env_reset_screenshot"},
        # These fields exist in the monitor UI; keep them stable even if unused for genv_local.
        "prehook_executed": False,
        "prehook_output": None,
    }
    _update_rollout_execution_details(
        rollout_recorder,
        execution_details_turns=[],
        env_build=env_build,
    )

    # Build initial system prompt.
    screen_w, screen_h = env.adb_device.get_screen_size()

    # Provider-aware eval-only mode: core/train.py (baseline_only + provider!=tinker) passes a dummy
    # sampling_client through TinkerTokenCompleter. We detect provider here so system prompt uses the
    # correct tool-call format (e.g. Doubao magic tags).
    provider = "tinker"
    provider_model_name = None
    provider_base_url = None
    provider_api_key = None
    if isinstance(policy, TinkerTokenCompleter):
        sc = getattr(policy, "sampling_client", None)
        provider = str(getattr(sc, "provider", "tinker") or "tinker").lower()
        provider_model_name = getattr(sc, "model_path", None) or getattr(sc, "model_name", None)
        provider_base_url = getattr(sc, "provider_base_url", None)
        provider_api_key = getattr(sc, "provider_api_key", None)
    use_provider_inference = provider != "tinker"

    system_prompt = create_system_prompt(
        task_description=getattr(task, "description", ""),
        max_turns=max_turns,
        box_type="android",
        provider=provider,
        coordinate_mode="direct",
        # Direct mode in this codebase typically uses 0-1000 scaled coordinates.
        # We'll convert to genv's normalized (0-1) in action_adapter.py.
        coordinate_scale=True,
        screen_width=screen_w,
        screen_height=screen_h,
        cua_guide="You need to operate the Umetrip app to complete the task.",
    )

    messages: list[renderers.Message] = [renderers.Message(role="system", content=system_prompt)]

    def _truncate_messages(msgs: list[Any]) -> list[Any]:
        # Keep system prompt + last N turns (approx).
        if not msgs:
            return msgs
        if len(msgs) <= 1:
            return msgs
        # naive: keep last 2*max_recent_turns + system
        return [msgs[0]] + msgs[-(2 * max_recent_turns) :]

    stop = renderer.get_stop_sequences()

    trajectory_turns: list[tuple[int, tinker.ModelInput, TokensWithLogprobs, float, bool]] = []
    # training-monitor expects rollout.trajectory_data_json.execution_details.turns[*].action_results[*].coordinates
    execution_details_turns: list[dict[str, Any]] = []
    total_reward = 0.0
    final_result: Optional[dict[str, Any]] = None
    # NOTE:
    # For genv_local, we intentionally do NOT treat env-provided terminated/truncated as a hard stop.
    # This rollout loop should only stop when:
    #   - the model calls `finish`, OR
    #   - we hit max_turns
    #
    # We still record env termination flags for debugging and for downstream summaries, but they do
    # not control the loop.
    terminated = False
    truncated = False
    last_obs: Optional[dict[str, Any]] = None
    last_response_text = ""

    # Start from reset screenshot for turn 1.
    screenshot = reset_obs.get("screenshot")

    for turn in range(1, max_turns + 1):
        turn_start = time.time()

        user_text = f"Turn {turn}/{max_turns}. Analyze the screenshot and take the next action to complete the task."
        # Build message with screenshot + text.
        from PIL import Image as PILImage
        import numpy as np

        if isinstance(screenshot, np.ndarray) and screenshot.size > 0:
            pil_img = PILImage.fromarray(screenshot.astype("uint8"))
            user_msg = renderers.Message(
                role="user",
                content=[
                    renderers.ImagePart(type="image", image=pil_img),
                    renderers.TextPart(type="text", text=user_text),
                ],
            )
        else:
            user_msg = renderers.Message(role="user", content=user_text)

        messages.append(user_msg)

        truncated_messages = _truncate_messages(messages)

        if use_provider_inference:
            # This rollout code stores (turn, model_input, tokens/logprobs) in trajectory_turns.
            # For provider inference we don't have a "true" token-level observation prompt, so
            # we store an empty ModelInput (benchmark-only path).
            model_input = tinker.ModelInput.empty()

            from tinker_cookbook.recipes.cua_rl.agent.inference_client_factory import create_inference_client

            inf = create_inference_client(
                provider=provider,
                model_name=str(provider_model_name or model_path),
                base_url=provider_base_url,
                api_key=provider_api_key,
                base_model_name=getattr(getattr(policy, "sampling_client", None), "model_name", None)
                or "Qwen/Qwen3-VL-30B-A3B-Instruct",
                renderer=renderer,
                tokenizer=renderer.tokenizer,
            )
            text = await inf.generate_text(
                messages=truncated_messages,  # type: ignore[arg-type]
                max_tokens=getattr(policy, "max_tokens", 2048),
                temperature=getattr(policy, "temperature", 1.0),
            )

            tokens = renderer.tokenizer.encode(text, add_special_tokens=False)
            # Best-effort append stop sequence for renderers that require one.
            try:
                stops = renderer.get_stop_sequences()
                if stops:
                    if isinstance(stops[0], int):
                        tokens = tokens + [stops[0]]
                    elif isinstance(stops[0], str):
                        tokens = tokens + renderer.tokenizer.encode(stops[0], add_special_tokens=False)
            except Exception:
                pass

            sampled = TokensWithLogprobs(tokens=tokens, maybe_logprobs=None)
            parsed_message, parse_success = renderer.parse_response(tokens)
            last_response_text = str(parsed_message.get("content") or "")
        else:
            model_input = renderer.build_generation_prompt(truncated_messages)
            # Sample tokens/logprobs.
            sampled = await policy(model_input, stop)
            parsed_message, parse_success = renderer.parse_response(sampled.tokens)
            last_response_text = str(parsed_message.get("content") or "")

        tool_calls = _tool_calls_from_renderer(parsed_message)
        if not tool_calls:
            tool_calls = _parse_tool_calls_from_text(last_response_text)

        # Always append assistant message text so the model has its own reasoning/history.
        messages.append(renderers.Message(role="assistant", content=last_response_text))

        # Execute tool calls (we only affect the env for 'action'; 'finish' ends the rollout loop).
        step_reward = 0.0
        turn_action_results: list[dict[str, Any]] = []
        turn_parse_error: str | None = None
        finish_called = False
        # Track whether env ever reported done during this turn (for logging only).
        turn_env_terminated = False
        turn_env_truncated = False
        # For action overlay, prefer the current screenshot's true pixel size (may differ from adb reported size).
        img_wh = _image_wh_from_ndarray(screenshot)
        img_w = img_wh[0] if img_wh else screen_w
        img_h = img_wh[1] if img_wh else screen_h
        for tc in tool_calls:
            name = str(tc.get("name") or "")
            args = tc.get("args") or {}
            if not isinstance(args, dict):
                args = {}

            if name == "finish":
                finish_called = True
                # Mark rollout as "truncated" in the conventional RL sense (agent chose to stop),
                # but do not rely on env termination to end the episode.
                truncated = True
                break

            genv_action = tool_call_to_genv_action(
                name,
                args,
                screen_width=screen_w,
                screen_height=screen_h,
                adb_device=env.adb_device,
            )

            # Build action_result for training-monitor (used to render the overlay).
            # Note: monitor reads these from rollout.trajectory_data_json (NOT the action table).
            action_result: dict[str, Any] = {
                "action_type": _action_type_str_from_genv_action(genv_action),
            }
            coords = _coords_from_genv_action(genv_action, screen_w=img_w, screen_h=img_h)
            if coords is not None:
                action_result["coordinates"] = coords
                action_result["coordinates_scaled"] = False
                raw_xy = _extract_raw_xy_from_tool_args(args)
                if raw_xy is not None:
                    # Expose the raw model-emitted coords (often 0-1000) for debugging.
                    action_result["original_coordinates"] = raw_xy
                    action_result["coordinates_scaled"] = True
            if "text" in genv_action and genv_action.get("text") is not None:
                action_result["text"] = genv_action.get("text")
            if "key" in genv_action and genv_action.get("key") is not None:
                action_result["key"] = genv_action.get("key")
            if "duration_ms" in genv_action and genv_action.get("duration_ms") is not None:
                action_result["duration"] = genv_action.get("duration_ms")
            if "error" in genv_action:
                # This indicates we couldn't map/execute the requested tool call.
                action_result["error"] = str(genv_action.get("error_message") or genv_action.get("error") or "")
                turn_parse_error = action_result["error"] or "action_mapping_error"
            turn_action_results.append(action_result)

            # Record action + mapping.
            if rollout_recorder is not None:
                try:
                    rollout_recorder.record_action(
                        turn=turn,
                        action_type=str(genv_action.get("type") or "unknown"),
                        tool_name=name,
                        tool_args=args,
                        tokens=sampled.tokens,
                        logprobs=sampled.maybe_logprobs,
                    )
                    rollout_recorder.record_json_obs(turn=turn, obs_type="genv_action", payload=genv_action)
                except Exception:
                    pass

            # If the requested action isn't supported by genv, do NOT step the env.
            # Instead, provide explicit feedback to the model in the next turn.
            if name == "action":
                supported_types = {"tap", "swipe", "type", "key", "long_press"}
                a_type = genv_action.get("type")
                if (
                    "error" in genv_action
                    or not isinstance(a_type, str)
                    or a_type not in supported_types
                ):
                    err_msg = str(
                        genv_action.get("error_message")
                        or genv_action.get("error")
                        or f"Unsupported genv action type: {a_type!r}"
                    )
                    messages.append(
                        renderers.Message(
                            role="user",
                            content=(
                                "Environment does not support this operation. "
                                f"Reason: {err_msg}\n"
                                "Supported actions: tap, swipe, type, key(home/back/menu), long_press. "
                                "Please choose a supported action."
                            ),
                        )
                    )
                    break

            try:
                obs, _env_reward, env_terminated, env_truncated, info = env.step(genv_action)
            except Exception as e:
                # If the environment errors (e.g. stepping after it has internally terminated),
                # do not crash the rollout. We keep looping until finish/max_turns.
                err_msg = f"env_step_error: {e}"
                action_result["error"] = err_msg
                turn_parse_error = turn_parse_error or err_msg
                continue

            last_obs = obs
            # Record env termination flags for debugging, but do NOT break on them.
            turn_env_terminated = turn_env_terminated or bool(env_terminated)
            turn_env_truncated = turn_env_truncated or bool(env_truncated)
            terminated = terminated or bool(env_terminated)
            truncated = truncated or bool(env_truncated)

            # Compute reward from checks (genv default reward impl is inconsistent with status ints).
            checks = obs.get("checks") or {}
            step_reward += _compute_step_reward_from_checks(checks)

            # Screenshots (see genv.sdk.envs.types.StepObservation)
            screenshot_before = obs.get("screenshot_before")
            screenshot_after = obs.get("screenshot_after")
            screenshot_before_annotated = obs.get("screenshot_before_annotated")
            screenshot = _pick_step_current_screenshot(obs, screenshot)

            if rollout_recorder is not None:
                try:
                    if screenshot_before is not None:
                        rollout_recorder.record_screenshot(turn=turn, obs_type="screenshot_before", screenshot=screenshot_before)  # type: ignore[arg-type]
                    if screenshot_after is not None:
                        rollout_recorder.record_screenshot(turn=turn, obs_type="screenshot_after", screenshot=screenshot_after)  # type: ignore[arg-type]
                    if screenshot_before_annotated is not None:
                        rollout_recorder.record_screenshot(
                            turn=turn,
                            obs_type="screenshot_before_annotated",
                            screenshot=screenshot_before_annotated,
                        )
                    rollout_recorder.record_json_obs(turn=turn, obs_type="checks", payload=checks)
                    if "final_result" in obs:
                        rollout_recorder.record_json_obs(turn=turn, obs_type="final_result", payload=obs.get("final_result"))
                except Exception:
                    pass

            # IMPORTANT: Do NOT break on env termination; keep looping until finish/max_turns.

        total_reward += step_reward
        # Only stop the episode when the model calls finish, or we hit max_turns.
        episode_done = bool(finish_called or turn >= max_turns)
        if rollout_recorder is not None:
            try:
                rollout_recorder.record_turn_end(
                    turn=turn,
                    reward=float(step_reward),
                    episode_done=episode_done,
                    model_response_text=last_response_text,
                    metrics={
                        "parse_success": bool(parse_success),
                        # For monitoring/debugging: env flags are still recorded, but they do not end the loop.
                        "terminated": bool(turn_env_terminated),
                        "truncated": bool(turn_env_truncated or finish_called),
                        "finish_called": bool(finish_called),
                        "step_reward": float(step_reward),
                    },
                    turn_time=time.time() - turn_start,
                )
            except Exception:
                pass
        trajectory_turns.append((turn, model_input, sampled, step_reward, episode_done))
        execution_details_turns.append(
            {
                "turn_num": int(turn),
                # training-monitor uses parse_success/parse_error to display "Action Parsed" status.
                "parse_success": bool(tool_calls) and (turn_parse_error is None),
                "parse_error": turn_parse_error,
                "action_results": turn_action_results,
            }
        )
        # Update DB after each turn so the monitor can show action parsing/coords immediately.
        _update_rollout_execution_details(
            rollout_recorder,
            execution_details_turns=execution_details_turns,
            env_build=env_build,
        )
        if episode_done:
            break

    # Final evaluation snapshot
    if last_obs and "final_result" in last_obs:
        final_result = last_obs.get("final_result")
    # Ensure final execution_details are persisted.
    _update_rollout_execution_details(
        rollout_recorder,
        execution_details_turns=execution_details_turns,
        env_build=env_build,
    )

    # Build Trajectory (per-turn rewards).
    transitions: list[Transition] = []
    for turn, ob, ac, r, done in trajectory_turns:
        transitions.append(
            Transition(
                ob=ob,
                ac=TokensWithLogprobs(tokens=ac.tokens, maybe_logprobs=ac.maybe_logprobs),
                reward=float(r),
                episode_done=bool(done),
                metrics={
                    "turn": turn,
                    "step_reward": float(r),
                    "terminated": float(terminated) if done else 0.0,
                    "truncated": float(truncated) if done else 0.0,
                },
            )
        )

    traj = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())
    rollout_time = time.time() - episode_start_time
    summary = {
        "rollout_id": rollout_id,
        "task_id": getattr(task, "id", "unknown"),
        "execution_id": execution_id,
        "num_turns": len(trajectory_turns),
        "terminated": terminated,
        "truncated": truncated,
        "total_reward": float(total_reward),
        "final_result": final_result,
        "rollout_time": float(rollout_time),
    }
    metrics = {
        "task_success": float(bool(final_result and final_result.get("success"))),
        "reward": float(total_reward),
        "num_turns": len(trajectory_turns),
    }

    return _EpisodeResult(trajectory=traj, total_reward=float(total_reward), metrics=metrics, summary=summary)


async def do_genv_group_rollout(
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter,
    model_path: str | None = None,
    step: int | None = None,
    batch: int | None = None,
    group: int | None = None,
    output_dir: str | None = None,
    is_eval: bool = False,
    db_session=None,
    step_id: int | None = None,
    eval_id: int | None = None,
    baseline_id: int | None = None,
) -> TrajectoryGroup:
    """
    Group rollout for genv_local.

    Important: this implementation runs rollouts **serially** within the group and also
    uses a global semaphore to avoid concurrent LocalEnv usage across groups.
    """
    _ = output_dir
    model_path_str = _get_model_path_from_policy(policy, model_path)

    # Create env wrappers (cheap).
    envs = await env_group_builder.make_envs()
    if not envs:
        raise ValueError("No envs created for group rollout")

    # Validate renderer/task presence.
    first_env = envs[0]
    task = getattr(first_env, "task", None)
    renderer = getattr(first_env, "renderer", None)
    max_turns = int(getattr(first_env, "max_turns", 20))
    max_recent_turns = int(getattr(first_env, "max_recent_turns", 5))
    if task is None or renderer is None:
        raise ValueError("GenvEnv instances must have task and renderer")

    # Determine source_type for DB grouping.
    source_type: Optional[str] = None
    # Priority: explicit IDs > context lookup. Never guess "eval" without an eval_id.
    if step_id is not None:
        source_type = "step"
    else:
        # Try context IDs when caller didn't pass them.
        if (baseline_id is None or eval_id is None) and is_eval:
            try:
                from tinker_cookbook.recipes.cua_rl.database.database_context import (
                    get_baseline_id,
                    get_eval_id,
                )

                baseline_id = baseline_id if baseline_id is not None else get_baseline_id()
                eval_id = eval_id if eval_id is not None else get_eval_id()
            except Exception:
                pass

        if baseline_id is not None:
            source_type = "baseline"
        elif eval_id is not None:
            source_type = "eval"
        else:
            # If we don't have a DB parent ID, we skip DB recording for this rollout.
            source_type = None

    # If caller didn't pass a db_session, try to get one from global database context.
    if db_session is None:
        try:
            from tinker_cookbook.recipes.cua_rl.database.database_context import get_database_session

            db_session = get_database_session()
        except Exception:
            db_session = None

    # Global serialization for LocalEnv usage.
    from tinker_cookbook.recipes.cua_rl.genv_local.env_manager import acquire_env_slot

    async with (await acquire_env_slot()):
        prepared = prepare_local_env(env_config=_load_env_config_for_task(task))
        try:
            results: list[_EpisodeResult] = []

            for env_idx in range(len(envs)):
                rollout_uuid = str(uuid.uuid4())

                # Per-rollout DB session (avoid cross-rollout contamination).
                rollout_db_session = None
                if db_session is not None and source_type is not None:
                    from tinker_cookbook.recipes.cua_rl.database.database import get_session_direct

                    rollout_db_session = get_session_direct()

                rollout_recorder = None
                rr: RolloutRecorder | None = None
                if rollout_db_session is not None and source_type is not None:
                    rr = RolloutRecorder(rollout_db_session, rollout_uuid)
                    ok = rr.start_rollout(
                        task_id_str=str(getattr(task, "id", "")),
                        task_description=str(getattr(task, "description", "")),
                        model_path=model_path_str,
                        env_type="android",
                        source_type=source_type,
                        step_id=step_id,
                        eval_id=eval_id,
                        baseline_id=baseline_id,
                        batch=batch,
                        group_num=group,
                        env_index=env_idx,
                        is_eval=is_eval,
                        group_id=None,
                        box_type="android",
                        max_turns=max_turns,
                    )
                    if ok:
                        rollout_recorder = GenvRolloutRecorder(recorder=rr, rollout_uuid=rollout_uuid)

                ep = await _run_single_episode(
                    task=task,
                    env=prepared.env,
                    policy=policy,
                    renderer=renderer,
                    model_path=model_path_str,
                    max_turns=max_turns,
                    max_recent_turns=max_recent_turns,
                    rollout_recorder=rollout_recorder,
                    rollout_id=rollout_uuid,
                    source_type=source_type or "unknown",
                )
                results.append(ep)

                # Mark rollout completion in DB.
                if rr is not None:
                    try:
                        final_result = ep.summary.get("final_result") or {}
                        success = bool(isinstance(final_result, dict) and final_result.get("success", False))
                        rr.record_validation(
                            success=success,
                            validation_query="genv_graphql",
                            expected_result=None,
                            actual_result=json.dumps(final_result, ensure_ascii=False, default=str),
                            execution_time=None,
                            error_message=None,
                            screenshot_uri=None,
                            details_json={"final_result": final_result, "summary": ep.summary},
                        )
                        rr.complete_rollout(
                            task_completed=bool(ep.summary.get("terminated", False)),
                            task_success=success,
                            agent_reported_success=success,
                            validation_passed=success,
                            num_turns=int(ep.summary.get("num_turns", 0)),
                            reward=float(ep.total_reward),
                            rollout_time=float(ep.summary.get("rollout_time", 0.0) or 0.0),
                            max_turns=max_turns,
                            model_path=model_path_str,
                        )
                    except Exception:
                        pass

                # Close per-rollout DB session.
                if rollout_db_session is not None:
                    rollout_db_session.close()

            return TrajectoryGroup(
                trajectories_G=[r.trajectory for r in results],
                final_rewards_G=[r.total_reward for r in results],
                metrics_G=[r.metrics for r in results],
            )
        finally:
            close_env(prepared.env)


def _load_env_config_for_task(task: Any) -> dict[str, Any]:
    """
    Load env_config dict for the given CUATask from genv tasks directory.
    """
    TaskLoader = _require_genv_task_loader()
    tasks_dir = getattr(task, "genv_tasks_dir", None)
    identifier = getattr(task, "genv_identifier", None)
    if not isinstance(tasks_dir, str) or not isinstance(identifier, str):
        raise ValueError("CUATask missing genv_tasks_dir/genv_identifier metadata")

    loader = TaskLoader(tasks_dir=tasks_dir)
    resolved = loader.find_task(identifier) or identifier
    task_obj = loader.load(resolved)
    task_data: dict[str, Any] = task_obj.data
    env_config = task_data.get("env") or {}
    if not isinstance(env_config, dict):
        raise ValueError("genv task env config is not a dict")
    return env_config

