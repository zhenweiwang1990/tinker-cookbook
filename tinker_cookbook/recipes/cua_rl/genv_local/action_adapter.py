from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedPoint:
    x_norm: float
    y_norm: float


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _px_to_norm(x_px: float, y_px: float, width: int, height: int) -> NormalizedPoint:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid screen size: {width}x{height}")
    return NormalizedPoint(x_norm=_clamp01(float(x_px) / float(width)), y_norm=_clamp01(float(y_px) / float(height)))


def _maybe_scale_1000_to_px(x: float, y: float, *, screen_width: int, screen_height: int) -> tuple[tuple[float, float], bool]:
    """
    In CUA direct mode, coordinates are often emitted in a normalized 0-1000 space.
    genv expects normalized (0-1), so we first map 0-1000 -> pixels -> normalized.

    Returns:
      - (x_px, y_px)
      - did_scale: whether a 0-1000 conversion was applied
    """
    # Heuristic: if both x/y look like 0-1000 coordinates and the device is larger than 1000px,
    # treat them as scaled.
    try:
        xf = float(x)
        yf = float(y)
    except Exception:
        return ((x, y), False)

    if screen_width > 1000 and screen_height > 1000 and 0.0 <= xf <= 1000.0 and 0.0 <= yf <= 1000.0:
        return ((xf / 1000.0) * float(screen_width), (yf / 1000.0) * float(screen_height)), True
    return ((xf, yf), False)


def _extract_xy_from_target(target: Any) -> Optional[tuple[float, float]]:
    """
    Extract pixel coordinates (x, y) from a target-like object/dict.

    We support the structures produced by TargetElement in tinker_cua_agent:
    - {"coordinates":[x,y]}
    - {"x":..., "y":...}
    - TargetElement(coordinates=[x,y])
    """
    if target is None:
        return None

    if isinstance(target, dict):
        coords = target.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return (float(coords[0]), float(coords[1]))
        if "x" in target and "y" in target:
            try:
                return (float(target["x"]), float(target["y"]))
            except Exception:
                return None
        return None

    # Pydantic / dataclass objects
    coords = getattr(target, "coordinates", None)
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return (float(coords[0]), float(coords[1]))

    x = getattr(target, "x", None)
    y = getattr(target, "y", None)
    if x is not None and y is not None:
        try:
            return (float(x), float(y))
        except Exception:
            return None

    return None


def _resolve_test_id_center(adb_device: Any, test_id: str) -> Optional[tuple[float, float]]:
    if not adb_device or not test_id:
        return None
    try:
        el = adb_device.find_element_by_testid(test_id)
        if not isinstance(el, dict):
            return None
        cx = el.get("center_x")
        cy = el.get("center_y")
        if cx is None or cy is None:
            return None
        return (float(cx), float(cy))
    except Exception:
        return None


def tool_call_to_genv_action(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    screen_width: int,
    screen_height: int,
    adb_device: Any,
) -> dict[str, Any]:
    """
    Convert a CUA tool call (renderer/tool schema) into a genv env action dict.

    genv supports: tap/swipe/type/key/long_press.
    We keep failure explicit by returning {"error":..., "error_message":...} which triggers truncated in genv StepMixin.
    """
    if tool_name == "wait":
        # genv does not have a native wait action, but StepMixin treats unknown types as no-op.
        # We still return something explicit for logging.
        duration_s = float(tool_args.get("duration", 1.0) or 1.0)
        return {"type": "wait", "duration_ms": int(duration_s * 1000)}

    if tool_name != "action":
        return {
            "error": "unknown_tool",
            "error_message": f"Unsupported tool_name={tool_name!r} (expected 'action' or 'wait')",
        }

    action_type = str(tool_args.get("action_type") or "").strip().lower()
    duration = tool_args.get("duration")
    duration_ms = int(float(duration) * 1000) if isinstance(duration, (int, float)) else None

    target = tool_args.get("target")
    start_target = tool_args.get("start_target")
    end_target = tool_args.get("end_target")

    # Optional: testID-based targeting (preferred over OCR selector in genv_local).
    test_id = None
    if isinstance(target, dict):
        test_id = target.get("test_id") or target.get("testId") or target.get("testID")

    # Resolve coordinates for single-point actions.
    xy_from_tool = _extract_xy_from_target(target)
    xy = xy_from_tool
    if xy is None and isinstance(test_id, str) and test_id:
        xy = _resolve_test_id_center(adb_device, test_id)

    if action_type in {"tap", "click"}:
        if xy is None:
            return {"error": "coordinates_missing", "error_message": "tap/click requires target coordinates or test_id"}
        # If coordinates came from tool args, they may be in 0-1000 space; convert first.
        if xy_from_tool is not None:
            (x_px, y_px), _did_scale = _maybe_scale_1000_to_px(
                xy[0], xy[1], screen_width=screen_width, screen_height=screen_height
            )
        else:
            x_px, y_px = xy[0], xy[1]
        pt = _px_to_norm(x_px, y_px, screen_width, screen_height)
        return {
            "type": "tap",
            "x_norm": pt.x_norm,
            "y_norm": pt.y_norm,
            "duration_ms": int(duration_ms or 100),
        }

    if action_type in {"long_press", "longpress"}:
        if xy is None:
            return {"error": "coordinates_missing", "error_message": "long_press requires target coordinates or test_id"}
        if xy_from_tool is not None:
            (x_px, y_px), _did_scale = _maybe_scale_1000_to_px(
                xy[0], xy[1], screen_width=screen_width, screen_height=screen_height
            )
        else:
            x_px, y_px = xy[0], xy[1]
        pt = _px_to_norm(x_px, y_px, screen_width, screen_height)
        return {
            "type": "long_press",
            "x_norm": pt.x_norm,
            "y_norm": pt.y_norm,
            "duration_ms": int(duration_ms or 600),
        }

    if action_type in {"type", "input"}:
        text = tool_args.get("text")
        if not isinstance(text, str) or not text:
            return {"error": "text_missing", "error_message": "type/input requires non-empty text"}
        return {"type": "type", "text": text}

    # In the original CUA tool schema, device key presses are expressed as "button_press"
    # (e.g. {"action_type":"button_press","button":"back"}). genv uses {"type":"key","key":...}.
    if action_type in {"key", "press_key", "key_press", "button_press"}:
        # Support both keys=[...] and button="BACK".
        keys = tool_args.get("keys")
        button = tool_args.get("button")
        key_name = None
        if isinstance(button, str) and button:
            key_name = button
        elif isinstance(keys, list) and keys and isinstance(keys[0], str):
            key_name = keys[0]
        if not key_name:
            return {"error": "key_missing", "error_message": "key action requires 'button' or non-empty 'keys' list"}
        return {"type": "key", "key": str(key_name)}

    if action_type in {"swipe", "drag", "scroll"}:
        # Prefer explicit start/end targets.
        start_xy_from_tool = _extract_xy_from_target(start_target) or (xy_from_tool if xy_from_tool is not None else None)
        end_xy_from_tool = _extract_xy_from_target(end_target)
        start_xy = start_xy_from_tool or (xy if xy is not None else None)
        end_xy = end_xy_from_tool

        if action_type == "scroll" and end_xy is None:
            # Convert scroll(direction,distance) into a swipe gesture.
            direction = str(tool_args.get("direction") or "down").lower()
            distance = float(tool_args.get("distance") or 0.5)
            distance = max(0.1, min(0.9, distance))
            cx = screen_width * 0.5
            cy = screen_height * 0.5
            if direction in {"down"}:
                start_xy = (cx, cy - screen_height * distance * 0.4)
                end_xy = (cx, cy + screen_height * distance * 0.4)
            elif direction in {"up"}:
                start_xy = (cx, cy + screen_height * distance * 0.4)
                end_xy = (cx, cy - screen_height * distance * 0.4)
            elif direction in {"left"}:
                start_xy = (cx + screen_width * distance * 0.4, cy)
                end_xy = (cx - screen_width * distance * 0.4, cy)
            elif direction in {"right"}:
                start_xy = (cx - screen_width * distance * 0.4, cy)
                end_xy = (cx + screen_width * distance * 0.4, cy)
            else:
                return {"error": "direction_invalid", "error_message": f"Unsupported scroll direction: {direction!r}"}

        if start_xy is None or end_xy is None:
            return {
                "error": "coordinates_missing",
                "error_message": "swipe/drag/scroll requires start_target and end_target (or scroll direction/distance)",
            }

        # If coordinates came from tool args, they may be in 0-1000 space; convert first.
        if start_xy_from_tool is not None:
            (sx, sy), _ = _maybe_scale_1000_to_px(start_xy[0], start_xy[1], screen_width=screen_width, screen_height=screen_height)
        else:
            sx, sy = start_xy[0], start_xy[1]
        if end_xy_from_tool is not None:
            (ex, ey), _ = _maybe_scale_1000_to_px(end_xy[0], end_xy[1], screen_width=screen_width, screen_height=screen_height)
        else:
            ex, ey = end_xy[0], end_xy[1]

        p1 = _px_to_norm(sx, sy, screen_width, screen_height)
        p2 = _px_to_norm(ex, ey, screen_width, screen_height)
        return {
            "type": "swipe",
            "x_norm": p1.x_norm,
            "y_norm": p1.y_norm,
            "x2_norm": p2.x_norm,
            "y2_norm": p2.y_norm,
            "duration_ms": int(duration_ms or 300),
        }

    return {
        "error": "action_type_unknown",
        "error_message": f"Unsupported action_type={action_type!r}",
    }

