"""Tool definitions and implementations for CUA Agent."""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from tinker_cookbook.recipes.cua_rl.gbox.client import CuaGBoxClient
from tinker_cookbook.recipes.cua_rl.gbox.coordinate import CuaGBoxCoordinateGenerator
from tinker_cookbook.recipes.cua_rl.core.rollout_logger import RolloutLogger

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Input Models
# ============================================================================

class TargetElement(BaseModel):
    """Target element description."""
    model_config = {"extra": "forbid"}
    
    element: str = Field(description="Description of the UI element (required)")
    label: Optional[str] = Field(None, description="Text label on the element")
    color: Optional[str] = Field(None, description="Color of the element")
    size: Optional[str] = Field(None, description="Size of the element (small, medium, large)")
    location: Optional[str] = Field(None, description="Location on screen")
    shape: Optional[str] = Field(None, description="Shape of the element")


# ============================================================================
# Tool Implementations
# ============================================================================

def target_to_description(target: Optional[TargetElement]) -> str:
    """Convert target to description string for coordinate generation."""
    if not target:
        return "center of the screen"
    
    parts = [target.element]
    if target.label:
        parts.append(f'labeled "{target.label}"')
    if target.color:
        parts.append(f"{target.color} colored")
    if target.size:
        parts.append(f"{target.size} sized")
    if target.location:
        parts.append(f"located at {target.location}")
    if target.shape:
        parts.append(f"with {target.shape} shape")
    
    return " ".join(parts)


async def perform_action_impl(
    action_type: str,
    option: Optional[str] = None,
    target: Optional[TargetElement] = None,
    start_target: Optional[TargetElement] = None,
    end_target: Optional[TargetElement] = None,
    direction: Optional[str] = None,
    distance: Optional[int] = None,
    text: Optional[str] = None,
    keys: Optional[List[str]] = None,
    button: Optional[Any] = None,  # Can be str or List[str] for multiple buttons
    duration: Optional[str] = None,
    gbox_client: Optional[CuaGBoxClient] = None,
    screenshot_uri: Optional[str] = None,
    coord_generator: Optional[CuaGBoxCoordinateGenerator] = None,
    rollout_logger: Optional[RolloutLogger] = None,
) -> Dict[str, Any]:
    """Execute a UI action on GBox environment."""
    action_start_time = time.time()
    if not rollout_logger:
        logger.info(f"[Tool: perform_action] Starting action execution")
        logger.info(f"[Tool: perform_action] Action type: {action_type}")
    
    if not gbox_client:
        raise ValueError("gbox_client is required")
    if not screenshot_uri:
        raise ValueError("screenshot_uri is required")
    if not coord_generator:
        raise ValueError("coord_generator is required")
    
    box = gbox_client._get_box()
    box_type = gbox_client.box_type
    
    if action_type == "tap":
        # Tap action for Android devices
        target_desc = target_to_description(target)
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Tap target: {target_desc}")
        
        # Generate coordinates
        coord_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Generating coordinates for tap...")
        result = await coord_generator.generate_coordinates(
            screenshot_uri=screenshot_uri,
            action_type="tap",
            target=target_desc,
            rollout_logger=rollout_logger,
        )
        coord_time = time.time() - coord_start
        
        coords = result.get("response", {}).get("coordinates", {}) or result.get("coordinates", {})
        x, y = coords.get("x", 0), coords.get("y", 0)
        
        # Execute tap using GBox tap API
        tap_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Executing tap action at ({x}, {y})...")
        result = box.action.tap(x=x, y=y)
        tap_time = time.time() - tap_start
        action_total_time = time.time() - action_start_time
        
        # Log using rollout_logger if available
        if rollout_logger:
            rollout_logger.log_action(
                action_type="tap",
                target_desc=target_desc,
                coordinates={"x": x, "y": y},
                coord_time=coord_time,
                exec_time=tap_time,
                total_time=action_total_time,
            )
        else:
            logger.info(f"[Tool: perform_action] ✓ Coordinates generated in {coord_time:.3f}s")
            logger.info(f"[Tool: perform_action] Generated coordinates: x={x}, y={y}")
            logger.info(f"[Tool: perform_action] ✓ Tap executed in {tap_time:.3f}s")
            logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
        
        return {"action": "tap", "target": target_desc, "coords": {"x": x, "y": y}}
    
    elif action_type == "click":
        target_desc = target_to_description(target)
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Click target: {target_desc}")
            logger.info(f"[Tool: perform_action] Click option: {option or 'left'}")
        
        # Generate coordinates
        coord_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Generating coordinates for click...")
        result = await coord_generator.generate_coordinates(
            screenshot_uri=screenshot_uri,
            action_type="click",
            target=target_desc,
            rollout_logger=rollout_logger,
        )
        coord_time = time.time() - coord_start
        
        coords = result.get("response", {}).get("coordinates", {}) or result.get("coordinates", {})
        x, y = coords.get("x", 0), coords.get("y", 0)
        
        button_type = option or "left"
        double_click = button_type == "double"
        
        # Execute click
        click_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Executing click action at ({x}, {y})...")
        result = box.action.click(x=x, y=y, button=button_type, double=double_click)
        click_time = time.time() - click_start
        action_total_time = time.time() - action_start_time
        
        # Log using rollout_logger if available
        if rollout_logger:
            rollout_logger.log_action(
                action_type="click",
                target_desc=target_desc,
                coordinates={"x": x, "y": y},
                coord_time=coord_time,
                exec_time=click_time,
                total_time=action_total_time,
            )
        else:
            logger.info(f"[Tool: perform_action] ✓ Coordinates generated in {coord_time:.3f}s")
            logger.info(f"[Tool: perform_action] Generated coordinates: x={x}, y={y}")
            logger.info(f"[Tool: perform_action] ✓ Click executed in {click_time:.3f}s")
            logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
        
        return {"action": "click", "target": target_desc, "coords": {"x": x, "y": y}}
    
    elif action_type == "swipe":
        start_desc = target_to_description(start_target)
        end_desc = target_to_description(end_target)
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Swipe start target: {start_desc}")
            logger.info(f"[Tool: perform_action] Swipe end target: {end_desc}")
        
        # Generate coordinates for drag
        coord_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Generating coordinates for swipe...")
        result = await coord_generator.generate_coordinates(
            screenshot_uri=screenshot_uri,
            action_type="drag",
            target=start_desc,
            end_target=end_desc,
            rollout_logger=rollout_logger,
        )
        coord_time = time.time() - coord_start
        
        response_data = result.get("response", {}) or result
        coordinates = response_data.get("coordinates", {})
        
        if "start" in coordinates and "end" in coordinates:
            start_coords = coordinates.get("start", {})
            end_coords = coordinates.get("end", {})
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Using combined coordinates response")
        else:
            # Fallback: separate calls
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Using fallback: separate coordinate calls")
            start_coord_start = time.time()
            start_result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="click",
                target=start_desc,
                rollout_logger=rollout_logger,
            )
            start_coord_time = time.time() - start_coord_start
            
            end_coord_start = time.time()
            end_result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="click",
                target=end_desc,
                rollout_logger=rollout_logger,
            )
            end_coord_time = time.time() - end_coord_start
            
            start_coords = (start_result.get("response", {}) or start_result).get("coordinates", {})
            end_coords = (end_result.get("response", {}) or end_result).get("coordinates", {})
            coord_time = start_coord_time + end_coord_time
        
        start_x, start_y = start_coords.get("x", 0), start_coords.get("y", 0)
        end_x, end_y = end_coords.get("x", 0), end_coords.get("y", 0)
        
        # Execute swipe
        swipe_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Executing swipe action...")
        result = box.action.swipe(
            start={"x": start_x, "y": start_y},
            end={"x": end_x, "y": end_y},
            duration="300ms",
        )
        swipe_time = time.time() - swipe_start
        action_total_time = time.time() - action_start_time
        
        # Log using rollout_logger if available
        if rollout_logger:
            rollout_logger.log_action(
                action_type="swipe",
                start_target=start_desc,
                end_target=end_desc,
                coordinates={"start": start_coords, "end": end_coords},
                coord_time=coord_time,
                exec_time=swipe_time,
                total_time=action_total_time,
            )
        else:
            logger.info(f"[Tool: perform_action] ✓ Coordinates generated in {coord_time:.3f}s")
            logger.info(f"[Tool: perform_action] Swipe coordinates: start=({start_x}, {start_y}), end=({end_x}, {end_y})")
            logger.info(f"[Tool: perform_action] ✓ Swipe executed in {swipe_time:.3f}s")
            logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
        
        return {"action": "swipe", "start": start_coords, "end": end_coords}
    
    elif action_type == "drag":
        # Drag action for both PC and Android
        start_desc = target_to_description(start_target)
        end_desc = target_to_description(end_target)
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Drag start target: {start_desc}")
            logger.info(f"[Tool: perform_action] Drag end target: {end_desc}")
        
        # Generate coordinates for drag
        coord_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Generating coordinates for drag...")
        result = await coord_generator.generate_coordinates(
            screenshot_uri=screenshot_uri,
            action_type="drag",
            target=start_desc,
            end_target=end_desc,
            rollout_logger=rollout_logger,
        )
        coord_time = time.time() - coord_start
        
        response_data = result.get("response", {}) or result
        coordinates = response_data.get("coordinates", {})
        
        if "start" in coordinates and "end" in coordinates:
            start_coords = coordinates.get("start", {})
            end_coords = coordinates.get("end", {})
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Using combined coordinates response")
        else:
            # Fallback: separate calls
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Using fallback: separate coordinate calls")
            start_coord_start = time.time()
            start_result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="click",
                target=start_desc,
                rollout_logger=rollout_logger,
            )
            start_coord_time = time.time() - start_coord_start
            
            end_coord_start = time.time()
            end_result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="click",
                target=end_desc,
                rollout_logger=rollout_logger,
            )
            end_coord_time = time.time() - end_coord_start
            
            start_coords = (start_result.get("response", {}) or start_result).get("coordinates", {})
            end_coords = (end_result.get("response", {}) or end_result).get("coordinates", {})
            coord_time = start_coord_time + end_coord_time
        
        start_x, start_y = start_coords.get("x", 0), start_coords.get("y", 0)
        end_x, end_y = end_coords.get("x", 0), end_coords.get("y", 0)
        
        # Execute drag
        drag_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Executing drag action...")
        drag_duration = duration or "50ms"
        result = box.action.drag(
            start={"x": start_x, "y": start_y},
            end={"x": end_x, "y": end_y},
            duration=drag_duration,
        )
        drag_time = time.time() - drag_start
        action_total_time = time.time() - action_start_time
        
        # Log using rollout_logger if available
        if rollout_logger:
            rollout_logger.log_action(
                action_type="drag",
                start_target=start_desc,
                end_target=end_desc,
                coordinates={"start": start_coords, "end": end_coords},
                coord_time=coord_time,
                exec_time=drag_time,
                total_time=action_total_time,
            )
        else:
            logger.info(f"[Tool: perform_action] ✓ Coordinates generated in {coord_time:.3f}s")
            logger.info(f"[Tool: perform_action] Drag coordinates: start=({start_x}, {start_y}), end=({end_x}, {end_y})")
            logger.info(f"[Tool: perform_action] ✓ Drag executed in {drag_time:.3f}s")
            logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
        
        return {"action": "drag", "start": start_coords, "end": end_coords}
    
    elif action_type == "scroll":
        target_desc = target_to_description(target)
        direction = direction or "down"
        scroll_distance = distance or 300
        
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Scroll target: {target_desc}")
            logger.info(f"[Tool: perform_action] Scroll direction: {direction}")
            logger.info(f"[Tool: perform_action] Scroll distance: {scroll_distance}")
        
        # Generate coordinates
        coord_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Generating coordinates for scroll...")
        result = await coord_generator.generate_coordinates(
            screenshot_uri=screenshot_uri,
            action_type="scroll",
            target=target_desc,
            direction=direction,
            rollout_logger=rollout_logger,
        )
        coord_time = time.time() - coord_start
        
        coords = result.get("response", {}).get("coordinates", {}) or result.get("coordinates", {})
        x, y = coords.get("x", 0), coords.get("y", 0)
        
        # Calculate scroll_x and scroll_y from direction
        scroll_x = 0
        scroll_y = 0
        if direction == "up":
            scroll_y = scroll_distance
        elif direction == "down":
            scroll_y = -scroll_distance
        elif direction == "left":
            scroll_x = scroll_distance
        elif direction == "right":
            scroll_x = -scroll_distance
        
        # Execute scroll using GBox scroll API
        scroll_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Executing scroll action at ({x}, {y}) with scroll_x={scroll_x}, scroll_y={scroll_y}...")
        result = box.action.scroll(
            x=x,
            y=y,
            scroll_x=scroll_x,
            scroll_y=scroll_y,
        )
        scroll_time = time.time() - scroll_start
        action_total_time = time.time() - action_start_time
        
        # Log using rollout_logger if available
        if rollout_logger:
            rollout_logger.log_action(
                action_type="scroll",
                target_desc=f"{target_desc} ({direction}, distance={scroll_distance})",
                coordinates={"x": x, "y": y},
                coord_time=coord_time,
                exec_time=scroll_time,
                total_time=action_total_time,
            )
        else:
            logger.info(f"[Tool: perform_action] ✓ Coordinates generated in {coord_time:.3f}s")
            logger.info(f"[Tool: perform_action] Generated coordinates: x={x}, y={y}")
            logger.info(f"[Tool: perform_action] ✓ Scroll executed in {scroll_time:.3f}s")
            logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
        
        return {"action": "scroll", "direction": direction, "coords": {"x": x, "y": y}, "scroll_x": scroll_x, "scroll_y": scroll_y}
    
    elif action_type == "input":
        text = text or ""
        target_dict = target
        
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Input text length: {len(text)} characters")
            logger.info(f"[Tool: perform_action] Input text (first 100 chars): {text[:100]}")
        
        if target_dict:
            target_desc = target_to_description(target_dict)
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Input target: {target_desc}")
            
            # Generate coordinates and click/tap first based on box type
            coord_start = time.time()
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Generating coordinates for input target...")
            result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="click" if box_type == "linux" else "tap",
                target=target_desc,
                rollout_logger=rollout_logger,
            )
            coord_time = time.time() - coord_start
            
            coords = result.get("response", {}).get("coordinates", {}) or result.get("coordinates", {})
            x, y = coords.get("x", 0), coords.get("y", 0)
            
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Clicking/tapping target at ({x}, {y}) before input...")
            click_start = time.time()
            if box_type == "android":
                box.action.tap(x=x, y=y)
            else:
                box.action.click(x=x, y=y)
            click_time = time.time() - click_start
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] ✓ Target clicked/tapped in {click_time:.3f}s")
        
        # Type text
        type_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Typing text...")
        result = box.action.type(text=text)
        type_time = time.time() - type_start
        action_total_time = time.time() - action_start_time
        
        # Log using rollout_logger if available
        if rollout_logger:
            # Calculate total coord time (includes click/tap if target exists)
            total_coord_time = coord_time if target_dict else 0
            total_exec_time = (click_time if target_dict else 0) + type_time
            rollout_logger.log_action(
                action_type="input",
                target_desc=target_desc if target_dict else f"text input ({len(text)} chars)",
                coordinates={"x": x, "y": y} if target_dict else None,
                coord_time=total_coord_time if target_dict else None,
                exec_time=total_exec_time,
                total_time=action_total_time,
            )
        else:
            logger.info(f"[Tool: perform_action] ✓ Text typed in {type_time:.3f}s")
            logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
        
        return {"action": "input", "text": text}
    
    elif action_type == "key_press":
        keys = keys or []
        
        key_start = time.time()
        result = box.action.press_key(keys=keys, combination=len(keys) > 1)
        key_time = time.time() - key_start
        action_total_time = time.time() - action_start_time
        
        # Log using rollout_logger if available
        if rollout_logger:
            rollout_logger.log_action(
                action_type="key_press",
                target_desc=f"keys={keys}",
                coordinates=None,
                coord_time=None,
                exec_time=key_time,
                total_time=action_total_time,
            )
        else:
            logger.info(f"[Tool: perform_action] ✓ Key press: {keys}, Combination: {len(keys) > 1}, Executed in {key_time:.3f}s, Total time: {action_total_time:.3f}s")
        
        return {"action": "key_press", "keys": keys}
    
    elif action_type == "button_press":
        # Valid button values according to GBox API
        VALID_BUTTONS = {
            "power", "volumeUp", "volumeDown", "volumeMute", 
            "home", "back", "menu", "appSwitch"
        }
        
        # Button name normalization map (common variations -> valid names)
        # Keys are lowercase for case-insensitive matching
        BUTTON_NORMALIZE = {
            "volume_up": "volumeUp",
            "volume-up": "volumeUp",
            "volume_down": "volumeDown",
            "volume-down": "volumeDown",
            "volume_mute": "volumeMute",
            "volume-mute": "volumeMute",
            "app_switch": "appSwitch",
            "app-switch": "appSwitch",
            "recent": "appSwitch",
            "recent_apps": "appSwitch",
        }
        
        # Support both single button and multiple buttons
        if button:
            buttons_list = [button] if isinstance(button, str) else button
        else:
            buttons_list = ["home"] if box_type == "android" else ["power"]
        
        # Normalize and validate button values
        normalized_buttons = []
        for btn in buttons_list:
            # First check if it's already a valid button (case-insensitive)
            btn_lower = btn.lower()
            if btn in VALID_BUTTONS:
                # Already valid, use as-is
                normalized_btn = btn
            elif btn_lower in BUTTON_NORMALIZE:
                # Normalize common variations
                normalized_btn = BUTTON_NORMALIZE[btn_lower]
            else:
                # Try case-insensitive match against valid buttons
                matched = None
                for valid_btn in VALID_BUTTONS:
                    if valid_btn.lower() == btn_lower:
                        matched = valid_btn
                        break
                if matched:
                    normalized_btn = matched
                else:
                    raise ValueError(
                        f"Invalid button value: '{btn}'. "
                        f"Valid values are: {sorted(VALID_BUTTONS)}"
                    )
            normalized_buttons.append(normalized_btn)
        
        buttons_list = normalized_buttons
        
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Button press: {buttons_list}")
        
        button_start = time.time()
        result = box.action.press_button(buttons=buttons_list)
        button_time = time.time() - button_start
        action_total_time = time.time() - action_start_time
        
        # Log using rollout_logger if available
        if rollout_logger:
            rollout_logger.log_action(
                action_type="button_press",
                target_desc=f"buttons={buttons_list}",
                coordinates=None,
                coord_time=None,
                exec_time=button_time,
                total_time=action_total_time,
            )
        else:
            logger.info(f"[Tool: perform_action] ✓ Button press executed in {button_time:.3f}s")
            logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
        
        return {"action": "button_press", "buttons": buttons_list}
    
    elif action_type == "touch":
        # Touch action for Android devices (multi-point touch gestures)
        # This is a simplified version - full touch API supports complex multi-point gestures
        # For now, we'll use it as an alternative to swipe with more control
        if box_type != "android":
            raise ValueError("touch action is only available for Android devices")
        
        start_desc = target_to_description(start_target)
        end_desc = target_to_description(end_target)
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Touch start target: {start_desc}")
            logger.info(f"[Tool: perform_action] Touch end target: {end_desc}")
        
        # Generate coordinates
        coord_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Generating coordinates for touch...")
        result = await coord_generator.generate_coordinates(
            screenshot_uri=screenshot_uri,
            action_type="drag",
            target=start_desc,
            end_target=end_desc,
            rollout_logger=rollout_logger,
        )
        coord_time = time.time() - coord_start
        
        response_data = result.get("response", {}) or result
        coordinates = response_data.get("coordinates", {})
        
        if "start" in coordinates and "end" in coordinates:
            start_coords = coordinates.get("start", {})
            end_coords = coordinates.get("end", {})
        else:
            # Fallback: separate calls
            start_result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="tap",
                target=start_desc,
                rollout_logger=rollout_logger,
            )
            end_result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="tap",
                target=end_desc,
                rollout_logger=rollout_logger,
            )
            start_coords = (start_result.get("response", {}) or start_result).get("coordinates", {})
            end_coords = (end_result.get("response", {}) or end_result).get("coordinates", {})
        
        start_x, start_y = start_coords.get("x", 0), start_coords.get("y", 0)
        end_x, end_y = end_coords.get("x", 0), end_coords.get("y", 0)
        
        # Execute touch using GBox touch API
        touch_start = time.time()
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Executing touch action...")
        touch_duration = duration or "200ms"
        result = box.action.touch(
            points=[
                {
                    "start": {"x": start_x, "y": start_y},
                    "actions": [
                        {"x": end_x, "y": end_y, "duration": touch_duration}
                    ]
                }
            ]
        )
        touch_time = time.time() - touch_start
        action_total_time = time.time() - action_start_time
        
        # Log using rollout_logger if available
        if rollout_logger:
            rollout_logger.log_action(
                action_type="touch",
                start_target=start_desc,
                end_target=end_desc,
                coordinates={"start": start_coords, "end": end_coords},
                coord_time=coord_time,
                exec_time=touch_time,
                total_time=action_total_time,
            )
        else:
            logger.info(f"[Tool: perform_action] ✓ Coordinates generated in {coord_time:.3f}s")
            logger.info(f"[Tool: perform_action] Touch coordinates: start=({start_x}, {start_y}), end=({end_x}, {end_y})")
            logger.info(f"[Tool: perform_action] ✓ Touch executed in {touch_time:.3f}s")
            logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
        
        return {"action": "touch", "start": start_coords, "end": end_coords}
    
    elif action_type == "long_press":
        target_desc = target_to_description(target)
        if not rollout_logger:
            logger.info(f"[Tool: perform_action] Long press target: {target_desc}")
            logger.info(f"[Tool: perform_action] Long press duration: {duration or 'default'}")
        
        # Generate coordinates if target is provided
        if target:
            coord_start = time.time()
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Generating coordinates for long press...")
            result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="click" if box_type == "linux" else "tap",
                target=target_desc,
                rollout_logger=rollout_logger,
            )
            coord_time = time.time() - coord_start
            
            coords = result.get("response", {}).get("coordinates", {}) or result.get("coordinates", {})
            x, y = coords.get("x", 0), coords.get("y", 0)
            
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] ✓ Coordinates generated in {coord_time:.3f}s")
                logger.info(f"[Tool: perform_action] Generated coordinates: x={x}, y={y}")
            
            # Execute long press with coordinates
            long_press_start = time.time()
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Executing long press action at ({x}, {y})...")
            if duration:
                result = box.action.long_press(x=x, y=y, duration=duration)
            else:
                result = box.action.long_press(x=x, y=y)
            long_press_time = time.time() - long_press_start
            action_total_time = time.time() - action_start_time
            
            # Log using rollout_logger if available
            if rollout_logger:
                rollout_logger.log_action(
                    action_type="long_press",
                    target_desc=target_desc,
                    coordinates={"x": x, "y": y},
                    coord_time=coord_time,
                    exec_time=long_press_time,
                    total_time=action_total_time,
                )
            else:
                logger.info(f"[Tool: perform_action] ✓ Long press executed in {long_press_time:.3f}s")
                logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
            
            return {"action": "long_press", "target": target_desc, "coords": {"x": x, "y": y}, "duration": duration}
        else:
            # Execute long press with natural language target (GBox SDK supports this)
            long_press_start = time.time()
            if not rollout_logger:
                logger.info(f"[Tool: perform_action] Executing long press action with target description...")
            if duration:
                result = box.action.long_press(target=target_desc, duration=duration)
            else:
                result = box.action.long_press(target=target_desc)
            long_press_time = time.time() - long_press_start
            action_total_time = time.time() - action_start_time
            
            # Log using rollout_logger if available
            if rollout_logger:
                rollout_logger.log_action(
                    action_type="long_press",
                    target_desc=target_desc,
                    coordinates=None,
                    coord_time=None,
                    exec_time=long_press_time,
                    total_time=action_total_time,
                )
            else:
                logger.info(f"[Tool: perform_action] ✓ Long press executed in {long_press_time:.3f}s")
                logger.info(f"[Tool: perform_action] Total action time: {action_total_time:.3f}s")
            
            return {"action": "long_press", "target": target_desc, "duration": duration}
    
    else:
        raise ValueError(f"Unknown action type: {action_type}")


async def sleep_impl(duration: float) -> Dict[str, Any]:
    """Wait for a specified amount of time."""
    logger.info(f"[Tool: wait] Waiting for {duration} seconds")
    sleep_start = time.time()
    await asyncio.sleep(duration)
    sleep_time = time.time() - sleep_start
    logger.info(f"[Tool: wait] ✓ Wait completed in {sleep_time:.3f}s")
    return {"status": "success", "action": "wait", "duration": duration}


# Tool function schemas for JSON schema generation
TOOL_SCHEMAS = {
    "action": {
        "type": "function",
        "function": {
            "name": "action",
            "description": "Execute a UI action on the device screen. Use this tool to interact with UI elements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["tap", "click", "touch", "swipe", "drag", "scroll", "input", "key_press", "button_press", "long_press"],
                        "description": "Type of action to perform. Use 'tap'/'touch'/'swipe' for Android, 'click'/'key_press' for PC/Linux."
                    },
                    "option": {
                        "type": "string",
                        "enum": ["left", "right", "double"],
                        "description": "Click option (for click action on PC/Linux devices, not used for tap on Android)"
                    },
                    "target": {
                        "type": "object",
                        "description": "Target element description",
                        "properties": {
                            "element": {"type": "string"},
                            "label": {"type": "string"},
                            "color": {"type": "string"},
                            "size": {"type": "string"},
                            "location": {"type": "string"},
                            "shape": {"type": "string"}
                        },
                        "required": ["element"]
                    },
                    "start_target": {"type": "object", "description": "Start target for swipe"},
                    "end_target": {"type": "object", "description": "End target for swipe"},
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                    "distance": {"type": "integer", "description": "Scroll distance in pixels"},
                    "text": {"type": "string", "description": "Text to input"},
                    "keys": {"type": "array", "items": {"type": "string"}},
                    "button": {
                        "oneOf": [
                            {
                                "type": "string",
                                "enum": ["back", "home", "menu", "power", "volumeUp", "volumeDown"],
                                "description": "Single button to press. For Android: back, home, menu, power, volumeUp, volumeDown. For PC: power."
                            },
                            {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["back", "home", "menu", "power", "volumeUp", "volumeDown"]
                                },
                                "description": "Multiple buttons to press simultaneously (Android only, e.g., [\"power\", \"volumeUp\"])"
                            }
                        ]
                    },
                    "duration": {
                        "type": "string",
                        "description": "Duration for long_press action (e.g., '500ms', '1s'). Optional."
                    }
                },
                "required": ["action_type"]
            }
        }
    },
    "wait": {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a specified amount of time before taking the next action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "Sleep duration in seconds (0.1 to 60.0)",
                        "minimum": 0.1,
                        "maximum": 60.0
                    }
                },
                "required": ["duration"]
            }
        }
    },
    "finish": {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Report that the task has been completed or cannot be completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task was completed successfully"
                    },
                    "result_message": {
                        "type": "string",
                        "description": "Summary of what was accomplished or why the task couldn't be completed"
                    }
                },
                "required": ["success", "result_message"]
            }
        }
    }
}

