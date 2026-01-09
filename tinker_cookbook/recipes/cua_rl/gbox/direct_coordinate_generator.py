"""Direct coordinate generator - extracts coordinates from model output."""

import logging
from typing import Dict, Any, Optional

from tinker_cookbook.recipes.cua_rl.gbox.base_coordinate_generator import BaseCoordinateGenerator

logger = logging.getLogger(__name__)


class DirectCoordinateGenerator(BaseCoordinateGenerator):
    """
    Direct coordinate generator that extracts coordinates from model output.
    
    In Direct mode, the VLM model directly outputs coordinates as part of its
    tool call response. This generator extracts those coordinates and optionally
    applies scaling transformation.
    
    Two coordinate modes:
    1. **No scaling** (coordinate_scale=False): Model outputs coordinates in actual screen pixels
       - System prompt includes actual screen dimensions
       - Example: Screen 1080x2400, model outputs [540, 1200]
       
    2. **With scaling** (coordinate_scale=True): Model outputs normalized coordinates (e.g., 0-1000)
       - System prompt does NOT include screen dimensions
       - Coordinates are scaled: actual_x = model_x * x_scale_ratio
       - Useful for models like Qwen3-VL that normalize images to 1000x1000
    
    Expected format in Direct mode:
    {
        "name": "action",
        "args": {
            "action_type": "tap",
            "target": {
                "element": "login button",
                "coordinates": [540, 1200]  # or "x": 540, "y": 1200
            }
        }
    }
    """
    
    def __init__(
        self,
        screen_width: int = 1080,
        screen_height: int = 2400,
        coordinate_scale: bool = False,
        x_scale_ratio: Optional[float] = None,
        y_scale_ratio: Optional[float] = None,
    ):
        """
        Initialize Direct coordinate generator.
        
        Args:
            screen_width: Actual screen width in pixels
            screen_height: Actual screen height in pixels
            coordinate_scale: Whether to apply coordinate scaling
            x_scale_ratio: X scaling ratio (default: screen_width / 1000)
            y_scale_ratio: Y scaling ratio (default: screen_height / 1000)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.default_x = screen_width // 2
        self.default_y = screen_height // 2
        
        # Coordinate scaling configuration
        self.coordinate_scale = coordinate_scale
        if coordinate_scale:
            # Default: assume model outputs in 1000x1000 space
            self.x_scale_ratio = x_scale_ratio if x_scale_ratio is not None else screen_width / 1000.0
            self.y_scale_ratio = y_scale_ratio if y_scale_ratio is not None else screen_height / 1000.0
            logger.info(
                f"[DirectCoordinateGenerator] Initialized with coordinate scaling: "
                f"screen={screen_width}x{screen_height}, "
                f"scale_ratios=({self.x_scale_ratio:.3f}, {self.y_scale_ratio:.3f}), "
                f"center=({self.default_x}, {self.default_y})"
            )
        else:
            self.x_scale_ratio = 1.0
            self.y_scale_ratio = 1.0
            logger.info(
                f"[DirectCoordinateGenerator] Initialized without coordinate scaling: "
                f"screen={screen_width}x{screen_height}, "
                f"center=({self.default_x}, {self.default_y})"
            )
    
    async def generate_coordinates(
        self,
        screenshot_uri: str,
        action_type: str,
        target: str,  # In Direct mode, this can be str or dict with coordinates
        end_target: Optional[str] = None,
        direction: Optional[str] = None,
        rollout_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Extract coordinates from target parameters.
        
        In Direct mode, the target and end_target contain the coordinates
        directly output by the VLM model. This method simply extracts them.
        
        Args:
            screenshot_uri: Screenshot URI (not used in Direct mode)
            action_type: Action type ("tap", "click", "drag", "swipe", "scroll")
            target: Target data - in Direct mode, should be dict with x, y coordinates
            end_target: End target for drag/swipe - should be dict with x, y coordinates
            direction: Direction for scroll (not used in Direct mode)
            rollout_logger: Optional logger
            
        Returns:
            Dictionary with coordinates in standard format
        """
        try:
            # Extract coordinates from target
            coords = self._extract_single_coords(target, "target")
            
            # For drag/swipe actions, extract end coordinates
            if action_type in ["drag", "swipe"] and end_target:
                end_coords = self._extract_single_coords(end_target, "end_target")
                
                # Log if available
                if rollout_logger:
                    rollout_logger.log(
                        f"[Direct Coordinate] Extracted {action_type} coordinates: "
                        f"start=({coords['x']}, {coords['y']}), "
                        f"end=({end_coords['x']}, {end_coords['y']})"
                    )
                else:
                    logger.info(
                        f"[Direct Coordinate] Extracted {action_type} coordinates: "
                        f"start=({coords['x']}, {coords['y']}), "
                        f"end=({end_coords['x']}, {end_coords['y']})"
                    )
                
                return {
                    "coordinates": {
                        "start": coords,
                        "end": end_coords
                    }
                }
            
            # For single-point actions (tap, click, scroll)
            if rollout_logger:
                rollout_logger.log(
                    f"[Direct Coordinate] Extracted {action_type} coordinates: "
                    f"({coords['x']}, {coords['y']})"
                )
            else:
                logger.info(
                    f"[Direct Coordinate] Extracted {action_type} coordinates: "
                    f"({coords['x']}, {coords['y']})"
                )
            
            return {"coordinates": coords}
            
        except Exception as e:
            logger.error(f"[Direct Coordinate] Failed to extract coordinates: {e}", exc_info=True)
            # Fallback: return default coordinates (screen center)
            logger.warning(
                f"[Direct Coordinate] Using fallback coordinates "
                f"(screen center): ({self.default_x}, {self.default_y})"
            )
            return {"coordinates": {"x": self.default_x, "y": self.default_y}}
    
    def _extract_single_coords(self, target_data: Any, param_name: str) -> Dict[str, Any]:
        """
        Extract x, y coordinates from target data and apply scaling if enabled.
        
        Args:
            target_data: Target data (can be dict, str, or other)
            param_name: Parameter name for logging (e.g., "target", "end_target")
            
        Returns:
            Dictionary with:
            - x, y: Final coordinates (scaled if coordinate_scale is enabled)
            - original_x, original_y: Original coordinates from model (if scaling enabled)
            - scaled: Boolean indicating if coordinates were scaled
            
        Raises:
            ValueError: If coordinates cannot be extracted
        """
        # Case 1: target_data is a dictionary
        if isinstance(target_data, dict):
            # First, try to extract from 'coordinates' array (preferred format)
            coordinates = target_data.get("coordinates")
            if coordinates and isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
                try:
                    # Extract raw coordinates from model
                    model_x = float(coordinates[0])
                    model_y = float(coordinates[1])
                    
                    # Apply scaling if enabled
                    if self.coordinate_scale:
                        x_int = int(round(model_x * self.x_scale_ratio))
                        y_int = int(round(model_y * self.y_scale_ratio))
                        logger.debug(
                            f"[Direct Coordinate] Scaled coordinates: "
                            f"model=[{model_x:.1f}, {model_y:.1f}] → "
                            f"screen=[{x_int}, {y_int}] "
                            f"(ratios: {self.x_scale_ratio:.3f}, {self.y_scale_ratio:.3f})"
                        )
                        result = {
                            "x": x_int,
                            "y": y_int,
                            "original_x": int(round(model_x)),
                            "original_y": int(round(model_y)),
                            "scaled": True
                        }
                    else:
                        x_int = int(round(model_x))
                        y_int = int(round(model_y))
                        result = {
                            "x": x_int,
                            "y": y_int,
                            "scaled": False
                        }
                    
                    # Sanity check: coordinates should be within screen bounds
                    if not (0 <= result["x"] <= self.screen_width and 0 <= result["y"] <= self.screen_height):
                        if result.get("scaled"):
                            logger.warning(
                                f"[Direct Coordinate] Scaled coordinates out of screen bounds "
                                f"(screen: {self.screen_width}x{self.screen_height}): "
                                f"original=[{result.get('original_x')}, {result.get('original_y')}] → "
                                f"scaled=[{result['x']}, {result['y']}] "
                                f"(ratios: {self.x_scale_ratio:.3f}, {self.y_scale_ratio:.3f})"
                            )
                        else:
                            logger.warning(
                                f"[Direct Coordinate] Coordinates out of screen bounds "
                                f"(screen: {self.screen_width}x{self.screen_height}): "
                                f"coordinates=[{result['x']}, {result['y']}] "
                                f"(scaling disabled or not in direct mode)"
                            )
                    
                    return result
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid coordinate values in {param_name}: coordinates={coordinates}. "
                        f"Coordinates must be numeric."
                    ) from e
            
            # Fallback: try to extract from separate 'x' and 'y' fields (legacy format)
            x = target_data.get("x")
            y = target_data.get("y")
            
            if x is not None and y is not None:
                # Validate coordinates are numeric
                try:
                    model_x = float(x)
                    model_y = float(y)
                    
                    # Apply scaling if enabled
                    if self.coordinate_scale:
                        x_int = int(round(model_x * self.x_scale_ratio))
                        y_int = int(round(model_y * self.y_scale_ratio))
                        logger.debug(
                            f"[Direct Coordinate] Scaled coordinates: "
                            f"model=({model_x:.1f}, {model_y:.1f}) → "
                            f"screen=({x_int}, {y_int}) "
                            f"(ratios: {self.x_scale_ratio:.3f}, {self.y_scale_ratio:.3f})"
                        )
                        result = {
                            "x": x_int,
                            "y": y_int,
                            "original_x": int(round(model_x)),
                            "original_y": int(round(model_y)),
                            "scaled": True
                        }
                    else:
                        x_int = int(round(model_x))
                        y_int = int(round(model_y))
                        result = {
                            "x": x_int,
                            "y": y_int,
                            "scaled": False
                        }
                    
                    # Sanity check: coordinates should be within screen bounds
                    if not (0 <= result["x"] <= self.screen_width and 0 <= result["y"] <= self.screen_height):
                        if result.get("scaled"):
                            logger.warning(
                                f"[Direct Coordinate] Scaled coordinates out of screen bounds "
                                f"(screen: {self.screen_width}x{self.screen_height}): "
                                f"original=({result.get('original_x')}, {result.get('original_y')}) → "
                                f"scaled=({result['x']}, {result['y']}) "
                                f"(ratios: {self.x_scale_ratio:.3f}, {self.y_scale_ratio:.3f})"
                            )
                        else:
                            logger.warning(
                                f"[Direct Coordinate] Coordinates out of screen bounds "
                                f"(screen: {self.screen_width}x{self.screen_height}): "
                                f"x={result['x']}, y={result['y']} "
                                f"(scaling disabled or not in direct mode)"
                            )
                    
                    return result
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid coordinate values in {param_name}: x={x}, y={y}. "
                        f"Coordinates must be numeric."
                    ) from e
            
            # If neither format found, log available keys for debugging
            available_keys = list(target_data.keys())
            raise ValueError(
                f"No coordinates found in {param_name} dictionary. "
                f"Expected 'coordinates' array [x, y] or separate 'x' and 'y' keys. "
                f"Found keys: {available_keys}"
            )
        
        # Case 2: target_data is a string (legacy format or error)
        elif isinstance(target_data, str):
            # Print full string for debugging
            logger.error(
                f"[Direct Coordinate] {param_name} is a string (full content): {target_data}"
            )
            raise ValueError(
                f"{param_name} is a string, expected dictionary with coordinates. "
                f"In Direct mode, the model must output coordinates as part of the target. "
                f"Received: {target_data}"
            )
        
        # Case 3: Unexpected type
        else:
            raise ValueError(
                f"Unexpected {param_name} type: {type(target_data)}. "
                f"Expected dictionary with coordinates."
            )

