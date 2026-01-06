"""Shared GBox coordinate generation logic."""

import json
import logging
from typing import Optional, Dict, Any

try:
    from gbox_sdk import GboxSDK
except ImportError:
    GboxSDK = None
    logging.warning("gbox_sdk not installed. Install with: pip install gbox-sdk")

from tinker_cookbook.recipes.cua_rl.core.rollout_logger import RolloutLogger

logger = logging.getLogger(__name__)


class CuaGBoxCoordinateGenerator:
    """Shared coordinate generation using GBox model."""
    
    def __init__(self, api_key: str, model: str = "gbox-handy-1"):
        """Initialize GBox coordinate generator.
        
        Args:
            api_key: GBox API key
            model: Model name for coordinate generation (default: gbox-handy-1)
        """
        if GboxSDK is None:
            raise ImportError(
                "gbox_sdk not installed. Install with: pip install gbox-sdk"
            )
        
        self.api_key = api_key
        self.model = model
        self._sdk = GboxSDK(api_key=api_key)
    
    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse SDK response to dictionary."""
        if hasattr(response, 'json'):
            return response.json()
        elif hasattr(response, 'data'):
            if hasattr(response.data, 'model_dump'):
                return response.data.model_dump()
            elif hasattr(response.data, 'dict'):
                return response.data.dict()
            else:
                return dict(response.data) if hasattr(response.data, '__dict__') else response.data
        elif hasattr(response, 'model_dump'):
            return response.model_dump()
        elif hasattr(response, 'dict'):
            return response.dict()
        else:
            return dict(response) if isinstance(response, dict) else {"response": response}
    
    async def generate_coordinates(
        self,
        screenshot_uri: str,
        action_type: str,
        target: str,
        end_target: Optional[str] = None,
        direction: Optional[str] = None,
        rollout_logger: Optional[RolloutLogger] = None,
    ) -> Dict[str, Any]:
        """Generate coordinates using gbox-handy-1 model.
        
        Args:
            screenshot_uri: Screenshot URI (base64 data URI or URL)
            action_type: Type of action ("click", "tap", "drag", "scroll")
            target: Target element description (natural language)
            end_target: End target for drag actions
            direction: Direction for scroll actions
            
        Returns:
            Coordinate generation response with coordinates
        """
        # Build action object based on type
        if action_type == "click":
            action = {
                "type": "click",
                "target": target,
            }
        elif action_type == "tap":
            # Tap is similar to click for coordinate generation
            action = {
                "type": "click",  # Use click type for coordinate generation
                "target": target,
            }
        elif action_type == "drag":
            action = {
                "type": "drag",
                "target": target,
                "destination": end_target or target,
            }
        elif action_type == "scroll":
            action = {
                "type": "scroll",
                "location": target,
                "direction": direction or "down",
            }
        else:
            raise ValueError(f"Unknown action type: {action_type}. Supported: click, tap, drag, scroll")
        
        try:
            # Only log if rollout_logger is not provided (to avoid duplicate logs)
            if not rollout_logger:
                logger.info(f"[GBox Coordinate] Generating coordinates: action_type={action_type}, target={target}")
                logger.debug(f"[GBox Coordinate] Action object: {action}")
                logger.debug(f"[GBox Coordinate] Screenshot URI length: {len(screenshot_uri)} chars")
                logger.info(f"[GBox Coordinate] Calling GBox model API: POST /model with model={self.model}")
            
            # Use SDK client to call model API
            result = self._sdk.client.post(
                "/model",
                cast_to=Dict[str, Any],
                body={
                    "model": self.model,
                    "screenshot": screenshot_uri,
                    "action": action,
                }
            )
            
            parsed_result = self._parse_response(result)
            if not rollout_logger:
                logger.info(f"[GBox Coordinate] Coordinate generation successful")
                logger.debug(f"[GBox Coordinate] Response: {json.dumps(parsed_result, indent=2, default=str)}")
            
            return parsed_result
        except Exception as e:
            logger.error(f"[GBox Coordinate] Failed to generate coordinates: {e}", exc_info=True)
            raise

