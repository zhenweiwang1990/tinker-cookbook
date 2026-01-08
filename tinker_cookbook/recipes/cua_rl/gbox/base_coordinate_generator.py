"""Base class for coordinate generators."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseCoordinateGenerator(ABC):
    """
    Abstract base class for coordinate generators.
    
    Coordinate generators are responsible for converting UI element descriptions
    into pixel coordinates that can be used for device interactions.
    
    Two implementations are provided:
    1. GBoxCoordinateGenerator: Uses GBox external model (gbox-handy-1)
    2. DirectCoordinateGenerator: Extracts coordinates from model output
    """
    
    @abstractmethod
    async def generate_coordinates(
        self,
        screenshot_uri: str,
        action_type: str,
        target: str,
        end_target: Optional[str] = None,
        direction: Optional[str] = None,
        rollout_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate coordinates for a UI action.
        
        Args:
            screenshot_uri: Screenshot URI (base64 data URI or URL)
            action_type: Type of action ("tap", "click", "drag", "swipe", "scroll")
            target: Target element description or coordinate data
            end_target: End target for drag/swipe actions
            direction: Direction for scroll actions
            rollout_logger: Optional logger for tracking generation process
            
        Returns:
            Dictionary with coordinates:
            
            For single-point actions (tap, click, scroll):
            {
                "coordinates": {"x": 100, "y": 200}
            }
            
            For two-point actions (drag, swipe):
            {
                "coordinates": {
                    "start": {"x": 100, "y": 200},
                    "end": {"x": 300, "y": 400}
                }
            }
        """
        pass

