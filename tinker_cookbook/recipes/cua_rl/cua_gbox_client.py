"""GBox API Client for box management and UI actions using official SDK."""

import base64
import logging
from typing import Optional, Dict, Any, List, Tuple

from gbox_sdk import GboxSDK

logger = logging.getLogger(__name__)


class CuaGBoxClient:
    """Client for interacting with GBox API using official SDK wrapper."""
    
    def __init__(
        self,
        api_key: str,
        box_type: str = "android",
        timeout: str = "60s",
        wait: bool = True,
        expires_in: str = "15m",
        labels: Optional[Dict[str, Any]] = None,
        envs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize GBox client."""
        self.api_key = api_key
        self.box_type = box_type
        self.timeout = timeout
        self.wait = wait
        self.expires_in = expires_in
        self.labels = labels or {}
        self.envs = envs or {}
        
        self.box_id: Optional[str] = None
        self._sdk = GboxSDK(api_key=api_key)
        self._box: Optional[Any] = None
    
    async def create_box(self, box_type: Optional[str] = None) -> Dict[str, Any]:
        """Create a new GBox environment."""
        box_type = box_type or self.box_type
        logger.debug(f"Creating {box_type} box...")
        
        box = self._sdk.create(
            type=box_type,
            wait=self.wait,
            timeout=self.timeout,
            config={
                "expiresIn": self.expires_in,
                **({"labels": self.labels} if self.labels else {}),
                **({"envs": self.envs} if self.envs else {}),
            }
        )
        
        self._box = box
        self.box_id = box.data.id
        logger.debug(f"Box created: {self.box_id}")
        return {"id": self.box_id}
    
    def _get_box(self, box_id: Optional[str] = None) -> Any:
        """Get box operator."""
        if box_id and box_id != self.box_id:
            return self._sdk.get(box_id)
        if self._box:
            return self._box
        if self.box_id:
            self._box = self._sdk.get(self.box_id)
            return self._box
        raise ValueError("No box available. Call create_box() first.")
    
    async def terminate_box(self, box_id: Optional[str] = None) -> Dict[str, Any]:
        """Terminate a GBox environment."""
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        logger.debug(f"Terminating box: {box_id}")
        box = self._get_box(box_id)
        box.terminate()
        
        if box_id == self.box_id:
            self.box_id = None
            self._box = None
        
        return {"id": box_id, "status": "terminated"}
    
    async def take_screenshot(
        self,
        box_id: Optional[str] = None,
        format: str = "png",
    ) -> Tuple[bytes, str]:
        """Take a screenshot of the box display."""
        box = self._get_box(box_id)
        result = box.action.screenshot(output_format="base64")
        
        screenshot_uri = result.uri
        
        if screenshot_uri.startswith("data:"):
            parts = screenshot_uri.split(",", 1)
            image_bytes = base64.b64decode(parts[1])
            return image_bytes, screenshot_uri
        else:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(screenshot_uri)
                resp.raise_for_status()
                image_bytes = resp.content
                data_uri = f"data:image/{format};base64,{base64.b64encode(image_bytes).decode()}"
                return image_bytes, data_uri
    
    async def close(self):
        """Close and terminate the box if active."""
        if self._box:
            try:
                await self.terminate_box()
            except Exception as e:
                logger.warning(f"Failed to terminate box on close: {e}")
        self._box = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

