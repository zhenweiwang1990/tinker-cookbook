"""
Tinker CUA Agent with support for multiple inference providers.

This agent supports:
- Tinker native API (for training with logprobs)
- vLLM (for local deployment)
- OpenRouter (for cloud API)
- OpenAI (for GPT models)
- Any OpenAI-compatible API

All providers use the same prompts, tool parsing, and coordinate handling logic.
"""

import asyncio
import base64
import io
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import tinker
from PIL import Image

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.image_processing_utils import get_image_processor

from tinker_cookbook.recipes.cua_rl.utils.cua_prompts import create_system_prompt
from tinker_cookbook.recipes.cua_rl.gbox.client import CuaGBoxClient
from tinker_cookbook.recipes.cua_rl.gbox.coordinate import CuaGBoxCoordinateGenerator
from tinker_cookbook.recipes.cua_rl.gbox.tools import (
    perform_action_impl,
    sleep_impl,
    TargetElement,
    TOOL_SCHEMAS,
)
from tinker_cookbook.recipes.cua_rl.core.rollout_logger import RolloutLogger
from tinker_cookbook.recipes.cua_rl.agent.base_inference_client import BaseInferenceClient
from tinker_cookbook.recipes.cua_rl.agent.inference_client_factory import create_inference_client

logger = logging.getLogger(__name__)


class TinkerCuaAgent:
    """
    CUA Agent with support for multiple inference providers.
    
    This agent:
    - Supports Tinker, vLLM, OpenRouter, OpenAI, and any OpenAI-compatible API
    - Uses prompt-based tool calling (preserves tinker-cookbook logic)
    - Manages conversation history
    - Handles tool calls via renderer parsing
    - Integrates with GBox for UI interactions
    - Preserves all coordinate handling (gbox and direct modes)
    - Maintains full database recording capabilities
    """
    
    def __init__(
        self,
        gbox_api_key: str,
        
        # NEW: Unified inference client interface
        inference_client: Optional[BaseInferenceClient] = None,
        
        # Legacy Tinker-specific parameters (backward compatible)
        tinker_api_key: Optional[str] = None,
        tinker_model_path: Optional[str] = None,
        
        # NEW: Provider-based creation parameters
        provider: Optional[str] = None,
        provider_model_name: Optional[str] = None,
        provider_base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        
        # Model configuration
        base_model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        renderer_name: Optional[str] = None,
        
        # Agent configuration
        max_turns: int = 20,
        box_type: str = "android",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        max_recent_turns: int = 5,
        
        # Logging and recording
        rollout_logger: Optional[RolloutLogger] = None,
        rollout_recorder = None,
        rollout_id: Optional[str] = None,
        
        # Timeouts
        max_task_time_seconds: int = 60 * 60,
        max_turn_time_seconds: int = 5 * 60,
        
        # Coordinate handling
        coordinate_mode: str = "gbox",
        coordinate_scale: Optional[bool] = None,
        x_scale_ratio: Optional[float] = None,
        y_scale_ratio: Optional[float] = None,
    ):
        """
        Initialize CUA Agent with flexible inference provider support.
        
        Three ways to initialize:
        1. Provide inference_client directly (most flexible)
        2. Provide tinker_api_key + tinker_model_path (backward compatible)
        3. Provide provider + provider_model_name (new unified way)
        
        Args:
            gbox_api_key: GBox API key
            
            inference_client: Pre-initialized inference client (optional)
            
            tinker_api_key: Tinker API key (legacy, for backward compatibility)
            tinker_model_path: Tinker checkpoint path (legacy)
            
            provider: Provider name ("tinker", "vllm", "openrouter", "openai")
            provider_model_name: Model name for the provider
            provider_base_url: API base URL (optional, uses provider default)
            provider_api_key: API key (optional, auto-detects from env)
            
            base_model_name: Base model name for tokenizer/renderer
            renderer_name: Renderer name (auto-detected if None)
            max_turns: Maximum number of turns
            box_type: Type of GBox environment (android or linux)
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            max_recent_turns: Number of recent turns to keep in context
            
            rollout_logger: Logger for detailed execution tracking
            rollout_recorder: Database recorder instance
            rollout_id: Rollout ID for database recording
            
            max_task_time_seconds: Maximum total time for task execution
            max_turn_time_seconds: Maximum time per turn for model inference
            
            coordinate_mode: Coordinate generation mode ("gbox" or "direct")
            coordinate_scale: Whether to apply coordinate scaling
            x_scale_ratio: X scaling ratio
            y_scale_ratio: Y scaling ratio
        """
        # Store basic parameters
        self.gbox_api_key = gbox_api_key
        self.base_model_name = base_model_name
        self.max_turns = max_turns
        self.box_type = box_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_recent_turns = max_recent_turns
        self.rollout_logger = rollout_logger
        self.rollout_recorder = rollout_recorder
        self.rollout_id = rollout_id
        self.max_task_time_seconds = max_task_time_seconds
        self.max_turn_time_seconds = max_turn_time_seconds
        self.coordinate_mode = coordinate_mode

        # Backward-compatible fields used in logging/debug output.
        # For non-Tinker providers, these may represent provider-native identifiers/keys.
        self.tinker_api_key = tinker_api_key or provider_api_key or ""
        self.tinker_model_path = tinker_model_path or provider_model_name or ""
        
        # Auto-detect coordinate_scale if not explicitly set
        if coordinate_scale is None:
            self.coordinate_scale = (coordinate_mode == "direct")
        else:
            self.coordinate_scale = coordinate_scale
        
        self.x_scale_ratio = x_scale_ratio
        self.y_scale_ratio = y_scale_ratio
        
        # Track termination reason for reporting
        self.termination_reason = None
        
        # Set up compatibility layer for old database recording code
        if rollout_recorder is not None:
            logger.info(f"[TinkerCuaAgent] Initializing with rollout_recorder, rollout_id={rollout_id}")
            from tinker_cookbook.recipes.cua_rl.database.compat import set_recorder
            set_recorder(rollout_recorder)
        else:
            logger.warning(f"[TinkerCuaAgent] rollout_recorder is None! No database recording will occur.")
        
        # Initialize tokenizer and renderer (needed for all providers)
        self.tokenizer = get_tokenizer(base_model_name)
        image_processor = get_image_processor(base_model_name)
        renderer_name = renderer_name or "qwen3_vl_instruct"
        self.renderer = renderers.get_renderer(
            renderer_name,
            tokenizer=self.tokenizer,
            image_processor=image_processor,
        )
        
        # Create or use inference client
        if inference_client is not None:
            # Method 1: Use provided inference client
            self.inference_client = inference_client
            self.provider = inference_client.get_provider_name()
            logger.info(f"[TinkerCuaAgent] Using provided inference client: {self.provider}")
            
        elif tinker_api_key and tinker_model_path:
            # Method 2: Create Tinker client (backward compatible)
            logger.info(f"[TinkerCuaAgent] Creating Tinker inference client (legacy mode)")
            self.inference_client = create_inference_client(
                provider="tinker",
                model_name=base_model_name,
                api_key=tinker_api_key,
                model_path=tinker_model_path,
                renderer=self.renderer,
                tokenizer=self.tokenizer,
            )
            self.provider = "tinker"
            # Store for backward compatibility
            self.tinker_api_key = tinker_api_key
            self.tinker_model_path = tinker_model_path
            
        elif provider and provider_model_name:
            # Method 3: Create client from provider settings
            logger.info(f"[TinkerCuaAgent] Creating {provider} inference client")
            self.inference_client = create_inference_client(
                provider=provider,
                model_name=provider_model_name,
                base_url=provider_base_url,
                api_key=provider_api_key,
                base_model_name=base_model_name,
                renderer=self.renderer,
                tokenizer=self.tokenizer,
            )
            self.provider = provider
            # Keep legacy fields in sync for downstream logging/compat.
            self.tinker_model_path = provider_model_name
            
        else:
            raise ValueError(
                "Must provide one of:\n"
                "1. inference_client\n"
                "2. tinker_api_key + tinker_model_path\n"
                "3. provider + provider_model_name"
            )
        
        logger.info(f"[TinkerCuaAgent] Inference provider: {self.provider}")
        
        # Initialize GBox components
        self.gbox_client = CuaGBoxClient(api_key=gbox_api_key, box_type=box_type)
        
        # Initialize coordinate generator based on mode
        if coordinate_mode == "gbox":
            logger.info(f"[Agent Init] Using GBox coordinate mode (external model)")
            self.coord_generator = CuaGBoxCoordinateGenerator(api_key=gbox_api_key)
        elif coordinate_mode == "direct":
            scale_info = "with scaling" if self.coordinate_scale else "without scaling"
            logger.info(f"[Agent Init] Using Direct coordinate mode ({scale_info})")
            from tinker_cookbook.recipes.cua_rl.gbox.direct_coordinate_generator import DirectCoordinateGenerator
            # Note: Screen dimensions will be updated dynamically after first screenshot
            self.coord_generator = DirectCoordinateGenerator(
                coordinate_scale=self.coordinate_scale,  # Use auto-detected value
                x_scale_ratio=self.x_scale_ratio,
                y_scale_ratio=self.y_scale_ratio,
            )
        else:
            raise ValueError(
                f"Unknown coordinate_mode: {coordinate_mode}. "
                f"Must be 'gbox' or 'direct'"
            )
        
        # Agent state
        self.task_completed = False
        self.task_success = False
        self.result_message = ""
        self.current_screenshot_uri: Optional[str] = None
        
        # Conversation history
        self.messages: List[renderers.Message] = []
        self.system_prompt: Optional[str] = None
        
        # Store trajectory data for token-level training
        # Each entry contains: (turn, observation_model_input, action_tokens, action_logprobs)
        self.trajectory_turns: List[Tuple[int, tinker.ModelInput, List[int], List[float]]] = []
        
        # Metrics tracking for reward calculation
        self.num_total_actions: int = 0
        self.consecutive_repeated_actions: int = 0
        self.parse_errors: int = 0
        self.tool_name_errors: int = 0
        self.tool_arg_errors: int = 0
        self.runtime_errors: int = 0
        self.turn_first_success: int = -1
        self.turn_task_completed: int = -1
        self.attempted_completion: bool = False
        
        # Track last action for detecting consecutive repeats
        self._last_action_signature: Optional[str] = None
        self._current_repeat_count: int = 0
        
        # Store task object if available (for app identification)
        self.task: Optional[Any] = None
    
    def _identify_app_from_task(self, task_description: str) -> Optional[str]:
        """
        Identify which app/category the task belongs to (demo, airbnb, instagram, etc.).
        
        This is used for system prompt generation and app identification.
        Note: APK installation is now controlled by task config, not this method.
        
        Args:
            task_description: Task description string
            
        Returns:
            "demo", "airbnb", "instagram", or None if cannot be determined
        """
        # First, try to identify from task object if available
        if self.task is not None:
            # Check if task has app_name attribute (set by task_loader)
            app_name = getattr(self.task, 'app_name', None)
            if app_name:
                return app_name
            
            # Check if task has module path or path attribute
            task_module = getattr(self.task, '__module__', None)
            if task_module:
                if 'demo' in task_module.lower():
                    return "demo"
                elif 'airbnb' in task_module.lower():
                    return "airbnb"
                elif 'instagram' in task_module.lower():
                    return "instagram"
            
            # Check task name or id
            task_name = getattr(self.task, 'name', '') or getattr(self.task, 'id', '')
            if task_name:
                task_name_lower = task_name.lower()
                if 'demo' in task_name_lower or 'settings' in task_name_lower:
                    return "demo"
                elif 'airbnb' in task_name_lower:
                    return "airbnb"
                elif 'instagram' in task_name_lower:
                    return "instagram"
            
            # Check tags
            tags = getattr(self.task, 'tags', [])
            tags_lower = [tag.lower() for tag in tags]
            if 'demo' in tags_lower:
                return "demo"
            elif 'airbnb' in tags_lower:
                return "airbnb"
            elif 'instagram' in tags_lower:
                return "instagram"
        
        # Fallback: identify from task description keywords
        task_lower = task_description.lower()
        
        # Demo task indicators (Android Settings)
        demo_keywords = [
            'settings', 'brightness', 'wifi', 'airplane mode', 'battery saver',
            'auto time', 'notifications', 'timeout', 'screen timeout', 'dnd',
            'do not disturb', 'volume', 'sound', 'display'
        ]
        
        # Strong Airbnb indicators (more specific)
        airbnb_strong_keywords = [
            'airbnb', 'listing', 'listings', 'host', 'booking', 'reservation',
            'save 3 listings', 'save a listing', 'book a listing', 
            'cancel reservation', 'cancel all', 'trip', 'vacation',
            'guest favourite', 'guest favorite', 'amazing pools', 'amazing views'
        ]
        
        # Strong Instagram indicators (more specific)
        instagram_strong_keywords = [
            'instagram', 'post', 'posts', 'reel', 'reels', 'follow', 'unfollow',
            'like', 'comment', 'profile', 'story', 'feed', 'first post',
            'first reel', 'search user', 'open my profile'
        ]
        
        # Check for strong indicators first
        for keyword in demo_keywords:
            if keyword in task_lower:
                return "demo"
        
        for keyword in airbnb_strong_keywords:
            if keyword in task_lower:
                return "airbnb"
        
        for keyword in instagram_strong_keywords:
            if keyword in task_lower:
                return "instagram"
        
        # If no clear match, return None
        return None
    
    def _data_uri_to_pil(self, data_uri: str) -> Image.Image:
        """Convert data URI to PIL Image."""
        if not data_uri.startswith("data:image"):
            raise ValueError(f"Invalid data URI: {data_uri[:50]}...")
        
        header, encoded = data_uri.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    
    def _get_screen_dimensions(self, screenshot_uri: str) -> tuple[int, int]:
        """
        Extract screen dimensions from screenshot.
        
        Args:
            screenshot_uri: Screenshot data URI
            
        Returns:
            Tuple of (width, height) in pixels
            
        Raises:
            RuntimeError: If screen dimensions cannot be extracted
        """
        try:
            img = self._data_uri_to_pil(screenshot_uri)
            width, height = img.size
            
            # Validate dimensions are reasonable
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid screen dimensions: {width}x{height}")
            
            return width, height
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract screen dimensions from screenshot: {e}. "
                f"In Direct coordinate mode, screen dimensions are required."
            ) from e
    
    def _build_messages_with_screenshot(
        self,
        text: str,
        screenshot_uri: str,
    ) -> List[renderers.Message]:
        """Build messages with screenshot for Tinker."""
        # Convert screenshot URI to PIL Image
        screenshot_img = self._data_uri_to_pil(screenshot_uri)
        
        # Build message with image and text
        message = renderers.Message(
            role="user",
            content=[
                renderers.ImagePart(type="image", image=screenshot_img),
                renderers.TextPart(type="text", text=text),
            ]
        )
        
        return [message]
    
    def _create_action_signature(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Create a signature for an action to detect consecutive repeats.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            
        Returns:
            String signature (normalized JSON of tool name + sorted arguments)
        """
        # Create normalized signature: sort keys and convert to JSON string
        normalized_args = json.dumps(arguments, sort_keys=True)
        return f"{tool_name}:{normalized_args}"
    
    def _extract_json_from_text(self, text: str, start_char: str = '{') -> Optional[Any]:
        """
        Extract a complete JSON object or array from text using Python's json module.
        
        This method uses json.JSONDecoder.raw_decode() which can parse JSON
        from any position in a string, handling all edge cases correctly.
        
        Args:
            text: Text to extract JSON from
            start_char: Starting character, either '{' for objects or '[' for arrays
            
        Returns:
            Parsed JSON object/array, or None if extraction fails
        """
        from tinker_cookbook.utils.json_repair import extract_first_json

        obj = extract_first_json(text, start_chars=(start_char,))
        if obj is None:
            logger.debug("JSON extract failed after repair attempts")
        return obj
    
    def _parse_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from model response.
        
        Supports multiple formats:
        1. <tool_call>{"name": "...", "args": {...}}</tool_call>
        2. <function_calls>...</function_calls> (OpenAI format)
        3. JSON tool calls embedded in text
        """
        tool_calls = []
        
        # Pattern 1: <tool_call>...</tool_call>
        pattern1 = r'<tool_call>(.*?)</tool_call>'
        matches1 = re.finditer(pattern1, response_text, re.DOTALL)
        for match in matches1:
            content = match.group(1).strip()
            # Common model glitch: emit an extra "{" before a sibling field, e.g.
            #   {"start_target": {...},{"end_target": {...}}}
            # which is invalid JSON. Best-effort fix: turn "},{\"key\"" into ",\"key\"".
            content = re.sub(r"\}\s*,\s*\{\s*(\"[a-zA-Z0-9_]+\"\s*:)", r"}, \1", content)
            # Try to extract JSON object
            tool_json = self._extract_json_from_text(content, start_char='{')
            if tool_json:
                tool_calls.append(tool_json)
            else:
                logger.warning(f"Could not extract JSON from tool_call content: {content}")
        
        # Pattern 2: <function_calls>...</function_calls> (OpenAI format)
        pattern2 = r'<function_calls>(.*?)</function_calls>'
        matches2 = re.finditer(pattern2, response_text, re.DOTALL)
        for match in matches2:
            content = match.group(1).strip()
            # Try to extract JSON array
            tools_list = self._extract_json_from_text(content, start_char='[')
            if tools_list and isinstance(tools_list, list):
                tool_calls.extend(tools_list)
            elif tools_list:
                logger.warning(f"function_calls content is not a list: {type(tools_list)}")
        
        # Pattern 3: Try to find JSON objects that look like tool calls in the raw text
        if not tool_calls:
            # Look for JSON objects with "name" and "args" keys directly in text
            json_obj = self._extract_json_from_text(response_text, start_char='{')
            if json_obj and "name" in json_obj and "args" in json_obj:
                tool_calls.append(json_obj)
        
        return tool_calls
    
    async def _execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool call and return result."""
        # Note: Detailed logging is handled in the tool implementation itself
        # This is just a routing function
        
        if tool_name == "action":
            # Convert dict arguments to TargetElement objects
            target = None
            if arguments.get("target"):
                target_arg = arguments["target"]
                # Handle case where target might be a list or other non-dict type
                if isinstance(target_arg, dict):
                    # Standardize on coordinates: [x, y]. If a model outputs x/y, convert.
                    if "coordinates" not in target_arg and "x" in target_arg and "y" in target_arg:
                        try:
                            target_arg = dict(target_arg)
                            target_arg["coordinates"] = [int(target_arg.pop("x")), int(target_arg.pop("y"))]
                        except Exception:
                            # Leave as-is; TargetElement validation will fail loudly.
                            pass
                    # Be forgiving: some models omit the 'element' field but provide coordinates.
                    if "element" not in target_arg and "coordinates" in target_arg:
                        try:
                            target_arg = dict(target_arg)
                            target_arg["element"] = "direct coordinates"
                        except Exception:
                            pass
                    target = TargetElement(**target_arg)
                elif isinstance(target_arg, (list, tuple)) and len(target_arg) >= 2:
                    # Be forgiving: some models output coordinates directly as [x, y].
                    try:
                        target = TargetElement(
                            element="direct coordinates",
                            coordinates=[int(target_arg[0]), int(target_arg[1])],
                        )
                    except Exception:
                        logger.error(f"[Tool Execution] Cannot convert target to TargetElement: {target_arg}")
                else:
                    logger.warning(
                        f"[Tool Execution] target argument is not a dict (type: {type(target_arg)}), "
                        f"value: {target_arg}. Expected dict with 'element' key."
                    )
                    # Try to create a simple TargetElement with just element field
                    if isinstance(target_arg, str):
                        target = TargetElement(element=target_arg)
                    else:
                        logger.error(f"[Tool Execution] Cannot convert target to TargetElement: {target_arg}")
            
            start_target = None
            if arguments.get("start_target"):
                start_arg = arguments["start_target"]
                if isinstance(start_arg, dict):
                    if "coordinates" not in start_arg and "x" in start_arg and "y" in start_arg:
                        try:
                            start_arg = dict(start_arg)
                            start_arg["coordinates"] = [int(start_arg.pop("x")), int(start_arg.pop("y"))]
                        except Exception:
                            pass
                    if "element" not in start_arg and "coordinates" in start_arg:
                        try:
                            start_arg = dict(start_arg)
                            start_arg["element"] = "direct coordinates"
                        except Exception:
                            pass
                    start_target = TargetElement(**start_arg)
                elif isinstance(start_arg, (list, tuple)) and len(start_arg) >= 2:
                    try:
                        start_target = TargetElement(
                            element="direct coordinates",
                            coordinates=[int(start_arg[0]), int(start_arg[1])],
                        )
                    except Exception:
                        logger.error(f"[Tool Execution] Cannot convert start_target to TargetElement: {start_arg}")
                else:
                    logger.warning(
                        f"[Tool Execution] start_target is not a dict (type: {type(start_arg)}), "
                        f"value: {start_arg}"
                    )
                    if isinstance(start_arg, str):
                        start_target = TargetElement(element=start_arg)
            
            end_target = None
            if arguments.get("end_target"):
                end_arg = arguments["end_target"]
                if isinstance(end_arg, dict):
                    if "coordinates" not in end_arg and "x" in end_arg and "y" in end_arg:
                        try:
                            end_arg = dict(end_arg)
                            end_arg["coordinates"] = [int(end_arg.pop("x")), int(end_arg.pop("y"))]
                        except Exception:
                            pass
                    if "element" not in end_arg and "coordinates" in end_arg:
                        try:
                            end_arg = dict(end_arg)
                            end_arg["element"] = "direct coordinates"
                        except Exception:
                            pass
                    end_target = TargetElement(**end_arg)
                elif isinstance(end_arg, (list, tuple)) and len(end_arg) >= 2:
                    try:
                        end_target = TargetElement(
                            element="direct coordinates",
                            coordinates=[int(end_arg[0]), int(end_arg[1])],
                        )
                    except Exception:
                        logger.error(f"[Tool Execution] Cannot convert end_target to TargetElement: {end_arg}")
                else:
                    logger.warning(
                        f"[Tool Execution] end_target is not a dict (type: {type(end_arg)}), "
                        f"value: {end_arg}"
                    )
                    if isinstance(end_arg, str):
                        end_target = TargetElement(element=end_arg)
            
            result = await perform_action_impl(
                action_type=arguments["action_type"],
                option=arguments.get("option"),
                target=target,
                start_target=start_target,
                end_target=end_target,
                direction=arguments.get("direction"),
                distance=arguments.get("distance"),
                text=arguments.get("text"),
                keys=arguments.get("keys"),
                button=arguments.get("button"),
                duration=arguments.get("duration"),
                gbox_client=self.gbox_client,
                screenshot_uri=self.current_screenshot_uri,
                coord_generator=self.coord_generator,
                rollout_logger=self.rollout_logger,
            )
            return result
        
        elif tool_name == "wait":
            duration = arguments.get("duration", 1.0)
            result = await sleep_impl(duration)
            return result
        
        elif tool_name == "finish":
            self.task_completed = True
            self.task_success = arguments.get("success", False)
            self.result_message = arguments.get("result_message", "Task completed")
            self.attempted_completion = True
            # Set termination reason based on success
            if self.task_success:
                self.termination_reason = "finish_success"
            else:
                self.termination_reason = "finish_failure"
            return {
                "status": "complete",
                "success": self.task_success,
                "message": self.result_message,
            }
        
        else:
            # Unknown tool name - this is a tool_name_error
            self.tool_name_errors += 1
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _get_message_role(self, message) -> str:
        """Get role from a message, handling both dict and Message object formats."""
        if isinstance(message, dict):
            return message.get("role", "")
        elif hasattr(message, "role"):
            return message.role
        else:
            return ""
    
    def _truncate_messages_to_recent_turns(
        self,
        messages: List[renderers.Message],
        max_turns: int,  # Must be provided explicitly (use self.max_recent_turns)
    ) -> List[renderers.Message]:
        """
        Truncate messages to keep only system prompt and recent N turns of conversation.
        
        Each turn typically consists of:
        - User message (with screenshot)
        - Assistant response
        - Optional: Tool execution result (user message)
        
        We identify turns by finding assistant messages and their preceding user messages.
        
        Args:
            messages: Full conversation history
            max_turns: Maximum number of recent turns to keep (default: 5)
            
        Returns:
            Truncated messages list with system prompt + recent turns
        """
        if not messages:
            return messages
        
        # Always keep system prompt (first message if it's a system message)
        system_prompt = []
        start_idx = 0
        
        # Check if first message is system prompt (handle both dict and Message object)
        if messages and self._get_message_role(messages[0]) == "system":
            system_prompt = [messages[0]]
            start_idx = 1
        
        # If there are no messages after system prompt, return just the system prompt
        if start_idx >= len(messages):
            return system_prompt
        
        # Find all assistant message indices (these mark the end of each turn)
        assistant_indices = []
        for i in range(start_idx, len(messages)):
            if self._get_message_role(messages[i]) == "assistant":
                assistant_indices.append(i)
        
        # If we have fewer turns than max_turns, keep all messages
        if len(assistant_indices) <= max_turns:
            return system_prompt + messages[start_idx:]
        
        # Keep only the last max_turns turns
        # Start from the (len(assistant_indices) - max_turns)-th assistant message
        first_turn_assistant_idx = assistant_indices[len(assistant_indices) - max_turns]
        
        # Find the start of this turn: look backwards from the assistant message
        # Each turn starts with a user message (with screenshot), followed by assistant response,
        # and optionally a tool result (also a user message)
        # We need to find the first user message that starts this turn
        turn_start_idx = first_turn_assistant_idx
        
        # Look backwards to find the user message that starts this turn
        # There may be a tool result (user) right before the assistant, but we want the
        # user message with screenshot that actually starts the turn
        # In practice, we'll just go back to the first user message before this assistant
        while turn_start_idx > start_idx:
            if self._get_message_role(messages[turn_start_idx - 1]) == "user":
                turn_start_idx = turn_start_idx - 1
                # Continue looking backwards until we find a non-user message or reach the start
                # This handles the case where there's a tool result (user) right before the assistant
                break
            turn_start_idx -= 1
        
        return system_prompt + messages[turn_start_idx:]
    
    async def _sample_with_model(
        self,
        messages: List[renderers.Message],
    ) -> Tuple[renderers.Message, bool, List[int], List[float]]:
        """
        Sample from model using inference client.
        
        This method works with any provider (Tinker, vLLM, OpenRouter, etc.)
        by following these steps:
        1. For Tinker: Get tokens and logprobs directly (needed for training)
        2. For others: Generate text, then tokenize
        3. Parse using renderer to extract tool calls
        
        Returns:
            (response_message, parse_success, tokens, logprobs)
            - response_message: Parsed message with content and optional tool_calls
            - parse_success: Whether parsing was successful
            - tokens: List of token IDs
            - logprobs: List of log probabilities (empty for non-Tinker providers)
        """
        # Truncate messages to keep only system prompt and recent N turns
        truncated_messages = self._truncate_messages_to_recent_turns(messages, max_turns=self.max_recent_turns)
        
        # Check if this is Tinker provider that supports logprobs
        use_tinker_logprobs = (
            self.provider == "tinker" and 
            hasattr(self.inference_client, 'get_tokens_and_logprobs')
        )
        
        if use_tinker_logprobs:
            # For Tinker: Get tokens and logprobs directly
            logger.debug(f"[_sample_with_model] Calling Tinker get_tokens_and_logprobs()...")
            tokens, logprobs = await self.inference_client.get_tokens_and_logprobs(
                messages=truncated_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            logger.debug(f"[_sample_with_model] Got {len(tokens)} tokens and {len(logprobs)} logprobs")
        else:
            # For other providers: Generate text then tokenize
            logger.debug(f"[_sample_with_model] Calling {self.provider} generate_text()...")
            text = await self.inference_client.generate_text(
                messages=truncated_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            logger.debug(f"[_sample_with_model] Generated text length: {len(text)} chars")
            
            # Tokenize the generated text.
            #
            # Important: OpenAI-compatible APIs return the assistant content *without* including the
            # model's EOT token in the text. Our renderers' parse_response() logic often expects an
            # explicit stop token (e.g., Qwen: <|im_end|>) to mark a well-formed message.
            #
            # To avoid spurious "parse_success=False" for HTTP providers, we append the stop sequence
            # to the token stream before parsing.
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            try:
                stops = self.renderer.get_stop_sequences()
                if stops:
                    if isinstance(stops[0], int):
                        tokens = tokens + [stops[0]]
                    elif isinstance(stops[0], str):
                        tokens = tokens + self.tokenizer.encode(stops[0], add_special_tokens=False)
            except Exception:
                # Best-effort; parsing may still succeed without explicit stop tokens for some renderers.
                pass
            logprobs = []  # No logprobs for non-Tinker providers
        
        # Parse response using renderer (extracts <tool_call> tags, etc.)
        logger.debug(f"[_sample_with_model] Parsing response with renderer...")
        response_message, parse_success = self.renderer.parse_response(tokens)
        
        logger.debug(f"[_sample_with_model] Parse success: {parse_success}, tokens: {len(tokens)}, logprobs: {len(logprobs)}")
        
        return response_message, parse_success, tokens, logprobs
    
    def _save_turn_data_to_db(self, turn: int):
        """Save the completed turn's data to database immediately."""
        if not self.rollout_logger or not self.rollout_recorder:
            return
        
        try:
            # Get the turn data that was just saved
            if (hasattr(self.rollout_logger, 'trajectory_data') and 
                'turns' in self.rollout_logger.trajectory_data):
                turns_data = self.rollout_logger.trajectory_data['turns']
                # Find the turn we just completed
                completed_turn_data = None
                for t in turns_data:
                    if t.get('turn_num') == turn:
                        completed_turn_data = t
                        break
                
                if completed_turn_data:
                    # Get existing trajectory data from database
                    from tinker_cookbook.recipes.cua_rl.database.database_dao import get_rollout_by_rollout_id
                    
                    existing_data = {}
                    rollout = get_rollout_by_rollout_id(self.rollout_recorder.session, self.rollout_id)
                    if rollout and rollout.trajectory_data_json:
                        try:
                            existing_data = json.loads(rollout.trajectory_data_json)
                        except Exception as parse_error:
                            logger.warning(f"[Turn {turn}] Failed to parse existing trajectory_data_json: {parse_error}")
                            existing_data = {}
                    
                    # Ensure execution_details exists
                    if 'execution_details' not in existing_data:
                        existing_data['execution_details'] = {}
                    if 'turns' not in existing_data['execution_details']:
                        existing_data['execution_details']['turns'] = []
                    
                    # Add or update this turn's data
                    turns_list = existing_data['execution_details']['turns']
                    # Check if turn already exists
                    turn_exists = False
                    for i, t in enumerate(turns_list):
                        if t.get('turn_num') == turn:
                            turns_list[i] = completed_turn_data
                            turn_exists = True
                            break
                    if not turn_exists:
                        turns_list.append(completed_turn_data)
                    
                    # Save updated data through rollout_recorder
                    trajectory_data_json = json.dumps(existing_data, default=str)
                    self.rollout_recorder.update(
                        trajectory_data_json=trajectory_data_json
                    )
                    logger.info(f"[Turn {turn}] ✓ Saved turn data to database (rollout {self.rollout_id})")
        except Exception as e:
            # Log error but don't raise - we don't want to break the rollout
            logger.error(f"[Turn {turn}] Failed to save turn data to database: {e}", exc_info=True)
    
    async def run_task(
        self,
        task_description: str,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a task with the agent.
        
        Args:
            task_description: Description of the task to complete
            verbose: Whether to print verbose output
            
        Returns:
            Dictionary with task results
        """
        task_start_time = time.time()
        
        # Get context info from rollout_logger if available
        context_parts = []
        if self.rollout_logger:
            if self.rollout_logger.step is not None:
                context_parts.append(f"Step: {self.rollout_logger.step}")
            if self.rollout_logger.batch is not None:
                context_parts.append(f"Batch: {self.rollout_logger.batch}")
            if self.rollout_logger.group is not None:
                context_parts.append(f"Group: {self.rollout_logger.group}")
            if self.rollout_logger.rollout_index is not None:
                context_parts.append(f"Rollout: {self.rollout_logger.rollout_index}")
        
        context_str = " | ".join(context_parts) if context_parts else ""
        
        # Get task name if available
        task_name_str = ""
        if self.task:
            task_name = getattr(self.task, 'name', None) or getattr(self.task, 'id', None)
            if task_name:
                task_name_str = f"Task: {task_name} | "
        
        # Force flush before starting task execution
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        logger.info("")
        logger.info("╔" + "=" * 118 + "╗")
        if context_str:
            # Format: "TASK EXECUTION START | Task: X | Step: Y | Batch: Z | Group: W | Rollout: V"
            title = f"TASK EXECUTION START | {task_name_str}{context_str}"
            # Calculate padding to center the text
            total_len = len(title)
            padding_left = (118 - total_len) // 2
            padding_right = 118 - total_len - padding_left
            logger.info(f"║{' ' * padding_left}{title}{' ' * padding_right}║")
            # Force flush after logging
            sys.stdout.flush()
            sys.stderr.flush()
        else:
            # If no context but have task name, show it
            if task_name_str:
                title = f"TASK EXECUTION START | {task_name_str.rstrip(' | ')}"
                total_len = len(title)
                padding_left = (118 - total_len) // 2
                padding_right = 118 - total_len - padding_left
                logger.info(f"║{' ' * padding_left}{title}{' ' * padding_right}║")
            else:
                logger.info("║" + " " * 48 + "TASK EXECUTION START" + " " * 50 + "║")
        logger.info("╠" + "=" * 118 + "╣")
        logger.info(f"║ Task Description: {task_description[:98]:<98} ║")
        logger.info(f"║ Max Turns: {self.max_turns:<105} ║")
        logger.info(f"║ Box Type: {self.box_type:<106} ║")
        logger.info(f"║ Model Path: {self.tinker_model_path[:104]:<104} ║")
        logger.info("╚" + "=" * 118 + "╝")
        
        # Initialize turn counter before try block so it's always defined
        turn = 0
        
        # Reset metrics for this rollout
        self.num_total_actions = 0
        self.consecutive_repeated_actions = 0
        self.parse_errors = 0
        self.tool_name_errors = 0
        self.tool_arg_errors = 0
        self.runtime_errors = 0
        self.turn_first_success = -1
        self.turn_task_completed = -1
        self.attempted_completion = False
        self._last_action_signature = None
        self._current_repeat_count = 0
        
        # Reset trajectory data for this rollout
        self.trajectory_turns.clear()
        
        # Track recording state (initialized before try block so it's accessible in finally)
        recording_started = False
        
        try:
            # Get APK configuration from task first (needed for both APK installation and system prompt)
            # Check both self.task and self.task._original_task (for CUATask wrapper)
            task_for_config = None
            if self.task is not None:
                # First try the task itself
                if hasattr(self.task, 'get_apk_config'):
                    task_for_config = self.task
                # Then try _original_task if it exists (for wrapped tasks)
                elif hasattr(self.task, '_original_task'):
                    original = self.task._original_task
                    if original is not None and hasattr(original, 'get_apk_config'):
                        task_for_config = original
            
            # Get APK config from task
            from tinker_cookbook.recipes.cua_rl.executor.base import ApkConfig
            if task_for_config is not None:
                try:
                    apk_config = task_for_config.get_apk_config()
                    logger.info(f"[Task Setup] Using config from task: app_name={apk_config.app_name}, requires_apk={apk_config.requires_apk}")
                except Exception as e:
                    logger.warning(f"[Task Setup] Failed to get config from task: {e}, using default")
                    apk_config = ApkConfig()
            else:
                # No task available, default config
                logger.warning(f"[Task Setup] No task object available, using default config")
                apk_config = ApkConfig()
            
            # Create system prompt using config
            prompt_start = time.time()
            self.system_prompt = create_system_prompt(
                task_description=task_description,
                max_turns=self.max_turns,
                cua_guide=apk_config.cua_guide,  # Use guide from config
                box_type=self.box_type,
                coordinate_mode=self.coordinate_mode,
                coordinate_scale=self.coordinate_scale,
            )
            prompt_time = time.time() - prompt_start
            
            # Initialize conversation history
            self.messages = [
                renderers.Message(role="system", content=self.system_prompt)
            ]
            
            # Create GBox environment
            box_creation_start = time.time()
            
            # Start env build logging
            if self.rollout_logger:
                self.rollout_logger.log_env_build_start()
            
            # apk_config is already loaded above, reuse it for environment setup
            
            # Create box without installing APK (we'll do it manually if needed)
            # Always use logger.info (not rollout_logger) for critical debug messages
            import sys
            logger.info(f"[Task Setup] Creating box (this may take a while)...")
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Log box creation start
            box_create_stage_start = time.time()
            if self.rollout_logger:
                self.rollout_logger.log_env_build_stage(
                    "Box Creation",
                    "in_progress",
                    details={"type": self.box_type}
                )
            
            box_info = await self.gbox_client.create_box(box_type=self.box_type, apk_paths=None)
            box_id = box_info.get("id") or self.gbox_client.box_id
            box_create_time = time.time() - box_create_stage_start
            
            # Log box creation success
            if self.rollout_logger:
                self.rollout_logger.log_env_build_stage(
                    "Box Creation",
                    "success",
                    duration=box_create_time,
                    details={
                        "box_id": box_id,
                        "box_type": self.box_type,
                    }
                )
            
            logger.info(f"[Task Setup] ✓ Box created: {box_id}")
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Install APK if required by task configuration
            if apk_config.requires_apk:
                if not apk_config.apk_url or not apk_config.package_name:
                    error_msg = f"[Task Setup] ✗ Task requires APK but config is incomplete: apk_url={apk_config.apk_url}, package_name={apk_config.package_name}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Always use logger.info for critical debug messages to ensure real-time output
                logger.info(f"[Task Setup] Installing APK from {apk_config.apk_url}...")
                sys.stdout.flush()
                sys.stderr.flush()
                
                # Log APK installation start
                apk_install_stage_start = time.time()
                if self.rollout_logger:
                    self.rollout_logger.log_env_build_stage(
                        "APK Installation",
                        "in_progress",
                        details={
                            "apk_url": apk_config.apk_url,
                            "package": apk_config.package_name,
                            "launch_after_install": apk_config.launch_after_install,
                        }
                    )
                
                try:
                    # Install and optionally launch the app
                    await self.gbox_client.install_apk(apk_config.apk_url, open_app=apk_config.launch_after_install)
                    apk_install_time = time.time() - apk_install_stage_start
                    
                    # Log APK installation success
                    if self.rollout_logger:
                        self.rollout_logger.log_env_build_stage(
                            "APK Installation",
                            "success",
                            duration=apk_install_time,
                            details={
                                "package": apk_config.package_name,
                                "app_opened": str(apk_config.launch_after_install),
                            }
                        )
                    
                    if apk_config.launch_after_install:
                        logger.info(f"[Task Setup] ✓ APK installed and app opened")
                    else:
                        logger.info(f"[Task Setup] ✓ APK installed (app not launched)")
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception as apk_error:
                    apk_install_time = time.time() - apk_install_stage_start
                    error_msg = f"[Task Setup] ✗ Failed to install APK: {apk_error}"
                    
                    # Log APK installation error
                    if self.rollout_logger:
                        self.rollout_logger.log_env_build_stage(
                            "APK Installation",
                            "error",
                            duration=apk_install_time,
                            error=str(apk_error)
                        )
                    
                    logger.error(error_msg, exc_info=True)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    raise  # Re-raise to be caught by outer exception handler
            else:
                logger.info(f"[Task Setup] No APK installation required for this task")
                sys.stdout.flush()
                sys.stderr.flush()
            
            box_creation_time = time.time() - box_creation_start
            if self.rollout_logger:
                self.rollout_logger.log(f"[Task Setup] Box created and environment prepared in {box_creation_time:.3f}s")
            else:
                logger.info(f"[Task Setup] Task executing on {apk_config.package_name or 'system'} with max {self.max_turns} turns on the box {box_id} (box prepared in {box_creation_time:.3f}s)")

            # Create AdbClient for prehook and app launch
            from tinker_cookbook.recipes.cua_rl.tasks.adb import AdbClient
            adb_client = AdbClient(gbox_client=self.gbox_client)
            
            # Wait additional time after app launch before executing prehook
            if self.rollout_logger:
                self.rollout_logger.log(f"[Task Setup] Waiting 3 seconds for app to stabilize...")
            else:
                logger.info(f"[Task Setup] Waiting 3 seconds for app to stabilize...")
            await asyncio.sleep(3.0)
            
            # Start screen recording
            try:
                box = self.gbox_client._get_box()
                box.action.recording.start()
                recording_started = True
                if self.rollout_logger:
                    self.rollout_logger.log(f"[Task Setup] Screen recording started")
                else:
                    logger.info(f"[Task Setup] Screen recording started")
            except Exception as e:
                if self.rollout_logger:
                    self.rollout_logger.log(f"[Task Setup] Failed to start recording: {e}")
                else:
                    logger.warning(f"[Task Setup] Failed to start recording: {e}")

            # Execute prehook after app launch and wait period
            # Check both self.task and self.task._original_task (for CUATask wrapper)
            pre_hook = None
            task_with_prehook = None
            
            if self.task is not None:
                # First, try to get prehook from self.task directly
                if hasattr(self.task, "get_pre_hook"):
                    pre_hook = self.task.get_pre_hook()
                    task_with_prehook = self.task
                # If not found, check _original_task (for CUATask wrapper from task_adapter)
                elif hasattr(self.task, "_original_task") and self.task._original_task is not None:
                    original_task = self.task._original_task
                    if hasattr(original_task, "get_pre_hook"):
                        pre_hook = original_task.get_pre_hook()
                        task_with_prehook = original_task
            
            if pre_hook is not None:
                prehook_start = time.time()
                if self.rollout_logger:
                    self.rollout_logger.log_env_build_stage(
                        "Prehook Execution",
                        "in_progress",
                    )
                
                try:
                    # Execute prehook
                    # Capture print output from prehook.run() if possible
                    import sys
                    from io import StringIO
                    old_stdout = sys.stdout
                    captured_output = StringIO()
                    sys.stdout = captured_output
                    
                    try:
                        pre_hook.run(adb_client)
                    finally:
                        sys.stdout = old_stdout
                        output = captured_output.getvalue()
                    
                    prehook_time = time.time() - prehook_start
                    
                    # Log prehook success
                    prehook_details = {}
                    if output.strip():
                        prehook_details["output"] = output.strip()[:200]  # Limit output length
                    
                    if self.rollout_logger:
                        self.rollout_logger.log_env_build_stage(
                            "Prehook Execution",
                            "success",
                            duration=prehook_time,
                            details=prehook_details
                        )
                        # Store full output in trajectory data
                        if "env_build" in self.rollout_logger.trajectory_data:
                            self.rollout_logger.trajectory_data["env_build"]["prehook_executed"] = True
                            self.rollout_logger.trajectory_data["env_build"]["prehook_output"] = output.strip() if output.strip() else None
                    else:
                        logger.info(f"[Task Setup] ✓ Prehook executed in {prehook_time:.3f}s")
                        if output.strip():
                            logger.info(f"[Task Setup] Prehook output: {output.strip()}")
                except Exception as e:
                    prehook_time = time.time() - prehook_start
                    error_msg = str(e)
                    
                    # Log prehook error
                    if self.rollout_logger:
                        self.rollout_logger.log_env_build_stage(
                            "Prehook Execution",
                            "error",
                            duration=prehook_time,
                            error=error_msg
                        )
                    else:
                        logger.error(f"[Task Setup] ✗ Prehook execution failed after {prehook_time:.3f}s: {error_msg}")
                    # Continue execution even if prehook fails
            
            # Complete env build logging
            if self.rollout_logger:
                self.rollout_logger.log_env_build_complete(
                    total_time=box_creation_time,
                    box_id=box_id,
                    box_type=self.box_type,
                    success=True
                )
                
                # Save env_build data to database immediately (so it's visible during rollout)
                if self.rollout_recorder is not None:
                    try:
                        # Create a partial trajectory_data_json with just env_build info
                        env_build_data = {
                            "execution_details": {
                                "env_build": self.rollout_logger.trajectory_data.get("env_build", {})
                            }
                        }
                        trajectory_data_json = json.dumps(env_build_data, default=str)
                        
                        # Update rollout with env_build data
                        self.rollout_recorder.update(
                            trajectory_data_json=trajectory_data_json
                        )
                        logger.debug(f"[Env Build] Saved env_build data to database for rollout {self.rollout_id}")
                    except Exception as e:
                        logger.warning(f"[Env Build] Failed to save env_build data to database: {e}")
            
            # Environment setup is now complete (APK installed, app launched, prehook executed)
            if self.rollout_logger:
                self.rollout_logger.log(f"[Task Setup] ✓ Environment setup complete")
            else:
                logger.info(f"[Task Setup] ✓ Environment setup complete")

            # Run task in turns
            turn = 0
            
            while turn < self.max_turns and not self.task_completed:
                # Check for timeout (configurable total task time)
                elapsed_time = time.time() - task_start_time
                if elapsed_time > self.max_task_time_seconds:
                    self.termination_reason = f"timeout_{self.max_task_time_seconds//60}min"
                    logger.warning(f"[Task Timeout] Task exceeded {self.max_task_time_seconds/60:.0f} minute timeout after {elapsed_time/60:.1f} minutes. Ending rollout.")
                    self.result_message = f"Task exceeded {self.max_task_time_seconds/60:.0f} minute timeout"
                    self.task_completed = False
                    self.task_success = False
                    break
                
                turn += 1
                # Debug: Log turn start
                logger.info(f"[Turn {turn}] Starting turn {turn}/{self.max_turns}, task_completed={self.task_completed}, elapsed_time={elapsed_time:.1f}s")
                turn_start_time = time.time()
                
                # Track timing for each stage
                stage_timings = {}
                
                # Record turn start in database
                if self.rollout_recorder is not None:
                    try:
                        turn_id = self.rollout_recorder.start_turn(turn)
                        if not turn_id:
                            logger.error(f"[Turn {turn}] Failed to start turn in database")
                    except Exception as e:
                        logger.warning(f"[Turn {turn}] Failed to record turn start in database: {e}", exc_info=True)
                
                # Start turn logging (includes turn header)
                if self.rollout_logger:
                    self.rollout_logger.start_turn(turn, self.max_turns)
                else:
                    logger.info("")
                    logger.info("┌" + "─" * 118 + "┐")
                    logger.info(f"│ Turn {turn}/{self.max_turns}" + " " * (118 - 12 - len(str(self.max_turns)) - len(str(turn)) - 1) + "│")
                    logger.info("└" + "─" * 118 + "┘")
                
                # Take screenshot (Before screenshot - what the model sees before making a decision)
                screenshot_start = time.time()
                if not self.rollout_logger:
                    logger.info(f"[Turn {turn}] Taking screenshot...")
                await asyncio.sleep(0.3)  # Small delay for stability
                logger.info(f"[Turn {turn}] About to call take_screenshot()...")
                screenshot_bytes, screenshot_uri = await self.gbox_client.take_screenshot()
                logger.info(f"[Turn {turn}] Screenshot call returned")
                self.current_screenshot_uri = screenshot_uri
                screenshot_time = time.time() - screenshot_start
                stage_timings['screenshot_before'] = screenshot_time
                if self.rollout_logger:
                    self.rollout_logger.log_screenshot(screenshot_uri, screenshot_time)
                else:
                    logger.info(f"[Turn {turn}] ✓ Screenshot taken in {screenshot_time:.3f}s")
                    logger.info(f"[Turn {turn}] Screenshot URI: {screenshot_uri[:100]}...")
                
                # On first turn in Direct mode, extract screen dimensions
                if turn == 1 and self.coordinate_mode == "direct":
                    # Extract screen dimensions - fail fast if cannot extract
                    screen_width, screen_height = self._get_screen_dimensions(screenshot_uri)
                    logger.info(f"[Turn {turn}] Detected screen dimensions: {screen_width}x{screen_height}")
                    
                    # Update DirectCoordinateGenerator with actual screen dimensions
                    # This is needed for both scaling (to compute ratios) and boundary checks
                    if hasattr(self.coord_generator, 'screen_width'):
                        self.coord_generator.screen_width = screen_width
                        self.coord_generator.screen_height = screen_height
                        self.coord_generator.default_x = screen_width // 2
                        self.coord_generator.default_y = screen_height // 2
                        
                        # Update scale ratios if coordinate scaling is enabled
                        if self.coordinate_scale:
                            # Use custom ratios if provided, otherwise default to screen/1000
                            self.coord_generator.x_scale_ratio = (
                                self.x_scale_ratio if self.x_scale_ratio is not None 
                                else screen_width / 1000.0
                            )
                            self.coord_generator.y_scale_ratio = (
                                self.y_scale_ratio if self.y_scale_ratio is not None 
                                else screen_height / 1000.0
                            )
                            logger.info(
                                f"[Turn {turn}] Updated coordinate scaling: "
                                f"ratios=({self.coord_generator.x_scale_ratio:.3f}, {self.coord_generator.y_scale_ratio:.3f})"
                            )
                    
                    # Update system prompt with actual screen dimensions (only if not using scaling)
                    if not self.coordinate_scale:
                        self.system_prompt = create_system_prompt(
                            task_description=task_description,
                            max_turns=self.max_turns,
                            app_name=app_name,
                            box_type=self.box_type,  # Pass box type
                            coordinate_mode=self.coordinate_mode,
                            coordinate_scale=self.coordinate_scale,
                            screen_width=screen_width,
                            screen_height=screen_height,
                        )
                        
                        # Update messages with new system prompt
                        self.messages[0] = renderers.Message(role="system", content=self.system_prompt)
                        
                        if self.rollout_logger:
                            self.rollout_logger.log(
                                f"[Turn {turn}] Updated system prompt with screen dimensions: {screen_width}x{screen_height}"
                            )
                    else:
                        if self.rollout_logger:
                            self.rollout_logger.log(
                                f"[Turn {turn}] Coordinate scaling enabled - screen dimensions not included in prompt"
                            )
                
                # Build user message with screenshot
                model_input_prep_start = time.time()
                user_text = f"Turn {turn}/{self.max_turns}. Analyze the screenshot and take the next action to complete the task."
                user_messages = self._build_messages_with_screenshot(user_text, screenshot_uri)
                
                # Add to conversation history
                self.messages.extend(user_messages)
                
                # Sample from model
                model_call_start = time.time()
                stage_timings['model_input_prep'] = model_call_start - model_input_prep_start
                if not self.rollout_logger:
                    logger.info(f"[Turn {turn}] Calling model for inference...")
                    logger.info(f"[Turn {turn}] Model input - Conversation length: {len(self.messages)} messages")
                
                # Build ModelInput for this turn (with message history truncation)
                # Only keep system prompt and recent N turns to match what the model actually sees
                truncated_messages_for_observation = self._truncate_messages_to_recent_turns(
                    self.messages, max_turns=self.max_recent_turns
                )
                turn_observation = self.renderer.build_generation_prompt(truncated_messages_for_observation)
                
                # Record Before screenshot and model_input in database (right after turn start)
                logger.info(f"[Turn {turn}] About to check rollout_recorder: {self.rollout_recorder is not None}")
                if self.rollout_recorder is not None:  # Using new rollout_recorder
                    logger.info(f"[Turn {turn}] rollout_recorder is not None, attempting to record observation")
                    try:
                        from tinker_cookbook.recipes.cua_rl.database.compat import record_observation
                        # NOTE: We intentionally avoid storing the raw ModelInput repr / token lists.
                        # It's huge, not human-friendly, and may contain non-JSON-serializable objects.
                        model_input_dict = None
                        
                        # Also save readable messages format for display
                        # Use truncated messages to match what the model actually sees
                        readable_messages = None
                        if truncated_messages_for_observation:
                            try:
                                # Convert truncated messages to readable format
                                readable_messages_list = []
                                for msg in truncated_messages_for_observation:
                                    # msg can be a renderers.Message, or a dict-like structure
                                    if hasattr(msg, "role") and hasattr(msg, "content"):
                                        role = getattr(msg, "role")
                                        content = getattr(msg, "content")
                                    elif isinstance(msg, dict):
                                        role = msg.get("role", "unknown")
                                        content = msg.get("content", "")
                                    else:
                                        role = "unknown"
                                        content = str(msg)

                                    msg_dict = {
                                        "role": role or "unknown",
                                        "content": [],
                                    }
                                    
                                    # Handle content - could be string, list, or dict
                                    if isinstance(content, str):
                                        msg_dict["content"].append({"type": "text", "text": content})
                                    elif isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict):
                                                if item.get("type") == "image_url" or "image_url" in item:
                                                    # Always reference the screenshot we store on disk (no base64 in model input)
                                                    msg_dict["content"].append({"type": "image", "url": "__SCREENSHOT_BEFORE__"})
                                                elif item.get("type") == "text" or "text" in item:
                                                    msg_dict["content"].append({"type": "text", "text": item.get("text", "")})
                                                else:
                                                    msg_dict["content"].append(item)
                                            elif isinstance(item, str):
                                                msg_dict["content"].append({"type": "text", "text": item})
                                            else:
                                                # PIL.Image.Image or other objects – avoid stringifying into "<PIL.Image.Image ...>"
                                                type_name = type(item).__name__
                                                if type_name.lower().endswith("image") or "PIL" in str(type(item)):
                                                    msg_dict["content"].append({"type": "image", "url": "__SCREENSHOT_BEFORE__"})
                                                else:
                                                    msg_dict["content"].append({"type": "text", "text": str(item)})
                                    elif isinstance(content, dict):
                                        # If this dict contains an image, normalize to screenshot placeholder
                                        if content.get("type") in ("image", "image_url") or "image_url" in content:
                                            msg_dict["content"].append({"type": "image", "url": "__SCREENSHOT_BEFORE__"})
                                        else:
                                            msg_dict["content"].append(content)
                                    
                                    readable_messages_list.append(msg_dict)
                                
                                readable_messages = {
                                    "messages": readable_messages_list,
                                    # Used by training-monitor to resolve the placeholder image URL
                                    "image_placeholders": {"__SCREENSHOT_BEFORE__": "screenshot_before"},
                                }
                            except Exception as e:
                                logger.warning(f"[Turn {turn}] Failed to create readable messages format: {e}")
                                readable_messages = {
                                    "messages": None,
                                }
                        
                        record_observation(
                            None,  # session parameter is ignored by compat layer
                            None,  # turn_id is ignored by compat layer
                            obs_type="screenshot_before",
                            screenshot_uri=screenshot_uri,
                            model_input=readable_messages if readable_messages else None,
                            rollout_id=self.rollout_id,
                            turn=turn,  # Add turn parameter for compat layer
                        )
                        pass  # Commit handled by rollout_recorder
                        logger.debug(f"[Turn {turn}] Recorded Before screenshot and model_input for turn {turn}")
                    except Exception as e:
                        logger.warning(f"[Turn {turn}] Failed to record Before screenshot/model_input: {e}")
                
                logger.info(f"[Turn {turn}] About to call _sample_with_model()...")
                # Add timeout for model inference (configurable per turn)
                try:
                    response_message, parse_success, action_tokens, action_logprobs = await asyncio.wait_for(
                        self._sample_with_model(self.messages),
                        timeout=self.max_turn_time_seconds
                    )
                    logger.info(f"[Turn {turn}] _sample_with_model() returned")
                except asyncio.TimeoutError:
                    self.termination_reason = f"model_timeout_turn_{turn}"
                    logger.error(f"[Turn {turn}] Model inference timed out after {self.max_turn_time_seconds/60:.0f} minutes. Ending rollout.")
                    self.result_message = f"Model inference timed out on turn {turn}"
                    self.task_completed = False
                    self.task_success = False
                    # Don't add trajectory data for this turn since we didn't get a response
                    # But we should still save any previous turns' trajectory data
                    break
                except Exception as e:
                    self.termination_reason = f"model_error_turn_{turn}"
                    logger.error(f"[Turn {turn}] Model inference failed with exception: {e}", exc_info=True)
                    self.result_message = f"Model inference failed on turn {turn}: {str(e)}"
                    self.task_completed = False
                    self.task_success = False
                    # Don't add trajectory data for this turn since we didn't get a valid response
                    break
                
                # If we get here, we have a valid response
                response_text = response_message.get("content", "")
                
                # Store trajectory data for this turn (only if we got a valid response)
                self.trajectory_turns.append((
                    turn,
                    turn_observation,
                    action_tokens,
                    action_logprobs,
                ))
                
                model_call_time = time.time() - model_call_start
                stage_timings['model_inference'] = model_call_time
                renderer_parse_success = bool(parse_success)
                
                # Parse tool calls - prefer renderer's parsed tool_calls, fallback to custom parsing
                parse_start = time.time()
                tool_calls_from_renderer = response_message.get("tool_calls", [])
                
                if tool_calls_from_renderer:
                    # Convert ToolCall objects to dict format
                    tool_calls = []
                    for tc in tool_calls_from_renderer:
                        if hasattr(tc, 'function') and hasattr(tc.function, 'name'):
                            # ToolCall object from renderer
                            tool_calls.append({
                                "name": tc.function.name,
                                "args": json.loads(tc.function.arguments),
                            })
                        else:
                            # Already in dict format
                            tool_calls.append(tc)
                    parser_type = "renderer"
                else:
                    # Fallback to custom parsing if renderer didn't find tool calls
                    tool_calls = self._parse_tool_calls(response_text)
                    parser_type = "custom_parser"
                
                parse_time = time.time() - parse_start
                stage_timings['action_parse'] = parse_time
                overall_parse_success = bool(renderer_parse_success or len(tool_calls) > 0)
                
                if self.rollout_logger:
                    # Log model inference with an "overall" parse bit so the UI doesn't show false negatives
                    # when renderer parsing fails but tool calls are still extractable from raw text.
                    if self.rollout_logger.current_turn is not None:
                        self.rollout_logger.current_turn["renderer_parse_success"] = renderer_parse_success
                        self.rollout_logger.current_turn["tool_call_parse_success"] = bool(len(tool_calls) > 0)
                        self.rollout_logger.current_turn["user_input"] = user_text
                    self.rollout_logger.log_model_inference(turn, response_text, overall_parse_success, model_call_time)
                    self.rollout_logger.log_tool_calls(turn, tool_calls, parse_time, parser_type)
                else:
                    logger.info(f"[Turn {turn}] Using tool_calls parsed by {parser_type}: {len(tool_calls)} found")
                    logger.info(f"[Turn {turn}] Tool call parsing completed in {parse_time:.3f}s")
                    logger.info(f"[Turn {turn}] Number of tool calls found: {len(tool_calls)}")
                    logger.info(f"[Turn {turn}] ✓ Model inference completed in {model_call_time:.3f}s")
                    logger.info(
                        f"[Turn {turn}] Parse success: overall={overall_parse_success} "
                        f"(renderer={renderer_parse_success}, tool_calls={len(tool_calls)})"
                    )
                    logger.info(f"[Turn {turn}] Model response length: {len(response_text)} characters")
                    logger.info(f"[Turn {turn}] Model response (full):")
                    logger.info("  " + "\n  ".join(response_text.split("\n")))
                
                if tool_calls:
                    # Execute tool calls
                    tool_results = []
                    for tool_call_idx, tool_call in enumerate(tool_calls):
                        tool_name = tool_call.get("name")
                        # Use "args" field (model outputs "args")
                        tool_args = tool_call.get("args", {})
                        
                        # Validate tool call format
                        if not tool_name:
                            error_msg = f"Tool call {tool_call_idx + 1} is missing 'name' field. Tool call format: {{'name': 'tool_name', 'args': {{...}}}}"
                            self.parse_errors += 1
                            if self.rollout_logger:
                                self.rollout_logger.log(f"[Turn {turn}] ⚠ Tool call parse error: {error_msg}")
                            else:
                                logger.warning(f"[Turn {turn}] ⚠ Tool call parse error: {error_msg}")
                            tool_results.append({
                                "tool": "unknown",
                                "error": error_msg,
                                "status": "parse_error",
                            })
                            continue
                        
                        if not isinstance(tool_args, dict):
                            error_msg = f"Tool call {tool_call_idx + 1} has invalid 'args' field (expected dict, got {type(tool_args)}). Tool call format: {{'name': 'tool_name', 'args': {{...}}}}"
                            self.parse_errors += 1
                            if self.rollout_logger:
                                self.rollout_logger.log(f"[Turn {turn}] ⚠ Tool call parse error: {error_msg}")
                            else:
                                logger.warning(f"[Turn {turn}] ⚠ Tool call parse error: {error_msg}")
                            tool_results.append({
                                "tool": tool_name,
                                "error": error_msg,
                                "status": "parse_error",
                            })
                            continue
                        
                        # Validate tool arguments (basic check for required fields)
                        if tool_name == "action" and "action_type" not in tool_args:
                            self.tool_arg_errors += 1
                            error_msg = f"Tool call {tool_call_idx + 1} (action) is missing required 'action_type' field in args. Tool call format: {{'name': 'action', 'args': {{'action_type': 'tap'|'swipe'|..., ...}}}}"
                            if self.rollout_logger:
                                self.rollout_logger.log(f"[Turn {turn}] ⚠ Tool call parse error: {error_msg}")
                            else:
                                logger.warning(f"[Turn {turn}] ⚠ Tool call parse error: {error_msg}")
                            tool_results.append({
                                "tool": tool_name,
                                "error": error_msg,
                                "status": "parse_error",
                            })
                            continue
                        elif tool_name == "wait" and "duration" not in tool_args:
                            # duration is optional for wait, so this is not an error
                            pass
                        elif tool_name == "finish":
                            # finish doesn't require specific args
                            pass
                        
                        if not self.rollout_logger:
                            logger.info(f"[Turn {turn}] Tool call {tool_call_idx + 1}/{len(tool_calls)}: {tool_name}")
                            logger.info(f"[Turn {turn}] Tool arguments: {json.dumps(tool_args, indent=2)}")
                        
                        tool_exec_start = time.time()
                        try:
                            # Track action for consecutive repeat detection (only for action and wait tools)
                            action_signature = None
                            if tool_name in ["action", "wait"]:
                                action_signature = self._create_action_signature(tool_name, tool_args)
                                
                                # Check for consecutive repeat
                                if action_signature == self._last_action_signature:
                                    self._current_repeat_count += 1
                                    # Only count as consecutive repeat if it's the 2nd or more repeat
                                    if self._current_repeat_count >= 2:
                                        self.consecutive_repeated_actions = max(
                                            self.consecutive_repeated_actions,
                                            self._current_repeat_count - 1  # -1 because first repeat doesn't count
                                        )
                                else:
                                    # Different action, reset repeat count
                                    self._current_repeat_count = 0
                                    self._last_action_signature = action_signature
                            
                            result = await self._execute_tool_call(tool_name, tool_args)
                            tool_exec_time = time.time() - tool_exec_start
                            
                            # Track action execution time with detailed breakdown
                            if tool_name == "action":
                                # Extract detailed timing from result if available
                                if isinstance(result, dict):
                                    if 'coord_time' in result:
                                        stage_timings['action_coord'] = result['coord_time']
                                    if 'exec_time' in result:
                                        stage_timings['action_exec'] = result['exec_time']
                                # Fallback: use total time
                                if 'action_coord' not in stage_timings and 'action_exec' not in stage_timings:
                                    stage_timings['action_execution'] = tool_exec_time
                            
                            # Track successful action (all tools count, including finish)
                            self.num_total_actions += 1
                            
                            # Track first success (for action/wait tools that succeed)
                            if tool_name in ["action", "wait"] and self.turn_first_success < 0:
                                # Check if result indicates success (no error status)
                                if isinstance(result, dict) and result.get("status") != "error":
                                    self.turn_first_success = turn
                            
                            tool_results.append({
                                "tool": tool_name,
                                "result": result,
                            })
                            
                            # Log using rollout_logger if available
                            if self.rollout_logger:
                                self.rollout_logger.log_tool_execution(
                                    turn_num=turn,
                                    tool_idx=tool_call_idx + 1,
                                    total_tools=len(tool_calls),
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    result=result,
                                    exec_time=tool_exec_time,
                                )
                            else:
                                logger.info(f"[Turn {turn}] ✓ Tool '{tool_name}' executed in {tool_exec_time:.3f}s")
                                logger.info(f"[Turn {turn}] Tool result: {json.dumps(result, indent=2, default=str)}")
                            
                            # If task completed, track completion turn
                            if self.task_completed and self.turn_task_completed < 0:
                                self.turn_task_completed = turn
                            
                            # If task completed, break from tool execution loop
                            # Note: We'll check this again after the loop to break from the while loop
                            if self.task_completed:
                                break
                                
                        except ValueError as e:
                            # ValueError from _execute_tool_call usually means tool_name_error
                            tool_exec_time = time.time() - tool_exec_start
                            error_msg = f"Tool execution error: {str(e)}"
                            
                            # Note: tool_name_errors is already incremented in _execute_tool_call
                            
                            if self.rollout_logger:
                                self.rollout_logger.log_tool_execution(
                                    turn_num=turn,
                                    tool_idx=tool_call_idx + 1,
                                    total_tools=len(tool_calls),
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    exec_time=tool_exec_time,
                                    error=str(e),
                                )
                                self.rollout_logger.log(f"[Turn {turn}] ⚠ Tool execution error (continuing): {str(e)}")
                            else:
                                logger.error(f"[Turn {turn}] ✗ Tool '{tool_name}' failed after {tool_exec_time:.3f}s: {e}", exc_info=True)
                                logger.warning(f"[Turn {turn}] ⚠ Tool execution error (continuing rollout): {str(e)}")
                            tool_results.append({
                                "tool": tool_name,
                                "error": str(e),
                                "status": "error",
                            })
                            # Continue to next tool call instead of breaking
                            
                        except Exception as e:
                            tool_exec_time = time.time() - tool_exec_start
                            # Runtime error during tool execution
                            self.runtime_errors += 1
                            error_msg = f"Tool execution error: {str(e)}"
                            
                            if self.rollout_logger:
                                self.rollout_logger.log_tool_execution(
                                    turn_num=turn,
                                    tool_idx=tool_call_idx + 1,
                                    total_tools=len(tool_calls),
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    exec_time=tool_exec_time,
                                    error=str(e),
                                )
                                self.rollout_logger.log(f"[Turn {turn}] ⚠ Tool execution error (continuing): {str(e)}")
                            else:
                                logger.error(f"[Turn {turn}] ✗ Tool '{tool_name}' failed after {tool_exec_time:.3f}s: {e}", exc_info=True)
                                logger.warning(f"[Turn {turn}] ⚠ Tool execution error (continuing rollout): {str(e)}")
                            tool_results.append({
                                "tool": tool_name,
                                "error": str(e),
                                "status": "error",
                            })
                            # Continue to next tool call instead of breaking
                    
                    # Add assistant response and tool results to history
                    assistant_msg = renderers.Message(
                        role="assistant",
                        content=response_text
                    )
                    self.messages.append(assistant_msg)
                    
                    # Add tool results as user message
                    if tool_results:
                        tool_result_text = f"Tool execution results: {json.dumps(tool_results, indent=2)}"
                        tool_result_msg = renderers.Message(
                            role="user",
                            content=tool_result_text
                        )
                        self.messages.append(tool_result_msg)
                    
                    # Record turn + action for every turn (even if not completed)
                    turn_time = time.time() - turn_start_time
                    if self.rollout_recorder is not None:  # Using new rollout_recorder
                        try:
                            from tinker_cookbook.recipes.cua_rl.database.compat import record_turn, record_action, record_observation
                            from datetime import datetime
                            
                            # CRITICAL: Verify rollout_id before recording
                            if not self.rollout_id:
                                logger.error(f"[Turn {turn}] CRITICAL: rollout_id is None when recording turn! Agent ID: {id(self)}")
                                raise ValueError(f"[Turn {turn}] rollout_id is None - cannot record turn")
                            
                            logger.info(f"[Turn {turn}] Recording turn: rollout_id={self.rollout_id}, agent_id={id(self)}, response_length={len(response_text) if response_text else 0}")
                            
                            # Get task_id for verification
                            task_id_str = None
                            if hasattr(self, 'task') and self.task:
                                task_id_str = getattr(self.task, 'id', None)
                            
                            # Record turn (this will create the turn if it doesn't exist)
                            final_turn_id = record_turn(
                                None,  # session parameter is ignored by compat layer
                                self.rollout_id,
                                turn,
                                reward=0.0,  # Reward will be calculated later
                                episode_done=self.task_completed,
                                end_time=datetime.utcnow(),
                                turn_time=turn_time,
                                model_response=response_text,  # Store full LLM output
                                expected_task_id_str=task_id_str,  # Pass task ID for verification
                                metrics={"stage_timings": stage_timings},  # Store precise timing for each stage
                            )
                            
                            if not final_turn_id:
                                logger.error(f"[Turn {turn}] CRITICAL: Failed to record turn. rollout_id={self.rollout_id}, agent_id={id(self)}")
                                raise ValueError(f"[Turn {turn}] Failed to record turn in database")
                            
                            if final_turn_id is not None:
                                if tool_calls:
                                    # Record first tool_call as action (each turn should have exactly one action)
                                    tool_call = tool_calls[0]
                                    tool_name = tool_call.get("name")
                                    tool_args = tool_call.get("args", {})
                                    record_action(
                                        None,  # session parameter is ignored by compat layer
                                        final_turn_id,
                                        action_type="tool_call",
                                        tool_name=tool_name,
                                        tool_args=tool_args,
                                        tokens=action_tokens,
                                        num_tokens=len(action_tokens) if action_tokens else 0,  # Add token count
                                        logprobs=action_logprobs,
                                        turn=turn,  # Add turn parameter for compat layer
                                    )
                                    if len(tool_calls) > 1:
                                        logger.warning(f"[Turn {turn}] WARNING: Model returned {len(tool_calls)} tool_calls, but only recording the first one. This should not happen for CUA tasks.")
                                
                                # Take and record "after" screenshot (after action execution)
                                try:
                                    logger.info(f"[Turn {turn}] Taking 'after' screenshot...")
                                    after_screenshot_start = time.time()
                                    await asyncio.sleep(0.3)  # Small delay for UI to stabilize
                                    after_screenshot_bytes, after_screenshot_uri = await self.gbox_client.take_screenshot()
                                    after_screenshot_time = time.time() - after_screenshot_start
                                    stage_timings['screenshot_after'] = after_screenshot_time
                                    
                                    # Record after screenshot
                                    record_observation(
                                        None,  # session parameter is ignored by compat layer
                                        None,  # turn_id ignored by compat
                                        obs_type="screenshot_after",
                                        screenshot_uri=after_screenshot_uri,
                                        turn=turn,
                                    )
                                    logger.info(f"[Turn {turn}] ✓ Recorded 'after' screenshot")
                                except Exception as e:
                                    logger.warning(f"[Turn {turn}] Failed to take/record 'after' screenshot: {e}")
                                
                                # Record model response as observation for real-time UI
                                try:
                                    record_observation(
                                        None,  # session parameter is ignored by compat layer
                                        None,  # turn_id ignored by compat
                                        obs_type="model_response",
                                        text_content=response_text,
                                        turn=turn,
                                    )
                                except Exception as e:
                                    logger.warning(f"[Turn {turn}] Failed to record model_response observation: {e}")
                            
                        except Exception as e:
                            logger.warning(f"[Turn {turn}] Failed to record turn/action in database: {e}")
                    
                    if self.rollout_logger:
                        self.rollout_logger.end_turn(turn)
                        # Save turn data to database immediately
                        self._save_turn_data_to_db(turn)
                    if not self.rollout_logger:
                        logger.info(f"[Turn {turn}] ══ Turn {turn} completed in {turn_time:.3f}s ══")
                    
                    # If task completed, break from the while loop immediately
                    if self.task_completed:
                        break
                else:
                    # No tool calls found: return error message and continue rollout
                    error_msg = "No tool calls found in model response. Please use the available tools (action, wait, finish) to complete the task."
                    
                    assistant_msg = renderers.Message(
                        role="assistant",
                        content=response_text
                    )
                    self.messages.append(assistant_msg)
                    
                    # Add error message as user message so model can retry
                    error_result_msg = renderers.Message(
                        role="user",
                        content=error_msg
                    )
                    self.messages.append(error_result_msg)
                    
                    if self.rollout_logger:
                        self.rollout_logger.log(f"[Turn {turn}] ⚠ No tool calls found, continuing rollout with error message")
                    else:
                        logger.warning(f"[Turn {turn}] ⚠ No tool calls found, continuing rollout with error message")
                    
                    # Continue to next iteration instead of breaking
                    turn_time = time.time() - turn_start_time
                    
                    # Record turn, actions, and observations in database
                    if self.rollout_recorder is not None:  # Using new rollout_recorder
                        try:
                            from tinker_cookbook.recipes.cua_rl.database.compat import record_turn, record_observation
                            from datetime import datetime
                            
                            # Get task_id for verification
                            task_id_str = None
                            if hasattr(self, 'task') and self.task:
                                task_id_str = getattr(self.task, 'id', None)
                            
                            # Record turn end (this will create the turn if it doesn't exist)
                            final_turn_id = record_turn(
                                None,  # session parameter is ignored by compat layer
                                self.rollout_id,
                                turn,
                                reward=0.0,  # Reward will be calculated later
                                episode_done=False,
                                end_time=datetime.utcnow(),
                                turn_time=turn_time,
                                model_response=response_text,  # Store full LLM output
                                expected_task_id_str=task_id_str,  # Pass task ID for verification
                                metrics={"stage_timings": stage_timings},  # Store precise timing for each stage
                            )
                            
                            if final_turn_id is not None:
                                # Note: Screenshot is already recorded as "screenshot_before" at turn start
                                pass
                            
                            pass  # Commit handled by rollout_recorder
                        except Exception as e:
                            logger.warning(f"[Turn {turn}] Failed to record turn/observation in database: {e}")
                    
                    if self.rollout_logger:
                        self.rollout_logger.end_turn(turn)
                        # Save turn data to database immediately
                        self._save_turn_data_to_db(turn)
                    if not self.rollout_logger:
                        logger.info(f"[Turn {turn}] ══ Turn {turn} completed in {turn_time:.3f}s ══")
                    
                    # Small delay before next turn
                    await asyncio.sleep(0.5)
                    # Debug: Log before continue
                    logger.info(f"[Turn {turn}] Completed turn {turn}/{self.max_turns}, task_completed={self.task_completed}, continuing to next turn...")
                    continue
                
                turn_time = time.time() - turn_start_time
                
                # Record turn, actions, and observations in database
                if self.rollout_id is not None:
                    try:
                        from tinker_cookbook.recipes.cua_rl.database.compat import record_turn, record_action, record_observation
                        from datetime import datetime
                        
                        # Get task_id for verification
                        task_id_str = None
                        if hasattr(self, 'task') and self.task:
                            task_id_str = getattr(self.task, 'id', None)
                        
                        # CRITICAL: Log rollout_id and task info before recording turn
                        task_id = getattr(self.task, 'id', 'no_task') if hasattr(self, 'task') and self.task else 'no_task'
                        task_desc = getattr(self.task, 'description', 'no_description')[:100] if hasattr(self, 'task') and self.task else 'no_description'
                        logger.debug(
                            f"[Turn {turn}] Recording turn end (error case): rollout_id={self.rollout_id}, "
                            f"agent_id={id(self)}, task_id={task_id}, response_preview={response_text[:100] if response_text else 'None'}"
                        )
                        
                        # Record turn end (this will create the turn if it doesn't exist)
                        final_turn_id = record_turn(
                            None,  # session parameter is ignored by compat layer
                            self.rollout_id,
                            turn,
                            reward=0.0,  # Reward will be calculated later
                            episode_done=False,
                            end_time=datetime.utcnow(),
                            turn_time=turn_time,
                            model_response=response_text,  # Store full LLM output
                            expected_task_id_str=task_id_str,  # Pass task ID for verification
                            metrics={"stage_timings": stage_timings} if 'stage_timings' in locals() else None,  # Store precise timing if available
                        )
                        
                        if final_turn_id is not None:
                            # Note: Screenshot is already recorded as "screenshot_before" at turn start
                            
                            # Record action (only the first tool_call - each turn should have exactly one action)
                            if tool_calls:
                                tool_call = tool_calls[0]  # Only record the first tool_call
                                tool_name = tool_call.get("name")
                                tool_args = tool_call.get("args", {})
                                record_action(
                                    None,  # session parameter is ignored by compat layer
                                    final_turn_id,
                                    action_type="tool_call",
                                    tool_name=tool_name,
                                    tool_args=tool_args,
                                    tokens=action_tokens,
                                    num_tokens=len(action_tokens) if action_tokens else 0,  # Add token count
                                    logprobs=action_logprobs,
                                    turn=turn,  # Add turn parameter for compat layer
                                )
                                # Log warning if multiple tool_calls are found (should not happen normally)
                                if len(tool_calls) > 1:
                                    logger.warning(f"[Turn {turn}] WARNING: Model returned {len(tool_calls)} tool_calls, but only recording the first one. This should not happen for CUA tasks.")
                        
                        pass  # Commit handled by rollout_recorder
                    except Exception as e:
                        logger.warning(f"[Turn {turn}] Failed to record turn/action/observation in database: {e}")
                
                if self.rollout_logger:
                    self.rollout_logger.end_turn(turn)
                if not self.rollout_logger:
                    logger.info(f"[Turn {turn}] ══ Turn {turn} completed in {turn_time:.3f}s ══")
                
                # Small delay between turns
                await asyncio.sleep(0.5)
                # Debug: Log after turn completion, before next iteration
                logger.info(f"[Turn {turn}] Turn {turn}/{self.max_turns} finished, checking loop condition: turn < max_turns = {turn} < {self.max_turns} = {turn < self.max_turns}, task_completed={self.task_completed}")
            
            # Debug: Log loop exit
            logger.info(f"[Task End] Exited while loop: turn={turn}, max_turns={self.max_turns}, task_completed={self.task_completed}")
            if not self.task_completed:
                # Set termination reason if not already set
                if self.termination_reason is None:
                    self.termination_reason = f"max_turns_reached_{turn}"
                self.result_message = f"Task not completed within {self.max_turns} turns"
                logger.warning(f"[Task End] Task did not complete within {self.max_turns} turns")
            
            # Perform validation BEFORE terminating the box (if task object is available)
            # This must happen in the try block, before the finally block, so gbox_client is still available
            # All tasks must have validator - check for either validation_query or _original_task with validator
            has_validation = False
            if hasattr(self, 'task') and self.task and self.rollout_logger:
                # Check for _original_task with validator
                if hasattr(self.task, '_original_task') and self.task._original_task:
                    if hasattr(self.task._original_task, 'get_validator'):
                        validator = self.task._original_task.get_validator()
                        if validator and hasattr(validator, 'validate'):
                            has_validation = True
                # Check for validation_query
                if not has_validation and self.task.validation_query:
                    has_validation = True
            
            # Take validation screenshot before performing validation
            validation_screenshot_uri = None
            if has_validation:
                try:
                    # Take screenshot for validation
                    logger.info(f"[Task Validation] Taking screenshot for validation...")
                    await asyncio.sleep(0.3)  # Small delay for stability
                    validation_screenshot_bytes, validation_screenshot_uri = await self.gbox_client.take_screenshot()
                    self.validation_screenshot_uri = validation_screenshot_uri
                    logger.info(f"[Task Validation] ✓ Validation screenshot taken")
                except Exception as e:
                    logger.warning(f"[Task Validation] Failed to take validation screenshot: {e}")
            
            if has_validation:
                try:
                    from tinker_cookbook.recipes.cua_rl.core.reward import validate_task_completion_with_details
                    
                    result_message = getattr(self, 'result_message', '')
                    validation_result = await validate_task_completion_with_details(
                        task=self.task,
                        gbox_client=self.gbox_client,
                        result_message=result_message,
                    )
                    
                    if validation_result:
                        self.rollout_logger.log_adb_validation(
                            command=validation_result.command,
                            expected_result=validation_result.expected_result,
                            actual_result=validation_result.actual_result,
                            success=validation_result.success,
                            execution_time=validation_result.execution_time,
                            validation_query=validation_result.validation_query,
                            screenshot_uri=self.validation_screenshot_uri,
                            termination_reason=self.termination_reason,  # Add termination reason
                        )
                    else:
                        validation_query_str = self.task.validation_query or "task_validator"
                        logger.warning(f"[Task Validation] Validation returned None for task {self.task.id} (query: {validation_query_str})")
                        self.rollout_logger.log_adb_validation_error(
                            error="Validation returned None (unsupported query or validation failed)",
                            validation_query=validation_query_str,
                            termination_reason=self.termination_reason,  # Add termination reason
                        )
                except Exception as e:
                    logger.warning(f"[Task Validation] Failed to perform ADB validation: {e}", exc_info=True)
                    if hasattr(self, 'task') and self.task:
                        self.rollout_logger.log_adb_validation_error(
                            error=str(e),
                            validation_query=self.task.validation_query or "task_validator",
                            termination_reason=self.termination_reason,  # Add termination reason
                        )
            else:
                # Task has no validator - this is an error
                task_id = getattr(self.task, 'id', 'unknown') if hasattr(self, 'task') and self.task else 'unknown'
                task_name = getattr(self.task, 'name', 'unknown') if hasattr(self, 'task') and self.task else 'unknown'
                logger.warning(
                    f"[Task Validation] Task {task_id} ({task_name}) does not have a validator! "
                    f"All tasks must have a validator. Setting task_success=False."
                )
                # Set task_success to False
                self.task_success = False
                # Log validation error
                if self.rollout_logger:
                    self.rollout_logger.log_adb_validation_error(
                        error="Task has no validator (all tasks must have a validator)",
                        validation_query=None,
                        termination_reason=self.termination_reason,
                    )
            
        except Exception as e:
            # Set termination reason for exception
            if self.termination_reason is None:
                self.termination_reason = f"exception_{type(e).__name__}"
            # Log to both logger and rollout_logger to ensure visibility
            error_msg = f"[Task End] ✗ Task failed with exception: {e}"
            logger.error(error_msg, exc_info=True)
            if self.rollout_logger:
                self.rollout_logger.log(error_msg)
                self.rollout_logger.log(f"[Task End] Exception type: {type(e).__name__}")
                self.rollout_logger.log(f"[Task End] Exception details: {str(e)}")
                # Log traceback
                import traceback
                tb_str = traceback.format_exc()
                for line in tb_str.split('\n'):
                    if line.strip():
                        self.rollout_logger.log(f"[Task End] {line}")
            self.result_message = f"Task failed with error: {str(e)}"
            self.task_success = False
            
            # Try to perform validation even if task failed (if task object is available)
            has_validation = False
            if hasattr(self, 'task') and self.task and self.rollout_logger:
                # Check for _original_task with validator
                if hasattr(self.task, '_original_task') and self.task._original_task:
                    if hasattr(self.task._original_task, 'get_validator'):
                        validator = self.task._original_task.get_validator()
                        if validator and hasattr(validator, 'validate'):
                            has_validation = True
                # Check for validation_query
                if not has_validation and self.task.validation_query:
                    has_validation = True
            
            if has_validation:
                try:
                    from tinker_cookbook.recipes.cua_rl.core.reward import validate_task_completion_with_details
                    
                    result_message = getattr(self, 'result_message', '')
                    validation_result = await validate_task_completion_with_details(
                        task=self.task,
                        gbox_client=self.gbox_client,
                        result_message=result_message,
                    )
                    
                    if validation_result:
                        self.rollout_logger.log_adb_validation(
                            command=validation_result.command,
                            expected_result=validation_result.expected_result,
                            actual_result=validation_result.actual_result,
                            success=validation_result.success,
                            execution_time=validation_result.execution_time,
                            validation_query=validation_result.validation_query,
                            screenshot_uri=getattr(self, 'validation_screenshot_uri', None),
                            termination_reason=self.termination_reason,
                        )
                    else:
                        self.rollout_logger.log_adb_validation_error(
                            error="Validation returned None (unsupported query or validation failed)",
                            validation_query=self.task.validation_query,
                            termination_reason=self.termination_reason,
                        )
                except Exception as validation_error:
                    logger.warning(f"[Task Validation] Failed to perform ADB validation after task error: {validation_error}", exc_info=True)
                    if hasattr(self, 'task') and self.task:
                        self.rollout_logger.log_adb_validation_error(
                            error=str(validation_error),
                            validation_query=self.task.validation_query or "task_validator",
                            termination_reason=self.termination_reason,
                        )
            else:
                # Task has no validator - this is an error
                task_id = getattr(self.task, 'id', 'unknown') if hasattr(self, 'task') and self.task else 'unknown'
                task_name = getattr(self.task, 'name', 'unknown') if hasattr(self, 'task') and self.task else 'unknown'
                logger.warning(
                    f"[Task Validation] Task {task_id} ({task_name}) does not have a validator! "
                    f"All tasks must have a validator. Setting task_success=False."
                )
                # Set task_success to False
                self.task_success = False
                # Log validation error
                if self.rollout_logger:
                    self.rollout_logger.log_adb_validation_error(
                        error="Task has no validator (all tasks must have a validator)",
                        validation_query=None,
                        termination_reason=self.termination_reason,
                    )
        
        finally:
            # Stop screen recording and save recording URL (only if recording was started)
            recording_url = None
            recording_storage_key = None
            if recording_started:
                try:
                    box = self.gbox_client._get_box()
                    recording_result = box.action.recording.stop()
                    if hasattr(recording_result, 'presignedUrl'):
                        recording_url = recording_result.presignedUrl
                    elif isinstance(recording_result, dict):
                        recording_url = recording_result.get('presignedUrl')
                        recording_storage_key = recording_result.get('storageKey')
                    
                    if recording_url:
                        if self.rollout_logger:
                            self.rollout_logger.log(f"[Task Cleanup] Screen recording stopped, URL: {recording_url[:100]}...")
                            # Store recording URL in trajectory data
                            if not hasattr(self.rollout_logger, 'trajectory_data'):
                                self.rollout_logger.trajectory_data = {}
                            self.rollout_logger.trajectory_data["recording"] = {
                                "presigned_url": recording_url,
                                "storage_key": recording_storage_key,
                            }
                        else:
                            logger.info(f"[Task Cleanup] Screen recording stopped, URL: {recording_url[:100]}...")
                    else:
                        if self.rollout_logger:
                            self.rollout_logger.log(f"[Task Cleanup] Screen recording stopped, but no URL returned")
                        else:
                            logger.warning(f"[Task Cleanup] Screen recording stopped, but no URL returned")
                except Exception as recording_error:
                    if self.rollout_logger:
                        self.rollout_logger.log(f"[Task Cleanup] Failed to stop recording: {recording_error}")
                    else:
                        logger.warning(f"[Task Cleanup] Failed to stop recording: {recording_error}")
            
            # Terminate box
            cleanup_start = time.time()
            try:
                # DO NOT TERMINATE THE BOX HERE, FOR DEBUGGING PURPOSES
                await self.gbox_client.terminate_box()
                cleanup_time = time.time() - cleanup_start
                logger.info(f"[Task Cleanup] ✓ GBox environment terminated in {cleanup_time:.3f}s")
            except Exception as cleanup_error:
                cleanup_time = time.time() - cleanup_start
                logger.warning(f"[Task Cleanup] Cleanup error after {cleanup_time:.3f}s: {cleanup_error}")
        
        total_task_time = time.time() - task_start_time
        
        # Set summary in rollout logger
        summary = {
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "result_message": self.result_message,
            "num_turns": turn,
            "max_turns": self.max_turns,
            "total_time": total_task_time,
            "avg_time_per_turn": total_task_time / max(turn, 1),
        }
        if self.rollout_logger:
            self.rollout_logger.set_summary(summary)
        
        if not self.rollout_logger:
            logger.info("")
            logger.info("╔" + "=" * 118 + "╗")
            logger.info("║" + " " * 47 + "TASK EXECUTION SUMMARY" + " " * 48 + "║")
            logger.info("╠" + "=" * 118 + "╣")
            logger.info(f"║ Task Completed: {str(self.task_completed):<100} ║")
            logger.info(f"║ Task Success: {str(self.task_success):<102} ║")
            logger.info(f"║ Result Message: {self.result_message[:100]:<100} ║")
            total_turns_str = f"{turn}/{self.max_turns}"
            logger.info(f"║ Total Turns: {total_turns_str:<103} ║")
            total_time_str = f"{total_task_time:.2f}s"
            logger.info(f"║ Total Time: {total_time_str:<104} ║")
            avg_time_str = f"{total_task_time / max(turn, 1):.2f}s"
            logger.info(f"║ Average Time per Turn: {avg_time_str:<93} ║")
            logger.info("╚" + "=" * 118 + "╝")
        
        return {
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "result_message": self.result_message,
            "num_turns": turn,
            "max_turns": self.max_turns,
            # Metrics for reward calculation
            "num_total_actions": self.num_total_actions,
            "consecutive_repeated_actions": self.consecutive_repeated_actions,
            "parse_errors": self.parse_errors,
            "tool_name_errors": self.tool_name_errors,
            "tool_arg_errors": self.tool_arg_errors,
            "runtime_errors": self.runtime_errors,
            "turn_first_success": self.turn_first_success,
            "turn_task_completed": self.turn_task_completed,
            "attempted_completion": self.attempted_completion,
        }
    
    async def close(self):
        """Close the agent and cleanup resources."""
        await self.gbox_client.close()

