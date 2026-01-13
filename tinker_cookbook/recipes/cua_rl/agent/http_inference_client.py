"""HTTP-based inference client for OpenAI-compatible APIs.

This client works with any OpenAI-compatible API including:
- vLLM (local deployment)
- OpenRouter (API service)
- OpenAI (GPT models)
- Any other compatible service
"""

import logging
from typing import List, Dict, Any, Optional

import httpx

from .base_inference_client import BaseInferenceClient

logger = logging.getLogger(__name__)


class HTTPInferenceClient(BaseInferenceClient):
    """Generic HTTP client for OpenAI-compatible APIs.
    
    Key design decisions:
    1. Does NOT pass tools parameter to API
       - Model follows tool examples in system prompt instead
       - This preserves tinker-cookbook's prompt-based approach
    
    2. Returns raw text
       - Caller tokenizes and uses renderer.parse_response()
       - This preserves all tool call parsing logic
    
    3. Handles multimodal messages (text + images)
       - Converts to OpenAI image_url format
    """
    
    def __init__(
        self,
        provider: str,
        model_name: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """Initialize HTTP inference client.
        
        Args:
            provider: Provider name (for logging and headers)
            model_name: Model name/identifier for the API
            base_url: API base URL (e.g., http://localhost:8000/v1)
            api_key: API key (optional for vLLM, required for most others)
            timeout: Request timeout in seconds
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.base_url = base_url
        
        # Build headers
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Provider-specific headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/ThinkingMachinesLab/tinker-cookbook"
            headers["X-Title"] = "Tinker Cookbook CUA Benchmark"
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
        
        logger.info(f"HTTPInferenceClient initialized: {provider} (model={model_name}, base_url={base_url})")
    
    async def generate_text(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate text using OpenAI-compatible API.
        
        CRITICAL: Does NOT pass tools parameter.
        The model follows tool examples in the system prompt.
        
        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (e.g., top_p)
            
        Returns:
            Generated text string
        """
        # Prepare messages for API (handle images)
        openai_messages = self._prepare_messages(messages)
        
        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 1.0),
            # NO tools parameter! Model follows prompt examples.
        }
        
        # Call API
        try:
            logger.debug(f"Calling {self.provider} API: model={self.model_name}, max_tokens={max_tokens}")
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract text from response
            text = result["choices"][0]["message"]["content"] or ""
            
            logger.debug(f"Generated text length: {len(text)} chars")
            
            return text
            
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_body = e.response.json()
                error_detail = f": {error_body}"
                
                # Helpful error for OpenRouter model not found
                if self.provider == "openrouter" and e.response.status_code == 404:
                    error_msg = error_body.get("error", {}).get("message", "")
                    if "No endpoints found" in error_msg:
                        logger.error(
                            f"Model '{self.model_name}' is not available on OpenRouter. "
                            f"Check available models at https://openrouter.ai/models"
                        )
            except:
                error_detail = f": {e.response.text}"
            
            logger.error(f"{self.provider} API error: {e.response.status_code}{error_detail}")
            raise RuntimeError(
                f"{self.provider} API error: {e.response.status_code}{error_detail}"
            ) from e
        
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to {self.provider} at {self.base_url}: {e}")
            raise RuntimeError(
                f"Cannot connect to {self.provider} at {self.base_url}. "
                f"Please ensure the server is running and accessible."
            ) from e
        
        except Exception as e:
            logger.error(f"Unexpected error calling {self.provider}: {e}")
            raise
    
    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI format with proper image handling.
        
        Handles:
        - Text messages: pass through
        - Multimodal messages: convert images to image_url format
        - Tool messages: pass through
        
        Args:
            messages: Messages in Tinker format
            
        Returns:
            Messages in OpenAI format
        """
        openai_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if isinstance(content, str):
                # Simple text message
                openai_messages.append({
                    "role": role,
                    "content": content,
                })
            
            elif isinstance(content, list):
                # Multimodal message (text + images)
                openai_content = []
                
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        
                        if part_type == "text":
                            openai_content.append({
                                "type": "text",
                                "text": part.get("text", "")
                            })
                        
                        elif part_type == "image":
                            # Convert image to OpenAI format
                            image_data = part.get("image", "")
                            
                            # Ensure data URI format (support base64 strings, URLs, bytes, or PIL images)
                            if not isinstance(image_data, str):
                                import base64
                                import io
                                from PIL import Image
                                
                                if isinstance(image_data, Image.Image):
                                    buf = io.BytesIO()
                                    image_data.save(buf, format="PNG")
                                    image_data = base64.b64encode(buf.getvalue()).decode("ascii")
                                elif isinstance(image_data, (bytes, bytearray)):
                                    image_data = base64.b64encode(bytes(image_data)).decode("ascii")
                                else:
                                    raise TypeError(
                                        f"Unsupported image type in message content: {type(image_data)}"
                                    )
                            
                            if not image_data.startswith("data:"):
                                if image_data.startswith("http://") or image_data.startswith("https://"):
                                    image_data = image_data
                                else:
                                    image_data = f"data:image/png;base64,{image_data}"
                            
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data
                                }
                            })
                
                openai_messages.append({
                    "role": role,
                    "content": openai_content,
                })
            
            # Handle tool messages (for multi-turn conversations)
            elif role == "tool":
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": str(content),
                })
            
            # Handle assistant messages with tool_calls
            if "tool_calls" in msg:
                # Ensure last message is from assistant
                if openai_messages and openai_messages[-1]["role"] == "assistant":
                    openai_messages[-1]["tool_calls"] = msg["tool_calls"]
                else:
                    openai_messages.append({
                        "role": "assistant",
                        "content": msg.get("content", ""),
                        "tool_calls": msg["tool_calls"],
                    })
        
        return openai_messages
    
    def get_provider_name(self) -> str:
        return self.provider
    
    def supports_logprobs(self) -> bool:
        # Most OpenAI-compatible APIs don't provide logprobs
        # Only OpenAI itself and some vLLM setups do
        return self.provider in ["openai"]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

