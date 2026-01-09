"""Factory for creating inference clients.

This module provides a simple factory function to create the appropriate
inference client based on the provider.
"""

import os
import logging
from typing import Optional

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer
from tinker_cookbook.image_processing_utils import get_image_processor

from .base_inference_client import BaseInferenceClient
from .tinker_inference_client import TinkerInferenceClient
from .http_inference_client import HTTPInferenceClient

logger = logging.getLogger(__name__)


# Provider default configurations
PROVIDER_DEFAULTS = {
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "requires_api_key": False,
        "env_key": None,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "requires_api_key": True,
        "env_key": "OPENROUTER_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "requires_api_key": True,
        "env_key": "OPENAI_API_KEY",
    },
}


def create_inference_client(
    provider: str,
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_path: Optional[str] = None,  # For Tinker only
    base_model_name: Optional[str] = None,  # For tokenizer/renderer
    renderer_name: Optional[str] = None,
    renderer: Optional[renderers.Renderer] = None,
    tokenizer: Optional[Tokenizer] = None,
    **kwargs
) -> BaseInferenceClient:
    """Create an inference client based on provider.
    
    Args:
        provider: Provider name ("tinker", "vllm", "openrouter", "openai")
        model_name: Model name or identifier
        base_url: API base URL (optional, uses provider default)
        api_key: API key (optional, auto-detects from environment)
        model_path: Tinker checkpoint path (required for "tinker" provider)
        base_model_name: Base model name for tokenizer/renderer (optional, defaults to model_name)
        renderer_name: Renderer name (optional, auto-detected)
        renderer: Pre-initialized renderer (optional)
        tokenizer: Pre-initialized tokenizer (optional)
        **kwargs: Additional parameters passed to client
        
    Returns:
        BaseInferenceClient instance
        
    Raises:
        ValueError: If required parameters are missing
    
    Examples:
        # Tinker
        client = create_inference_client(
            provider="tinker",
            model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
            api_key="your-tinker-key",
            model_path="tinker://...sampler_weights/000080"
        )
        
        # vLLM
        client = create_inference_client(
            provider="vllm",
            model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
            base_url="http://localhost:8000/v1"
        )
        
        # OpenRouter
        client = create_inference_client(
            provider="openrouter",
            model_name="qwen/qwen3-vl-30b-a3b-instruct",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    """
    provider = provider.lower()
    
    # Validate provider
    supported = ["tinker", "vllm", "openrouter", "openai"]
    if provider not in supported:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(supported)}"
        )
    
    # Auto-detect base_url for non-Tinker providers
    if provider != "tinker" and not base_url:
        if provider in PROVIDER_DEFAULTS:
            base_url = PROVIDER_DEFAULTS[provider]["base_url"]
            logger.info(f"Using default base_url for {provider}: {base_url}")
    
    # Auto-detect API key from environment
    if not api_key:
        if provider == "tinker":
            api_key = os.getenv("TINKER_API_KEY")
        elif provider in PROVIDER_DEFAULTS and PROVIDER_DEFAULTS[provider]["env_key"]:
            env_key = PROVIDER_DEFAULTS[provider]["env_key"]
            api_key = os.getenv(env_key)
            if not api_key and PROVIDER_DEFAULTS[provider]["requires_api_key"]:
                raise ValueError(
                    f"{provider} requires API key. "
                    f"Set it via api_key parameter or {env_key} environment variable"
                )
    
    # Create tokenizer and renderer if not provided
    if not base_model_name:
        base_model_name = model_name
    
    if not tokenizer:
        tokenizer = get_tokenizer(base_model_name)
    
    if not renderer:
        image_processor = get_image_processor(base_model_name)
        
        if not renderer_name:
            # Auto-detect renderer based on model name
            model_lower = base_model_name.lower()
            if "qwen3" in model_lower or "qwen-3" in model_lower:
                renderer_name = "qwen3_vl_instruct"
            elif "qwen2.5" in model_lower or "qwen-2.5" in model_lower:
                renderer_name = "qwen3_vl_instruct"  # Uses same renderer
            elif "qwen" in model_lower:
                renderer_name = "qwen3_vl_instruct"  # Default to qwen3
            else:
                renderer_name = "qwen3_vl_instruct"  # Fallback
            
            logger.info(f"Auto-detected renderer: {renderer_name}")
        
        renderer = renderers.get_renderer(
            renderer_name,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
    
    # Create appropriate client
    if provider == "tinker":
        if not model_path:
            raise ValueError("Tinker provider requires model_path parameter")
        
        if not api_key:
            raise ValueError("Tinker provider requires api_key parameter")
        
        logger.info(f"Creating TinkerInferenceClient: {model_path}")
        
        return TinkerInferenceClient(
            api_key=api_key,
            model_path=model_path,
            renderer=renderer,
            tokenizer=tokenizer,
        )
    
    else:
        # All other providers use HTTP client
        if not base_url:
            raise ValueError(f"{provider} provider requires base_url parameter")
        
        logger.info(f"Creating HTTPInferenceClient: {provider} ({base_url})")
        
        return HTTPInferenceClient(
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout=kwargs.get("timeout", 300.0),
        )

