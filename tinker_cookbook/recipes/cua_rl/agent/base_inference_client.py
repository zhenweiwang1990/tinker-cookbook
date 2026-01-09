"""Base inference client abstraction for CUA agent.

This module provides a simple abstraction for different inference providers.
The key insight is that all we need is: messages in, text out.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseInferenceClient(ABC):
    """Abstract base class for model inference clients.
    
    All inference clients must implement a simple interface:
    - Take messages (conversation history)
    - Return generated text
    
    The caller (TinkerCuaAgent) will:
    1. Call generate_text() to get raw text
    2. Tokenize the text
    3. Use renderer.parse_response() to extract tool calls
    
    This design preserves all tinker-cookbook logic (prompts, tool parsing,
    coordinate handling, database recording) regardless of provider.
    """
    
    @abstractmethod
    async def generate_text(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate text from the model.
        
        Args:
            messages: Conversation history in simple dict format
                      Each message has 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text string
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name (e.g., 'tinker', 'vllm', 'openrouter')."""
        pass
    
    def supports_logprobs(self) -> bool:
        """Whether this provider supports token logprobs.
        
        Only Tinker provides logprobs, which are needed for training.
        For benchmarking/evaluation, logprobs are not required.
        """
        return False

