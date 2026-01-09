"""Tinker native API inference client.

This client uses Tinker's native SamplingClient for inference,
which is needed for training (provides logprobs) but also works for benchmarking.
"""

import os
import logging
from typing import List, Dict, Any

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer
from .base_inference_client import BaseInferenceClient

logger = logging.getLogger(__name__)


class TinkerInferenceClient(BaseInferenceClient):
    """Tinker native API inference client.
    
    Uses Tinker's SamplingClient to generate tokens, then decodes to text.
    This preserves all Tinker functionality including logprobs for training.
    """
    
    def __init__(
        self,
        api_key: str,
        model_path: str,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
    ):
        """Initialize Tinker inference client.
        
        Args:
            api_key: Tinker API key
            model_path: Tinker checkpoint path (e.g., tinker://...sampler_weights/000080)
            renderer: Tinker renderer for building prompts
            tokenizer: Tokenizer for decoding tokens to text
        """
        self.api_key = api_key
        self.model_path = model_path
        self.renderer = renderer
        self.tokenizer = tokenizer
        
        # Initialize Tinker clients
        os.environ["TINKER_API_KEY"] = api_key
        self.service_client = tinker.ServiceClient()
        self.sampling_client = None
        self._checkpoint_loaded = False
        
        logger.info(f"TinkerInferenceClient initialized: {model_path}")
    
    def _ensure_checkpoint_loaded(self):
        """Ensure checkpoint is loaded (lazy loading)."""
        if self._checkpoint_loaded:
            return
        
        logger.info(f"Loading Tinker checkpoint: {self.model_path}")
        self.sampling_client = self.service_client.create_sampling_client(
            model_path=self.model_path
        )
        self._checkpoint_loaded = True
        logger.info("Tinker checkpoint loaded successfully")
    
    async def generate_text(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> str:
        """Generate text using Tinker native API.
        
        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (ignored for Tinker)
            
        Returns:
            Generated text string
        """
        self._ensure_checkpoint_loaded()
        
        # Build ModelInput using renderer
        model_input = self.renderer.build_generation_prompt(messages)
        
        # Sampling parameters
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )
        
        # Sample
        logger.debug(f"Calling Tinker sample_async with max_tokens={max_tokens}, temp={temperature}")
        sample_resp = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        
        # Decode tokens to text
        tokens = sample_resp.sequences[0].tokens
        text = self.tokenizer.decode(tokens)
        
        logger.debug(f"Generated text length: {len(text)} chars")
        
        return text
    
    def get_provider_name(self) -> str:
        return "tinker"
    
    def supports_logprobs(self) -> bool:
        return True
    
    async def get_tokens_and_logprobs(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> tuple[List[int], List[float]]:
        """Get raw tokens and logprobs (Tinker-specific method).
        
        This method is used by training code that needs logprobs.
        For benchmarking, use generate_text() instead.
        
        Returns:
            Tuple of (tokens, logprobs)
        """
        self._ensure_checkpoint_loaded()
        
        # Build ModelInput
        model_input = self.renderer.build_generation_prompt(messages)
        
        # Sampling parameters
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )
        
        # Sample
        logger.debug(f"Calling Tinker sample_async with params: max_tokens={max_tokens}, temp={temperature}")
        sample_resp = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        
        # Return raw tokens and logprobs
        seq = sample_resp.sequences[0]
        tokens = seq.tokens
        
        # Check if logprobs are available
        if seq.logprobs is None:
            logger.error(
                f"Tinker sampling did not return logprobs! "
                f"This is required for RL training. "
                f"Tokens length: {len(tokens)}, "
                f"Response: {sample_resp}"
            )
            raise RuntimeError(
                "Tinker sampling response did not include logprobs. "
                "This is required for token-level RL training."
            )
        
        logprobs = seq.logprobs
        
        return tokens, logprobs

