"""
Enhanced timeout monitoring for Tinker API calls.

This module provides wrappers and utilities to monitor and diagnose
timeout issues in Tinker API calls, especially for sample_async().
"""

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class TimeoutMonitor:
    """Monitor async operations with detailed timeout tracking."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.timeout_occurred = False
        self.exception_occurred = False
        self.exception: Optional[Exception] = None
    
    @asynccontextmanager
    async def monitor(self, timeout_seconds: float):
        """Context manager for monitoring an async operation with timeout."""
        self.start_time = time.time()
        
        logger.info(
            f"[TimeoutMonitor] Starting '{self.operation_name}' "
            f"(timeout: {timeout_seconds}s)"
        )
        
        try:
            yield self
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time
            
            logger.info(
                f"[TimeoutMonitor] ✓ '{self.operation_name}' completed "
                f"in {elapsed:.2f}s"
            )
            
        except asyncio.TimeoutError:
            self.timeout_occurred = True
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time
            
            logger.error(
                f"[TimeoutMonitor] ✗ '{self.operation_name}' TIMED OUT "
                f"after {elapsed:.2f}s (limit: {timeout_seconds}s)"
            )
            raise
            
        except Exception as e:
            self.exception_occurred = True
            self.exception = e
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time
            
            logger.error(
                f"[TimeoutMonitor] ✗ '{self.operation_name}' FAILED "
                f"after {elapsed:.2f}s with exception: {e}",
                exc_info=True
            )
            raise
    
    def get_elapsed_time(self) -> Optional[float]:
        """Get elapsed time if operation has ended."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return None


async def sample_with_monitoring(
    sampling_client,
    prompt,
    num_samples: int,
    sampling_params,
    timeout_seconds: float,
    context_info: Optional[dict] = None,
) -> Any:
    """
    Wrapper for sampling_client.sample_async() with enhanced monitoring.
    
    Args:
        sampling_client: Tinker SamplingClient
        prompt: ModelInput
        num_samples: Number of samples
        sampling_params: SamplingParams
        timeout_seconds: Timeout in seconds
        context_info: Additional context for logging (e.g., turn number, rollout_id)
    
    Returns:
        SampleResponse from Tinker API
        
    Raises:
        asyncio.TimeoutError: If operation times out
        Exception: Any other exception from Tinker API
    """
    context_str = ""
    if context_info:
        parts = [f"{k}={v}" for k, v in context_info.items()]
        context_str = f" [{', '.join(parts)}]"
    
    monitor = TimeoutMonitor(f"sample_async{context_str}")
    
    # Log input characteristics
    try:
        if hasattr(prompt, 'chunks'):
            num_chunks = len(prompt.chunks)
            chunk_types = [type(c).__name__ for c in prompt.chunks]
            logger.info(
                f"[sample_with_monitoring]{context_str} "
                f"Input: {num_chunks} chunks, types={chunk_types}"
            )
    except Exception as e:
        logger.warning(f"[sample_with_monitoring]{context_str} Could not inspect input: {e}")
    
    # Log sampling params
    logger.info(
        f"[sample_with_monitoring]{context_str} "
        f"Params: max_tokens={sampling_params.max_tokens}, "
        f"temperature={sampling_params.temperature}"
    )
    
    # Create a task for the API call so we can check its status
    api_call_task = asyncio.create_task(
        sampling_client.sample_async(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )
    )
    
    # Monitor with periodic status updates
    async def status_reporter():
        """Report status every 30 seconds."""
        interval = 30
        elapsed = 0
        while not api_call_task.done():
            await asyncio.sleep(interval)
            elapsed += interval
            logger.info(
                f"[sample_with_monitoring]{context_str} "
                f"Still waiting... ({elapsed}s elapsed, timeout at {timeout_seconds}s)"
            )
    
    status_task = asyncio.create_task(status_reporter())
    
    try:
        async with monitor.monitor(timeout_seconds):
            # Wait for API call with timeout
            result = await asyncio.wait_for(api_call_task, timeout=timeout_seconds)
            
            # Cancel status reporter
            status_task.cancel()
            
            # Log response characteristics
            try:
                num_sequences = len(result.sequences)
                if num_sequences > 0:
                    seq_len = len(result.sequences[0].tokens)
                    logger.info(
                        f"[sample_with_monitoring]{context_str} "
                        f"Response: {num_sequences} sequences, first has {seq_len} tokens"
                    )
            except Exception as e:
                logger.warning(f"[sample_with_monitoring]{context_str} Could not inspect response: {e}")
            
            return result
            
    except asyncio.TimeoutError:
        # Cancel the API call task
        api_call_task.cancel()
        status_task.cancel()
        
        logger.error(
            f"[sample_with_monitoring]{context_str} "
            f"API call did NOT complete within {timeout_seconds}s timeout"
        )
        
        # Log additional diagnostics
        logger.error(
            f"[sample_with_monitoring]{context_str} "
            f"Task state: cancelled={api_call_task.cancelled()}, "
            f"done={api_call_task.done()}"
        )
        
        raise
        
    except Exception as e:
        # Cancel tasks
        api_call_task.cancel()
        status_task.cancel()
        
        logger.error(
            f"[sample_with_monitoring]{context_str} "
            f"Exception during API call: {e}"
        )
        raise


class TinkerAPIHealthCheck:
    """Periodic health checks for Tinker API responsiveness."""
    
    def __init__(self, sampling_client, check_interval_seconds: float = 300):
        """
        Args:
            sampling_client: Tinker SamplingClient
            check_interval_seconds: Time between health checks (default: 5 minutes)
        """
        self.sampling_client = sampling_client
        self.check_interval = check_interval_seconds
        self.last_check_time: Optional[float] = None
        self.last_check_success: Optional[bool] = None
        self.last_check_latency: Optional[float] = None
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def quick_health_check(self, timeout_seconds: float = 30) -> bool:
        """
        Run a quick health check with a minimal prompt.
        
        Returns:
            True if health check passed, False otherwise
        """
        try:
            import tinker
            from tinker_cookbook.tokenizer_utils import get_tokenizer
            
            # Use a minimal prompt
            tokenizer = get_tokenizer("Qwen/Qwen3-VL-30B-A3B-Instruct")
            tokens = tokenizer.encode("Hi")
            prompt = tinker.ModelInput.from_ints(tokens)
            
            sampling_params = tinker.SamplingParams(
                max_tokens=5,
                temperature=1.0,
            )
            
            logger.info("[HealthCheck] Running quick health check...")
            start = time.time()
            
            await asyncio.wait_for(
                self.sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=sampling_params,
                ),
                timeout=timeout_seconds
            )
            
            latency = time.time() - start
            
            self.last_check_time = time.time()
            self.last_check_success = True
            self.last_check_latency = latency
            
            logger.info(f"[HealthCheck] ✓ Health check passed (latency: {latency:.2f}s)")
            return True
            
        except Exception as e:
            self.last_check_time = time.time()
            self.last_check_success = False
            
            logger.error(f"[HealthCheck] ✗ Health check failed: {e}")
            return False
    
    def start_background_checks(self):
        """Start background health checks."""
        if self._health_check_task is not None:
            logger.warning("[HealthCheck] Background checks already running")
            return
        
        async def _run_checks():
            while True:
                try:
                    await self.quick_health_check()
                    await asyncio.sleep(self.check_interval)
                except asyncio.CancelledError:
                    logger.info("[HealthCheck] Background checks cancelled")
                    break
                except Exception as e:
                    logger.error(f"[HealthCheck] Error in background check: {e}")
                    await asyncio.sleep(self.check_interval)
        
        self._health_check_task = asyncio.create_task(_run_checks())
        logger.info(f"[HealthCheck] Started background checks (interval: {self.check_interval}s)")
    
    def stop_background_checks(self):
        """Stop background health checks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
            logger.info("[HealthCheck] Stopped background checks")

