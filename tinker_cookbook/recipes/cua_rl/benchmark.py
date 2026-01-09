"""
Benchmark evaluation script for CUA agents.

This script is a simplified wrapper around the training script that:
1. Creates a minimal training session (for database tracking)
2. Runs only the baseline evaluation part
3. Saves results to database

It reuses all the existing training infrastructure.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.cua_rl.train import CLIConfig, cli_main

logger = logging.getLogger(__name__)


@chz.chz
class BenchmarkConfig:
    """Configuration for benchmark evaluation - supports multiple providers."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_path: str | None = None  # For Tinker: checkpoint path
    
    # Provider configuration (NEW)
    provider: str = "tinker"  # "tinker", "vllm", "openrouter", "openai"
    provider_base_url: str | None = None  # Optional: API base URL (for vLLM, OpenRouter, etc.)
    provider_api_key: str | None = None  # Optional: API key (for OpenRouter, OpenAI, etc.)
    
    # Evaluation dataset
    eval_tasks: dict | list[dict] | None = None
    
    # Benchmark settings
    benchmark_name: str | None = None
    seed: int = 42
    max_turns: int = 20
    temperature: float = 1.0
    max_tokens: int = 2048
    
    # Box configuration
    box_type: str = "android"
    gbox_api_key: str = ""
    tinker_api_key: str = ""  # For Tinker provider only
    
    # Concurrency and timeout
    max_concurrent_rollouts: int = 8
    max_task_time_seconds: int = 30 * 60
    max_turn_time_seconds: int = 5 * 60
    
    # Coordinate generation mode
    coordinate_mode: str = "gbox"  # "gbox" or "direct"
    coordinate_scale: bool | None = None  # Auto-detect based on mode if None
    
    # Logging
    log_path: str = "./benchmark_logs"
    
    # Database
    database_url: str | None = None


async def run_benchmark(config: BenchmarkConfig) -> dict:
    """
    Run benchmark evaluation by wrapping the training script.
    
    This creates a training config that:
    - Skips actual training (groups_per_batch=0)
    - Runs only baseline evaluation
    - Uses the specified provider and model
    """
    # Build benchmark name
    if config.benchmark_name:
        benchmark_name = config.benchmark_name
    else:
        model_tag = config.model_name.replace("/", "-")
        date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        benchmark_name = f"benchmark-{config.provider}-{model_tag}-{date_and_time}"
    
    logger.info(f"Running benchmark: {benchmark_name}")
    logger.info(f"Provider: {config.provider}")
    logger.info(f"Model: {config.model_name}")
    if config.model_path:
        logger.info(f"Checkpoint: {config.model_path}")
    
    # Convert benchmark config to training config
    # The trick: set groups_per_batch=0 to skip training, only run baseline
    cli_config = CLIConfig(
        # Model settings
        model_name=config.model_name,
        load_checkpoint_path=config.model_path,  # For Tinker: checkpoint path
        lora_rank=32,  # Need valid rank even though we won't train
        
        # Provider settings (NEW)
        provider=config.provider,
        provider_base_url=config.provider_base_url,
        provider_api_key=config.provider_api_key,
        
        # Box settings
        gbox_api_key=config.gbox_api_key,
        tinker_api_key=config.tinker_api_key,  # Only used for Tinker provider
        box_type=config.box_type,
        
        # Data settings  
        # Provide minimal training task config (required by dataset builder, but won't be used)
        tasks={"source_type": "demo_training", "limit": 1},  # Minimal config
        eval_tasks=config.eval_tasks,
        use_default_eval_tasks=(config.eval_tasks is None),
        seed=config.seed,
        max_turns=config.max_turns,
        
        # Training settings (not used since baseline_only=True)
        group_size=1,
        groups_per_batch=1,
        learning_rate=1e-5,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        
        # Logging settings
        log_path=config.log_path,
        wandb_project=None,
        wandb_name=benchmark_name,
        eval_every=1,
        save_every=0,
        num_groups_to_log=0,
        
        # Execution settings
        behavior_if_log_dir_exists="delete",  # Each benchmark is independent
        max_concurrent_rollouts=config.max_concurrent_rollouts,
        max_task_time_seconds=config.max_task_time_seconds,
        max_turn_time_seconds=config.max_turn_time_seconds,
        coordinate_mode=config.coordinate_mode,  # Pass coordinate mode
        coordinate_scale=config.coordinate_scale,  # Pass coordinate scale
        
        # CRITICAL: Only run baseline, no training!
        skip_baseline=False,  # Run baseline evaluation
        baseline_only=True,   # Exit after baseline (don't train)
    )
    
    # Check log directory
    cli_utils.check_log_dir(
        config.log_path,
        behavior_if_exists="delete"  # Each benchmark run is independent
    )
    
    # Run the training script (which will only do baseline evaluation)
    logger.info("Starting baseline evaluation via training script...")
    logger.info("=" * 80)
    
    try:
        await cli_main(cli_config)
        logger.info("=" * 80)
        logger.info("Benchmark completed successfully!")
        return {"status": "completed"}
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise


def main_wrapper(config: BenchmarkConfig) -> None:
    """Wrapper function for CLI entry point."""
    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main_wrapper, allow_hyphens=True)
