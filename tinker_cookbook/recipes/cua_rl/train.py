"""
Training script for CUA (Computer Use Agent) RL.

This script trains a CUA model using Tinker's RL framework with GBoxAgent for rollouts.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.cua_rl.cua_env import CUADatasetBuilder
from tinker_cookbook.recipes.cua_rl.rollout import (
    do_cua_group_rollout,
    set_rollout_output_dir,
    set_rollout_context,
)
# IMPORTANT: Override rollouts.do_group_rollout BEFORE importing train module
# because train imports metric_util which binds do_group_rollout at import time
from tinker_cookbook.rl import rollouts
from tinker_cookbook.rl.types import EnvGroupBuilder
from tinker_cookbook.completers import TokenCompleter

logger = logging.getLogger(__name__)

# Global variable to store current model_path for evaluation
# This is set during baseline evaluation and regular evaluation
_current_eval_model_path: str | None = None


def set_eval_model_path(model_path: str | None):
    """Set the current model_path for evaluation rollouts."""
    global _current_eval_model_path
    _current_eval_model_path = model_path


async def _cua_group_rollout_for_eval(
    env_group_builder: EnvGroupBuilder, 
    policy: TokenCompleter
):
    """
    Wrapper for do_cua_group_rollout that sets is_eval=True for evaluation.
    This is used when RLTestSetEvaluator calls do_group_rollout.
    """
    global _current_eval_model_path
    
    # Use the global model_path if available, otherwise try to extract from policy
    model_path = _current_eval_model_path
    
    return await do_cua_group_rollout(
        env_group_builder=env_group_builder,
        policy=policy,
        model_path=model_path,  # Use global model_path if set, otherwise will try to extract from policy
        step=None,
        batch=None,
        group=None,
        output_dir=None,
        is_eval=True,  # Always True for evaluation
    )

# Override rollouts.do_group_rollout BEFORE importing train module
# This ensures that when train imports metric_util, it will use our custom function
rollouts.do_group_rollout = _cua_group_rollout_for_eval

# Now import train module (which will import metric_util)
from tinker_cookbook.rl import train

# Also update metric_util's reference to do_group_rollout
# because it was bound at import time, we need to update it explicitly
from tinker_cookbook.rl import metric_util
metric_util.do_group_rollout = _cua_group_rollout_for_eval


@chz.chz
class CLIConfig:
    """Command-line configuration for CUA RL training."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # Model for training (also used for rollout in on-policy RL)
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None  # Optional: path to a specific checkpoint to load. If None, will auto-resume from log_path if checkpoint exists. If set, will load weights from this checkpoint (fresh optimizer, starts from batch 0).
    
    # GBox configuration
    gbox_api_key: str = ""  # Will use GBOX_API_KEY env var if empty
    tinker_api_key: str = ""  # Will use TINKER_API_KEY env var if empty (same as gbox_api_key for Tinker)
    box_type: str = "android"  # android or linux
    
    # Data / environment configuration
    # Tasks are configured using TaskSourceConfig:
    # - dict: Single TaskSourceConfig (e.g., {"source_type": "demo_training"})
    # - List[dict]: Multiple TaskSourceConfig objects
    #   Example: [{"source_type": "demo_training"}, {"source_type": "demo_eval"}]
    tasks: dict | list[dict] = chz.field(default_factory=lambda: {"source_type": "demo_training"})  # TaskSourceConfig dict(s) for training
    eval_tasks: dict | list[dict] | None = None  # Optional TaskSourceConfig dict(s) for evaluation. If None and use_default_eval_tasks=True, will use default (10 random tasks from demo_eval)
    use_default_eval_tasks: bool = True  # If True and eval_tasks is None, will use default: 10 random tasks from demo_eval
    seed: int = 0
    max_turns: int = 8
    
    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 2
    learning_rate: float = 1e-5
    max_tokens: int = 2048
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    
    # Logging / eval / checkpoints
    log_dir: str | None = None  # Directory to create run in (will create subdirectory with run name)
    log_path: str | None = None  # Full path to log directory (overrides log_dir). If this path contains checkpoints, training will automatically resume from the latest checkpoint.
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 10
    save_every: int = 10  # Save checkpoint every N batches (0 = disabled)
    num_groups_to_log: int = 1
    
    # Service configuration
    base_url: str | None = None
    
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    
    # Async rollout configuration
    max_steps_off_policy: int | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    """Main training function."""
    import os
    
    # Get API keys from environment if not provided
    gbox_api_key = cli_config.gbox_api_key or os.getenv("GBOX_API_KEY")
    if not gbox_api_key:
        raise ValueError("GBOX_API_KEY must be provided via config or environment variable")
    
    # Tinker API key (for OpenAI-compatible API)
    # Usually same as TINKER_API_KEY, but can be different
    # If not provided, default to gbox_api_key (they are often the same for Tinker)
    tinker_api_key = cli_config.tinker_api_key or os.getenv("TINKER_API_KEY") or gbox_api_key
    
    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    
    # Build run name
    model_tag = cli_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"cua_rl-{model_tag}-"
        f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
        f"{cli_config.group_size}group-{cli_config.groups_per_batch}batch-"
        f"seed{cli_config.seed}-{date_and_time}"
    )
    
    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    elif cli_config.log_dir is not None:
        log_path = str(Path(cli_config.log_dir) / run_name)
    else:
        log_path = f"/tmp/tinker-examples/cua_rl/{run_name}"
    
    wandb_name = cli_config.wandb_name or run_name
    
    # Check log directory
    # Note: If log_path contains checkpoints, training will automatically resume from the latest checkpoint
    # To enable resume, either:
    # 1. Set log_path to a fixed path (not using run_name with timestamp), or
    # 2. Use behavior_if_log_dir_exists="resume" to allow using existing log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    
    # Set default eval_tasks if needed
    eval_tasks = cli_config.eval_tasks
    if eval_tasks is None and cli_config.use_default_eval_tasks:
        eval_tasks = {"source_type": "demo_eval", "limit": 10, "seed": cli_config.seed}
        logger.info(f"Using default eval_tasks: randomly sampling 10 tasks from demo_eval (seed={cli_config.seed})")
    elif eval_tasks is None:
        logger.info("No eval_tasks configured, evaluation will be disabled")
    
    # Build dataset builder
    dataset_builder = CUADatasetBuilder(
        tasks=cli_config.tasks,
        eval_tasks=eval_tasks,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        gbox_api_key=gbox_api_key,
        tinker_api_key=tinker_api_key,  # Tinker API key for OpenAI-compatible API
        rollout_model_name=None,  # Not used, rollout uses dynamic checkpoint path from training
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_turns=cli_config.max_turns,
        box_type=cli_config.box_type,
        seed=cli_config.seed,
    )
    
    # Override train.do_group_rollout with our custom function for training rollouts
    # Note: rollouts.do_group_rollout was already overridden at module import time
    # to ensure metric_util uses our custom function
    train.do_group_rollout = do_cua_group_rollout
    
    # Set output directory for trajectory saving
    set_rollout_output_dir(log_path)
    
    # Build training config
    config = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        num_groups_to_log=cli_config.num_groups_to_log,
        async_config=train.AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
    )
    
    # Run training
    await train.main(config)


def main_wrapper(cli_config: CLIConfig) -> None:
    """Wrapper function for nested_entrypoint."""
    asyncio.run(cli_main(cli_config))


if __name__ == "__main__":
    chz.nested_entrypoint(main_wrapper, allow_hyphens=True)

