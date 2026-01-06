"""
Training script for CUA (Computer Use Agent) RL.

This script trains a CUA model using Tinker's RL framework with GBoxAgent for rollouts.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.cua_rl.agent.cua_env import CUADatasetBuilder
from tinker_cookbook.recipes.cua_rl.core.rollout import (
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


# Global variables for evaluation
_current_eval_model_path: str | None = None
_eval_group_counter: dict | None = None

def set_eval_model_path(model_path: str | None):
    """Set the current model_path for evaluation rollouts."""
    global _current_eval_model_path, _eval_group_counter
    _current_eval_model_path = model_path
    # Reset group counter when setting new eval model path
    _eval_group_counter = None

def reset_eval_group_counter():
    """Reset the evaluation group counter. Call this at the start of each evaluation."""
    global _eval_group_counter
    _eval_group_counter = None


async def _cua_group_rollout_for_eval(
    env_group_builder: EnvGroupBuilder, 
    policy: TokenCompleter
):
    """
    Wrapper for do_cua_group_rollout that sets is_eval=True for evaluation.
    This is used when RLTestSetEvaluator calls do_group_rollout.
    
    Note: group number cannot be determined from env_group_builder alone.
    We use a global counter to track group numbers across parallel evaluations.
    """
    global _current_eval_model_path, _eval_group_counter
    
    # Use the global model_path if available, otherwise try to extract from policy
    model_path = _current_eval_model_path
    
    # Use a global counter to track group numbers across parallel evaluations
    # This is a workaround since do_group_rollout doesn't accept group parameter
    if _eval_group_counter is None:
        _eval_group_counter = {'value': 0, 'lock': asyncio.Lock()}
    
    async with _eval_group_counter['lock']:
        current_group = _eval_group_counter['value']
        _eval_group_counter['value'] += 1
    
    # Try to get eval_id from context to pass to do_cua_group_rollout
    eval_id = None
    try:
        from tinker_cookbook.recipes.cua_rl.database.database_context import get_eval_id
        eval_id = get_eval_id()
        if eval_id:
            logger.debug(f"[Eval Rollout] Retrieved eval_id={eval_id} from context for group {current_group}")
    except Exception as e:
        logger.warning(f"[Eval Rollout] Failed to get eval_id from context: {e}")
    
    return await do_cua_group_rollout(
        env_group_builder=env_group_builder,
        policy=policy,
        model_path=model_path,  # Use global model_path if set, otherwise will try to extract from policy
        step=None,
        batch=None,
        group=current_group,  # Use counter-based group number
        output_dir=None,
        is_eval=True,  # Always True for evaluation
        eval_id=eval_id,  # Pass eval_id explicitly
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
    max_turns: int = 3
    
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
    
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "resume"
    
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
    
    # Initialize database
    from tinker_cookbook.recipes.cua_rl.database.database import init_database, get_session, get_session_direct
    from tinker_cookbook.recipes.cua_rl.database.database_dao import (
        create_training,
        get_training_by_run_name,
        update_training,
    )
    import json
    
    # Initialize PostgreSQL database
    # Priority: DATABASE_URL > individual POSTGRES_* env vars
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        # Construct from individual environment variables (with defaults matching docker-compose.yml)
        postgres_host = os.getenv("POSTGRES_HOST", "127.0.0.1")
        postgres_port = os.getenv("POSTGRES_PORT", "5433")  # Use 5433 to avoid conflict with Cursor
        postgres_db = os.getenv("POSTGRES_DB", "training_db")
        postgres_user = os.getenv("POSTGRES_USER", "training_user")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "training_password")
        database_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        logger.info(f"Using PostgreSQL database from environment variables: {postgres_host}:{postgres_port}/{postgres_db}")
    else:
        logger.info(f"Using PostgreSQL database from DATABASE_URL")
    
    init_database(database_url, echo=False)
    
    # Create or load training record
    with get_session() as session:
        existing_training = get_training_by_run_name(session, run_name)
        if existing_training:
            logger.info(f"Found existing training record: {existing_training.id}")
            training_id = existing_training.id
            # Update status to initializing
            update_training(
                session,
                training_id,
                status="initializing",
                current_phase="initialization",
                status_message="Resuming training",
            )
        else:
            # Create new training record
            config_dict = {
                "model_name": cli_config.model_name,
                "lora_rank": cli_config.lora_rank,
                "learning_rate": cli_config.learning_rate,
                "batch_size": cli_config.groups_per_batch,
                "group_size": cli_config.group_size,
                "groups_per_batch": cli_config.groups_per_batch,
                "max_tokens": cli_config.max_tokens,
                "temperature": cli_config.temperature,
                "kl_penalty_coef": cli_config.kl_penalty_coef,
                "num_substeps": cli_config.num_substeps,
                "max_turns": cli_config.max_turns,
                "seed": cli_config.seed,
                "box_type": cli_config.box_type,
                "renderer_name": renderer_name,
                "wandb_project": cli_config.wandb_project,
                "wandb_name": wandb_name,
                "eval_every": cli_config.eval_every,
                "save_every": cli_config.save_every,
            }
            training = create_training(
                session,
                run_name=run_name,
                log_path=log_path,
                model_name=cli_config.model_name,
                lora_rank=cli_config.lora_rank,
                learning_rate=cli_config.learning_rate,
                batch_size=cli_config.groups_per_batch,
                group_size=cli_config.group_size,
                groups_per_batch=cli_config.groups_per_batch,
                max_tokens=cli_config.max_tokens,
                temperature=cli_config.temperature,
                kl_penalty_coef=cli_config.kl_penalty_coef,
                num_substeps=cli_config.num_substeps,
                max_turns=cli_config.max_turns,
                seed=cli_config.seed,
                box_type=cli_config.box_type,
                renderer_name=renderer_name,
                wandb_project=cli_config.wandb_project,
                wandb_name=wandb_name,
                status="initializing",
                current_phase="initialization",
                config_json=json.dumps(config_dict),
                start_time=datetime.utcnow(),
            )
            training_id = training.id
            logger.info(f"Created new training record: {training_id}")
    
    # Get database session for dataset builder (will be closed after dataset is built)
    db_session = get_session_direct()
    
    # Set default eval_tasks if needed
    eval_tasks = cli_config.eval_tasks
    if eval_tasks is None and cli_config.use_default_eval_tasks:
        eval_tasks = {"source_type": "demo_eval", "limit": 10, "seed": cli_config.seed}
        logger.info(f"Using default eval_tasks: randomly sampling 10 tasks from demo_eval (seed={cli_config.seed})")
    elif eval_tasks is None:
        logger.info("No eval_tasks configured, evaluation will be disabled")
    
    # Build dataset builder
    # Note: db_session is passed but will be used later when dataset_builder.__call__() is invoked
    # We don't close the session here because tasks are saved during __call__() which happens later
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
        db_session=db_session,  # Pass database session for task saving (will be used in __call__)
        training_id=training_id,  # Pass training ID
    )
    # Don't close db_session here - it will be used when dataset_builder.__call__() is invoked
    # The session will be managed by the global database context
    
    # Override train.do_group_rollout with our custom function for training rollouts
    # Note: rollouts.do_group_rollout was already overridden at module import time
    # to ensure metric_util uses our custom function
    # Wrap do_cua_group_rollout to add database recording
    original_do_cua_group_rollout = do_cua_group_rollout
    
    async def do_cua_group_rollout_with_db(
        env_group_builder,
        policy,
        model_path=None,
        step=None,
        batch=None,
        group=None,
        output_dir=None,
        is_eval=False,
    ):
        """Wrapper that adds database recording to rollout."""
        from tinker_cookbook.recipes.cua_rl.database.database_context import get_database_session, get_baseline_id, get_training_id
        from tinker_cookbook.recipes.cua_rl.database.database_training_hooks import (
            record_step_before_rollout,
            record_step_after_rollout,
        )
        from tinker_cookbook.recipes.cua_rl.database.database_dao import get_step_by_training_and_step, get_eval_by_training_and_step
        from tinker_cookbook.recipes.cua_rl.core.rollout import _rollout_step, _rollout_batch
        from tinker_cookbook.utils.trace import get_scope_context
        
        # Auto-detect if this is a baseline evaluation rollout
        # If step is None and we have a baseline_id in context, this is a baseline evaluation
        baseline_id_from_context = get_baseline_id()
        if not is_eval and step is None and baseline_id_from_context is not None:
            # This is a baseline evaluation rollout
            is_eval = True
            logger.info(f"[Rollout DB] Auto-detected baseline evaluation rollout (baseline_id={baseline_id_from_context})")
        
        # Get step from parameter, global rollout context, or scope context
        if step is None:
            # Try global rollout context first
            step = _rollout_step
            if step is None:
                # Try scope context
                try:
                    scope_ctx = get_scope_context()
                    step = scope_ctx.attributes.get("step")
                except:
                    pass
        
        # Get batch from parameter or global rollout context
        if batch is None:
            batch = _rollout_batch
        
        # Record step before rollout if this is a training step
        step_id = None
        if not is_eval and step is not None:
            step_id = record_step_before_rollout(step, batch=batch)
            logger.info(f"[Rollout DB] Recorded step {step} (batch={batch}) to database, step_id={step_id}")
        
        try:
            # Get database session and step_id for rollout
            db_session = get_database_session()
            eval_id = None
            baseline_id = None
            
            # Write debug info to file
            import os
            from datetime import datetime
            debug_file = os.path.join(os.getenv("CUA_DEBUG_LOG_DIR", "/tmp"), "cua_rollout_debug.log")
            try:
                os.makedirs(os.path.dirname(debug_file), exist_ok=True)
                with open(debug_file, "a") as f:
                    f.write(f"[{datetime.now().isoformat()}] [Rollout DB] do_cua_group_rollout_with_db: is_eval={is_eval}, step={step}, db_session={db_session is not None}\n")
                    f.flush()
            except Exception:
                pass
            
            # If this is baseline evaluation (is_eval=True but no step), get baseline_id from context
            if is_eval and step is None:
                baseline_id = baseline_id_from_context or get_baseline_id()
                if baseline_id:
                    logger.info(f"[Rollout DB] Using baseline_id={baseline_id} for baseline evaluation rollout")
                    try:
                        with open(debug_file, "a") as f:
                            f.write(f"[{datetime.now().isoformat()}] [Rollout DB] Using baseline_id={baseline_id} for baseline evaluation rollout\n")
                            f.flush()
                    except Exception:
                        pass
                else:
                    logger.warning(f"[Rollout DB] Baseline evaluation rollout but baseline_id is None! is_eval={is_eval}, step={step}")
                    try:
                        with open(debug_file, "a") as f:
                            f.write(f"[{datetime.now().isoformat()}] [Rollout DB] WARNING: Baseline evaluation rollout but baseline_id is None! is_eval={is_eval}, step={step}\n")
                            f.flush()
                    except Exception:
                        pass
            
            # If this is regular eval (is_eval=True and step is not None), try to get eval_id
            elif is_eval and step is not None:
                training_id = get_training_id()
                if training_id and db_session:
                    eval_obj = get_eval_by_training_and_step(db_session, training_id, step)
                    if eval_obj:
                        eval_id = eval_obj.id
            
            result = await original_do_cua_group_rollout(
                env_group_builder=env_group_builder,
                policy=policy,
                model_path=model_path,
                step=step,
                batch=batch,
                group=group,
                output_dir=output_dir,
                is_eval=is_eval,
                db_session=db_session,
                step_id=step_id,
                eval_id=eval_id,
                baseline_id=baseline_id,
            )
            
            # Record step after rollout
            if not is_eval and step is not None:
                record_step_after_rollout(step, model_path=model_path)
            
            return result
        except Exception as e:
            logger.error(f"[Rollout DB] Error in rollout with database recording: {e}", exc_info=True)
            raise
    
    train.do_group_rollout = do_cua_group_rollout_with_db
    
    # Set output directory for trajectory saving
    set_rollout_output_dir(log_path)
    
    # Check database for latest completed step to resume from
    # ONLY use database, do not check log files
    db_checkpoint_path = None
    db_resume_step = None
    db_state_path = None
    
    with get_session() as session:
        from tinker_cookbook.recipes.cua_rl.database.database_dao import get_latest_completed_step
        latest_step = get_latest_completed_step(session, training_id)
        if latest_step and latest_step.checkpoint_path:
            db_checkpoint_path = latest_step.checkpoint_path
            db_resume_step = latest_step.step
            # checkpoint_path should be state_path for resuming training
            db_state_path = latest_step.checkpoint_path
            logger.info(f"[Resume DB] Found latest completed step {db_resume_step} in database")
            logger.info(f"[Resume DB] Using checkpoint_path from database: {db_checkpoint_path}")
        else:
            logger.info(f"[Resume DB] No completed step with checkpoint_path found in database, starting from scratch")
    
    # Use database checkpoint if available, otherwise use explicit config
    # If both are None, will start from scratch
    resume_checkpoint_path = db_checkpoint_path or cli_config.load_checkpoint_path
    
    # Override checkpoint_utils.get_last_checkpoint to return database info instead of log file
    # This ensures train.main() uses database info for resuming
    original_get_last_checkpoint = None
    if db_state_path and db_resume_step is not None:
        from tinker_cookbook import checkpoint_utils
        original_get_last_checkpoint = checkpoint_utils.get_last_checkpoint
        
        def get_last_checkpoint_from_db(log_dir: str, required_key: str = "state_path") -> dict[str, Any] | None:
            """Override to return checkpoint info from database instead of log file."""
            if required_key == "state_path" and db_state_path:
                logger.info(f"[Resume DB] Returning checkpoint from database: step={db_resume_step}, state_path={db_state_path}")
                return {
                    "state_path": db_state_path,
                    "batch": db_resume_step + 1,  # Resume from next step after completed step
                }
            # For other keys (like "sampler_path"), return None to use default behavior
            return None
        
        checkpoint_utils.get_last_checkpoint = get_last_checkpoint_from_db
        logger.info(f"[Resume DB] Overridden checkpoint_utils.get_last_checkpoint to use database")
    
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
        load_checkpoint_path=resume_checkpoint_path,  # Use database checkpoint if available
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
    
    # Update training status to running
    with get_session() as session:
        update_training(
            session,
            training_id,
            status="running",
            current_phase="training",
            status_message="Training started",
            start_time=datetime.utcnow(),
        )
    
    # Set global database context for training loop
    from tinker_cookbook.recipes.cua_rl.database.database_context import set_database_context
    db_session_for_training = get_session_direct()
    set_database_context(db_session_for_training, training_id)
    
    # Hook evaluation functions to record to database
    from tinker_cookbook.rl import train as rl_train
    original_run_evaluations = rl_train.run_evaluations_parallel
    original_run_baseline_evaluations = rl_train.run_baseline_evaluations_parallel
    
    async def run_evaluations_with_db(
        evaluators,
        sampling_client,
        cfg,
        i_batch,
        model_path=None,
    ):
        """Wrapper to record evaluations to database."""
        from tinker_cookbook.recipes.cua_rl.database.database_eval import (
            get_or_create_eval,
            record_eval_completion,
        )
        from tinker_cookbook.recipes.cua_rl.database.database_context import get_database_session, get_training_id
        
        session = get_database_session()
        training_id = get_training_id()
        
        # Create eval record
        eval_id = None
        if session and training_id and model_path:
            try:
                # Count total tasks from evaluators
                total_tasks = 0
                for evaluator in evaluators:
                    if hasattr(evaluator, 'env_group_builders_P'):
                        total_tasks += len(evaluator.env_group_builders_P)
                
                eval_id = get_or_create_eval(
                    session,
                    training_id,
                    i_batch,
                    model_path,
                    total_tasks=total_tasks,
                )
                session.commit()
                
                # Set eval_id in context so _cua_group_rollout_for_eval can access it
                from tinker_cookbook.recipes.cua_rl.database.database_context import set_eval_id
                set_eval_id(eval_id)
                logger.info(f"[Eval DB] Set eval_id={eval_id} in context for evaluation rollout")
            except Exception as e:
                logger.warning(f"Failed to create eval record: {e}")
                session.rollback()
        
        try:
            # Run original evaluation
            metrics = await original_run_evaluations(
                evaluators,
                sampling_client,
                cfg,
                i_batch,
                model_path=model_path,
            )
            
            # Record eval completion
            if session and eval_id:
                try:
                    # Extract metrics
                    success_rate = metrics.get("test/success_rate", 0.0)
                    avg_reward = metrics.get("test/reward_mean", 0.0)
                    avg_turns = metrics.get("test/turns_mean", 0.0)
                    successful_tasks = int(success_rate * total_tasks) if total_tasks > 0 else 0
                    
                    record_eval_completion(
                        session,
                        eval_id,
                        success_rate=success_rate,
                        avg_reward=avg_reward,
                        avg_turns=avg_turns,
                        successful_tasks=successful_tasks,
                        metrics=metrics,
                    )
                    session.commit()
                except Exception as e:
                    logger.warning(f"Failed to record eval completion: {e}")
                    session.rollback()
            
            return metrics
        except Exception as e:
            # Record eval failure
            if session and eval_id:
                try:
                    from tinker_cookbook.recipes.cua_rl.database.database_dao import update_eval
                    update_eval(
                        session,
                        eval_id,
                        status="failed",
                        error_message=str(e),
                        end_time=datetime.utcnow(),
                    )
                    session.commit()
                except:
                    session.rollback()
            raise
        finally:
            # Clear eval_id from context after evaluation completes
            from tinker_cookbook.recipes.cua_rl.database.database_context import set_eval_id
            set_eval_id(None)
    
    async def run_baseline_evaluations_with_db(
        evaluators,
        sampling_client,
        cfg,
        model_path=None,
    ):
        """Wrapper to record baseline evaluations to database."""
        from tinker_cookbook.recipes.cua_rl.database.database_eval import (
            record_baseline_start,
            record_baseline_completion,
        )
        from tinker_cookbook.recipes.cua_rl.database.database_context import get_database_session, get_training_id
        
        logger.info(f"[Baseline DB] run_baseline_evaluations_with_db called, model_path={model_path}")
        session = get_database_session()
        training_id = get_training_id()
        logger.info(f"[Baseline DB] session={session is not None}, training_id={training_id}, model_path={model_path is not None}")
        
        # Write debug info to file
        import os
        from datetime import datetime
        debug_file = os.path.join(os.getenv("CUA_DEBUG_LOG_DIR", "/tmp"), "cua_rollout_debug.log")
        try:
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            with open(debug_file, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [Baseline DB] run_baseline_evaluations_with_db: session={session is not None}, training_id={training_id}, model_path={model_path is not None}\n")
                f.flush()
        except Exception:
            pass
        
        # Create baseline record
        baseline_id = None
        if session and training_id and model_path:
            try:
                # Count total tasks
                total_tasks = 0
                for evaluator in evaluators:
                    if hasattr(evaluator, 'env_group_builders_P'):
                        total_tasks += len(evaluator.env_group_builders_P)
                
                baseline_id = record_baseline_start(
                    session,
                    training_id,
                    model_path,
                    total_tasks=total_tasks,
                )
                # Retry commit with exponential backoff for database lock
                import time
                max_retries = 5
                retry_delay = 0.1
                committed = False
                for attempt in range(max_retries):
                    try:
                        session.commit()
                        committed = True
                        break
                    except Exception as commit_error:
                        if "locked" in str(commit_error).lower() and attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"[Baseline DB] Database locked, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise
                
                if committed:
                    # Set baseline_id in global context so rollout functions can access it
                    from tinker_cookbook.recipes.cua_rl.database.database_context import set_baseline_id
                    set_baseline_id(baseline_id)
                    logger.info(f"[Baseline DB] Created baseline record: baseline_id={baseline_id}, total_tasks={total_tasks}")
                    
                    # Write to debug file
                    try:
                        with open(debug_file, "a") as f:
                            f.write(f"[{datetime.now().isoformat()}] [Baseline DB] Created baseline record: baseline_id={baseline_id}, total_tasks={total_tasks}\n")
                            f.flush()
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"[Baseline DB] Failed to create baseline record: {e}", exc_info=True)
                session.rollback()
        else:
            logger.warning(f"[Baseline DB] Skipping baseline record creation: session={session is not None}, training_id={training_id}, model_path={model_path is not None}")
        
        try:
            # Run original baseline evaluation
            metrics = await original_run_baseline_evaluations(
                evaluators,
                sampling_client,
                cfg,
                model_path=model_path,
            )
            
            # Record baseline completion
            if session and baseline_id:
                try:
                    # Extract metrics
                    success_rate = metrics.get("test/success_rate", 0.0)
                    avg_reward = metrics.get("test/reward_mean", 0.0)
                    avg_turns = metrics.get("test/turns_mean", 0.0)
                    successful_tasks = int(success_rate * total_tasks) if total_tasks > 0 else 0
                    
                    record_baseline_completion(
                        session,
                        baseline_id,
                        success_rate=success_rate,
                        avg_reward=avg_reward,
                        avg_turns=avg_turns,
                        successful_tasks=successful_tasks,
                        metrics=metrics,
                    )
                    session.commit()
                except Exception as e:
                    logger.warning(f"Failed to record baseline completion: {e}")
                    session.rollback()
            
            return metrics
        except Exception as e:
            # Record baseline failure
            if session and baseline_id:
                try:
                    from tinker_cookbook.recipes.cua_rl.database.database_dao import update_baseline
                    update_baseline(
                        session,
                        baseline_id,
                        status="failed",
                        error_message=str(e),
                        end_time=datetime.utcnow(),
                    )
                    session.commit()
                except:
                    session.rollback()
            raise
        finally:
            # Clear baseline_id from context after evaluation completes
            from tinker_cookbook.recipes.cua_rl.database.database_context import set_baseline_id
            set_baseline_id(None)
    
    # Replace evaluation functions
    rl_train.run_evaluations_parallel = run_evaluations_with_db
    rl_train.run_baseline_evaluations_parallel = run_baseline_evaluations_with_db
    
    # Wrap save_checkpoint_and_get_sampling_client to record checkpoint_path to database
    original_save_checkpoint = rl_train.save_checkpoint_and_get_sampling_client
    
    async def save_checkpoint_with_db_recording(
        training_client,
        i_batch: int,
        log_path: str,
        save_every: int,
        start_batch: int = 0,
    ):
        """Wrapper that records checkpoint_path to database after saving."""
        result = await original_save_checkpoint(
            training_client, i_batch, log_path, save_every, start_batch
        )
        sampling_client, model_path, metrics = result
        
        # If this is a checkpoint batch (not _tmp), record checkpoint_path to database
        if save_every > 0 and i_batch > start_batch and i_batch % save_every == 0:
            try:
                # Read checkpoints.jsonl directly to get state_path (bypassing our override)
                import os
                import json
                checkpoints_file = os.path.join(log_path, "checkpoints.jsonl")
                state_path = None
                if os.path.exists(checkpoints_file):
                    with open(checkpoints_file, 'r') as f:
                        lines = f.readlines()
                        # Get the last line that has state_path
                        for line in reversed(lines):
                            try:
                                checkpoint_info = json.loads(line.strip())
                                if "state_path" in checkpoint_info and checkpoint_info.get("name") == f"{i_batch:06d}":
                                    state_path = checkpoint_info["state_path"]
                                    break
                            except json.JSONDecodeError:
                                continue
                
                if state_path:
                    # Record checkpoint_path to database
                    from tinker_cookbook.recipes.cua_rl.database.database_training_hooks import record_step_after_training
                    # Get metrics from training step if available
                    step_metrics = metrics.copy() if metrics else None
                    record_step_after_training(
                        step=i_batch,
                        model_path=model_path,
                        checkpoint_path=state_path,  # Use state_path for resuming training
                        metrics=step_metrics,
                    )
                    logger.info(f"[Checkpoint DB] Recorded checkpoint_path for step {i_batch}: {state_path}")
                else:
                    logger.warning(f"[Checkpoint DB] Could not find state_path for step {i_batch} in checkpoints.jsonl")
            except Exception as e:
                logger.warning(f"[Checkpoint DB] Failed to record checkpoint_path for step {i_batch}: {e}", exc_info=True)
        
        return result
    
    rl_train.save_checkpoint_and_get_sampling_client = save_checkpoint_with_db_recording
    
    # Force flush before starting training
    import sys
    logger.info("=" * 80)
    logger.info("About to call train.main(config)...")
    logger.info(f"Config: model={config.model_name}, log_path={config.log_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        # Run training
        logger.info("Calling train.main(config) now...")
        sys.stdout.flush()
        await train.main(config)
        
        # Restore original get_last_checkpoint if we overrode it
        if original_get_last_checkpoint is not None:
            from tinker_cookbook import checkpoint_utils
            checkpoint_utils.get_last_checkpoint = original_get_last_checkpoint
            logger.info(f"[Resume DB] Restored original checkpoint_utils.get_last_checkpoint")
        
        # Update training status to completed
        with get_session() as session:
            update_training(
                session,
                training_id,
                status="completed",
                current_phase="completed",
                status_message="Training completed successfully",
                end_time=datetime.utcnow(),
                progress_percent=100.0,
            )
    except Exception as e:
        # Restore original get_last_checkpoint if we overrode it
        if original_get_last_checkpoint is not None:
            from tinker_cookbook import checkpoint_utils
            checkpoint_utils.get_last_checkpoint = original_get_last_checkpoint
            logger.info(f"[Resume DB] Restored original checkpoint_utils.get_last_checkpoint after error")
        
        # Update training status to failed
        with get_session() as session:
            update_training(
                session,
                training_id,
                status="failed",
                current_phase="failed",
                status_message=f"Training failed: {str(e)}",
                error_message=str(e),
                end_time=datetime.utcnow(),
            )
        raise
    finally:
        # Clean up database context
        from tinker_cookbook.recipes.cua_rl.database.database_context import clear_database_context
        clear_database_context()
        if db_session_for_training:
            db_session_for_training.close()


def main_wrapper(cli_config: CLIConfig) -> None:
    """Wrapper function for nested_entrypoint."""
    asyncio.run(cli_main(cli_config))


if __name__ == "__main__":
    chz.nested_entrypoint(main_wrapper, allow_hyphens=True)

