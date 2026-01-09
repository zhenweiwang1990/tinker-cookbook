"""
CUA (Computer Use Agent) Environment for RL training.

This environment wraps GBoxAgent to provide an RL-compatible interface.
"""

import asyncio
import logging
import time
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Union

import chz
import tinker
from sqlalchemy.orm import Session

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.recipes.cua_rl.utils.vision_utils import convert_openai_responses_to_message
from tinker_cookbook.recipes.cua_rl.agent.tinker_cua_agent import TinkerCuaAgent
from tinker_cookbook.recipes.cua_rl.demo_tasks import CUATask

logger = logging.getLogger(__name__)


class CUAEnv(ProblemEnv):
    """
    Environment for Computer Use Agent tasks.
    
    This environment uses GBoxAgent to interact with a device screen and complete tasks.
    The agent receives screenshots and takes actions to complete the task.
    """
    
    def __init__(
        self,
        task_description: str,
        gbox_api_key: str,
        
        # Provider configuration (NEW)
        provider: str = "tinker",
        provider_base_url: Optional[str] = None,
        provider_api_key: Optional[str] = None,
        
        # Legacy parameters (backward compatible)
        tinker_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        
        rollout_model_name: str = "gpt-4o",
        renderer: Optional[renderers.Renderer] = None,
        convo_prefix: Optional[List[renderers.Message]] = None,
        max_turns: int = 20,
        box_type: str = "android",
        format_coef: float = 0.0,  # No format penalty for CUA
        task: Optional[CUATask] = None,  # Optional task object for validation
        max_recent_turns: int = 5,  # Maximum number of recent turns to keep in message history
        max_task_time_seconds: int = 30 * 60,  # Maximum total time for task execution (default: 30 minutes)
        max_turn_time_seconds: int = 5 * 60,  # Maximum time per turn for model inference (default: 5 minutes)
        coordinate_mode: str = "gbox",  # Coordinate generation mode: "gbox" or "direct"
        coordinate_scale: Optional[bool] = None,  # Auto: True for direct, False for gbox
        x_scale_ratio: Optional[float] = None,  # X scaling ratio
        y_scale_ratio: Optional[float] = None,  # Y scaling ratio
    ):
        """
        Initialize CUA Environment.
        
        Args:
            task_description: Description of the task to complete
            gbox_api_key: GBox API key
            
            provider: Inference provider ("tinker", "vllm", "openrouter", "openai")
            provider_base_url: API base URL (for vLLM, OpenRouter, etc.)
            provider_api_key: API key (for OpenRouter, OpenAI, etc.)
            
            tinker_api_key: Tinker API key (legacy, for backward compatibility)
            openai_api_key: OpenAI API key (legacy, not used)
            openai_api_base: OpenAI API base URL (legacy, not used)
            
            rollout_model_name: Not used, kept for compatibility
            renderer: Renderer instance (for training, not used in rollout)
            convo_prefix: Conversation prefix (for training, not used in rollout)
            max_turns: Maximum number of turns
            box_type: Type of GBox environment (android or linux)
            format_coef: Format coefficient (not used for CUA)
            max_task_time_seconds: Maximum total time for task execution (default: 30 minutes)
            max_turn_time_seconds: Maximum time per turn for model inference (default: 5 minutes)
            coordinate_mode: Coordinate generation mode ("gbox" or "direct")
        """
        # Initialize parent with renderer (needed for training conversion)
        super().__init__(renderer or renderers.RoleColonRenderer(None), convo_prefix, format_coef=format_coef)
        
        self.task_description = task_description
        self.gbox_api_key = gbox_api_key
        
        # Provider settings
        self.provider = provider
        self.provider_base_url = provider_base_url
        self.provider_api_key = provider_api_key
        
        # Legacy support
        self.tinker_api_key = tinker_api_key
        
        self.max_turns = max_turns
        self.box_type = box_type
        self.max_recent_turns = max_recent_turns
        self.task = task  # Store task object for validation
        self.max_task_time_seconds = max_task_time_seconds
        self.max_turn_time_seconds = max_turn_time_seconds
        self.coordinate_mode = coordinate_mode  # Store coordinate mode
        self.coordinate_scale = coordinate_scale
        self.x_scale_ratio = x_scale_ratio
        self.y_scale_ratio = y_scale_ratio
        
        # TinkerCuaAgent instance (created lazily during rollout)
        self._agent: Optional[TinkerCuaAgent] = None
        
        # Track rollout state
        self._rollout_messages: List[Dict[str, Any]] = []
        self._rollout_result: Optional[Dict[str, Any]] = None
        
        # Trajectory data for token-level training (saved from agent before cleanup)
        self._trajectory_turns: List = []
    
    def get_question(self) -> str:
        """Return the task description."""
        return self.task_description
    
    def check_format(self, sample_str: str) -> bool:
        """CUA doesn't have strict format requirements."""
        return True
    
    def check_answer(self, sample_str: str) -> bool:
        """Check if task was completed successfully."""
        if self._rollout_result is None:
            return False
        return bool(self._rollout_result.get("task_success", False))
    
    def get_reference_answer(self) -> str:
        """Return the result message."""
        if self._rollout_result is None:
            return "Task not completed"
        return self._rollout_result.get("result_message", "Task not completed")
    
    
    async def run_rollout_with_tinker_model(
        self,
        tinker_model_path: str,
        tinker_api_key: str,
        base_model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        # base_model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct",
        renderer_name: Optional[str] = None,
        rollout_logger = None,
        rollout_recorder = None,  # RolloutRecorder instance for database recording
        rollout_id: Optional[str] = None,  # Rollout ID (UUID) for database recording
    ) -> Dict[str, Any]:
        """
        Run a rollout using TinkerCuaAgent with flexible inference provider.
        
        This method supports multiple providers (Tinker, vLLM, OpenRouter, etc.).
        For Tinker provider, tinker_model_path is the checkpoint path.
        For other providers, tinker_model_path is used as the model name.
        
        The provider is determined from self.provider (set during __init__).
        
        Args:
            tinker_model_path: Model path or identifier
                - For Tinker: checkpoint path (e.g., tinker://.../sampler_weights/000080)
                - For others: model name (e.g., Qwen/Qwen3-VL-30B-A3B-Instruct)
            tinker_api_key: API key (Tinker or provider-specific)
            base_model_name: Base model name for tokenizer/renderer
            renderer_name: Renderer name (auto-detected if None)
            
        Returns:
            Dictionary with rollout results
        """
        # Create TinkerCuaAgent based on provider
        # The agent supports multiple inference backends while preserving all functionality
        # (prompts, tool parsing, coordinate handling, database recording)
        agent_init_start = time.time()
        
        # Log environment initialization
        if rollout_logger:
            task_desc_short = self.task_description[:80] + "..." if len(self.task_description) > 80 else self.task_description
            rollout_logger.log(f"[CUAEnv] Task: {self.task_description}")
            rollout_logger.log(
                f"[CUAEnv] Config: provider={self.provider} | model={base_model_name} | "
                f"renderer={renderer_name or 'auto'} | box={self.box_type} | max_turns={self.max_turns}"
            )
        else:
            logger.info(f"[CUAEnv] Initializing environment for rollout")
            logger.info(f"[CUAEnv] Provider: {self.provider}")
            logger.info(f"[CUAEnv] Task: {self.task_description}")
            logger.info(f"[CUAEnv] Model path: {tinker_model_path}")
            logger.info(f"[CUAEnv] Base model: {base_model_name}")
            logger.info(f"[CUAEnv] Renderer: {renderer_name or 'auto'}")
            logger.info(f"[CUAEnv] Box type: {self.box_type}")
            logger.info(f"[CUAEnv] Max turns: {self.max_turns}")
        
        # Create agent based on provider
        if self.provider == "tinker":
            # Tinker provider: use legacy parameters
            self._agent = TinkerCuaAgent(
                gbox_api_key=self.gbox_api_key,
                tinker_api_key=tinker_api_key,
                tinker_model_path=tinker_model_path,
                base_model_name=base_model_name,
                renderer_name=renderer_name,
                max_turns=self.max_turns,
                box_type=self.box_type,
                max_recent_turns=self.max_recent_turns,
                rollout_logger=rollout_logger,
                rollout_recorder=rollout_recorder,
                rollout_id=rollout_id,
                max_task_time_seconds=self.max_task_time_seconds,
                max_turn_time_seconds=self.max_turn_time_seconds,
                coordinate_mode=self.coordinate_mode,
                coordinate_scale=self.coordinate_scale,
                x_scale_ratio=self.x_scale_ratio,
                y_scale_ratio=self.y_scale_ratio,
            )
        else:
            # Other providers: use new provider-based parameters
            self._agent = TinkerCuaAgent(
                gbox_api_key=self.gbox_api_key,
                provider=self.provider,
                provider_model_name=tinker_model_path,  # For non-Tinker, this is the model name
                provider_base_url=self.provider_base_url,
                provider_api_key=self.provider_api_key or tinker_api_key,
                base_model_name=base_model_name,
                renderer_name=renderer_name,
                max_turns=self.max_turns,
                box_type=self.box_type,
                max_recent_turns=self.max_recent_turns,
                rollout_logger=rollout_logger,
                rollout_recorder=rollout_recorder,
                rollout_id=rollout_id,
                max_task_time_seconds=self.max_task_time_seconds,
                max_turn_time_seconds=self.max_turn_time_seconds,
                coordinate_mode=self.coordinate_mode,
                coordinate_scale=self.coordinate_scale,
                x_scale_ratio=self.x_scale_ratio,
                y_scale_ratio=self.y_scale_ratio,
            )
        # Pass task object to agent for validation
        if self.task:
            self._agent.task = self.task
        agent_init_time = time.time() - agent_init_start
        if rollout_logger:
            rollout_logger.log(f"[CUAEnv] ✓ TinkerCuaAgent created in {agent_init_time:.3f}s")
        else:
            logger.info(f"[CUAEnv] ✓ TinkerCuaAgent created in {agent_init_time:.3f}s")
        
        # Force flush before starting task
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        logger.info(f"[CUAEnv] About to call run_task()...")
        sys.stdout.flush()
        
        try:
            # Run task with current training model
            task_start = time.time()
            result = await self._agent.run_task(
                task_description=self.task_description,
                verbose=False,
            )
            task_time = time.time() - task_start
            if rollout_logger:
                rollout_logger.log(f"[CUAEnv] ✓ Task execution completed in {task_time:.3f}s")
            else:
                logger.info(f"[CUAEnv] ✓ Task execution completed in {task_time:.3f}s")
            
            self._rollout_result = result
            
            # Save trajectory data from agent before cleanup (for token-level training)
            if self._agent and hasattr(self._agent, 'trajectory_turns'):
                self._trajectory_turns = self._agent.trajectory_turns.copy()
            else:
                self._trajectory_turns = []
            
            # Note: Validation is now performed inside run_task() before the box is terminated
            # This ensures gbox_client is still available when validation executes
            
            return result
        finally:
            # Cleanup
            cleanup_start = time.time()
            await self._agent.close()
            cleanup_time = time.time() - cleanup_start
            if rollout_logger:
                rollout_logger.log(f"[CUAEnv] ✓ Agent cleanup completed in {cleanup_time:.3f}s")
            else:
                logger.info(f"[CUAEnv] ✓ Agent cleanup completed in {cleanup_time:.3f}s")
            self._agent = None
    
    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        """
        Initial observation for training.
        
        Note: This is used during training data preparation, not during rollout.
        The actual rollout is done via run_rollout().
        """
        # For training, we need to convert the rollout messages to ModelInput
        # This will be called after rollout is complete
        if not self._rollout_messages:
            # If no messages yet, return empty (shouldn't happen in normal flow)
            return tinker.ModelInput.empty(), self.stop_condition
        
        # Convert OpenAI format messages to Tinker Message format
        tinker_messages = [
            convert_openai_responses_to_message(msg) for msg in self._rollout_messages
        ]
        
        # Build ModelInput using renderer
        model_input = self.renderer.build_generation_prompt(tinker_messages)
        return model_input, self.stop_condition
    
    async def step(self, action: Action) -> StepResult:
        """
        Step function for training.
        
        Note: This is used during training data preparation, not during rollout.
        The actual rollout is done via run_rollout().
        """
        # For training, we parse the action and check if task was successful
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.ensure_text(message["content"])
        
        # Check if task was completed successfully
        task_success = self.check_answer(content)
        
        # Reward is 1.0 if task succeeded, 0.0 otherwise
        reward = 1.0 if task_success else 0.0
        
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "task_success": float(task_success),
                "task_completed": float(self._rollout_result.get("task_completed", False) if self._rollout_result else False),
            },
        )


class CUADataset(RLDataset):
    """Dataset for CUA tasks."""
    
    def __init__(
        self,
        tasks: Union[List[str], List[CUATask]],  # Can be list of strings or CUATask objects
        batch_size: int,
        group_size: int,
        gbox_api_key: str,
        tinker_api_key: Optional[str] = None,
        rollout_model_name: Optional[str] = None,  # Not used, kept for compatibility
        renderer: Optional[renderers.Renderer] = None,
        convo_prefix: Optional[List[renderers.Message]] = None,
        max_turns: int = 20,
        box_type: str = "android",
        seed: int = 0,
        max_recent_turns: int = 5,  # Maximum number of recent turns to keep in message history
        max_task_time_seconds: int = 30 * 60,  # Maximum total time for task execution (default: 30 minutes)
        max_turn_time_seconds: int = 5 * 60,  # Maximum time per turn for model inference (default: 5 minutes)
        coordinate_mode: str = "gbox",  # Coordinate generation mode: "gbox" or "direct"
        coordinate_scale: Optional[bool] = None,  # Auto: True for direct, False for gbox
        x_scale_ratio: Optional[float] = None,  # X scaling ratio
        y_scale_ratio: Optional[float] = None,  # Y scaling ratio
    ):
        self.tasks = tasks
        self.batch_size = batch_size
        self.group_size = group_size
        self.gbox_api_key = gbox_api_key
        self.tinker_api_key = tinker_api_key
        self.rollout_model_name = rollout_model_name
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.max_turns = max_turns
        self.box_type = box_type
        self.seed = seed
        self.max_recent_turns = max_recent_turns
        self.max_task_time_seconds = max_task_time_seconds
        self.max_turn_time_seconds = max_turn_time_seconds
        self.coordinate_mode = coordinate_mode  # Store coordinate mode
        self.coordinate_scale = coordinate_scale
        self.x_scale_ratio = x_scale_ratio
        self.y_scale_ratio = y_scale_ratio
        
        # Shuffle tasks with seed
        import random
        rng = random.Random(seed)
        self.tasks = tasks.copy()
        rng.shuffle(self.tasks)
    
    def __len__(self) -> int:
        return (len(self.tasks) + self.batch_size - 1) // self.batch_size
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.tasks))
        if start >= end:
            raise IndexError(f"Incorrect batch index {index} for CUADataset")
        
        builders: List[EnvGroupBuilder] = []
        for task in self.tasks[start:end]:
            builder = self._make_env_group_builder(task, self.group_size)
            builders.append(builder)
        
        return builders
    
    def _make_env_group_builder(
        self, task: Any, group_size: int  # task can be CUATask or str
    ) -> ProblemGroupBuilder:
        # Extract task description and task object
        if hasattr(task, 'description'):
            # It's a CUATask object
            task_description = task.description
            # Make a deep copy to ensure independence
            import copy
            task_obj = copy.deepcopy(task)
        else:
            # It's a string
            task_description = str(task)
            task_obj = None
        
        # CRITICAL: Use a closure factory instead of partial to ensure
        # each environment gets the correct task parameters.
        # We capture task_description and task_obj at THIS moment.
        _captured_desc = task_description
        _captured_task = task_obj
        
        def make_env():
            """Factory function that creates a CUAEnv with captured parameters."""
            return CUAEnv(
                task_description=_captured_desc,
                gbox_api_key=self.gbox_api_key,
                tinker_api_key=self.tinker_api_key,
                rollout_model_name=self.rollout_model_name,
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
                max_turns=self.max_turns,
                box_type=self.box_type,
                task=_captured_task,
            max_recent_turns=self.max_recent_turns,
            max_task_time_seconds=self.max_task_time_seconds,
            max_turn_time_seconds=self.max_turn_time_seconds,
            coordinate_mode=self.coordinate_mode,  # Pass coordinate mode
            coordinate_scale=self.coordinate_scale,
            x_scale_ratio=self.x_scale_ratio,
            y_scale_ratio=self.y_scale_ratio,
        )
        
        return ProblemGroupBuilder(
            env_thunk=make_env,
            num_envs=group_size,
            dataset_name="cua",
        )


@chz.chz
class CUADatasetBuilder(RLDatasetBuilder):
    """Builder for CUA dataset.
    
    Tasks are configured using TaskSourceConfig:
    - dict: Single TaskSourceConfig (e.g., {"source_type": "demo_training"})
    - List[dict]: Multiple TaskSourceConfig objects
    
    Evaluation dataset (eval_tasks) is optional and uses the same configuration format.
    """
    
    tasks: dict | list[dict]  # TaskSourceConfig dict(s) for training
    eval_tasks: Optional[dict | list[dict]] = None  # Optional TaskSourceConfig dict(s) for evaluation
    batch_size: int
    group_size: int
    gbox_api_key: str
    tinker_api_key: Optional[str] = None
    rollout_model_name: Optional[str] = None  # Not used, kept for compatibility
    model_name_for_tokenizer: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    # model_name_for_tokenizer: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    renderer_name: Optional[str] = None
    convo_prefix: Optional[List[renderers.Message]] = None
    max_turns: int = 20
    box_type: str = "android"
    seed: int = 0
    max_recent_turns: int = 5  # Maximum number of recent turns to keep in message history
    db_session: Optional[Session] = None  # Database session for saving tasks
    training_id: Optional[int] = None  # Training ID for database records
    max_task_time_seconds: int = 30 * 60  # Maximum total time for task execution (default: 30 minutes)
    max_turn_time_seconds: int = 5 * 60  # Maximum time per turn for model inference (default: 5 minutes)
    coordinate_mode: str = "gbox"  # Coordinate generation mode: "gbox" or "direct"
    coordinate_scale: Optional[bool] = None  # Auto: True for direct, False for gbox
    x_scale_ratio: Optional[float] = None  # X scaling ratio (default: screen_width / 1000)
    y_scale_ratio: Optional[float] = None  # Y scaling ratio (default: screen_height / 1000)
    
    async def __call__(self) -> tuple[CUADataset, CUADataset | None]:
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from tinker_cookbook import model_info
        from tinker_cookbook.image_processing_utils import get_image_processor
        from tinker_cookbook.recipes.cua_rl.task_loader import (
            TaskSourceConfig,
            load_tasks_from_config,
            load_tasks_from_multiple_sources,
        )
        
        # Load tasks from TaskSourceConfig
        # Save to database if db_session is provided
        # If db_session is None or closed, try to get from global context
        db_session = self.db_session
        if db_session is None:
            from tinker_cookbook.recipes.cua_rl.database.database_context import get_database_session
            db_session = get_database_session()
        
        save_to_db = db_session is not None
        if isinstance(self.tasks, dict):
            # Single TaskSourceConfig
            config = TaskSourceConfig(**self.tasks)
            tasks = load_tasks_from_config(config, save_to_db=save_to_db, db_session=db_session)
        elif isinstance(self.tasks, list):
            if len(self.tasks) == 0:
                raise ValueError("tasks list cannot be empty")
            # List of TaskSourceConfig dicts
            configs = [TaskSourceConfig(**item) for item in self.tasks]
            tasks = load_tasks_from_multiple_sources(configs, save_to_db=save_to_db, db_session=db_session)
        else:
            raise ValueError(
                f"Invalid tasks format: {type(self.tasks)}. "
                "Expected: dict (TaskSourceConfig) or List[dict] (multiple TaskSourceConfigs)"
            )
        
        # Commit task saves if we saved any tasks
        if save_to_db and db_session is not None and tasks:
            try:
                db_session.commit()
                logger.info(f"[Dataset Builder] Committed {len(tasks)} tasks to database")
            except Exception as e:
                logger.error(f"[Dataset Builder] Failed to commit tasks to database: {e}", exc_info=True)
                db_session.rollback()
        
        # Get renderer
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name_for_tokenizer
        )
        # Get image processor if model is vision-language
        attributes = model_info.get_model_attributes(self.model_name_for_tokenizer)
        image_processor = get_image_processor(self.model_name_for_tokenizer) if attributes.is_vl else None
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer, image_processor=image_processor)
        
        dataset = CUADataset(
            tasks=tasks,
            batch_size=self.batch_size,
            group_size=self.group_size,
            gbox_api_key=self.gbox_api_key,
            tinker_api_key=self.tinker_api_key,
            rollout_model_name=self.rollout_model_name,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            max_turns=self.max_turns,
            box_type=self.box_type,
            seed=self.seed,
            max_recent_turns=self.max_recent_turns,
            max_task_time_seconds=self.max_task_time_seconds,
            max_turn_time_seconds=self.max_turn_time_seconds,
            coordinate_mode=self.coordinate_mode,  # Pass coordinate mode
            coordinate_scale=self.coordinate_scale,
            x_scale_ratio=self.x_scale_ratio,
            y_scale_ratio=self.y_scale_ratio,
        )
        
        # Load evaluation tasks if provided
        eval_dataset = None
        if self.eval_tasks is not None:
            if isinstance(self.eval_tasks, dict):
                eval_config = TaskSourceConfig(**self.eval_tasks)
                eval_tasks = load_tasks_from_config(eval_config, save_to_db=save_to_db, db_session=db_session)
            elif isinstance(self.eval_tasks, list):
                if len(self.eval_tasks) == 0:
                    raise ValueError("eval_tasks list cannot be empty")
                eval_configs = [TaskSourceConfig(**item) for item in self.eval_tasks]
                eval_tasks = load_tasks_from_multiple_sources(eval_configs, save_to_db=save_to_db, db_session=db_session)
            
            # Commit eval task saves if we saved any tasks
            if save_to_db and db_session is not None and eval_tasks:
                try:
                    db_session.commit()
                    logger.info(f"[Dataset Builder] Committed {len(eval_tasks)} eval tasks to database")
                except Exception as e:
                    logger.error(f"[Dataset Builder] Failed to commit eval tasks to database: {e}", exc_info=True)
                    db_session.rollback()
            else:
                raise ValueError(
                    f"Invalid eval_tasks format: {type(self.eval_tasks)}. "
                    "Expected: dict (TaskSourceConfig) or List[dict] (multiple TaskSourceConfigs)"
                )
            
            # For evaluation, use group_size=1 (each task runs independently)
            # and batch_size=1 (each task is a separate batch for simpler evaluation)
            eval_dataset = CUADataset(
                tasks=eval_tasks,
                batch_size=1,  # Each task is a separate batch for evaluation
                group_size=1,  # Each task runs independently, no group comparison needed
                gbox_api_key=self.gbox_api_key,
                tinker_api_key=self.tinker_api_key,
                rollout_model_name=self.rollout_model_name,
                renderer=renderer,
                convo_prefix=self.convo_prefix,
                max_turns=self.max_turns,
                box_type=self.box_type,
                max_recent_turns=self.max_recent_turns,
                seed=self.seed + 9999,  # Use different seed for eval to ensure different shuffling
                max_task_time_seconds=self.max_task_time_seconds,
                max_turn_time_seconds=self.max_turn_time_seconds,
                coordinate_mode=self.coordinate_mode,  # Pass coordinate mode
            )
        
        # Force flush before returning
        import sys
        logger.info(f"[Dataset Builder] Returning dataset with {len(tasks)} tasks, eval_dataset={'with tasks' if eval_dataset else None}")
        sys.stdout.flush()
        sys.stderr.flush()
        
        return dataset, eval_dataset

