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

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.recipes.cua_rl.vision_utils import convert_openai_responses_to_message
from tinker_cookbook.recipes.cua_rl.tinker_cua_agent import TinkerCuaAgent
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
    ):
        """
        Initialize CUA Environment.
        
        Args:
            task_description: Description of the task to complete
            gbox_api_key: GBox API key
            tinker_api_key: Tinker API key for OpenAI-compatible API (for rollout)
            rollout_model_name: Not used, kept for compatibility
            renderer: Renderer instance (for training, not used in rollout)
            convo_prefix: Conversation prefix (for training, not used in rollout)
            max_turns: Maximum number of turns
            box_type: Type of GBox environment (android or linux)
            format_coef: Format coefficient (not used for CUA)
        """
        # Initialize parent with renderer (needed for training conversion)
        super().__init__(renderer or renderers.RoleColonRenderer(None), convo_prefix, format_coef=format_coef)
        
        self.task_description = task_description
        self.gbox_api_key = gbox_api_key
        self.tinker_api_key = tinker_api_key
        self.max_turns = max_turns
        self.box_type = box_type
        self.task = task  # Store task object for validation
        
        # TinkerCuaAgent instance (created lazily during rollout)
        self._agent: Optional[TinkerCuaAgent] = None
        
        # Track rollout state
        self._rollout_messages: List[Dict[str, Any]] = []
        self._rollout_result: Optional[Dict[str, Any]] = None
    
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
        renderer_name: Optional[str] = None,
        rollout_logger = None,
    ) -> Dict[str, Any]:
        """
        Run a rollout using TinkerCuaAgent with Tinker's native API.
        
        This allows using the current training model for rollout (on-policy RL).
        The model_path is a tinker://... checkpoint path that dynamically updates
        as training progresses.
        
        Uses Tinker's native API which supports multimodal inputs (images),
        unlike the OpenAI-compatible API which only supports text.
        
        Args:
            tinker_model_path: Tinker checkpoint path (e.g., tinker://.../sampler_weights/000080)
            tinker_api_key: Tinker API key (same as TINKER_API_KEY)
            base_model_name: Base model name for tokenizer/renderer
            renderer_name: Renderer name (auto-detected if None)
            
        Returns:
            Dictionary with rollout results
        """
        # Create TinkerCuaAgent instance with Tinker's native API
        # This allows using the current training model checkpoint for rollout
        # The model_path dynamically updates as training progresses
        # Native API supports multimodal inputs (images) unlike OpenAI-compatible API
        agent_init_start = time.time()
        
        # Log environment initialization (2 lines: config + agent creation)
        if rollout_logger:
            # Truncate task description if too long
            task_desc_short = self.task_description[:80] + "..." if len(self.task_description) > 80 else self.task_description
            rollout_logger.log(f"[CUAEnv] Task: {self.task_description}")
            rollout_logger.log(
                f"[CUAEnv] Config: model={base_model_name} | "
                f"renderer={renderer_name or 'auto'} | box={self.box_type} | max_turns={self.max_turns}"
            )
        else:
            logger.info(f"[CUAEnv] Initializing environment for rollout")
            logger.info(f"[CUAEnv] Task: {self.task_description}")
            logger.info(f"[CUAEnv] Model path: {tinker_model_path}")
            logger.info(f"[CUAEnv] Base model: {base_model_name}")
            logger.info(f"[CUAEnv] Renderer: {renderer_name or 'auto'}")
            logger.info(f"[CUAEnv] Box type: {self.box_type}")
            logger.info(f"[CUAEnv] Max turns: {self.max_turns}")
        
        self._agent = TinkerCuaAgent(
            gbox_api_key=self.gbox_api_key,
            tinker_api_key=tinker_api_key,
            tinker_model_path=tinker_model_path,  # Use checkpoint path (dynamically updates)
            base_model_name=base_model_name,
            renderer_name=renderer_name,
            max_turns=self.max_turns,
            box_type=self.box_type,
            rollout_logger=rollout_logger,
        )
        agent_init_time = time.time() - agent_init_start
        if rollout_logger:
            rollout_logger.log(f"[CUAEnv] ✓ TinkerCuaAgent created in {agent_init_time:.3f}s")
        else:
            logger.info(f"[CUAEnv] ✓ TinkerCuaAgent created in {agent_init_time:.3f}s")
        
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
            
            # Perform ADB validation if task has validation_query (before agent is closed)
            if self.task and self.task.validation_query and rollout_logger:
                try:
                    from tinker_cookbook.recipes.cua_rl.reward import validate_task_completion_with_details
                    
                    # Get result_message for validation if needed
                    result_message = result.get("result_message", "") if isinstance(result, dict) else ""
                    
                    validation_result = await validate_task_completion_with_details(
                        task=self.task,
                        gbox_client=self._agent.gbox_client,
                        result_message=result_message,
                    )
                    
                    if validation_result:
                        rollout_logger.log_adb_validation(
                            command=validation_result.command,
                            expected_result=validation_result.expected_result,
                            actual_result=validation_result.actual_result,
                            success=validation_result.success,
                            execution_time=validation_result.execution_time,
                            validation_query=validation_result.validation_query,
                        )
                except Exception as e:
                    logger.warning(f"Failed to perform ADB validation: {e}")
            
            return result
        finally:
            # Cleanup
            cleanup_start = time.time()
            if rollout_logger:
                rollout_logger.log(f"[CUAEnv] Cleaning up agent...")
            else:
                logger.info(f"[CUAEnv] Cleaning up agent...")
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
            task_obj = task
        else:
            # It's a string
            task_description = str(task)
            task_obj = None
        
        return ProblemGroupBuilder(
            env_thunk=partial(
                CUAEnv,
                task_description,
                self.gbox_api_key,
                tinker_api_key=self.tinker_api_key,
                rollout_model_name=self.rollout_model_name,
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
                max_turns=self.max_turns,
                box_type=self.box_type,
                task=task_obj,  # Pass task object for validation (None if task is string)
            ),
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
    renderer_name: Optional[str] = None
    convo_prefix: Optional[List[renderers.Message]] = None
    max_turns: int = 20
    box_type: str = "android"
    seed: int = 0
    
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
        if isinstance(self.tasks, dict):
            # Single TaskSourceConfig
            config = TaskSourceConfig(**self.tasks)
            tasks = load_tasks_from_config(config)
        elif isinstance(self.tasks, list):
            if len(self.tasks) == 0:
                raise ValueError("tasks list cannot be empty")
            # List of TaskSourceConfig dicts
            configs = [TaskSourceConfig(**item) for item in self.tasks]
            tasks = load_tasks_from_multiple_sources(configs)
        else:
            raise ValueError(
                f"Invalid tasks format: {type(self.tasks)}. "
                "Expected: dict (TaskSourceConfig) or List[dict] (multiple TaskSourceConfigs)"
            )
        
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
        )
        
        # Load evaluation tasks if provided
        eval_dataset = None
        if self.eval_tasks is not None:
            if isinstance(self.eval_tasks, dict):
                eval_config = TaskSourceConfig(**self.eval_tasks)
                eval_tasks = load_tasks_from_config(eval_config)
            elif isinstance(self.eval_tasks, list):
                if len(self.eval_tasks) == 0:
                    raise ValueError("eval_tasks list cannot be empty")
                eval_configs = [TaskSourceConfig(**item) for item in self.eval_tasks]
                eval_tasks = load_tasks_from_multiple_sources(eval_configs)
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
                seed=self.seed + 9999,  # Use different seed for eval to ensure different shuffling
            )
        
        return dataset, eval_dataset

