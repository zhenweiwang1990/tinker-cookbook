from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Union

import chz
import tinker
from sqlalchemy.orm import Session

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.recipes.cua_rl.demo_tasks import CUATask

logger = logging.getLogger(__name__)


class GenvEnv(ProblemEnv):
    """
    Thin wrapper representing a genv task for rollout.

    Rollouts are executed by a custom do_group_rollout (genv_local/rollout.py),
    not by ProblemEnv.step(). This class exists to plug into the existing dataset
    / EnvGroupBuilder interfaces.
    """

    def __init__(
        self,
        task: CUATask,
        renderer: renderers.Renderer,
        convo_prefix: Optional[List[renderers.Message]] = None,
        max_turns: int = 20,
        max_recent_turns: int = 5,
        format_coef: float = 0.0,
    ):
        super().__init__(renderer, convo_prefix, format_coef=format_coef)
        self.task = task
        self.max_turns = max_turns
        self.max_recent_turns = max_recent_turns

    def get_question(self) -> str:
        return self.task.description

    def check_format(self, sample_str: str) -> bool:
        return True

    def check_answer(self, sample_str: str) -> bool:
        # Ground truth is computed by genv evaluator checks, not by text.
        return False

    def get_reference_answer(self) -> str:
        return ""


class GenvDataset(RLDataset):
    def __init__(
        self,
        tasks: list[CUATask],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: Optional[List[renderers.Message]] = None,
        max_turns: int = 20,
        max_recent_turns: int = 5,
        seed: int = 0,
    ):
        self.tasks = tasks
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.max_turns = max_turns
        self.max_recent_turns = max_recent_turns
        self.seed = seed

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
            raise IndexError(f"Incorrect batch index {index} for GenvDataset")

        builders: list[EnvGroupBuilder] = []
        for task in self.tasks[start:end]:
            builders.append(self._make_env_group_builder(task))
        return builders

    def _make_env_group_builder(self, task: CUATask) -> ProblemGroupBuilder:
        import copy

        task_copy = copy.deepcopy(task)

        def make_env() -> GenvEnv:
            return GenvEnv(
                task=task_copy,
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
                max_turns=self.max_turns,
                max_recent_turns=self.max_recent_turns,
            )

        return ProblemGroupBuilder(env_thunk=make_env, num_envs=self.group_size, dataset_name="genv_umetrip")


@chz.chz
class GenvDatasetBuilder(RLDatasetBuilder):
    """
    Dataset builder for genv-umetrip tasks.

    Uses the same TaskSourceConfig format as CUADatasetBuilder, but expects tasks to
    include `source_type="genv_umetrip"` and a valid `tasks_dir`.
    """

    tasks: dict | list[dict]
    eval_tasks: Optional[dict | list[dict]] = None
    batch_size: int = 1
    group_size: int = 1
    model_name_for_tokenizer: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    renderer_name: Optional[str] = None
    convo_prefix: Optional[List[renderers.Message]] = None
    max_turns: int = 20
    max_recent_turns: int = 5
    seed: int = 0
    db_session: Optional[Session] = None

    async def __call__(self) -> tuple[GenvDataset, GenvDataset | None]:
        from tinker_cookbook import model_info
        from tinker_cookbook.image_processing_utils import get_image_processor
        from tinker_cookbook.recipes.cua_rl.task_loader import (
            TaskSourceConfig,
            load_tasks_from_config,
            load_tasks_from_multiple_sources,
        )
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        # DB session (optional) for task saving.
        db_session = self.db_session
        if db_session is None:
            from tinker_cookbook.recipes.cua_rl.database.database_context import get_database_session

            db_session = get_database_session()

        save_to_db = db_session is not None

        if isinstance(self.tasks, dict):
            config = TaskSourceConfig(**self.tasks)
            tasks = load_tasks_from_config(config, save_to_db=save_to_db, db_session=db_session)
        elif isinstance(self.tasks, list):
            configs = [TaskSourceConfig(**item) for item in self.tasks]
            tasks = load_tasks_from_multiple_sources(configs, save_to_db=save_to_db, db_session=db_session)
        else:
            raise ValueError(f"Invalid tasks format: {type(self.tasks)}")

        if save_to_db and db_session is not None and tasks:
            try:
                db_session.commit()
            except Exception:
                db_session.rollback()

        hf_tokenizer_model_name = model_info.get_tokenizer_model_name(self.model_name_for_tokenizer)
        tokenizer = get_tokenizer(hf_tokenizer_model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name_for_tokenizer)
        attributes = model_info.get_model_attributes(self.model_name_for_tokenizer)
        image_processor = get_image_processor(hf_tokenizer_model_name) if attributes.is_vl else None
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer, image_processor=image_processor)

        dataset = GenvDataset(
            tasks=tasks,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            max_turns=self.max_turns,
            max_recent_turns=self.max_recent_turns,
            seed=self.seed,
        )

        eval_dataset = None
        if self.eval_tasks is not None:
            if isinstance(self.eval_tasks, dict):
                eval_config = TaskSourceConfig(**self.eval_tasks)
                eval_tasks = load_tasks_from_config(
                    eval_config, save_to_db=save_to_db, db_session=db_session
                )
            else:
                eval_configs = [TaskSourceConfig(**item) for item in self.eval_tasks]
                eval_tasks = load_tasks_from_multiple_sources(
                    eval_configs, save_to_db=save_to_db, db_session=db_session
                )

            if eval_tasks:
                eval_dataset = GenvDataset(
                    tasks=eval_tasks,
                    batch_size=1,
                    group_size=1,
                    renderer=renderer,
                    convo_prefix=self.convo_prefix,
                    max_turns=self.max_turns,
                    max_recent_turns=self.max_recent_turns,
                    seed=self.seed + 9999,
                )

        return dataset, eval_dataset

