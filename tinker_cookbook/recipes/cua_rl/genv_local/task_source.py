from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy.orm import Session

from tinker_cookbook.recipes.cua_rl.database.database_dao import create_or_get_task, create_or_get_validator
from tinker_cookbook.recipes.cua_rl.demo_tasks import CUATask, TaskCategory, TaskDifficulty

logger = logging.getLogger(__name__)


def _require_genv_task_loader() -> Any:
    try:
        from genv.sdk.tasks import TaskLoader  # type: ignore

        return TaskLoader
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "genv is required for genv-umetrip task loading. "
            "Install it with: pip install -e /Users/zhenwei/workspace/genv-umetrip/rl"
        ) from e


@dataclass(frozen=True)
class GenvTaskRef:
    tasks_dir: str
    identifier: str  # directory name (TaskLoader.find_task input)


def list_genv_task_identifiers(tasks_dir: str) -> list[str]:
    """List available task directory names (not meta.id)."""
    TaskLoader = _require_genv_task_loader()
    loader = TaskLoader(tasks_dir=tasks_dir)
    return loader.list_tasks()


def load_genv_task_ref(tasks_dir: str, identifier: str) -> tuple[CUATask, dict[str, Any], GenvTaskRef]:
    """
    Load a genv task config and convert it into a CUATask (+ raw config).

    Returns:
      - CUATask with id set to meta.id (e.g. task-001)
      - raw_task_data dict (includes env/evaluation/checks)
      - GenvTaskRef (tasks_dir + identifier) for re-loading later
    """
    TaskLoader = _require_genv_task_loader()
    loader = TaskLoader(tasks_dir=tasks_dir)
    resolved = loader.find_task(identifier)
    if resolved is None:
        raise ValueError(f"genv task not found: {identifier!r} in tasks_dir={tasks_dir!r}")

    task_obj = loader.load(resolved)
    data: dict[str, Any] = task_obj.data

    meta = data.get("meta") or {}
    task_id = str(meta.get("id") or resolved)
    task_name = str(meta.get("name") or resolved)
    instruction = str(data.get("instruction") or data.get("description") or task_name)

    # Keep max steps aligned with cua_rl semantics.
    timeout_sec = None
    try:
        timeout_sec = int(meta.get("timeoutSec")) if meta.get("timeoutSec") is not None else None
    except Exception:
        timeout_sec = None
    max_steps = 20 if timeout_sec is None else max(10, min(60, timeout_sec // 10))

    cua_task = CUATask(
        id=task_id,
        name=task_name,
        description=instruction,
        difficulty=TaskDifficulty.MEDIUM,
        category=TaskCategory.APP,
        max_steps=max_steps,
        validation_type="genv_graphql",
        validation_query=None,
        expected_result=None,
        tags=["genv_umetrip"],
    )

    # Attach refs for runtime.
    cua_task.genv_tasks_dir = tasks_dir
    cua_task.genv_identifier = resolved

    return cua_task, data, GenvTaskRef(tasks_dir=tasks_dir, identifier=resolved)


def _iter_split(
    identifiers: list[str],
    *,
    seed: int,
    train_ratio: float,
    split_type: Optional[str],
) -> list[str]:
    rng = random.Random(seed)
    shuffled = identifiers.copy()
    rng.shuffle(shuffled)
    if split_type is None:
        return shuffled
    split_type_norm = split_type.strip().lower()
    if split_type_norm not in {"train", "eval"}:
        raise ValueError(f"Invalid split_type: {split_type!r} (expected 'train' or 'eval' or None)")
    cut = int(len(shuffled) * float(train_ratio))
    if split_type_norm == "train":
        return shuffled[:cut]
    return shuffled[cut:]


def load_genv_tasks(
    *,
    tasks_dir: str,
    seed: int,
    train_ratio: float = 0.8,
    split_type: Optional[str] = None,
    limit: Optional[int] = None,
    save_to_db: bool = False,
    db_session: Optional[Session] = None,
) -> list[CUATask]:
    """
    Load genv-umetrip tasks and optionally save them into cua_rl database.
    """
    tasks_dir_path = Path(tasks_dir)
    if not tasks_dir_path.exists():
        raise FileNotFoundError(f"tasks_dir does not exist: {tasks_dir}")

    all_identifiers = list_genv_task_identifiers(tasks_dir)
    chosen_identifiers = _iter_split(all_identifiers, seed=seed, train_ratio=train_ratio, split_type=split_type)
    if limit is not None and limit < len(chosen_identifiers):
        rng = random.Random(seed)
        chosen_identifiers = rng.sample(chosen_identifiers, int(limit))

    tasks: list[CUATask] = []
    for identifier in chosen_identifiers:
        cua_task, raw_data, _ref = load_genv_task_ref(tasks_dir, identifier)
        tasks.append(cua_task)
        if save_to_db and db_session is not None:
            save_genv_task_to_database(db_session, cua_task, raw_data, source_type="genv_umetrip")

    if save_to_db and db_session is not None:
        db_session.commit()

    logger.info(
        "[genv_local] Loaded %d tasks from %s (split=%s, limit=%s)",
        len(tasks),
        tasks_dir,
        split_type,
        limit,
    )
    return tasks


def save_genv_task_to_database(
    session: Session,
    task: CUATask,
    raw_task_data: dict[str, Any],
    *,
    source_type: str,
) -> int:
    """
    Save task + GraphQL validator config to database.

    We store evaluation config in Validator.config_json for inspection/debugging.
    """
    db_task = create_or_get_task(
        session,
        task_id=task.id,
        name=task.name,
        description=task.description,
        difficulty=str(task.difficulty.value if hasattr(task.difficulty, "value") else task.difficulty),
        category=str(task.category.value if hasattr(task.category, "value") else task.category),
        max_steps=task.max_steps,
        validation_type=task.validation_type,
        validation_query=None,
        expected_result=None,
        tags=task.tags,
        prerequisites=task.prerequisites,
        app_name=getattr(task, "app_name", None),
        source_type=source_type,
    )

    evaluation = raw_task_data.get("evaluation") or {}
    graphql = evaluation.get("graphql") or {}
    checks = evaluation.get("checks") or []
    success_criteria = evaluation.get("successCriteria") or None

    validator_config = {
        "graphql": graphql,
        "checks": checks,
        "successCriteria": success_criteria,
        "taskMeta": raw_task_data.get("meta") or {},
    }

    # Save a single validator per task.
    create_or_get_validator(
        session,
        task_id=db_task.id,
        validator_type="genv_graphql",
        validation_method="graphql_checks",
        validation_query=json.dumps(
            {"endpoint": graphql.get("endpoint", ""), "checksCount": len(checks)},
            ensure_ascii=False,
        ),
        config_json=validator_config,
    )

    return int(db_task.id)

