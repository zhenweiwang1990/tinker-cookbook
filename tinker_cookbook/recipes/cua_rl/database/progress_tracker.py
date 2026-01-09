"""
Progress tracking system for CUA RL training.

This module implements a comprehensive progress tracking system that tracks all levels:
- Training (contains N steps + N/eval_every eval)
- Baseline (contains Task-count rollouts)
- Eval (contains Task-count rollouts)  
- Step (contains Task-count groups)
- Group (contains group_size rollouts)
- Rollout (contains max_turns turns)

All progress is based on turns for consistent estimation, and includes:
- Status tracking (pending/running/completed/failed)
- Progress percentage (0-100%)
- Estimated total time (based on average turn time)
- Estimated remaining time
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from sqlalchemy.orm import Session

from tinker_cookbook.recipes.cua_rl.database.database_dao import (
    update_training,
    update_baseline,
    update_eval,
    update_step,
    update_group,
    update_rollout,
    get_training,
    get_baseline,
    get_eval,
    get_step,
    get_group,
    get_rollout,
    list_groups_by_baseline,
    list_groups_by_eval,
    list_groups_by_step,
    list_rollouts_by_group,
)

logger = logging.getLogger(__name__)


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""
    completed_turns: int = 0
    total_turns: int = 0
    progress_percent: float = 0.0
    avg_turn_time: Optional[float] = None  # Average time per turn in seconds
    estimated_total_time: Optional[float] = None  # Estimated total time in seconds
    estimated_remaining_time: Optional[float] = None  # Estimated remaining time in seconds
    status: str = "pending"  # pending, running, completed, failed


class ProgressTracker:
    """
    Unified progress tracker for all hierarchical levels.
    
    Hierarchy:
    Training
      ├─ Baseline (1)
      │   └─ Groups (task_count)
      │       └─ Rollouts (group_size per group)
      │           └─ Turns (max_turns per rollout)
      ├─ Steps (N)
      │   └─ Groups (task_count per step)
      │       └─ Rollouts (group_size per group)
      │           └─ Turns (max_turns per rollout)
      └─ Evals (N/eval_every)
          └─ Groups (task_count per eval)
              └─ Rollouts (group_size per group)
                  └─ Turns (max_turns per rollout)
    """
    
    def __init__(self, session: Session):
        """Initialize progress tracker with database session."""
        self.session = session
        # Cache for average turn times (entity_type -> entity_id -> avg_time)
        self._turn_time_cache: Dict[str, Dict[int, float]] = {}

    def _get_rollout_turn_progress(
        self,
        rollout: Any,
        default_max_turns: int = 20,
    ) -> tuple[int, int]:
        """
        Return (completed_turns, effective_total_turns) for a rollout, strictly turn-based.

        Notes:
        - For finished rollouts (completed/failed/cancelled), effective_total_turns is the actual
          turns taken (num_turns / current_turn), so that aggregates reach 100% once all rollouts finish.
        - For unfinished rollouts, effective_total_turns is the budgeted max_turns since we don't yet
          know whether it will early-stop.
        - We clamp completed_turns into [0, max_turns] to avoid bad data inflating progress.
        """
        max_turns = int(getattr(rollout, "max_turns", None) or default_max_turns)

        status = getattr(rollout, "status", None)
        if status == "running":
            completed_turns = int(getattr(rollout, "current_turn", None) or 0)
        elif status in ("completed", "failed", "cancelled"):
            # Prefer the authoritative num_turns; fallback to current_turn; last resort 0.
            completed_turns = getattr(rollout, "num_turns", None)
            if completed_turns is None:
                completed_turns = getattr(rollout, "current_turn", None)
            completed_turns = int(completed_turns or 0)
        else:
            # pending / env_creation / agent_init / etc.
            completed_turns = int(getattr(rollout, "current_turn", None) or 0)

        # Clamp to prevent negative or >max values.
        if completed_turns < 0:
            completed_turns = 0
        if max_turns > 0 and completed_turns > max_turns:
            completed_turns = max_turns

        # Effective total turns:
        # - finished rollouts shrink denominator to actual turns (if >0), otherwise fall back to max_turns
        # - unfinished rollouts keep denominator at max_turns
        if status in ("completed", "failed", "cancelled"):
            effective_total_turns = completed_turns if completed_turns > 0 else max_turns
        else:
            effective_total_turns = max_turns

        return completed_turns, effective_total_turns
    
    def _calculate_avg_turn_time(
        self,
        entity_type: str,
        entity_id: int,
        fallback: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate average turn time for an entity.
        
        Args:
            entity_type: Type of entity ('rollout', 'group', 'step', 'eval', 'baseline', 'training')
            entity_id: Database ID of entity
            fallback: Fallback value if no data available
            
        Returns:
            Average turn time in seconds, or None if no data
        """
        # Check cache first
        if entity_type in self._turn_time_cache:
            if entity_id in self._turn_time_cache[entity_type]:
                return self._turn_time_cache[entity_type][entity_id]
        
        # Calculate based on entity type
        avg_time = None
        
        if entity_type == 'rollout':
            # Get rollout and calculate from turns
            from tinker_cookbook.recipes.cua_rl.database.database_dao import list_turns_by_rollout
            rollout = get_rollout(self.session, entity_id)
            if rollout and rollout.num_turns and rollout.rollout_time:
                avg_time = rollout.rollout_time / rollout.num_turns
            else:
                # Try to calculate from turn records
                turns = list_turns_by_rollout(self.session, entity_id)
                if turns:
                    total_time = sum(t.turn_time for t in turns if t.turn_time is not None)
                    count = sum(1 for t in turns if t.turn_time is not None)
                    if count > 0:
                        avg_time = total_time / count
        
        elif entity_type == 'group':
            # Calculate from completed rollouts in group
            rollouts = list_rollouts_by_group(self.session, entity_id)
            completed_rollouts = [r for r in rollouts if r.status == 'completed' and r.num_turns and r.rollout_time]
            if completed_rollouts:
                total_turns = sum(r.num_turns for r in completed_rollouts)
                total_time = sum(r.rollout_time for r in completed_rollouts)
                if total_turns > 0:
                    avg_time = total_time / total_turns
        
        elif entity_type == 'step':
            # Calculate from completed groups in step
            groups = list_groups_by_step(self.session, entity_id)
            turn_times = []
            for group in groups:
                group_avg = self._calculate_avg_turn_time('group', group.id)
                if group_avg is not None:
                    turn_times.append(group_avg)
            if turn_times:
                avg_time = sum(turn_times) / len(turn_times)
        
        elif entity_type == 'eval':
            # Calculate from completed groups in eval
            groups = list_groups_by_eval(self.session, entity_id)
            turn_times = []
            for group in groups:
                group_avg = self._calculate_avg_turn_time('group', group.id)
                if group_avg is not None:
                    turn_times.append(group_avg)
            if turn_times:
                avg_time = sum(turn_times) / len(turn_times)
        
        elif entity_type == 'baseline':
            # Calculate from completed groups in baseline
            groups = list_groups_by_baseline(self.session, entity_id)
            turn_times = []
            for group in groups:
                group_avg = self._calculate_avg_turn_time('group', group.id)
                if group_avg is not None:
                    turn_times.append(group_avg)
            if turn_times:
                avg_time = sum(turn_times) / len(turn_times)
        
        elif entity_type == 'training':
            # Calculate from all completed entities (baseline, steps, evals)
            training = get_training(self.session, entity_id)
            if training:
                turn_times = []
                
                # Get baseline avg
                for baseline in training.baselines:
                    baseline_avg = self._calculate_avg_turn_time('baseline', baseline.id)
                    if baseline_avg is not None:
                        turn_times.append(baseline_avg)
                
                # Get steps avg
                for step in training.steps:
                    step_avg = self._calculate_avg_turn_time('step', step.id)
                    if step_avg is not None:
                        turn_times.append(step_avg)
                
                # Get evals avg
                for eval_obj in training.evals:
                    eval_avg = self._calculate_avg_turn_time('eval', eval_obj.id)
                    if eval_avg is not None:
                        turn_times.append(eval_avg)
                
                if turn_times:
                    avg_time = sum(turn_times) / len(turn_times)
        
        # Use fallback if no data
        if avg_time is None:
            avg_time = fallback
        
        # Cache the result
        if avg_time is not None:
            if entity_type not in self._turn_time_cache:
                self._turn_time_cache[entity_type] = {}
            self._turn_time_cache[entity_type][entity_id] = avg_time
        
        return avg_time

    def _get_training_concurrency(self, training: Any) -> int:
        """Best-effort max concurrency for wall-time ETA."""
        try:
            import json

            cfg = json.loads(training.config_json) if getattr(training, "config_json", None) else {}
            v = cfg.get("max_concurrent_rollouts") or cfg.get("max_concurrent") or 1
            v = int(v)
            return v if v > 0 else 1
        except Exception:
            return 1
    
    def update_rollout_progress(
        self,
        rollout_id: int,
        current_turn: int,
        max_turns: int,
        status: Optional[str] = None,
        turn_time: Optional[float] = None,
    ) -> ProgressStats:
        """
        Update rollout progress based on current turn.
        
        Args:
            rollout_id: Database rollout ID
            current_turn: Current turn number (0-indexed)
            max_turns: Maximum number of turns
            status: Optional status override
            turn_time: Time taken for this turn (for averaging)
            
        Returns:
            ProgressStats with updated progress information
        """
        rollout = get_rollout(self.session, rollout_id)
        if not rollout:
            logger.warning(f"Rollout {rollout_id} not found")
            return ProgressStats(status="failed")
        
        # Calculate progress (turn-based; finished rollouts should reach 100% even if early-stop)
        completed_turns = max(0, int(current_turn))
        max_turns_i = max(0, int(max_turns))
        if max_turns_i > 0 and completed_turns > max_turns_i:
            completed_turns = max_turns_i

        if status in ("completed", "failed", "cancelled"):
            effective_total_turns = completed_turns if completed_turns > 0 else max_turns_i
        else:
            effective_total_turns = max_turns_i

        progress_percent = (
            (completed_turns / effective_total_turns * 100.0) if effective_total_turns > 0 else 0.0
        )
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('rollout', rollout_id, fallback=30.0)  # 30s default
        
        # Estimate times
        estimated_total_time = effective_total_turns * avg_turn_time if avg_turn_time else None
        estimated_remaining_time = (effective_total_turns - completed_turns) * avg_turn_time if avg_turn_time else None
        
        # Determine status
        if status is None:
            if completed_turns >= total_turns:
                status = "completed"
            elif completed_turns > 0:
                status = "running"
            else:
                status = rollout.status or "pending"
        
        # Update database
        update_kwargs = {
            'current_turn': current_turn,
            'progress_percent': progress_percent,
            'status': status,
        }
        
        update_rollout(self.session, rollout_id, **update_kwargs)
        
        stats = ProgressStats(
            completed_turns=completed_turns,
            total_turns=effective_total_turns,
            progress_percent=progress_percent,
            avg_turn_time=avg_turn_time,
            estimated_total_time=estimated_total_time,
            estimated_remaining_time=estimated_remaining_time,
            status=status,
        )
        
        # Update parent group progress
        if rollout.group_id:
            self.update_group_progress(rollout.group_id)
        
        return stats
    
    def update_group_progress(
        self,
        group_id: int,
        force_recalculate: bool = True,
    ) -> ProgressStats:
        """
        Update group progress based on its rollouts.
        
        Args:
            group_id: Database group ID
            force_recalculate: If True, recalculate from rollouts
            
        Returns:
            ProgressStats with updated progress information
        """
        group = get_group(self.session, group_id)
        if not group:
            logger.warning(f"Group {group_id} not found")
            return ProgressStats(status="failed")
        
        # Get all rollouts in this group
        rollouts = list_rollouts_by_group(self.session, group_id)
        
        if not rollouts:
            logger.warning(f"Group {group_id} has no rollouts")
            return ProgressStats(status="pending")
        
        # Calculate aggregate progress from rollouts
        total_turns = 0
        completed_turns = 0
        completed_rollouts = 0
        running_rollouts = 0
        failed_rollouts = 0
        
        for rollout in rollouts:
            rollout_completed_turns, rollout_effective_total_turns = self._get_rollout_turn_progress(
                rollout, default_max_turns=20
            )
            total_turns += rollout_effective_total_turns
            completed_turns += rollout_completed_turns

            if rollout.status == "completed":
                completed_rollouts += 1
            elif rollout.status == "running":
                running_rollouts += 1
            elif rollout.status == "failed":
                failed_rollouts += 1
        
        progress_percent = (completed_turns / total_turns * 100.0) if total_turns > 0 else 0.0
        
        # Determine status
        # All rollouts finished (completed or failed) -> group is done
        finished_rollouts = completed_rollouts + failed_rollouts
        if finished_rollouts >= len(rollouts):
            # If all finished, status depends on whether any succeeded
            if completed_rollouts > 0:
                status = "completed"
            else:
                status = "failed"
        elif running_rollouts > 0 or completed_rollouts > 0:
            status = "running"
        else:
            status = "pending"
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('group', group_id, fallback=30.0)
        
        # Estimate times (wall-time: divide by global concurrency)
        concurrency = 1
        try:
            if group.step_id:
                training = get_training(self.session, get_step(self.session, group.step_id).training_id)
                concurrency = self._get_training_concurrency(training) if training else 1
            elif group.eval_id:
                training = get_training(self.session, get_eval(self.session, group.eval_id).training_id)
                concurrency = self._get_training_concurrency(training) if training else 1
            elif group.baseline_id:
                training = get_training(self.session, get_baseline(self.session, group.baseline_id).training_id)
                concurrency = self._get_training_concurrency(training) if training else 1
        except Exception:
            concurrency = 1

        estimated_total_time = (total_turns * avg_turn_time / concurrency) if avg_turn_time else None
        estimated_remaining_time = ((total_turns - completed_turns) * avg_turn_time / concurrency) if avg_turn_time else None
        
        # Update database
        update_kwargs = {
            'num_rollouts': len(rollouts),
            'completed_rollouts': completed_rollouts,
            'progress_percent': progress_percent,
            'status': status,
        }
        
        update_group(self.session, group_id, **update_kwargs)
        
        stats = ProgressStats(
            completed_turns=completed_turns,
            total_turns=total_turns,
            progress_percent=progress_percent,
            avg_turn_time=avg_turn_time,
            estimated_total_time=estimated_total_time,
            estimated_remaining_time=estimated_remaining_time,
            status=status,
        )
        
        # Update parent (step/eval/baseline) progress
        if group.source_type == 'step' and group.step_id:
            self.update_step_progress(group.step_id)
        elif group.source_type == 'eval' and group.eval_id:
            self.update_eval_progress(group.eval_id)
        elif group.source_type == 'baseline' and group.baseline_id:
            self.update_baseline_progress(group.baseline_id)
        
        return stats
    
    def update_step_progress(
        self,
        step_id: int,
    ) -> ProgressStats:
        """
        Update step progress based on its rollouts (pure turn-based).

        NOTE: This intentionally ignores "training phase" progress and treats progress
        strictly as fraction of turns completed over turns budgeted for this step.

        Args:
            step_id: Database step ID
            
        Returns:
            ProgressStats with updated progress information
        """
        step = get_step(self.session, step_id)
        if not step:
            logger.warning(f"Step {step_id} not found")
            return ProgressStats(status="failed")

        training = get_training(self.session, step.training_id) if step.training_id else None
        default_max_turns = int(getattr(training, "max_turns", None) or 20)
        expected_groups = int(getattr(training, "groups_per_batch", None) or 0)
        expected_rollouts_per_group = int(getattr(training, "group_size", None) or 0)
        
        # Get all groups in this step
        groups = list_groups_by_step(self.session, step_id)
        
        if not groups:
            logger.warning(f"Step {step_id} has no groups")
            return ProgressStats(status="pending")
        
        # Calculate aggregate progress from groups
        total_turns = 0
        completed_turns = 0
        completed_groups = 0
        running_groups = 0
        failed_groups = 0
        
        for group in groups:
            # Expected rollouts for this group: prefer group.num_rollouts; fallback to training.group_size; else len(existing).
            group_rollouts = list_rollouts_by_group(self.session, group.id)
            group_expected_rollouts = int(getattr(group, "num_rollouts", None) or 0)
            if group_expected_rollouts <= 0:
                group_expected_rollouts = (
                    expected_rollouts_per_group if expected_rollouts_per_group > 0 else len(group_rollouts)
                )

            # Sum effective totals for existing rollouts; add budget for missing rollouts not yet created.
            existing_rollouts = len(group_rollouts)
            missing_rollouts = max(0, group_expected_rollouts - existing_rollouts)
            total_turns += missing_rollouts * default_max_turns

            for rollout in group_rollouts:
                rollout_completed_turns, rollout_effective_total_turns = self._get_rollout_turn_progress(
                    rollout, default_max_turns=default_max_turns
                )
                total_turns += rollout_effective_total_turns
                completed_turns += rollout_completed_turns
            
            # Finished groups count based on their status
            if group.status == 'completed':
                completed_groups += 1
            elif group.status == 'running':
                running_groups += 1
            elif group.status == 'failed':
                failed_groups += 1
        
        # Add not-yet-created groups into denominator (0 turns completed).
        if expected_groups > 0:
            missing_groups = max(0, expected_groups - len(groups))
            # Missing groups budget: expected_rollouts_per_group * max_turns.
            if expected_rollouts_per_group > 0:
                total_turns += missing_groups * expected_rollouts_per_group * default_max_turns
            else:
                # If we don't know group size, assume 1 rollout per missing group.
                total_turns += missing_groups * default_max_turns

        progress_percent = (completed_turns / total_turns * 100.0) if total_turns > 0 else 0.0
        
        # Determine status
        finished_groups = completed_groups + failed_groups
        expected_group_count = expected_groups if expected_groups > 0 else len(groups)
        if step.status == 'completed' or (expected_group_count > 0 and finished_groups >= expected_group_count):
            # Respect DB completion marker, or infer completion once all expected groups have finished.
            status = "completed" if completed_groups > 0 else "failed"
        elif running_groups > 0 or completed_groups > 0:
            status = "running"
        else:
            status = "pending"
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('step', step_id, fallback=30.0)
        
        # Estimate times (wall-time: divide by global concurrency)
        concurrency = self._get_training_concurrency(training) if training else 1
        estimated_total_time = (total_turns * avg_turn_time / concurrency) if avg_turn_time else None
        estimated_remaining_time = ((total_turns - completed_turns) * avg_turn_time / concurrency) if avg_turn_time else None
        
        # Update database
        update_kwargs = {
            'progress_percent': progress_percent,
            'status': status,
        }
        
        update_step(self.session, step_id, **update_kwargs)
        
        stats = ProgressStats(
            completed_turns=completed_turns,
            total_turns=total_turns,
            progress_percent=progress_percent,
            avg_turn_time=avg_turn_time,
            estimated_total_time=estimated_total_time,
            estimated_remaining_time=estimated_remaining_time,
            status=status,
        )
        
        # Update parent training progress
        if step.training_id:
            self.update_training_progress(step.training_id)
        
        return stats
    
    def update_eval_progress(
        self,
        eval_id: int,
    ) -> ProgressStats:
        """
        Update eval progress based on its groups.
        
        Args:
            eval_id: Database eval ID
            
        Returns:
            ProgressStats with updated progress information
        """
        eval_obj = get_eval(self.session, eval_id)
        if not eval_obj:
            logger.warning(f"Eval {eval_id} not found")
            return ProgressStats(status="failed")
        
        # Get all groups in this eval
        groups = list_groups_by_eval(self.session, eval_id)
        
        if not groups:
            logger.warning(f"Eval {eval_id} has no groups")
            return ProgressStats(status="pending")
        
        # Calculate aggregate progress from groups
        total_turns = 0
        completed_turns = 0
        completed_groups = 0
        running_groups = 0
        failed_groups = 0
        
        # Prefer known total_tasks to keep denominator stable even before groups are created.
        # This is critical for correct early-run progress: not-yet-started tasks must count as 0 turns done.
        training = get_training(self.session, eval_obj.training_id) if eval_obj.training_id else None
        default_max_turns = int(getattr(training, "max_turns", None) or 20)
        expected_tasks = int(eval_obj.total_tasks or 0)

        for group in groups:
            group_rollouts = list_rollouts_by_group(self.session, group.id)
            group_completed = 0
            group_total = 0
            for rollout in group_rollouts:
                rollout_completed_turns, rollout_effective_total_turns = self._get_rollout_turn_progress(
                    rollout, default_max_turns=default_max_turns
                )
                group_completed += rollout_completed_turns
                group_total += rollout_effective_total_turns

            # Eval semantics: one task per group; cap to max_turns
            group_total = min(default_max_turns, group_total or default_max_turns)
            group_completed = min(group_total, group_completed)
            total_turns += group_total
            completed_turns += group_completed
            
            if group.status == 'completed':
                completed_groups += 1
            elif group.status == 'running':
                running_groups += 1
            elif group.status == 'failed':
                failed_groups += 1
        
        # Add missing (not-yet-created) tasks into denominator.
        if expected_tasks > 0:
            missing_tasks = max(0, expected_tasks - len(groups))
            total_turns += missing_tasks * default_max_turns

        progress_percent = (completed_turns / total_turns * 100.0) if total_turns > 0 else 0.0
        
        # Determine status (must consider tasks not yet started)
        expected_group_count = expected_tasks if expected_tasks > 0 else len(groups)
        finished_groups = completed_groups + failed_groups
        if expected_group_count > 0 and finished_groups >= expected_group_count:
            # All expected tasks finished
            status = "completed" if completed_groups > 0 else "failed"
        elif running_groups > 0 or completed_groups > 0:
            status = "running"
        else:
            status = "pending"
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('eval', eval_id, fallback=30.0)
        
        # Estimate times (wall-time: divide by global concurrency)
        concurrency = self._get_training_concurrency(training) if training else 1
        estimated_total_time = (total_turns * avg_turn_time / concurrency) if avg_turn_time else None
        estimated_remaining_time = ((total_turns - completed_turns) * avg_turn_time / concurrency) if avg_turn_time else None
        
        # Update database
        update_kwargs = {
            'progress_percent': progress_percent,
            'status': status,
            'completed_tasks': completed_groups,
        }
        
        update_eval(self.session, eval_id, **update_kwargs)
        
        stats = ProgressStats(
            completed_turns=completed_turns,
            total_turns=total_turns,
            progress_percent=progress_percent,
            avg_turn_time=avg_turn_time,
            estimated_total_time=estimated_total_time,
            estimated_remaining_time=estimated_remaining_time,
            status=status,
        )
        
        # Update parent training progress
        if eval_obj.training_id:
            self.update_training_progress(eval_obj.training_id)
        
        return stats
    
    def update_baseline_progress(
        self,
        baseline_id: int,
    ) -> ProgressStats:
        """
        Update baseline progress based on its groups.
        
        Args:
            baseline_id: Database baseline ID
            
        Returns:
            ProgressStats with updated progress information
        """
        baseline = get_baseline(self.session, baseline_id)
        if not baseline:
            logger.warning(f"Baseline {baseline_id} not found")
            return ProgressStats(status="failed")
        
        # Get all groups in this baseline
        groups = list_groups_by_baseline(self.session, baseline_id)
        
        if not groups:
            logger.warning(f"Baseline {baseline_id} has no groups")
            return ProgressStats(status="pending")
        
        # Calculate aggregate progress from groups
        total_turns = 0
        completed_turns = 0
        completed_groups = 0
        running_groups = 0
        failed_groups = 0
        
        # Prefer known total_tasks to keep denominator stable even before groups are created.
        training = get_training(self.session, baseline.training_id) if baseline.training_id else None
        default_max_turns = int(getattr(training, "max_turns", None) or 20)
        expected_tasks = int(baseline.total_tasks or 0)

        for group in groups:
            group_rollouts = list_rollouts_by_group(self.session, group.id)
            group_completed = 0
            group_total = 0
            for rollout in group_rollouts:
                rollout_completed_turns, rollout_effective_total_turns = self._get_rollout_turn_progress(
                    rollout, default_max_turns=default_max_turns
                )
                group_completed += rollout_completed_turns
                group_total += rollout_effective_total_turns

            # Baseline semantics: one task per group; cap to max_turns
            group_total = min(default_max_turns, group_total or default_max_turns)
            group_completed = min(group_total, group_completed)
            total_turns += group_total
            completed_turns += group_completed
            
            if group.status == 'completed':
                completed_groups += 1
            elif group.status == 'running':
                running_groups += 1
            elif group.status == 'failed':
                failed_groups += 1

        # Add missing (not-yet-created) tasks into denominator.
        if expected_tasks > 0:
            missing_tasks = max(0, expected_tasks - len(groups))
            total_turns += missing_tasks * default_max_turns

        progress_percent = (completed_turns / total_turns * 100.0) if total_turns > 0 else 0.0
        
        # Determine status (must consider tasks not yet started)
        expected_group_count = expected_tasks if expected_tasks > 0 else len(groups)
        finished_groups = completed_groups + failed_groups
        if expected_group_count > 0 and finished_groups >= expected_group_count:
            # All expected tasks finished
            status = "completed" if completed_groups > 0 else "failed"
        elif running_groups > 0 or completed_groups > 0:
            status = "running"
        else:
            status = "pending"
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('baseline', baseline_id, fallback=30.0)
        
        # Estimate times (wall-time: divide by global concurrency)
        concurrency = self._get_training_concurrency(training) if training else 1
        estimated_total_time = (total_turns * avg_turn_time / concurrency) if avg_turn_time else None
        estimated_remaining_time = ((total_turns - completed_turns) * avg_turn_time / concurrency) if avg_turn_time else None
        
        # Update database
        update_kwargs = {
            'progress_percent': progress_percent,
            'status': status,
            'completed_tasks': completed_groups,
        }
        
        update_baseline(self.session, baseline_id, **update_kwargs)
        
        stats = ProgressStats(
            completed_turns=completed_turns,
            total_turns=total_turns,
            progress_percent=progress_percent,
            avg_turn_time=avg_turn_time,
            estimated_total_time=estimated_total_time,
            estimated_remaining_time=estimated_remaining_time,
            status=status,
        )
        
        # Update parent training progress
        if baseline.training_id:
            self.update_training_progress(baseline.training_id)
        
        return stats
    
    def update_training_progress(
        self,
        training_id: int,
    ) -> ProgressStats:
        """
        Update training progress based on baseline, steps, and evals (pure turn-based).

        Progress is computed as:
          completed_turns / total_turns

        where total_turns includes not-yet-started work:
        - Baseline: baseline.total_tasks * max_turns (group_size=1 in eval/baseline)
        - Steps: total_steps * groups_per_batch * group_size * max_turns
        - Evals: sum(eval.total_tasks * max_turns) for known evals, plus missing evals if inferable
        
        Args:
            training_id: Database training ID
            
        Returns:
            ProgressStats with updated progress information
        """
        training = get_training(self.session, training_id)
        if not training:
            logger.warning(f"Training {training_id} not found")
            return ProgressStats(status="failed")

        default_max_turns = int(getattr(training, "max_turns", None) or 20)
        groups_per_batch = int(getattr(training, "groups_per_batch", None) or 0)
        group_size = int(getattr(training, "group_size", None) or 0)

        total_turns = 0
        completed_turns = 0
        
        # Baseline turns (effective total shrinks on early-stop)
        if training.baselines:
            baseline = training.baselines[0]  # Should only be one baseline

            # Denominator should include not-yet-created tasks
            expected_tasks = int(getattr(baseline, "total_tasks", None) or 0)

            baseline_groups = list_groups_by_baseline(self.session, baseline.id)
            total_groups_expected = expected_tasks if expected_tasks > 0 else len(baseline_groups)
            # For tasks not started yet
            total_turns += max(0, total_groups_expected - len(baseline_groups)) * default_max_turns

            # Count completed + effective totals for existing groups (cap per task to max_turns)
            for group in baseline_groups:
                group_completed = 0
                group_total = 0
                for rollout in list_rollouts_by_group(self.session, group.id):
                    rollout_completed, rollout_total = self._get_rollout_turn_progress(
                        rollout, default_max_turns=default_max_turns
                    )
                    group_completed += rollout_completed
                    group_total += rollout_total
                group_total = min(default_max_turns, group_total or default_max_turns)
                group_completed = min(group_total, group_completed)
                total_turns += group_total
                completed_turns += group_completed
        
        # Steps turns (effective total shrinks on early-stop; missing rollouts are budgeted)
        expected_steps = int(getattr(training, "total_steps", None) or 0)
        expected_step_rollouts_per_step = groups_per_batch * group_size if groups_per_batch > 0 and group_size > 0 else 0

        if training.steps:
            # Existing steps: compute completed turns from rollouts
            for step in training.steps:
                step_groups = list_groups_by_step(self.session, step.id)

                # Count existing rollouts and add their effective totals
                existing_rollouts = 0
                for group in step_groups:
                    for rollout in list_rollouts_by_group(self.session, group.id):
                        existing_rollouts += 1
                        rollout_completed, rollout_total = self._get_rollout_turn_progress(
                            rollout, default_max_turns=default_max_turns
                        )
                        completed_turns += rollout_completed
                        total_turns += rollout_total

                # Budget for missing rollouts in this step
                if expected_step_rollouts_per_step > 0:
                    missing = max(0, expected_step_rollouts_per_step - existing_rollouts)
                    total_turns += missing * default_max_turns

            # Missing (not-yet-created) steps: budget full turns
            if expected_steps > 0 and expected_step_rollouts_per_step > 0:
                missing_steps = max(0, expected_steps - len(training.steps))
                total_turns += missing_steps * expected_step_rollouts_per_step * default_max_turns
        else:
            # No step rows yet, but if total_steps is known, include full budget as 0 completed turns
            if expected_steps > 0 and expected_step_rollouts_per_step > 0:
                total_turns += expected_steps * expected_step_rollouts_per_step * default_max_turns
        
        # Evals turns (effective total shrinks on early-stop; missing tasks are budgeted)
        if training.evals:
            # Use the first known eval.total_tasks as default for missing evals (best-effort).
            default_eval_tasks = int(getattr(training.evals[0], "total_tasks", None) or 0)

            for eval_obj in training.evals:
                expected_tasks = int(getattr(eval_obj, "total_tasks", None) or 0)
                groups = list_groups_by_eval(self.session, eval_obj.id)
                # Budget for missing tasks not yet created
                total_turns += max(0, expected_tasks - len(groups)) * default_max_turns

                for group in groups:
                    group_completed = 0
                    group_total = 0
                    for rollout in list_rollouts_by_group(self.session, group.id):
                        rollout_completed, rollout_total = self._get_rollout_turn_progress(
                            rollout, default_max_turns=default_max_turns
                        )
                        group_completed += rollout_completed
                        group_total += rollout_total
                    group_total = min(default_max_turns, group_total or default_max_turns)
                    group_completed = min(group_total, group_completed)
                    total_turns += group_total
                    completed_turns += group_completed

            # Missing evals (best-effort via config_json.eval_every if present)
            try:
                import json
                cfg = json.loads(training.config_json) if training.config_json else {}
                eval_every = int(cfg.get("eval_every") or 0)
                if eval_every > 0 and expected_steps > 0 and default_eval_tasks > 0:
                    # Approximate expected eval count: every eval_every steps (not including step 0)
                    expected_eval_count = (expected_steps + eval_every - 1) // eval_every
                    missing_evals = max(0, expected_eval_count - len(training.evals))
                    total_turns += missing_evals * default_eval_tasks * default_max_turns
            except Exception:
                pass

        progress_percent = (completed_turns / total_turns * 100.0) if total_turns > 0 else 0.0
        
        # Determine status
        status = training.status
        if progress_percent >= 100.0:
            status = "completed"
        elif progress_percent > 0:
            status = "running"
        
        # Calculate average turn time across all completed turns
        avg_turn_time = self._calculate_avg_turn_time('training', training_id, fallback=30.0)
        
        # Estimate times (wall-time: divide by global concurrency)
        concurrency = self._get_training_concurrency(training)
        estimated_total_time = (total_turns * avg_turn_time / concurrency) if avg_turn_time else None
        estimated_remaining_time = ((total_turns - completed_turns) * avg_turn_time / concurrency) if avg_turn_time else None
        
        # Update database
        update_kwargs = {
            'progress_percent': progress_percent,
            'status': status,
        }
        
        update_training(self.session, training_id, **update_kwargs)
        
        return ProgressStats(
            completed_turns=completed_turns,
            total_turns=total_turns,
            progress_percent=progress_percent,
            avg_turn_time=avg_turn_time,
            estimated_total_time=estimated_total_time,
            estimated_remaining_time=estimated_remaining_time,
            status=status,
        )
    
    def format_time_estimate(self, seconds: Optional[float]) -> str:
        """Format time estimate in human-readable format."""
        if seconds is None:
            return "N/A"
        
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"

