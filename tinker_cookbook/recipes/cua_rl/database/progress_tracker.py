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
        
        # Calculate progress
        completed_turns = current_turn  # 0-indexed, so current_turn is number of completed turns
        total_turns = max_turns
        progress_percent = min(100.0, (completed_turns / total_turns * 100.0) if total_turns > 0 else 0.0)
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('rollout', rollout_id, fallback=30.0)  # 30s default
        
        # Estimate times
        estimated_total_time = total_turns * avg_turn_time if avg_turn_time else None
        estimated_remaining_time = (total_turns - completed_turns) * avg_turn_time if avg_turn_time else None
        
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
            total_turns=total_turns,
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
            max_turns = rollout.max_turns or 20  # Default 20 turns
            total_turns += max_turns
            
            if rollout.status == 'completed':
                completed_turns += rollout.num_turns or max_turns
                completed_rollouts += 1
            elif rollout.status == 'running':
                completed_turns += rollout.current_turn or 0
                running_rollouts += 1
            elif rollout.status == 'failed':
                failed_rollouts += 1
            # pending rollouts contribute 0 completed turns
        
        progress_percent = (completed_turns / total_turns * 100.0) if total_turns > 0 else 0.0
        
        # Determine status
        if completed_rollouts >= len(rollouts):
            status = "completed"
        elif failed_rollouts >= len(rollouts):
            status = "failed"
        elif running_rollouts > 0 or completed_rollouts > 0:
            status = "running"
        else:
            status = "pending"
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('group', group_id, fallback=30.0)
        
        # Estimate times
        estimated_total_time = total_turns * avg_turn_time if avg_turn_time else None
        estimated_remaining_time = (total_turns - completed_turns) * avg_turn_time if avg_turn_time else None
        
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
        Update step progress based on its groups (rollout phase) and training metrics.
        
        A step has two phases:
        1. Rollout phase: Running groups to collect trajectories
        2. Training phase: Training the model on collected trajectories
        
        Args:
            step_id: Database step ID
            
        Returns:
            ProgressStats with updated progress information
        """
        step = get_step(self.session, step_id)
        if not step:
            logger.warning(f"Step {step_id} not found")
            return ProgressStats(status="failed")
        
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
            # Get group's total and completed turns
            group_rollouts = list_rollouts_by_group(self.session, group.id)
            for rollout in group_rollouts:
                max_turns = rollout.max_turns or 20
                total_turns += max_turns
                
                if rollout.status == 'completed':
                    completed_turns += rollout.num_turns or max_turns
                elif rollout.status == 'running':
                    completed_turns += rollout.current_turn or 0
            
            if group.status == 'completed':
                completed_groups += 1
            elif group.status == 'running':
                running_groups += 1
            elif group.status == 'failed':
                failed_groups += 1
        
        # Progress is weighted: 80% rollout, 20% training
        # This reflects that rollout is the more time-consuming phase
        rollout_progress = (completed_turns / total_turns * 80.0) if total_turns > 0 else 0.0
        
        # Training progress: if training has started, add remaining 20%
        training_progress = 0.0
        if step.current_phase == 'training' or step.status == 'completed':
            training_progress = 20.0
        elif step.current_phase == 'rollout' and completed_groups >= len(groups):
            # Rollout completed, training about to start
            training_progress = 0.0
        
        progress_percent = min(100.0, rollout_progress + training_progress)
        
        # Determine status
        if step.status == 'completed':
            status = "completed"
        elif failed_groups >= len(groups):
            status = "failed"
        elif running_groups > 0 or completed_groups > 0:
            status = "running"
        else:
            status = "pending"
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('step', step_id, fallback=30.0)
        
        # Estimate times (only for rollout phase, training time is harder to estimate)
        estimated_total_time = total_turns * avg_turn_time if avg_turn_time else None
        estimated_remaining_time = (total_turns - completed_turns) * avg_turn_time if avg_turn_time else None
        
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
        
        for group in groups:
            # Get group's total and completed turns
            group_rollouts = list_rollouts_by_group(self.session, group.id)
            for rollout in group_rollouts:
                max_turns = rollout.max_turns or 20
                total_turns += max_turns
                
                if rollout.status == 'completed':
                    completed_turns += rollout.num_turns or max_turns
                elif rollout.status == 'running':
                    completed_turns += rollout.current_turn or 0
            
            if group.status == 'completed':
                completed_groups += 1
            elif group.status == 'running':
                running_groups += 1
            elif group.status == 'failed':
                failed_groups += 1
        
        progress_percent = (completed_turns / total_turns * 100.0) if total_turns > 0 else 0.0
        
        # Determine status
        if completed_groups >= len(groups):
            status = "completed"
        elif failed_groups >= len(groups):
            status = "failed"
        elif running_groups > 0 or completed_groups > 0:
            status = "running"
        else:
            status = "pending"
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('eval', eval_id, fallback=30.0)
        
        # Estimate times
        estimated_total_time = total_turns * avg_turn_time if avg_turn_time else None
        estimated_remaining_time = (total_turns - completed_turns) * avg_turn_time if avg_turn_time else None
        
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
        
        for group in groups:
            # Get group's total and completed turns
            group_rollouts = list_rollouts_by_group(self.session, group.id)
            for rollout in group_rollouts:
                max_turns = rollout.max_turns or 20
                total_turns += max_turns
                
                if rollout.status == 'completed':
                    completed_turns += rollout.num_turns or max_turns
                elif rollout.status == 'running':
                    completed_turns += rollout.current_turn or 0
            
            if group.status == 'completed':
                completed_groups += 1
            elif group.status == 'running':
                running_groups += 1
            elif group.status == 'failed':
                failed_groups += 1
        
        progress_percent = (completed_turns / total_turns * 100.0) if total_turns > 0 else 0.0
        
        # Determine status
        if completed_groups >= len(groups):
            status = "completed"
        elif failed_groups >= len(groups):
            status = "failed"
        elif running_groups > 0 or completed_groups > 0:
            status = "running"
        else:
            status = "pending"
        
        # Calculate average turn time
        avg_turn_time = self._calculate_avg_turn_time('baseline', baseline_id, fallback=30.0)
        
        # Estimate times
        estimated_total_time = total_turns * avg_turn_time if avg_turn_time else None
        estimated_remaining_time = (total_turns - completed_turns) * avg_turn_time if avg_turn_time else None
        
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
        Update training progress based on baseline, steps, and evals.
        
        Training progress formula:
        - Baseline: 10% (if exists)
        - Steps: 80% (N steps, weighted equally)
        - Evals: 10% (N/eval_every evals, weighted equally)
        
        Args:
            training_id: Database training ID
            
        Returns:
            ProgressStats with updated progress information
        """
        training = get_training(self.session, training_id)
        if not training:
            logger.warning(f"Training {training_id} not found")
            return ProgressStats(status="failed")
        
        total_progress = 0.0
        total_turns = 0
        completed_turns = 0
        
        # Baseline progress (10%)
        if training.baselines:
            baseline = training.baselines[0]  # Should only be one baseline
            if baseline.progress_percent is not None:
                total_progress += baseline.progress_percent * 0.1
            
            # Count baseline turns
            baseline_groups = list_groups_by_baseline(self.session, baseline.id)
            for group in baseline_groups:
                group_rollouts = list_rollouts_by_group(self.session, group.id)
                for rollout in group_rollouts:
                    max_turns = rollout.max_turns or 20
                    total_turns += max_turns
                    if rollout.status == 'completed':
                        completed_turns += rollout.num_turns or max_turns
                    elif rollout.status == 'running':
                        completed_turns += rollout.current_turn or 0
        
        # Steps progress (80%)
        if training.steps and training.total_steps:
            step_progress_sum = sum(
                (step.progress_percent or 0.0) for step in training.steps
            )
            avg_step_progress = step_progress_sum / training.total_steps
            total_progress += avg_step_progress * 0.8
            
            # Count step turns
            for step in training.steps:
                step_groups = list_groups_by_step(self.session, step.id)
                for group in step_groups:
                    group_rollouts = list_rollouts_by_group(self.session, group.id)
                    for rollout in group_rollouts:
                        max_turns = rollout.max_turns or 20
                        total_turns += max_turns
                        if rollout.status == 'completed':
                            completed_turns += rollout.num_turns or max_turns
                        elif rollout.status == 'running':
                            completed_turns += rollout.current_turn or 0
        
        # Evals progress (10%)
        if training.evals and training.total_steps:
            # Number of expected evals based on eval_every
            # (assume eval_every is constant for simplicity)
            eval_progress_sum = sum(
                (eval_obj.progress_percent or 0.0) for eval_obj in training.evals
            )
            # Estimate number of evals based on total_steps and current eval count
            # For simplicity, weight by actual evals completed
            if training.evals:
                avg_eval_progress = eval_progress_sum / len(training.evals)
                total_progress += avg_eval_progress * 0.1
            
            # Count eval turns
            for eval_obj in training.evals:
                eval_groups = list_groups_by_eval(self.session, eval_obj.id)
                for group in eval_groups:
                    group_rollouts = list_rollouts_by_group(self.session, group.id)
                    for rollout in group_rollouts:
                        max_turns = rollout.max_turns or 20
                        total_turns += max_turns
                        if rollout.status == 'completed':
                            completed_turns += rollout.num_turns or max_turns
                        elif rollout.status == 'running':
                            completed_turns += rollout.current_turn or 0
        
        progress_percent = min(100.0, total_progress)
        
        # Determine status
        status = training.status
        if progress_percent >= 100.0:
            status = "completed"
        elif progress_percent > 0:
            status = "running"
        
        # Calculate average turn time across all completed turns
        avg_turn_time = self._calculate_avg_turn_time('training', training_id, fallback=30.0)
        
        # Estimate times
        estimated_total_time = total_turns * avg_turn_time if avg_turn_time else None
        estimated_remaining_time = (total_turns - completed_turns) * avg_turn_time if avg_turn_time else None
        
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

