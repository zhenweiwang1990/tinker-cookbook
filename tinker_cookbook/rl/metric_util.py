import asyncio
import itertools
import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, TrajectoryGroup
from tinker_cookbook.utils.misc_utils import all_same, dict_mean
from tinker_cookbook.utils import logtree
from tinker_cookbook.completers import TokenCompleter

logger = logging.getLogger(__name__)


def _compute_by_group_metrics(trajectory_groups_P: List[TrajectoryGroup], good_thresh: float = 0.5):
    n_groups = len(trajectory_groups_P)
    n_mixed = n_good = n_bad = 0
    for tg in trajectory_groups_P:
        grp_rewards = tg.get_total_rewards()
        if all_same(grp_rewards):
            if grp_rewards[0] >= good_thresh:
                n_good += 1
            else:
                n_bad += 1
        else:
            n_mixed += 1
    return {
        "by_group/frac_mixed": n_mixed / n_groups,
        "by_group/frac_all_good": n_good / n_groups,
        "by_group/frac_all_bad": n_bad / n_groups,
    }


def compute_trajectory_metrics(
    trajectory_groups_P: List[TrajectoryGroup], taglist_P: List[list[str]]
) -> Dict[str, float]:
    tag2trajgroups = defaultdict(list)
    for taglist, trajectory_group in zip(taglist_P, trajectory_groups_P):
        for tag in taglist:
            tag2trajgroups[tag].append(trajectory_group)
    out = {}
    have_nontrivial_tags = any(
        len(trajgroups) < len(trajectory_groups_P) for trajgroups in tag2trajgroups.values()
    )  # check if any tag gives us a strict subset of the full trajectory groups
    if have_nontrivial_tags:
        for tag, trajectory_groups in tag2trajgroups.items():
            prefixed_metrics = {
                f"env/{tag}/{k}": v
                for k, v in _compute_trajectory_metrics(trajectory_groups).items()
            }
            out.update(prefixed_metrics)
    out.update(
        {f"env/all/{k}": v for k, v in _compute_trajectory_metrics(trajectory_groups_P).items()}
    )
    return out


def _compute_trajectory_metrics(trajectory_groups_P: List[TrajectoryGroup]) -> Dict[str, float]:
    """Compute metrics for the trajectory groups."""
    flat_trajs_PG = [traj for tg in trajectory_groups_P for traj in tg.trajectories_G]
    ac_tokens_by_turn = [
        len(transition.ac.tokens) for traj in flat_trajs_PG for transition in traj.transitions
    ]
    ob_tokens_by_turn = [
        transition.ob.length for traj in flat_trajs_PG for transition in traj.transitions
    ]
    turns_by_trajectory = [len(traj.transitions) for traj in flat_trajs_PG]
    total_turns = sum(turns_by_trajectory)
    num_episodes = len(flat_trajs_PG)
    # Compute metrics with guards against division by zero
    metrics = {
        "ac_tokens_per_turn": sum(ac_tokens_by_turn) / total_turns if total_turns > 0 else 0.0,
        "ob_tokens_per_turn": sum(ob_tokens_by_turn) / total_turns if total_turns > 0 else 0.0,
        "turns_per_episode": total_turns / num_episodes if num_episodes > 0 else 0.0,
        "total_episodes": num_episodes,
        "total_turns": total_turns,
        "total_ac_tokens": sum(ac_tokens_by_turn),
        "total_ob_tokens": sum(ob_tokens_by_turn),
    }
    all_rewards = [reward for tg in trajectory_groups_P for reward in tg.get_total_rewards()]
    metrics["reward/total"] = np.mean(all_rewards).item() if len(all_rewards) > 0 else 0.0
    # Per-transition metrics
    transition_metrics = [
        transition.metrics
        for tg in trajectory_groups_P
        for traj in tg.trajectories_G
        for transition in traj.transitions
    ]
    traj_metrics = [metrics for tg in trajectory_groups_P for metrics in tg.metrics_G]
    metrics.update(dict_mean(transition_metrics + traj_metrics))
    # combine traj_metrics and transition_metrics in case there's some key
    # (like format error) that appears in the per-step metrics for some envs
    # but the compute_group_rewards metric for other envs.
    metrics.update(_compute_by_group_metrics(trajectory_groups_P))
    return metrics


def dataset_to_env_group_builders(dataset: RLDataset) -> list[EnvGroupBuilder]:
    """
    Get the whole dataset as a list of env group builders.
    """
    return list(itertools.chain(*[dataset.get_batch(i) for i in range(len(dataset))]))


class RLTestSetEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        dataset: RLDataset,
        max_tokens: int,
        name: str = "test",
        num_groups_to_log: int = 4,
    ):
        self.env_group_builders_P = dataset_to_env_group_builders(dataset)
        self.max_tokens = max_tokens
        self.name = name
        self.num_groups_to_log = num_groups_to_log

    async def eval_token_completer(self, policy: TokenCompleter) -> dict[str, float]:
        async def run_group_rollout(builder, i):
            enable_logging = i < self.num_groups_to_log
            with logtree.optional_enable_logging(enable=enable_logging):
                return await do_group_rollout(builder, policy)

        trajectory_groups_P = await asyncio.gather(
            *[run_group_rollout(builder, i) for i, builder in enumerate(self.env_group_builders_P)]
        )
        taglist_P = [builder.logging_tags() for builder in self.env_group_builders_P]
        metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)

        # Print comprehensive evaluation results table
        self._print_evaluation_results_table(trajectory_groups_P)

        metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return metrics
    
    def _print_evaluation_results_table(self, trajectory_groups_P: List[TrajectoryGroup]):
        """Print a comprehensive evaluation results table."""
        # Collect results from all trajectory groups
        all_results = []
        for tg in trajectory_groups_P:
            for traj in tg.trajectories_G:
                # Extract metrics from trajectory transitions
                # Metrics are stored in transition.metrics
                task_success = False
                task_completed = False
                num_turns = len(traj.transitions)
                reward = 0.0
                rollout_time = 0.0
                
                # Get metrics from the last transition (which contains the final metrics)
                if traj.transitions:
                    last_transition = traj.transitions[-1]
                    if hasattr(last_transition, 'metrics') and last_transition.metrics:
                        task_success = bool(last_transition.metrics.get("task_success", False))
                        task_completed = bool(last_transition.metrics.get("task_completed", False))
                        num_turns = int(last_transition.metrics.get("num_turns", num_turns))
                        # rollout_time might not be in metrics, use 0.0 as fallback
                        rollout_time = float(last_transition.metrics.get("rollout_time", 0.0))
                    
                    # Get reward from transition
                    if hasattr(last_transition, 'reward'):
                        reward = float(last_transition.reward)
                
                all_results.append({
                    "success": task_success,
                    "completed": task_completed,
                    "turns": num_turns,
                    "reward": reward,
                    "time": rollout_time,
                })
        
        if not all_results:
            return
        
        # Calculate summary statistics
        total_tasks = len(all_results)
        success_count = sum(1 for r in all_results if r["success"])
        completed_count = sum(1 for r in all_results if r["completed"])
        error_count = total_tasks - success_count
        success_rate = (success_count / total_tasks * 100) if total_tasks > 0 else 0.0
        total_turns = sum(r["turns"] for r in all_results)
        avg_turns = total_turns / total_tasks if total_tasks > 0 else 0.0
        total_time = sum(r["time"] for r in all_results)
        avg_time = total_time / total_tasks if total_tasks > 0 else 0.0
        avg_reward = sum(r["reward"] for r in all_results) / total_tasks if total_tasks > 0 else 0.0
        
        # Print evaluation results table
        logger.info("")
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("=" * 80)
        
        # Print summary statistics
        enable_color = True  # You can make this configurable
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        
        success_str = f"{GREEN}✓{RESET}" if success_count > 0 else f"{RED}✗{RESET}"
        
        logger.info(f"Total Tasks: {total_tasks}")
        logger.info(f"Successful: {success_str} {success_count}/{total_tasks} ({success_rate:.1f}%)")
        logger.info(f"Completed: {completed_count}/{total_tasks} ({completed_count/total_tasks*100:.1f}%)")
        logger.info(f"Failed: {error_count}/{total_tasks} ({error_count/total_tasks*100:.1f}%)")
        logger.info(f"Total Turns: {total_turns}")
        logger.info(f"Average Turns per Task: {avg_turns:.2f}")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(f"Average Time per Task: {avg_time:.2f}s")
        logger.info(f"Average Reward: {avg_reward:.4f}")
        logger.info("=" * 80)
        
        # Print detailed table for each task
        logger.info("")
        logger.info("DETAILED TASK RESULTS")
        logger.info("=" * 80)
        header = f"{'#':<4} {'Success':<8} {'Completed':<10} {'Turns':<7} {'Reward':<8} {'Time (s)':<10}"
        logger.info(header)
        logger.info("-" * 80)
        
        for i, result in enumerate(all_results):
            success_str = f"{GREEN}✓{RESET}" if result["success"] else f"{RED}✗{RESET}"
            completed_str = f"{GREEN}✓{RESET}" if result["completed"] else f"{RED}✗{RESET}"
            
            row = (
                f"{i+1:<4} "
                f"{success_str:<12} "  # Account for ANSI codes
                f"{completed_str:<14} "  # Account for ANSI codes
                f"{result['turns']:<7} "
                f"{result['reward']:<8.4f} "
                f"{result['time']:<10.2f}"
            )
            logger.info(row)
        
        logger.info("=" * 80)
        logger.info("")

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        return await self.eval_token_completer(policy)
