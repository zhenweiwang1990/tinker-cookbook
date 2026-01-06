"""
Main entry point for CUA RL training.

This module re-exports the training functionality from core.train for backward compatibility.
"""

import asyncio
import chz
from tinker_cookbook.recipes.cua_rl.core.train import CLIConfig, cli_main, set_eval_model_path, reset_eval_group_counter


def main_wrapper(cli_config: CLIConfig) -> None:
    """Wrapper function for nested_entrypoint."""
    asyncio.run(cli_main(cli_config))


if __name__ == "__main__":
    chz.nested_entrypoint(main_wrapper, allow_hyphens=True)

__all__ = ["CLIConfig", "cli_main", "set_eval_model_path", "reset_eval_group_counter"]

