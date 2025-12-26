import asyncio
import json
import logging
import os
import re
from typing import Any, Literal

import tinker

from tinker_cookbook.utils.file_utils import read_jsonl
from tinker_cookbook.utils.trace import scope, update_scope_context

CHECKPOINTS_BASE_NAME = "checkpoints.jsonl"

logger = logging.getLogger(__name__)


@scope
def load_checkpoints_file(log_dir: str) -> list[dict[str, Any]]:
    checkpoint_path = os.path.join(log_dir, CHECKPOINTS_BASE_NAME)
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoints found at {checkpoint_path}")
        return []

    logger.info(f"Reading checkpoints from {checkpoint_path}")
    update_scope_context({"checkpoint_path": checkpoint_path})
    return read_jsonl(checkpoint_path)


@scope
def get_last_checkpoint(log_dir: str, required_key: str = "state_path") -> dict[str, Any] | None:
    """
    Get the last checkpoint from the checkpoints.jsonl file in the specified log directory.

    Args:
        log_dir: The directory to check.
        required_key: The key to check for in the checkpoint.
            We might save partial checkpoints (e.g. sampler) in the same file,
            so we need to filter to the rows that have a fully-resumable checkpoint.

    Returns:
        The last checkpoint, or None if no checkpoint is found.
    """
    checkpoints = load_checkpoints_file(log_dir)
    checkpoints_with_key = [c for c in checkpoints if required_key in c]
    if checkpoints_with_key:
        logger.info(
            f"Found {len(checkpoints_with_key)} valid checkpoints with key '{required_key}' in {log_dir}"
        )
        logger.info(f"Using last checkpoint: {checkpoints_with_key[-1]}")
        return checkpoints_with_key[-1]
    else:
        logger.info(f"No checkpoints found with key {required_key} in {log_dir}")
        return None


@scope
async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    futures = {}
    try:
        if kind in ["state", "both"]:
            futures["state"] = await training_client.save_state_async(name)
        if kind in ["sampler", "both"]:
            futures["sampler"] = await training_client.save_weights_for_sampler_async(name)

        results = {k: await v.result_async() for k, v in futures.items()}
        paths = {k + "_path": v.path for k, v in results.items()}
    except tinker.ConflictError as e:
        # Handle the case where checkpoint already exists (e.g., from a previous run)
        # For temporary checkpoints (_tmp suffix), we can reuse the existing checkpoint
        if name.endswith("_tmp"):
            logger.warning(
                f"Checkpoint '{name}' already exists. Attempting to reuse existing checkpoint paths."
            )
            
            # First, check if we have a previous checkpoint with this name in checkpoints.jsonl
            checkpoints = load_checkpoints_file(log_path)
            existing_checkpoint = next((c for c in checkpoints if c.get("name") == name), None)
            
            if existing_checkpoint:
                # Use the existing checkpoint paths
                paths = {}
                if kind in ["state", "both"] and "state_path" in existing_checkpoint:
                    paths["state_path"] = existing_checkpoint["state_path"]
                if kind in ["sampler", "both"] and "sampler_path" in existing_checkpoint:
                    paths["sampler_path"] = existing_checkpoint["sampler_path"]
                logger.info(f"Reusing existing checkpoint paths: {paths}")
            else:
                # Try to extract model identifier from error message and construct path
                # Error format: "Checkpoint 'name' already exists for model {model_id} in {storage_type}."
                error_str = str(e)
                model_match = re.search(r"for model ([^\s]+)", error_str)
                if model_match:
                    model_id = model_match.group(1)
                    paths = {}
                    if kind in ["state", "both"]:
                        paths["state_path"] = f"tinker://{model_id}/state/{name}"
                    if kind in ["sampler", "both"]:
                        paths["sampler_path"] = f"tinker://{model_id}/sampler_weights/{name}"
                    logger.info(f"Constructed checkpoint paths from error message: {paths}")
                else:
                    # Couldn't extract model ID, re-raise the error
                    logger.error(
                        f"ConflictError for temporary checkpoint '{name}'. "
                        f"Could not extract model identifier from error: {e}. "
                        f"Please ensure the checkpoint name is unique or delete the existing checkpoint."
                    )
                    raise
            
            # For temporary checkpoints that already exist, skip writing to checkpoints.jsonl
            # to avoid duplicate entries
            if paths:
                update_scope_context(paths)
                logger.info(f"Reused existing checkpoints: {paths}")
                return paths
            else:
                # If we couldn't construct paths, re-raise
                raise
        else:
            # For non-temporary checkpoints, re-raise the error
            raise
    
    update_scope_context(paths)
    logger.info(f"Saved checkpoints: {paths}")
    full_dict = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths


@scope
def save_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    return asyncio.run(
        save_checkpoint_async(
            training_client, name=name, log_path=log_path, kind=kind, loop_state=loop_state
        )
    )
