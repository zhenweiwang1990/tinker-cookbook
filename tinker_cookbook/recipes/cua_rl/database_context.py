"""
Global database context for training.

This module provides a global context to access database session and training ID
during training without passing them through all function calls.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Global context
_db_session: Optional[Session] = None
_training_id: Optional[int] = None
_baseline_id: Optional[int] = None  # Current baseline ID for baseline evaluation


def set_database_context(session: Session, training_id: int) -> None:
    """Set the global database context."""
    global _db_session, _training_id
    _db_session = session
    _training_id = training_id
    logger.debug(f"Database context set: training_id={training_id}")


def set_baseline_id(baseline_id: Optional[int]) -> None:
    """Set the current baseline ID for baseline evaluation."""
    global _baseline_id
    _baseline_id = baseline_id
    logger.debug(f"Baseline ID set: {baseline_id}")


def get_database_session() -> Optional[Session]:
    """Get the global database session."""
    return _db_session


def get_training_id() -> Optional[int]:
    """Get the global training ID."""
    return _training_id


def get_baseline_id() -> Optional[int]:
    """Get the current baseline ID."""
    return _baseline_id


def clear_database_context() -> None:
    """Clear the global database context."""
    global _db_session, _training_id, _baseline_id
    _db_session = None
    _training_id = None
    _baseline_id = None

