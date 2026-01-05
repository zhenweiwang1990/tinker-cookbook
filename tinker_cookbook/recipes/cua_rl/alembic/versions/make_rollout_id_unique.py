"""Make rollout_id unique

Revision ID: make_rollout_id_unique
Revises: e8e01da37736
Create Date: 2026-01-04 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'make_rollout_id_unique'
down_revision: Union[str, Sequence[str], None] = 'e8e01da37736'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Make rollout_id unique by dropping and recreating the index with unique constraint."""
    # Drop existing non-unique index
    try:
        op.drop_index(op.f('ix_rollout_rollout_id'), table_name='rollout')
    except Exception:
        # Index might not exist or have different name
        pass
    
    # Create unique index
    op.create_index(op.f('ix_rollout_rollout_id'), 'rollout', ['rollout_id'], unique=True)


def downgrade() -> None:
    """Revert rollout_id to non-unique."""
    # Drop unique index
    try:
        op.drop_index(op.f('ix_rollout_rollout_id'), table_name='rollout')
    except Exception:
        pass
    
    # Create non-unique index
    op.create_index(op.f('ix_rollout_rollout_id'), 'rollout', ['rollout_id'], unique=False)

