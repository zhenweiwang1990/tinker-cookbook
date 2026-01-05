"""Add trajectory_data_json to rollout (direct, skip unique index issue)

Revision ID: add_trajectory_data_json_direct
Revises: make_rollout_id_unique
Create Date: 2026-01-05 01:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_trajectory_data_json_direct'
down_revision: Union[str, Sequence[str], None] = 'make_rollout_id_unique'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add trajectory_data_json column to rollout table."""
    # Check if column already exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('rollout')]
    
    if 'trajectory_data_json' not in columns:
        op.add_column('rollout', sa.Column('trajectory_data_json', sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove trajectory_data_json column from rollout table."""
    # Check if column exists before dropping
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('rollout')]
    
    if 'trajectory_data_json' in columns:
        op.drop_column('rollout', 'trajectory_data_json')

