"""merge_branches

Revision ID: 9be8671e04bc
Revises: add_trajectory_data_json_direct, add_progress_tracking
Create Date: 2026-01-06 21:03:23.879341

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9be8671e04bc'
down_revision: Union[str, Sequence[str], None] = ('add_trajectory_data_json_direct', 'add_progress_tracking')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
