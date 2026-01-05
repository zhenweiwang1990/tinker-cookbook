"""add_model_response_to_turn

Revision ID: e8e01da37736
Revises: e3988accc2ef
Create Date: 2026-01-04 01:21:59.591977

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8e01da37736'
down_revision: Union[str, Sequence[str], None] = 'e3988accc2ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add model_response column to turn table to store full LLM output."""
    op.add_column('turn', sa.Column('model_response', sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove model_response column from turn table."""
    op.drop_column('turn', 'model_response')
