"""add_progress_tracking_fields

Revision ID: add_progress_tracking
Revises: e8e01da37736
Create Date: 2025-01-06

Adds progress tracking fields to support turn-based progress calculation
and estimated time tracking for all entities:
- Training, Baseline, Eval, Step, Group
- avg_turn_time: Average time per turn in seconds
- estimated_total_time: Estimated total time in seconds
- estimated_remaining_time: Estimated remaining time in seconds
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_progress_tracking'
down_revision = 'e8e01da37736'
branch_labels = None
depends_on = None


def upgrade():
    """Add progress tracking fields to all entities."""
    
    # Training table
    op.add_column('training', sa.Column('avg_turn_time', sa.Float(), nullable=True))
    op.add_column('training', sa.Column('estimated_total_time', sa.Float(), nullable=True))
    op.add_column('training', sa.Column('estimated_remaining_time', sa.Float(), nullable=True))
    
    # Baseline table
    op.add_column('baseline', sa.Column('avg_turn_time', sa.Float(), nullable=True))
    op.add_column('baseline', sa.Column('estimated_total_time', sa.Float(), nullable=True))
    op.add_column('baseline', sa.Column('estimated_remaining_time', sa.Float(), nullable=True))
    
    # Eval table
    op.add_column('eval', sa.Column('avg_turn_time', sa.Float(), nullable=True))
    op.add_column('eval', sa.Column('estimated_total_time', sa.Float(), nullable=True))
    op.add_column('eval', sa.Column('estimated_remaining_time', sa.Float(), nullable=True))
    
    # Step table
    op.add_column('step', sa.Column('avg_turn_time', sa.Float(), nullable=True))
    op.add_column('step', sa.Column('estimated_total_time', sa.Float(), nullable=True))
    op.add_column('step', sa.Column('estimated_remaining_time', sa.Float(), nullable=True))
    
    # Group table
    op.add_column('group', sa.Column('avg_turn_time', sa.Float(), nullable=True))
    op.add_column('group', sa.Column('estimated_total_time', sa.Float(), nullable=True))
    op.add_column('group', sa.Column('estimated_remaining_time', sa.Float(), nullable=True))


def downgrade():
    """Remove progress tracking fields from all entities."""
    
    # Training table
    op.drop_column('training', 'estimated_remaining_time')
    op.drop_column('training', 'estimated_total_time')
    op.drop_column('training', 'avg_turn_time')
    
    # Baseline table
    op.drop_column('baseline', 'estimated_remaining_time')
    op.drop_column('baseline', 'estimated_total_time')
    op.drop_column('baseline', 'avg_turn_time')
    
    # Eval table
    op.drop_column('eval', 'estimated_remaining_time')
    op.drop_column('eval', 'estimated_total_time')
    op.drop_column('eval', 'avg_turn_time')
    
    # Step table
    op.drop_column('step', 'estimated_remaining_time')
    op.drop_column('step', 'estimated_total_time')
    op.drop_column('step', 'avg_turn_time')
    
    # Group table
    op.drop_column('group', 'estimated_remaining_time')
    op.drop_column('group', 'estimated_total_time')
    op.drop_column('group', 'avg_turn_time')

