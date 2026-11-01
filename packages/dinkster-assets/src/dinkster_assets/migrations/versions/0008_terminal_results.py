"""Retain immutable terminal publication sets independently of execution history."""

import sqlalchemy as sa
from alembic import op

revision = "0008_terminal_results"
down_revision = "0007_result_offers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asset_terminal_results",
        sa.Column("scope", sa.Text(), primary_key=True),
        sa.Column("job_ref", sa.Text(), primary_key=True),
        sa.Column("manifest", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("asset_terminal_results")
