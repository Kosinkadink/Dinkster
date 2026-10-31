"""Persist the operator's produced-asset size limit independently of eligibility."""

import sqlalchemy as sa
from alembic import op

revision = "0006_cleanup_policy"
down_revision = "0005_retention_completeness"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asset_cleanup_policy",
        sa.Column("singleton", sa.Integer(), primary_key=True),
        sa.Column("limit_bytes", sa.Integer(), nullable=True),
        sa.CheckConstraint("singleton = 1"),
        sa.CheckConstraint("limit_bytes IS NULL OR limit_bytes > 0"),
    )
    op.execute("INSERT INTO asset_cleanup_policy VALUES (1, 21474836480)")


def downgrade() -> None:
    op.drop_table("asset_cleanup_policy")
