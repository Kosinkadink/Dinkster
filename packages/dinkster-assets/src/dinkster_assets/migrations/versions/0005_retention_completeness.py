"""Distinguish declared complete owner manifests from unknown references."""

import sqlalchemy as sa
from alembic import op

revision = "0005_retention_completeness"
down_revision = "0004_derivation_references"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "asset_retentions",
        sa.Column("complete", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("asset_retentions", "complete")
