"""Preserve departed clients' possible holdings without a permanent census veto."""

import sqlalchemy as sa
from alembic import op

revision = "0012_retention_departure"
down_revision = "0011_retention_clients"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "asset_retention_clients",
        sa.Column("last_activity", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "asset_retention_clients",
        sa.Column("departed", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "asset_retention_quarantine",
        sa.Column("client_id", sa.Text(), primary_key=True),
        sa.Column("digest", sa.Text(), primary_key=True),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.CheckConstraint("size >= 0"),
    )


def downgrade() -> None:
    op.drop_table("asset_retention_quarantine")
    op.drop_column("asset_retention_clients", "departed")
    op.drop_column("asset_retention_clients", "last_activity")
