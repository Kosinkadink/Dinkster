"""Persist explicit scoped owners of retained asset values."""

import sqlalchemy as sa
from alembic import op

revision = "0003_asset_retentions"
down_revision = "0002_asset_observations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asset_retentions",
        sa.Column("scope", sa.Text(), primary_key=True),
        sa.Column("owner_kind", sa.Text(), primary_key=True),
        sa.Column("owner_id", sa.Text(), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("digests", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("asset_retentions")
