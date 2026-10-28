"""Preserve scoped asset names, origins and observed virtual locations."""

import sqlalchemy as sa
from alembic import op

revision = "0002_asset_observations"
down_revision = "0001_asset_records"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asset_observations",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("digest", sa.Text(), nullable=False),
        sa.Column("original_name", sa.Text(), nullable=False),
        sa.Column("stored_name", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("virtual_path", sa.Text(), nullable=False),
        sa.Column("origin", sa.Text(), nullable=False),
        sa.Column("project", sa.Text(), nullable=False),
        sa.Column("created", sa.REAL[float](), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "asset_observations_scope_digest", "asset_observations", ["scope", "digest", "created"]
    )


def downgrade() -> None:
    op.drop_index("asset_observations_scope_digest", table_name="asset_observations")
    op.drop_table("asset_observations")
