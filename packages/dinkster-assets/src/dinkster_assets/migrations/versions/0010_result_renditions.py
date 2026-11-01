"""Retain derived document resources under their exact result source."""

import sqlalchemy as sa
from alembic import op

revision = "0010_result_renditions"
down_revision = "0009_artifact_offers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asset_result_renditions",
        sa.Column("scope", sa.Text(), primary_key=True),
        sa.Column("source_token", sa.Text(), primary_key=True),
        sa.Column("cache_key", sa.Text(), primary_key=True),
        sa.Column("document_digest", sa.Text(), nullable=False),
        sa.Column("digests", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("asset_result_renditions")
