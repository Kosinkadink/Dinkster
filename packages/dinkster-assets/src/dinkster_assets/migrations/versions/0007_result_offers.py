"""Persist immutable output identities before a client claims their ownership."""

import sqlalchemy as sa
from alembic import op

revision = "0007_result_offers"
down_revision = "0006_cleanup_policy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("asset_retentions", sa.Column("source_token", sa.Text(), nullable=True))
    op.create_table(
        "asset_result_offers",
        sa.Column("token", sa.Text(), primary_key=True),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("job_ref", sa.Text(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("node_id", sa.Text(), nullable=False),
        sa.Column("output_id", sa.Text(), nullable=False),
        sa.Column("publication", sa.Integer(), nullable=False),
        sa.Column("digests", sa.Text(), nullable=False),
        sa.Column("complete", sa.Integer(), nullable=False),
        sa.Column("claimed", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("asset_result_offers_source", "asset_result_offers", ["scope", "job_ref"])


def downgrade() -> None:
    op.drop_index("asset_result_offers_source", table_name="asset_result_offers")
    op.drop_table("asset_result_offers")
    op.drop_column("asset_retentions", "source_token")
