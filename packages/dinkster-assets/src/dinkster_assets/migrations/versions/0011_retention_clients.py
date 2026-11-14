"""Persist server-admitted clients and their revision-bound holder census."""

import sqlalchemy as sa
from alembic import op

revision = "0011_retention_clients"
down_revision = "0010_result_renditions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "asset_retention_universe",
        sa.Column("singleton", sa.Integer(), primary_key=True),
        sa.Column("closed", sa.Integer(), nullable=False),
        sa.Column("legacy", sa.Integer(), nullable=False),
        sa.CheckConstraint("singleton = 1"),
        sa.CheckConstraint("closed IN (0, 1) AND legacy IN (0, 1)"),
    )
    op.execute("INSERT INTO asset_retention_universe VALUES (1, 0, 0)")
    op.create_table(
        "asset_retention_clients",
        sa.Column("client_id", sa.Text(), primary_key=True),
        sa.Column("principal_id", sa.Text(), nullable=False),
        sa.Column("generation", sa.Text(), nullable=False, unique=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("connected", sa.Integer(), nullable=False),
        sa.Column("acknowledged", sa.Integer(), nullable=False),
        sa.Column("pending_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.CheckConstraint("pending_requests >= 0"),
        sa.CheckConstraint("revision > 0"),
        sa.CheckConstraint("connected IN (0, 1) AND acknowledged IN (0, 1)"),
    )
    op.create_table(
        "asset_retention_members",
        sa.Column("scope", sa.Text(), primary_key=True),
        sa.Column("owner_kind", sa.Text(), primary_key=True),
        sa.Column("owner_id", sa.Text(), primary_key=True),
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("holder_class", sa.Integer(), nullable=False),
        sa.CheckConstraint("holder_class BETWEEN 1 AND 8"),
    )


def downgrade() -> None:
    op.drop_table("asset_retention_members")
    op.drop_table("asset_retention_clients")
    op.drop_table("asset_retention_universe")
