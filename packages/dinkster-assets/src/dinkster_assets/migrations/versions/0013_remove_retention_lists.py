"""Remove per-window retained asset lists."""

from alembic import op

revision = "0013_remove_retention_lists"
down_revision = "0012_retention_departure"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("asset_retention_quarantine")
    op.drop_table("asset_retention_members")
    op.drop_table("asset_retention_clients")
    op.drop_table("asset_retention_universe")
    op.drop_table("asset_retentions")


def downgrade() -> None:
    raise NotImplementedError("retention lists cannot be restored after removal")
