"""Bind terminal artifact offers to their exact saved-file descriptors."""

import sqlalchemy as sa
from alembic import op

revision = "0009_artifact_offers"
down_revision = "0008_terminal_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("asset_result_offers", sa.Column("artifact", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("asset_result_offers", "artifact")
