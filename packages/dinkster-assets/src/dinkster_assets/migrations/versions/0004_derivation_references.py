"""Record explicit external identities owned by cached derivations."""

import sqlalchemy as sa
from alembic import op

revision = "0004_derivation_references"
down_revision = "0003_asset_retentions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("asset_derivations", sa.Column("asset_digests", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("asset_derivations", "asset_digests")
