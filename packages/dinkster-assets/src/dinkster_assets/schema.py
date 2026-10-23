"""Asset record schema and Alembic configuration, independent of record CRUD."""

from pathlib import Path

import sqlalchemy as sa
from alembic.config import Config

metadata = sa.MetaData()
records = sa.Table(
    "records",
    metadata,
    sa.Column("id", sa.Text, primary_key=True, nullable=True),
    sa.Column("scope", sa.Text, nullable=False),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("digest", sa.Text, nullable=False),
    sa.Column("media_type", sa.Text, nullable=False),
    sa.Column("labels", sa.Text, nullable=False),
    sa.Column("folder", sa.Text, nullable=False),
    sa.Column("created", sa.REAL[float], nullable=False),
    sa.Column("modified", sa.REAL[float], nullable=False),
    sa.Column("revision", sa.Integer, nullable=False),
)
sa.Index("records_scope_modified", records.c.scope, records.c.modified.desc(), records.c.id.desc())
sa.Table(
    "media_grants",
    metadata,
    sa.Column("scope", sa.Text, primary_key=True),
    sa.Column("digest", sa.Text, primary_key=True),
    sa.Column("kind", sa.Text, primary_key=True),
    sa.Column("media_type", sa.Text, nullable=False),
    sa.Column("extension", sa.Text, nullable=False),
    sa.Column("byte_size", sa.Integer, nullable=False),
)
sa.Table(
    "asset_dependencies",
    metadata,
    sa.Column("digest", sa.Text, primary_key=True, nullable=True),
    sa.Column("manifest", sa.Text, nullable=False),
)
sa.Table(
    "asset_derivations",
    metadata,
    sa.Column("cache_key", sa.Text, primary_key=True, nullable=True),
    sa.Column("manifest", sa.Text, nullable=False),
)


def migration_config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).with_name("migrations")))
    return config
