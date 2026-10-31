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
    sa.Column("asset_digests", sa.Text, nullable=True),
)
asset_observations = sa.Table(
    "asset_observations",
    metadata,
    sa.Column("id", sa.Text, primary_key=True, nullable=False),
    sa.Column("scope", sa.Text, nullable=False),
    sa.Column("digest", sa.Text, nullable=False),
    sa.Column("original_name", sa.Text, nullable=False),
    sa.Column("stored_name", sa.Text, nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("media_type", sa.Text, nullable=False),
    sa.Column("byte_size", sa.Integer, nullable=False),
    sa.Column("virtual_path", sa.Text, nullable=False),
    sa.Column("origin", sa.Text, nullable=False),
    sa.Column("project", sa.Text, nullable=False),
    sa.Column("created", sa.REAL[float], nullable=False),
)
sa.Index(
    "asset_observations_scope_digest",
    asset_observations.c.scope,
    asset_observations.c.digest,
    asset_observations.c.created,
)
sa.Table(
    "asset_retentions",
    metadata,
    sa.Column("scope", sa.Text, primary_key=True),
    sa.Column("owner_kind", sa.Text, primary_key=True),
    sa.Column("owner_id", sa.Text, primary_key=True),
    sa.Column("revision", sa.Integer, nullable=False),
    sa.Column("digests", sa.Text, nullable=False),
    sa.Column("complete", sa.Integer, nullable=False, server_default="0"),
    sa.Column("source_token", sa.Text, nullable=True),
)
sa.Table(
    "asset_cleanup_policy",
    metadata,
    sa.Column("singleton", sa.Integer, primary_key=True),
    sa.Column("limit_bytes", sa.Integer, nullable=True),
    sa.CheckConstraint("singleton = 1"),
    sa.CheckConstraint("limit_bytes IS NULL OR limit_bytes > 0"),
)
sa.Table(
    "asset_result_offers",
    metadata,
    sa.Column("token", sa.Text, primary_key=True),
    sa.Column("scope", sa.Text, nullable=False),
    sa.Column("job_ref", sa.Text, nullable=False),
    sa.Column("attempt", sa.Integer, nullable=False),
    sa.Column("node_id", sa.Text, nullable=False),
    sa.Column("output_id", sa.Text, nullable=False),
    sa.Column("publication", sa.Integer, nullable=False),
    sa.Column("digests", sa.Text, nullable=False),
    sa.Column("complete", sa.Integer, nullable=False),
    sa.Column("claimed", sa.Integer, nullable=False, server_default="0"),
)
sa.Index(
    "asset_result_offers_source",
    metadata.tables["asset_result_offers"].c.scope,
    metadata.tables["asset_result_offers"].c.job_ref,
)
sa.Table(
    "asset_terminal_results",
    metadata,
    sa.Column("scope", sa.Text, primary_key=True),
    sa.Column("job_ref", sa.Text, primary_key=True),
    sa.Column("manifest", sa.Text, nullable=False),
)


def migration_config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).with_name("migrations")))
    return config
