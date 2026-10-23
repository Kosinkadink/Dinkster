"""Run catalog revisions on an application connection or an explicit CLI catalog."""

from typing import Literal

import sqlalchemy as sa
from alembic import context
from dinkster_assets.schema import metadata
from sqlalchemy.engine import Connection


def render_item(kind: str, item: object, _context: object) -> str | Literal[False]:
    if kind == "type" and isinstance(item, sa.REAL):
        return "sa.REAL[float]()"
    return False


def run(connection: Connection) -> None:
    if not connection.in_transaction():
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    context.configure(connection=connection, target_metadata=metadata, render_item=render_item)
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if connection is not None:
    run(connection)
else:
    catalog = context.get_x_argument(as_dictionary=True).get("catalog")
    if not catalog:
        raise ValueError("pass -x catalog=/path/to/library.sqlite")
    engine = sa.create_engine(sa.URL.create("sqlite", database=catalog))
    try:
        with engine.connect() as connection:
            run(connection)
            connection.commit()
    finally:
        engine.dispose()
