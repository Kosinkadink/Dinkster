"""First-party text manipulation and mounted text output."""

from .save_text import SaveText
from .text import TEXT_NODES

NODES = (*TEXT_NODES, SaveText)

__all__ = ["NODES"]
