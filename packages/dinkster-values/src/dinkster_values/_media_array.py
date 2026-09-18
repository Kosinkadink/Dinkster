"""Optional numpy runtime carrier; imported only when an array is annotated."""

import numpy as np


class MediaArray(np.ndarray):
    _dinkster_media: dict[str, object]
