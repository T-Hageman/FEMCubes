from __future__ import annotations

from typing import Any, Dict
from dolfinx import fem


class _MeshEnd:
    """Terminal mix-in that absorbs the cooperative ``super().__init__`` call.

    Because `Mesh` uses cooperative multiple inheritance, the MRO chain
    must end with a class whose ``__init__`` does *not* call ``super()``.
    """

    def __init__(self, params: Any) -> None:
        pass
