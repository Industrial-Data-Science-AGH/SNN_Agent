"""Controlled failures raised before a session can start.

Every failure carries a stable ``code`` so the backend can map it onto an Error
payload without parsing prose, and so tests assert on the code rather than on a
message that will be reworded.
"""

from __future__ import annotations


class RuntimeLoadError(ValueError):
    """A model package was rejected. The runtime never starts half-loaded."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class RuntimeStateError(RuntimeError):
    """A call arrived in the wrong order, e.g. step() before load()."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
