"""Stateful Lu.i simulator and the digital-twin contract (task T02).

P1 scope, implemented here: accept or reject a model package before anything
starts. The integrator itself arrives in P2. Nothing in this package imports
torch, so it stays runnable inside the W0 environment.
"""

from .errors import RuntimeLoadError, RuntimeStateError
from .manifest import LoadedModel, PortBinding, load_manifest
from .runtime import LuiRuntime

__all__ = [
    "LoadedModel",
    "LuiRuntime",
    "PortBinding",
    "RuntimeLoadError",
    "RuntimeStateError",
    "load_manifest",
]
