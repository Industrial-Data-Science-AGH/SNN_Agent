"""Stateful Lu.i simulator and the digital-twin contract (task T02).

P1: accept or reject a model package before anything starts (``manifest.py``).
P2: carry one continuous session on it (``integrator.py``, ``decoder.py``,
``runtime.py``). Nothing in this package imports torch or numpy, so it stays
runnable inside the W0 environment, in the pr-gate and on the Pi.
"""

from .decoder import KOfWDecoder
from .errors import RuntimeLoadError, RuntimeStateError
from .integrator import LuiIntegrator, NetworkPlan, NeuronParams, Wire, build_plan
from .manifest import LoadedModel, PortBinding, load_manifest
from .runtime import LuiRuntime

__all__ = [
    "KOfWDecoder",
    "LoadedModel",
    "LuiIntegrator",
    "LuiRuntime",
    "NetworkPlan",
    "NeuronParams",
    "PortBinding",
    "RuntimeLoadError",
    "RuntimeStateError",
    "Wire",
    "build_plan",
    "load_manifest",
]
