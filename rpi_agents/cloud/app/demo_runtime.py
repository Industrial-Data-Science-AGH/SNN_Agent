"""A stand-in SNN runtime for demos and tests. It is NOT an SNN and says nothing about glass detection.

It fires whenever a batch carries at least `threshold` spikes, and labels every decision `provenance: demo`. The real
runtime is the module named by SNN_RUNTIME; this one is only used when SNN_RUNTIME=demo AND SNN_ALLOW_DEMO_RUNTIME=1,
so it cannot be mistaken for the real thing in a deployment.
"""

from __future__ import annotations


class DemoRuntime:
    threshold = 3

    def __init__(self):
        self._manifest, self._epoch = None, None

    def load(self, manifest: dict) -> None:
        self._manifest = manifest

    def reset(self, *, epoch: int, source_time_us: int) -> None:
        self._epoch = epoch

    def step(self, batch: dict) -> dict:
        count = len(batch["spikes"])
        return {"trigger": count >= self.threshold, "status": "valid", "score": float(count), "score_kind": "spike_count", "provenance": "demo"}

    def snapshot(self) -> dict:
        return {}

    def checkpoint(self) -> bytes:
        return b""

    def restore(self, checkpoint: bytes) -> None:
        pass


class DemoVision:
    """A scripted vision provider for demos and tests: it answers the same thing for every photo, whatever is in it.

    It exists so the alarm path can be shown without an Azure vision deployment. Every result names the deployment
    `demo-vision` and the prompt version `demo`, and it needs SNN_ALLOW_DEMO_VISION=1, so it cannot pass for a model."""

    deployment, prompt_version = "demo-vision", "demo"
    ANSWERS = {
        "glass_person": (True, True, "good", "Demo answer: glass and a person."),
        "glass_only": (True, False, "good", "Demo answer: glass, no person."),
        "person_only": (False, True, "good", "Demo answer: a person, no glass."),
        "nothing": (False, False, "good", "Demo answer: nothing of interest."),
        "poor_quality": (True, True, "poor", "Demo answer: the image is unreadable."),
    }

    def __init__(self, answer: str):
        if answer not in self.ANSWERS and answer != "unavailable":
            raise ValueError(f"unknown demo vision answer {answer!r}")
        self._answer = answer

    def analyze(self, jpeg: bytes):
        from rpi_agents.cloud.app.vision import Observation, VisionUnavailable

        if self._answer == "unavailable":
            raise VisionUnavailable("VISION_DEMO_UNAVAILABLE")
        return Observation(*self.ANSWERS[self._answer])
