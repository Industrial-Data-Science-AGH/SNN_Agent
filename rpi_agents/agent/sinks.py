"""Image sinks for capture commands. Standard library only.

BackendImageSink uploads the JPEG to the backend and returns the image_id the backend assigned. It is the sink
for real deployments: the ack that a capture is "completed" then names an image that really is stored.

LocalDirSink keeps JPEGs on the device itself. It is a DEMO and bring-up sink, not an upload: the backend
never receives the image, so an ack that references it only proves the photo was taken and stored locally.
The directory is private (0700), files are 0600 and written atomically, and only the newest `keep` images
are retained so the SD card cannot fill up. Photos can show people: enable it deliberately.
"""

from __future__ import annotations

import hashlib
import os
import re

from rpi_agents.agent.api import ApiClient, Outcome
from rpi_agents.agent.commands import SinkUnavailable

_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class LocalDirSink:
    def __init__(self, directory: str, keep: int = 20):
        if keep < 1:
            raise ValueError("keep must be at least 1")
        self._dir, self._keep = directory, keep
        os.makedirs(directory, mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)

    def store(self, *, event_id: str, command_id: str, index: int, jpeg: bytes, captured_at: str) -> str:
        image_id = f"{command_id}-{index}"
        if not _SAFE.fullmatch(image_id) or len(image_id) > 64:
            raise SinkUnavailable("image id is not a safe file name")
        path = os.path.join(self._dir, f"{image_id}.jpg")
        tmp = path + ".tmp"
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(jpeg)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            self._prune()
        except OSError as exc:
            raise SinkUnavailable(f"cannot store image: {exc.strerror}") from None
        return image_id

    def _prune(self) -> None:
        files = [e for e in os.scandir(self._dir) if e.name.endswith(".jpg")]
        files.sort(key=lambda e: (e.stat().st_mtime_ns, e.name))
        for entry in files[: max(0, len(files) - self._keep)]:
            os.unlink(entry.path)


class BackendImageSink:
    """Uploads each image to the backend. A failure is SinkUnavailable, so the command fails visibly."""

    def __init__(self, api: ApiClient, attempts: int = 2):
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self._api, self._attempts = api, attempts

    def store(self, *, event_id: str, command_id: str, index: int, jpeg: bytes, captured_at: str) -> str:
        digest = hashlib.sha256(jpeg).hexdigest()
        code = "UNKNOWN"
        for _ in range(self._attempts):
            result = self._api.upload_image(event_id, index, jpeg, sha256=digest, captured_at=captured_at)
            if result.outcome is Outcome.OK and isinstance(result.body, dict) and isinstance(result.body.get("image_id"), str):
                return result.body["image_id"]
            code = result.code or result.detail
            if result.outcome is Outcome.PERMANENT:
                break  # the backend refused this image; sending it again cannot help
        raise SinkUnavailable(f"image upload failed: {code}")
