"""Image upload for events: verify, store once, hand to vision.

- The bytes are inspected as a JPEG (structure, size, dimensions) and must match the SHA-256 the device sent.
- The blob write is create-only. A retry after a crash finds the blob already there, checks that it is the same
  content, and completes the bookkeeping: the upload is idempotent and can never replace an image.
- Recording the image and requesting the analysis happen in ONE transaction (the event row plus the outbox
  row), so an image is never stored without the analysis being owed, and never analysed twice by accident.
- A device can only reach its own events; anything else looks like a missing event.
"""

from __future__ import annotations

import copy
import hashlib

from contracts.validation import ContractError
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.imaging import MAX_IMAGE_BYTES, InvalidImage, inspect_jpeg
from rpi_agents.cloud.app.publisher import Publisher, outbox_op
from rpi_agents.cloud.app.records import CORE, IMAGES, T_EVENTS, event_rk, image_name, parse_utc
from rpi_agents.cloud.app.storage import Conflict, NotFound, Op, PreconditionFailed

_CAS_ATTEMPTS = 8
_OPEN = ("photo_requested", "analyzing")


class ImageService:
    def __init__(self, ctx: Context, publisher: Publisher):
        self.ctx, self.publisher = ctx, publisher

    def upload(self, device_id: str, event_id: str, *, index: int, sha256: str, captured_at: str, data: bytes) -> dict:
        try:
            info = inspect_jpeg(data)
        except InvalidImage as exc:
            raise ContractError(exc.code, str(exc), 413 if exc.code == "IMAGE_TOO_LARGE" else 422) from None
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.ctx.settings.capture_frames:
            raise ContractError("BAD_IMAGE_INDEX", "Image index is outside the frames requested", 409)
        digest = hashlib.sha256(data).hexdigest()
        if sha256.lower() != digest:
            raise ContractError("HASH_MISMATCH", "The SHA-256 does not match the bytes", 422)
        try:
            parse_utc(captured_at)
        except ValueError:
            raise ContractError("BAD_TIMESTAMP", "captured_at must be RFC 3339 UTC", 422) from None
        if not captured_at.endswith("Z"):
            raise ContractError("BAD_TIMESTAMP", "captured_at must be RFC 3339 UTC", 422)

        event = self._owned_event(device_id, event_id)
        if event.data["event"]["status"] not in _OPEN:
            raise ContractError("EVENT_CLOSED", "The event no longer accepts images", 409)
        self._store_blob(event_id, index, data, digest)
        record = {"image_id": f"{event_id}-{index}", "index": index, "sha256": digest, "bytes": len(data),
                  "captured_at": captured_at, "width": info.width, "height": info.height}  # fmt: skip
        self._record(device_id, event_id, record)
        self.publisher.publish_pending()
        return {"schema_version": "1.0", "image_id": record["image_id"], "event_id": event_id, "bytes": len(data), "sha256": digest,
                "status": "queued" if index == 0 else "stored"}  # fmt: skip

    def _owned_event(self, device_id: str, event_id: str):
        found = self.ctx.storage.tables.get(T_EVENTS, CORE, event_rk(event_id))
        if found is None or found.data["event"]["device_id"] != device_id:
            raise ContractError("NOT_FOUND", "Event not found", 404)  # another device's event looks the same as none
        return found

    def _store_blob(self, event_id: str, index: int, data: bytes, digest: str) -> None:
        blobs, name = self.ctx.storage.blobs, image_name(event_id, index)
        try:
            blobs.put(IMAGES, name, data, "image/jpeg")
        except Conflict:
            if hashlib.sha256(blobs.get(IMAGES, name)).hexdigest() != digest:
                raise ContractError("IMAGE_CONFLICT", "A different image was already stored for this slot", 409) from None

    def _record(self, device_id: str, event_id: str, record: dict) -> None:
        tables = self.ctx.storage.tables
        for _ in range(_CAS_ATTEMPTS):
            found = self._owned_event(device_id, event_id)
            meta = found.data["meta"]
            existing = next((i for i in meta["image_ids"] if i["index"] == record["index"]), None)
            if existing is not None:
                if existing["sha256"] != record["sha256"]:
                    raise ContractError("IMAGE_CONFLICT", "A different image was already recorded for this slot", 409)
                return  # an exact retry: already recorded
            event = copy.deepcopy(found.data["event"])
            meta = copy.deepcopy(meta) | {"image_ids": meta["image_ids"] + [record]}
            ops = []
            if record["index"] == 0:
                event["status"] = "analyzing"
                ops.append(outbox_op(self.ctx, event_id, 1, "vision_job", {"event_id": event_id, "image_id": record["image_id"]}))
            try:
                tables.transaction(T_EVENTS, CORE, [Op("replace", event_rk(event_id), {"event": event, "meta": meta}, found.etag), *ops])
                return
            except PreconditionFailed:
                continue
        raise ContractError("BUSY", "Could not record the image; retry", 409)

    def read(self, event_id: str, index: int) -> bytes:
        try:
            return self.ctx.storage.blobs.get(IMAGES, image_name(event_id, index))
        except NotFound:
            raise ContractError("NOT_FOUND", "Image not found", 404) from None


__all__ = ["ImageService", "MAX_IMAGE_BYTES"]
