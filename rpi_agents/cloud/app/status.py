"""The latest DeviceStatus heartbeat of each device. Latest wins; this is not an audit log."""

from __future__ import annotations

import copy

from contracts.validation import ContractError, validate
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.storage import Conflict, PreconditionFailed

T_STATUS = "devicestatus"


class StatusService:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def report(self, device_id: str, body: dict) -> dict:
        validate("DeviceStatus", body)
        if body["device_id"] != device_id:
            raise ContractError("DEVICE_MISMATCH", "Path and device ID differ", 409)
        tables = self.ctx.storage.tables
        for _ in range(8):
            row = tables.get(T_STATUS, device_id, "latest")
            try:
                if row is None:
                    tables.insert(T_STATUS, device_id, "latest", body)
                else:
                    tables.replace(T_STATUS, device_id, "latest", body, row.etag)
                return copy.deepcopy(body)
            except (Conflict, PreconditionFailed):
                continue
        raise ContractError("BUSY", "Could not record the status; retry", 409)

    def latest(self, device_id: str) -> dict:
        row = self.ctx.storage.tables.get(T_STATUS, device_id, "latest")
        if row is None:
            raise ContractError("NOT_FOUND", "No status reported by this device", 404)
        return validate("DeviceStatus", copy.deepcopy(row.data))
