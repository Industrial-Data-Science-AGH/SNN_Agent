"""Read side for events: what the dashboard lists and shows. No mutation here."""

from __future__ import annotations

import copy

from contracts.validation import ContractError, validate
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.records import CORE, T_EVENTS, event_rk


class EventReader:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def get(self, event_id: str) -> dict:
        found = self.ctx.storage.tables.get(T_EVENTS, CORE, event_rk(event_id))
        if found is None:
            raise ContractError("NOT_FOUND", "Event not found", 404)
        return validate("Event", copy.deepcopy(found.data["event"]))

    def meta(self, event_id: str) -> dict:
        found = self.ctx.storage.tables.get(T_EVENTS, CORE, event_rk(event_id))
        if found is None:
            raise ContractError("NOT_FOUND", "Event not found", 404)
        return copy.deepcopy(found.data["meta"])

    def list(self, limit: int = 20, offset: int = 0) -> dict:
        if not 1 <= limit <= 100 or offset < 0:
            raise ContractError("INVALID_PAGINATION", "Use limit 1..100 and offset >= 0", 422)
        rows = self.ctx.storage.tables.query(T_EVENTS, CORE, rk_prefix="list:", limit=offset + limit + 1)
        page = rows[offset : offset + limit]
        items = [self.get(row.data["event_id"]) for row in page]
        return {"schema_version": "1.0", "items": items, "next_offset": offset + limit if len(rows) > offset + limit else None}
