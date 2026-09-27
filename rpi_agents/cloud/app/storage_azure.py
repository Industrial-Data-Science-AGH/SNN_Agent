"""Azure Table, Blob and Queue Storage behind the storage interfaces (storage.py).

Needs azure-data-tables, azure-storage-blob, azure-storage-queue and (in Azure) azure-identity. Nothing else in the
backend imports this module unless SNN_STORAGE=azure, so the rest runs and is tested without the SDKs.

Authentication in Azure is a managed identity through DefaultAzureCredential: no account key or connection string
is configured. A connection string is accepted only for the local Azurite emulator.

Table notes, all consequences of what Azure Tables are:
- values are stored as JSON split into 30,000-character chunks (a string property is limited to 64 KiB);
- a transaction is limited to one partition and 100 operations, which is what storage.py promises;
- there is no reverse ordering: `reverse=True` reads the matching rows and reverses them here. The backend's own
  keys are built so that it never needs that on a large partition (list rows carry a reversed timestamp).
Queue visibility is whole seconds, so sub-second delays are rounded up.
"""

from __future__ import annotations

import json
import math
from typing import Callable

from azure.core import MatchConditions
from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
)
from azure.data.tables import TableServiceClient, TableTransactionError, UpdateMode
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.queue import QueueClient

from rpi_agents.cloud.app.storage import (
    Conflict,
    NotFound,
    Op,
    PreconditionFailed,
    QueueMessage,
    Row,
    Storage,
)

CHUNK = 30_000
MAX_TRANSACTION_OPS = 100
_MOVED = MatchConditions.IfNotModified


def _entity(pk: str, rk: str, data: dict) -> dict:
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    chunks = [text[i : i + CHUNK] for i in range(0, len(text), CHUNK)] or [""]
    return {"PartitionKey": pk, "RowKey": rk, "n": len(chunks), **{f"d{i}": c for i, c in enumerate(chunks)}}


def _data(entity) -> dict:
    return json.loads("".join(entity[f"d{i}"] for i in range(int(entity["n"]))))


def _etag(entity) -> str:
    return entity.metadata["etag"]


def _map_table_error(exc: Exception) -> Exception:
    if isinstance(exc, ResourceExistsError):
        return Conflict(str(exc))
    if isinstance(exc, ResourceNotFoundError):
        return NotFound(str(exc))
    if isinstance(exc, ResourceModifiedError):
        return PreconditionFailed(str(exc))
    if isinstance(exc, TableTransactionError):
        code = getattr(exc, "error_code", None)
        code = getattr(code, "value", code)
        text = str(code or exc)
        if "InvalidDuplicateRow" in text:
            return ValueError("a row can appear only once in a transaction")
        if "EntityAlreadyExists" in text:
            return Conflict(text)
        if "ResourceNotFound" in text or "EntityNotFound" in text:
            return NotFound(text)
        if "UpdateConditionNotSatisfied" in text or "ConditionNotMet" in text:
            return PreconditionFailed(text)
    if isinstance(exc, HttpResponseError):
        status = getattr(exc, "status_code", None)
        if status == 412:
            return PreconditionFailed(str(exc))
        if status == 404:
            return NotFound(str(exc))
        if status == 409:
            return Conflict(str(exc))
    return exc


def _upper_bound(prefix: str) -> str:
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


class AzureTables:
    def __init__(self, service: TableServiceClient):
        self._service, self._clients = service, {}

    def _client(self, table: str):
        if table not in self._clients:
            self._service.create_table_if_not_exists(table)
            self._clients[table] = self._service.get_table_client(table)
        return self._clients[table]

    def get(self, table, pk, rk):
        try:
            entity = self._client(table).get_entity(pk, rk)
        except ResourceNotFoundError:
            return None
        return Row(pk, rk, _data(entity), _etag(entity))

    def insert(self, table, pk, rk, data):
        return self.transaction(table, pk, [Op("insert", rk, data)])[0]

    def replace(self, table, pk, rk, data, etag):
        return self.transaction(table, pk, [Op("replace", rk, data, etag)])[0]

    def delete(self, table, pk, rk, etag=None):
        # Through a transaction on purpose: the single-entity call silently ignores a missing row, a transaction
        # reports it, and this interface promises NotFound.
        self.transaction(table, pk, [Op("delete", rk, None, etag)])

    def query(self, table, pk, *, rk_prefix=None, rk_from=None, limit=100, reverse=False):
        clauses, params = ["PartitionKey eq @pk"], {"pk": pk}
        if rk_prefix is not None:
            clauses += ["RowKey ge @prefix", "RowKey lt @prefix_end"]
            params |= {"prefix": rk_prefix, "prefix_end": _upper_bound(rk_prefix)}
        if rk_from is not None:  # forward: rows at or after it; reverse: rows at or before it
            clauses.append("RowKey le @from" if reverse else "RowKey ge @from")
            params["from"] = rk_from
        pages = self._client(table).query_entities(" and ".join(clauses), parameters=params, results_per_page=min(max(limit, 1), 1000))
        rows: list[Row] = []
        for entity in pages:
            rows.append(Row(pk, entity["RowKey"], _data(entity), _etag(entity)))
            if not reverse and len(rows) >= limit:
                break
        return rows[::-1][:limit] if reverse else rows

    def transaction(self, table, pk, ops):
        if not ops:
            return []
        if len(ops) > MAX_TRANSACTION_OPS:
            raise ValueError(f"a transaction is limited to {MAX_TRANSACTION_OPS} operations")
        if len({op.rk for op in ops}) != len(ops):
            raise ValueError("a row can appear only once in a transaction")
        actions = []
        for op in ops:
            entity = _entity(pk, op.rk, op.data if op.data is not None else {})
            if op.kind == "insert":
                actions.append(("create", entity))
            elif op.kind == "replace":
                if op.etag is None:
                    raise PreconditionFailed("replace needs an etag")
                actions.append(("update", entity, {"mode": UpdateMode.REPLACE, "etag": op.etag, "match_condition": _MOVED}))
            elif op.kind == "delete":
                kwargs = {"etag": op.etag, "match_condition": _MOVED} if op.etag is not None else {}
                actions.append(("delete", {"PartitionKey": pk, "RowKey": op.rk}, kwargs))
            else:
                raise ValueError(f"unknown operation {op.kind!r}")
        try:
            results = self._client(table).submit_transaction(actions)
        except Exception as exc:
            raise _map_table_error(exc) from None
        rows = []
        for op, result in zip(ops, results):
            if op.kind != "delete":
                rows.append(Row(pk, op.rk, op.data, result["etag"]))
        return rows


class AzureBlobs:
    def __init__(self, service: BlobServiceClient):
        self._service, self._made = service, set()

    def _container(self, name: str):
        if name not in self._made:
            try:
                self._service.create_container(name)
            except ResourceExistsError:
                pass
            self._made.add(name)
        return self._service.get_container_client(name)

    def put(self, container, name, data, content_type):
        try:
            self._container(container).upload_blob(name, bytes(data), overwrite=False, content_settings=ContentSettings(content_type=content_type))
        except ResourceExistsError:
            raise Conflict(f"{container}/{name} exists") from None

    def get(self, container, name):
        try:
            return self._container(container).download_blob(name).readall()
        except ResourceNotFoundError:
            raise NotFound(f"{container}/{name}") from None

    def exists(self, container, name):
        return self._container(container).get_blob_client(name).exists()

    def delete(self, container, name):
        try:
            self._container(container).delete_blob(name)
        except ResourceNotFoundError:
            pass


class AzureQueues:
    def __init__(self, factory: Callable[[str], QueueClient]):
        self._factory, self._clients = factory, {}

    def _client(self, queue: str) -> QueueClient:
        if queue not in self._clients:
            client = self._factory(queue)
            try:
                client.create_queue()
            except ResourceExistsError:
                pass
            self._clients[queue] = client
        return self._clients[queue]

    @staticmethod
    def _seconds(value: float) -> int:
        return max(0, math.ceil(value))

    def send(self, queue, body, *, delay_s=0.0):
        sent = self._client(queue).send_message(json.dumps(body, separators=(",", ":")), visibility_timeout=self._seconds(delay_s) or None)
        return sent.id

    def receive(self, queue, *, visibility_s, max_messages=1):
        got = self._client(queue).receive_messages(messages_per_page=min(max(max_messages, 1), 32), visibility_timeout=max(1, self._seconds(visibility_s)), max_messages=max_messages)
        out = []
        for message in got:
            out.append(QueueMessage(message.id, message.pop_receipt, json.loads(message.content), int(message.dequeue_count or 1)))
            if len(out) >= max_messages:
                break
        return out

    def renew(self, queue, message, visibility_s):
        try:
            updated = self._client(queue).update_message(message.id, message.receipt, visibility_timeout=max(1, self._seconds(visibility_s)))
        except Exception as exc:
            raise self._map(exc) from None
        return QueueMessage(message.id, updated.pop_receipt, message.body, message.dequeue_count)

    def delete(self, queue, message):
        try:
            self._client(queue).delete_message(message.id, message.receipt)
        except Exception as exc:
            raise self._map(exc) from None

    @staticmethod
    def _map(exc: Exception) -> Exception:
        if isinstance(exc, ResourceNotFoundError):
            return NotFound(str(exc))
        if isinstance(exc, HttpResponseError):
            code = str(getattr(getattr(exc, "error", None), "code", "") or "")
            if "PopReceiptMismatch" in code or getattr(exc, "status_code", None) == 400:
                return PreconditionFailed(str(exc))
            if "MessageNotFound" in code or getattr(exc, "status_code", None) == 404:
                return NotFound(str(exc))
        return exc

    def depth(self, queue):
        return int(self._client(queue).get_queue_properties().approximate_message_count)


def azure_storage(*, account: str | None = None, connection_string: str | None = None, credential=None) -> Storage:
    """Storage on Azure (managed identity) or on the Azurite emulator (a development connection string)."""
    if (account is None) == (connection_string is None):
        raise ValueError("give either an account name or a connection string")
    if connection_string is not None:
        tables = TableServiceClient.from_connection_string(connection_string)
        blobs = BlobServiceClient.from_connection_string(connection_string)
        queue = lambda name: QueueClient.from_connection_string(connection_string, name)  # noqa: E731
    else:
        if credential is None:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
        tables = TableServiceClient(endpoint=f"https://{account}.table.core.windows.net", credential=credential)
        blobs = BlobServiceClient(account_url=f"https://{account}.blob.core.windows.net", credential=credential)
        queue = lambda name: QueueClient(account_url=f"https://{account}.queue.core.windows.net", queue_name=name, credential=credential)  # noqa: E731
    return Storage(AzureTables(tables), AzureBlobs(blobs), AzureQueues(queue))
