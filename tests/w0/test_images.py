import hashlib

import pytest

from contracts.validation import ContractError
from rpi_agents.cloud.app.records import CORE, IMAGES, Q_VISION, T_EVENTS
from rpi_agents.cloud.app.sessions import provision_device
from tests.w0.backend_env import Env
from tests.w0.fakes import make_jpeg


def code(exc_info):
    return (exc_info.value.code, exc_info.value.status)


def with_event(**settings):
    env = Env(**settings)
    state = env.open_session()
    event_id, command = env.trigger(state)
    return env, state, event_id, command


def event_record(env, event_id):
    return env.storage.tables.get(T_EVENTS, CORE, f"event:{event_id}").data


def test_an_uploaded_photo_is_stored_recorded_and_handed_to_vision_in_one_step():
    env, state, event_id, command = with_event()
    jpeg = make_jpeg(1280, 720)
    result = env.upload(event_id, data=jpeg)
    assert result == {"schema_version": "1.0", "image_id": f"{event_id}-0", "event_id": event_id, "bytes": len(jpeg),
                      "sha256": hashlib.sha256(jpeg).hexdigest(), "status": "queued"}  # fmt: skip
    assert env.storage.blobs.get(IMAGES, f"{event_id}/0.jpg") == jpeg
    record = event_record(env, event_id)
    assert record["event"]["status"] == "analyzing" and record["meta"]["image_ids"][0]["width"] == 1280
    (message,) = env.storage.queues.receive(Q_VISION, visibility_s=30)
    assert message.body == {"event_id": event_id, "image_id": f"{event_id}-0"}
    assert env.publisher.drained()
    assert env.images.read(event_id, 0) == jpeg


def test_a_retry_after_success_returns_the_same_answer_and_never_a_second_analysis():
    env, state, event_id, command = with_event()
    first = env.upload(event_id)
    assert env.upload(event_id) == first
    assert env.storage.queues.depth(Q_VISION) == 1


def test_a_crash_after_the_blob_was_written_but_before_it_was_recorded_is_completed_by_the_retry():
    env, state, event_id, command = with_event()
    real = env.images._record
    calls = {"n": 0}

    def dying(*a, **k):
        calls["n"] += 1
        raise KeyboardInterrupt

    env.images._record = dying
    with pytest.raises(KeyboardInterrupt):
        env.upload(event_id)
    assert env.storage.blobs.exists(IMAGES, f"{event_id}/0.jpg") and event_record(env, event_id)["meta"]["image_ids"] == []
    assert env.storage.queues.depth(Q_VISION) == 0
    env.images._record = real
    assert env.upload(event_id)["status"] == "queued"  # the retry finds the blob, checks it, and finishes the bookkeeping
    assert env.storage.queues.depth(Q_VISION) == 1 and event_record(env, event_id)["event"]["status"] == "analyzing"


def test_after_a_crash_a_different_image_cannot_be_recorded_over_bytes_that_are_already_stored():
    env, state, event_id, command = with_event()
    stored = make_jpeg(640, 480)
    env.images.ctx.storage.blobs.put(IMAGES, f"{event_id}/0.jpg", stored, "image/jpeg")  # the crash left only the blob
    with pytest.raises(ContractError) as error:
        env.upload(event_id, data=make_jpeg(800, 600))
    assert code(error) == ("IMAGE_CONFLICT", 409)
    assert event_record(env, event_id)["meta"]["image_ids"] == []  # no metadata describing the wrong bytes
    assert env.images.read(event_id, 0) == stored and env.storage.queues.depth(Q_VISION) == 0


def test_a_crash_after_recording_but_before_publishing_leaves_a_row_the_reconciler_publishes():
    env, state, event_id, command = with_event()
    env.publisher._publish = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("queue down"))
    assert env.upload(event_id)["status"] == "queued"  # the upload itself succeeded
    assert env.storage.queues.depth(Q_VISION) == 0 and not env.publisher.drained()
    del env.publisher._publish
    assert env.publisher.publish_pending() == 1 and env.storage.queues.depth(Q_VISION) == 1


def test_a_different_image_for_the_same_slot_is_a_conflict_and_the_first_one_stays():
    env, state, event_id, command = with_event()
    first = make_jpeg(640, 480)
    env.upload(event_id, data=first)
    with pytest.raises(ContractError) as error:
        env.upload(event_id, data=make_jpeg(800, 600))
    assert code(error) == ("IMAGE_CONFLICT", 409) and env.images.read(event_id, 0) == first


@pytest.mark.parametrize(
    "data,changes,expected",
    [(b"\x89PNG" + b"x" * 50, {}, ("NOT_JPEG", 422)),
     (make_jpeg(eoi=False), {}, ("TRUNCATED_JPEG", 422)),
     (make_jpeg(8, 8), {}, ("BAD_DIMENSIONS", 422)),
     (make_jpeg(scan=b"\x00" * 1_100_000), {}, ("IMAGE_TOO_LARGE", 413)),
     (None, {"sha256": "0" * 64}, ("HASH_MISMATCH", 422)),
     (None, {"captured_at": "yesterday"}, ("BAD_TIMESTAMP", 422)),
     (None, {"captured_at": "2026-09-24T12:00:04+00:00"}, ("BAD_TIMESTAMP", 422)),
     (None, {"index": 1}, ("BAD_IMAGE_INDEX", 409)),
     (None, {"index": -1}, ("BAD_IMAGE_INDEX", 409)),
     (None, {"index": True}, ("BAD_IMAGE_INDEX", 409))],
)  # fmt: skip
def test_a_photo_that_is_not_acceptable_is_refused_before_anything_is_stored(data, changes, expected):
    env, state, event_id, command = with_event()
    with pytest.raises(ContractError) as error:
        env.upload(event_id, data=data, **changes)
    assert code(error) == expected
    assert not env.storage.blobs.exists(IMAGES, f"{event_id}/0.jpg") and env.storage.queues.depth(Q_VISION) == 0


def test_the_hash_check_is_case_insensitive_and_covers_the_actual_bytes():
    env, state, event_id, command = with_event()
    data = make_jpeg()
    assert env.upload(event_id, data=data, sha256=hashlib.sha256(data).hexdigest().upper())["status"] == "queued"


def test_more_frames_are_stored_without_a_second_analysis_and_the_limit_is_the_request():
    env, state, event_id, command = with_event(capture_frames=2)
    env.upload(event_id, 0)
    assert env.upload(event_id, 1, data=make_jpeg(320, 240))["status"] == "stored"
    assert env.storage.queues.depth(Q_VISION) == 1  # one analysis per event
    with pytest.raises(ContractError) as error:
        env.upload(event_id, 2)
    assert code(error) == ("BAD_IMAGE_INDEX", 409)
    assert [i["index"] for i in event_record(env, event_id)["meta"]["image_ids"]] == [0, 1]


def test_another_devices_event_and_an_unknown_event_look_exactly_the_same():
    env, state, event_id, command = with_event()
    provision_device(env.ctx, "other-pi")
    with pytest.raises(ContractError) as foreign:
        env.images.upload("other-pi", event_id, index=0, sha256=hashlib.sha256(make_jpeg()).hexdigest(),
                          captured_at="2026-09-24T12:00:04Z", data=make_jpeg())  # fmt: skip
    with pytest.raises(ContractError) as unknown:
        env.upload("no-such-event")
    assert code(foreign) == code(unknown) == ("NOT_FOUND", 404) and foreign.value.args == unknown.value.args
    assert not env.storage.blobs.exists(IMAGES, f"{event_id}/0.jpg")


def test_a_closed_event_accepts_no_more_images():
    env, state, event_id, command = with_event()
    found = env.storage.tables.get(T_EVENTS, CORE, f"event:{event_id}")
    closed = found.data["event"] | {"status": "no_alarm"}
    env.storage.tables.replace(T_EVENTS, CORE, f"event:{event_id}", found.data | {"event": closed}, found.etag)
    with pytest.raises(ContractError) as error:
        env.upload(event_id)
    assert code(error) == ("EVENT_CLOSED", 409)


def test_reading_an_image_that_does_not_exist_is_not_found():
    env, state, event_id, command = with_event()
    with pytest.raises(ContractError) as error:
        env.images.read(event_id, 0)
    assert code(error) == ("NOT_FOUND", 404)


# --------------------------------------------------------------------------------------- event reader


def test_events_are_listed_newest_first_with_paging():
    env = Env(event_cooldown_s=0)
    state = env.open_session()
    ids = []
    for seq in range(5):
        env.clock.advance(1)
        ids.append(env.trigger(state, seq)[0])
    page = env.events.list(limit=2)
    assert [e["event_id"] for e in page["items"]] == [ids[4], ids[3]] and page["next_offset"] == 2
    page2 = env.events.list(limit=2, offset=2)
    assert [e["event_id"] for e in page2["items"]] == [ids[2], ids[1]] and page2["next_offset"] == 4
    last = env.events.list(limit=2, offset=4)
    assert [e["event_id"] for e in last["items"]] == [ids[0]] and last["next_offset"] is None
    assert env.events.get(ids[0])["event_id"] == ids[0]


@pytest.mark.parametrize("limit,offset", [(0, 0), (101, 0), (10, -1)])
def test_bad_paging_is_refused(limit, offset):
    with pytest.raises(ContractError) as error:
        Env().events.list(limit=limit, offset=offset)
    assert code(error) == ("INVALID_PAGINATION", 422)


def test_an_unknown_event_is_not_found():
    with pytest.raises(ContractError) as error:
        Env().events.get("nope")
    assert code(error) == ("NOT_FOUND", 404)
