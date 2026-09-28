"""A sketch from the editor is not a model (task P3, point 3).

Three things have to hold, and each one has a way of going wrong that would be
invisible until somebody soldered from it:

* the wiring has to be one a Lu.i rack could carry — three ports a board, one
  wire a port, at most fifty boards;
* zero boards is an empty canvas and has to say so, rather than looking like a
  network that detects nothing;
* redrawing the network must not touch the champion that is loaded.
"""

from __future__ import annotations

import pytest

from contracts.validation import ContractError, fixture, validate
from snn_runtime import LuiRuntime, RuntimeLoadError
from snn_runtime.topology import MAX_BOARDS, review_draft


def draft(boards, connections=(), draft_id="draft-test"):
    return {
        "schema_version": "1.0",
        "draft_id": draft_id,
        "boards": [{"id": b, "label": b, "x": 10.0 * i, "y": 20.0} for i, b in enumerate(boards)],
        "connections": list(connections),
    }


def wire(cid, source, target, port, *, kind="excitatory", source_kind="board"):
    return {
        "id": cid, "source": source, "source_kind": source_kind,
        "target": target, "target_port": port, "kind": kind,
    }


# --------------------------------------------------------------- the contract


def test_the_published_fixture_is_a_valid_draft():
    validate("TopologyDraft", fixture("topology-draft"))


def test_fifty_boards_are_allowed_and_fifty_one_are_not():
    names = [f"B{i}" for i in range(MAX_BOARDS)]
    assert review_draft(draft(names)).boards == MAX_BOARDS
    with pytest.raises(ContractError):
        review_draft(draft(names + ["B50"]))


def test_two_wires_into_one_port_are_refused():
    """The board has three screw terminals; a port takes one wire."""
    sketch = draft(["A", "B", "C"], [wire("w1", "A", "C", 1), wire("w2", "B", "C", 1)])
    with pytest.raises(ContractError):
        review_draft(sketch)


def test_three_ports_are_the_fan_in_limit():
    ok = draft(["A", "B", "C", "D"], [wire(f"w{p}", src, "D", p) for p, src in enumerate("ABC", 1)])
    assert review_draft(ok).connections == 3
    too_many = draft(
        ["A", "B", "C", "D"],
        [wire(f"w{p}", src, "D", p) for p, src in enumerate("ABC", 1)] + [wire("w4", "A", "D", 4)],
    )
    with pytest.raises(ContractError):
        review_draft(too_many)


def test_a_wire_to_a_board_that_is_not_there_is_refused():
    with pytest.raises(ContractError):
        review_draft(draft(["A"], [wire("w1", "A", "ghost", 1)]))


def test_a_channel_source_does_not_have_to_be_a_board():
    """Encoder channel names are not known while somebody is still drawing."""
    sketch = draft(["A"], [wire("w1", "hf_hi", "A", 1, source_kind="channel")])
    assert review_draft(sketch).wiring_ok


def test_duplicate_board_ids_are_refused():
    sketch = draft(["A", "A"])
    with pytest.raises(ContractError):
        review_draft(sketch)


# ----------------------------------------------------------------- the verdict


def test_an_empty_editor_says_it_is_an_empty_editor():
    review = review_draft(draft([]))
    assert review.empty and not review.runnable
    assert "blank canvas" in " ".join(review.reasons)


def test_even_a_complete_sketch_is_not_runnable():
    """The sketch has no weights, no decoder and no checkpoint. It never runs."""
    sketch = draft(["A", "D"], [wire("w1", "A", "D", 1)])
    review = review_draft(sketch)
    assert not review.runnable
    assert "ModelManifest" in review.reasons[-1]


def test_a_board_with_nothing_wired_into_it_is_named():
    review = review_draft(draft(["A", "B"], [wire("w1", "A", "B", 1)]))
    assert any("no input wired" in reason and "A" in reason for reason in review.reasons)


# ------------------------------------------------------------- and the runtime


def test_the_runtime_refuses_a_draft_by_name(lui8):
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    with pytest.raises(RuntimeLoadError) as err:
        runtime.load(fixture("topology-draft"))
    assert err.value.code == "DRAFT_NOT_A_MODEL"


def test_redrawing_the_network_cannot_replace_the_champion(lui8):
    """The editor may send anything; the loaded model is what it was."""
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(lui8)
    champion = runtime.model.model_hash

    review_draft(draft([f"B{i}" for i in range(MAX_BOARDS)]))
    with pytest.raises(RuntimeLoadError):
        runtime.load(draft(["A", "D"], [wire("w1", "A", "D", 1)]))

    assert runtime.model.model_hash == champion
