"""J2 format-family tests: async-profiler collapsed + ``jfr print --json``.

Binding contract: ``docs/units/UNIT-J2-JFR-PLAN.md`` sections 1 and 2 (module
``src/toothpick/profile_jfr.py``, chunk 1) with the eight acceptance cases of
section 3.3 -- async basic, async alloc, malformed counting, JFR samples, JFR
allocation, unknown events ignored, ``detect_format`` both/``None``, and the
end-to-end ``import_samples`` slice.  Design anchors: design document section 1
(no fabricated rows -- every row traces to parsed bytes) and section 19.

The same rules as ``tests/test_profile.py`` apply: every store is
context-managed and released before teardown (trap 7) and the suite keeps
``-p no:faulthandler`` (trap 11).  Nothing here invents a ``funcs`` row, and no
test asserts anything about ``ProfileFormat.coerce`` routing -- that is chunk 2.

Trap 14 is asserted structurally in ``test_diagnostics_are_json_plain``: every
value in a diagnostics tree must be ``str``/``int``/``float``/``bool``/``None``
(plus list/dict containers), never a tuple or a set, so ``json.dumps``
round-trips everything the parsers return.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from toothpick.profile import (
    ProfileError,
    ProfileSample,
    import_samples,
)
from toothpick.profile_jfr import (
    ASYNC_FORMAT,
    JFR_FORMAT,
    detect_format,
    parse_async_collapsed,
    parse_jfr_json,
)
from toothpick.storage import Database, FunctionRecord

SHA = "sha_jfr_test"

#: Two leaf spellings, both alloc column spellings, one address leaf, one
#: malformed line and one zero-weight line.  Joined (no trailing newline) so the
#: diagnostics counters below are exact.
ASYNC = "\n".join(
    [
        "# async-profiler collapsed",
        "--- Execution profile ---",
        "com/a.A;com/b.B.hot 42",
        "com/a.A;com/b.B.hot;0x140001000 7 1024",
        "com/a.A;com/b.B.hot;0x140002000 5 2048B",
        "com/a.A;com/b.B.hot 1500_512B",
        "nonsense-not-a-stack",
        "com/a.A;com/b.B 0",
    ]
)

MALFORMED_ASYNC = "\n".join(
    [
        "nonsense-not-a-stack",
        "",
        "# comment",
        "no_count_here",
        "stack 5 x",
    ]
)

JFR: Any = {
    "recording": {
        "events": [
            {
                "type": "jdk.ExecutionSample",
                "values": {
                    "stackTrace": {
                        "frames": [
                            {"method": {"name": "hot", "class": {"name": "com/b/B"}}},
                            {"method": {"name": "run", "class": {"name": "com/b/B"}}},
                        ]
                    }
                },
            },
            {
                "type": "jdk.ExecutionSample",
                "values": {
                    "stackTrace": {
                        "frames": [
                            {"method": {"name": "hot", "class": {"name": "com/b/B"}}}
                        ]
                    }
                },
            },
            {
                "type": "jdk.ObjectAllocationInNewTLAB",
                "values": {
                    "tlabSize": 4096,
                    "stackTrace": {
                        "frames": [
                            {"method": {"name": "hot", "class": {"name": "com/b/B"}}}
                        ]
                    },
                },
            },
            {
                "type": "jdk.ObjectAllocationOutsideTLAB",
                "values": {
                    "allocationSize": 1024,
                    "stackTrace": {
                        "frames": [
                            {"method": {"name": "hot", "class": {"name": "com/b/B"}}}
                        ]
                    },
                },
            },
            {"type": "jdk.CPULoad", "values": {"jvmUser": 0.1}},
            {"type": "jdk.ThreadStart", "values": {}},
        ]
    }
}


def _seed(db: Database, sha: str = SHA) -> None:
    """Two known functions -- the only rows attribution may ever touch."""
    for addr, name in (
        (0x140001000, "com.b.B.hot"),
        (0x140002000, "sub_140002000"),
    ):
        db.upsert_function(FunctionRecord(binary_sha256=sha, func_addr=addr, name=name))


def _scalars_only(node: Any) -> bool:
    """True when a diagnostics tree holds only JSON-plain values (trap 14)."""
    if isinstance(node, dict):
        return all(
            isinstance(key, str) and _scalars_only(value) for key, value in node.items()
        )
    if isinstance(node, list):
        return all(_scalars_only(value) for value in node)
    return node is None or isinstance(node, (str, int, float, bool))


# --------------------------------------------------------------------------- #
# async-profiler collapsed
# --------------------------------------------------------------------------- #


def test_async_basic_leaf_symbol_and_samples() -> None:
    rows, diag = parse_async_collapsed(ASYNC)
    name_only = [r for r in rows if r.symbol == "com/b.B.hot" and r.func_addr is None]
    assert sorted((r.samples, r.alloc_bytes) for r in name_only) == [
        (42, 0),  # bare count column -> no allocation
        (1500, 512),  # ``<count>_<bytes>B`` -> both columns from one token
    ]
    assert diag["format"] == ASYNC_FORMAT
    assert diag["rows"] == len(rows) == 5
    assert diag["rejected"] == 1  # ``nonsense-not-a-stack`` carries no count
    assert diag["lines"] == 6  # the comment and banner lines are skipped
    assert diag["skipped"] == 2


def test_async_alloc_both_spellings_and_address_leaf() -> None:
    rows, diag = parse_async_collapsed(ASYNC)
    by_addr = {r.func_addr: (r.samples, r.alloc_bytes) for r in rows if r.func_addr}
    assert by_addr == {0x140001000: (7, 1024), 0x140002000: (5, 2048)}
    # The address frame is its own symbol evidence; nothing is invented for it.
    assert {r.symbol for r in rows if r.func_addr == 0x140001000} == {"0x140001000"}
    # The attached ``<count>_<bytes>B`` spelling lands in the same two columns.
    attached = [r for r in rows if r.symbol == "com/b.B.hot" and r.samples == 1500]
    assert [r.alloc_bytes for r in attached] == [512]
    assert diag["alloc_bytes"] == 1024 + 2048 + 512
    assert diag["samples"] == 42 + 7 + 5 + 1500 + 0


def test_async_malformed_lines_counted_and_never_raise() -> None:
    rows, diag = parse_async_collapsed(MALFORMED_ASYNC)
    assert all(isinstance(row, ProfileSample) for row in rows)
    assert diag["rejected"] == 3  # two bare words + a non-numeric weight column
    assert diag["rows"] == len(rows) == 0
    assert diag["skipped"] == 2  # blank line + comment
    assert diag["lines"] == 3
    assert diag["samples"] == 0 and diag["alloc_bytes"] == 0


def test_async_leaf_offset_symbol_survives_verbatim() -> None:
    rows, diag = parse_async_collapsed("a;b;func+0x10 3")
    assert [r.symbol for r in rows] == ["func+0x10"]
    assert rows[0].func_addr is None  # an offset is a symbol, never an address
    assert rows[0].samples == 3 and rows[0].alloc_bytes == 0
    assert diag["rejected"] == 0


def test_async_accepts_a_path(tmp_path: Path) -> None:
    path = tmp_path / "collapsed.txt"
    path.write_text(ASYNC, encoding="utf-8")
    rows, diag = parse_async_collapsed(path)
    assert diag["rows"] == len(rows) == 5
    assert diag["samples"] == 42 + 7 + 5 + 1500  # the zero-weight row adds nothing


def test_async_rows_are_address_first_then_symbol() -> None:
    rows, _ = parse_async_collapsed("z;b.sym 1\nz;0x20 1\ny;0x10 1\n")
    assert [(r.symbol, r.func_addr) for r in rows] == [
        ("0x10", 0x10),
        ("0x20", 0x20),
        ("b.sym", None),
    ]


# --------------------------------------------------------------------------- #
# JFR JSON
# --------------------------------------------------------------------------- #


def test_jfr_execution_samples_plus_one_top_frame_symbol() -> None:
    rows, diag = parse_jfr_json(json.dumps(JFR))
    hot = [r for r in rows if r.symbol == "com.b.B.hot"]
    assert len(hot) == 1
    assert hot[0].samples == 4  # 2 sample events + 2 allocation events, one key
    assert hot[0].func_addr is None
    assert diag["events"] == 6
    assert diag["sample_events"] == 2
    assert diag["alloc_events"] == 2
    assert diag["samples"] == 4
    assert diag["rejected"] == 0
    assert diag["rows"] == 1


def test_jfr_allocation_bytes_from_tlab_size_then_allocation_size() -> None:
    rows, diag = parse_jfr_json(json.dumps(JFR))
    assert sum(r.alloc_bytes for r in rows) == 4096 + 1024
    assert diag["alloc_bytes"] == 4096 + 1024
    # ``tlabSize`` wins over a sibling ``allocationSize`` on the same event.
    only_tlab = json.dumps(
        [
            {
                "type": "jdk.ObjectAllocationInNewTLAB",
                "values": {
                    "tlabSize": 512,
                    "allocationSize": 999,
                    "stackTrace": {"frames": [{"method": {"name": "hot"}}]},
                },
            }
        ]
    )
    tlab_rows, _ = parse_jfr_json(only_tlab)
    assert [(r.symbol, r.samples, r.alloc_bytes) for r in tlab_rows] == [
        ("hot", 1, 512)
    ]


def test_jfr_unknown_event_types_ignored_and_counted() -> None:
    rows, diag = parse_jfr_json(json.dumps(JFR))
    assert diag["ignored"] == 2
    assert diag["ignored_types"] == {"jdk.CPULoad": 1, "jdk.ThreadStart": 1}
    assert len(rows) == 1
    # The ignored types are counted, not rows: an unknown event never rejects.
    assert diag["rejected"] == 0


def test_jfr_aggregates_by_symbol_and_address() -> None:
    events = [
        {
            "type": "jdk.ExecutionSample",
            "values": {
                "stackTrace": {
                    "frames": [{"method": {"name": "hot", "class": {"name": "com/b/B"}}}]
                }
            },
        },
        {
            "type": "jdk.ExecutionSample",
            "values": {
                "stackTrace": {
                    "frames": [{"method": {"name": "hot", "class": {"name": "com/b/B"}}}]
                }
            },
        },
        {
            "type": "jdk.ExecutionSample",
            "values": {
                "stackTrace": {
                    "frames": [
                        {
                            "method": {"name": "hot", "class": {"name": "com/b/B"}},
                            "address": "0x140001000",
                        }
                    ]
                }
            },
        },
        {
            "type": "jdk.ExecutionSample",
            "values": {"stackTrace": {"frames": ["com/b/B.verbatim"]}},
        },
    ]
    rows, diag = parse_jfr_json(json.dumps(events))
    assert {(r.symbol, r.func_addr): r.samples for r in rows} == {
        ("com.b.B.hot", None): 2,
        ("com.b.B.hot", 0x140001000): 1,
        ("com/b/B.verbatim", None): 1,
    }
    assert diag["rows"] == 3 and diag["samples"] == 4 and diag["rejected"] == 0
    # Address-bearing rows sort ahead of symbol-only rows (P1 sort key).
    assert rows[0].func_addr == 0x140001000


def test_jfr_dateless_and_typeless_events_are_rejected_not_fatal() -> None:
    payload = json.dumps(
        [
            "not-an-event",
            {"values": {"x": 1}},
            {"type": "jdk.ExecutionSample", "values": {"thread": {}}},
        ]
    )
    rows, diag = parse_jfr_json(payload)
    assert rows == []
    assert diag["events"] == 3
    # Two structurally unusable events plus one sample with no stack trace.
    assert diag["rejected"] == 3
    assert diag["ignored"] == 0


def test_jfr_refuses_non_json_and_event_less_json() -> None:
    with pytest.raises(ProfileError):
        parse_jfr_json("not json at all")
    with pytest.raises(ProfileError):
        parse_jfr_json("")
    with pytest.raises(ProfileError):
        parse_jfr_json('{"foo": 1}')


def test_diagnostics_are_json_plain() -> None:
    _, async_diag = parse_async_collapsed(ASYNC)
    _, jfr_diag = parse_jfr_json(json.dumps(JFR))
    for diag in (async_diag, jfr_diag):
        assert _scalars_only(diag)
        assert json.loads(json.dumps(diag)) == diag


# --------------------------------------------------------------------------- #
# detect_format
# --------------------------------------------------------------------------- #


def test_detect_format_both_and_none() -> None:
    assert detect_format(ASYNC) == ASYNC_FORMAT
    assert detect_format("--- Execution profile ---\ncom/a;com/b 5") == ASYNC_FORMAT
    assert detect_format(json.dumps(JFR)) == JFR_FORMAT
    assert detect_format('{"events": []}') == JFR_FORMAT
    assert detect_format('[{"type": "jdk.ExecutionSample", "values": {}}]') == JFR_FORMAT
    assert detect_format("alpha,beta\n1,2") is None
    assert detect_format("") is None
    assert detect_format("single_frame_no_stack 5") is None


def test_detect_format_accepts_a_path(tmp_path: Path) -> None:
    jfr_path = tmp_path / "hotspots.jfr.json"
    jfr_path.write_text(json.dumps(JFR), encoding="utf-8")
    async_path = tmp_path / "collapsed.txt"
    async_path.write_text(ASYNC, encoding="utf-8")
    assert detect_format(jfr_path) == JFR_FORMAT
    assert detect_format(async_path) == ASYNC_FORMAT


# --------------------------------------------------------------------------- #
# interface contract + end to end
# --------------------------------------------------------------------------- #


def test_reexports_profile_types_without_redefinition() -> None:
    import toothpick.profile as profile_module
    import toothpick.profile_jfr as jfr_module

    assert jfr_module.ProfileSample is profile_module.ProfileSample
    assert jfr_module.ProfileFormat is profile_module.ProfileFormat
    assert jfr_module.ProfileSample is ProfileSample
    assert set(jfr_module.__all__) == {
        "ASYNC_FORMAT",
        "JFR_FORMAT",
        "ProfileFormat",
        "detect_format",
        "parse_async_collapsed",
        "parse_jfr_json",
    }


def test_end_to_end_import_samples_attributes_jfr_and_async(tmp_path: Path) -> None:
    with Database(tmp_path / "graph.db") as db:
        _seed(db)

        jfr_rows, jfr_diag = parse_jfr_json(json.dumps(JFR))
        result = import_samples(db, SHA, jfr_rows, create=True)
        assert jfr_diag["rows"] == 1
        assert result["imported"] == 1  # the J1-style dotted name matches a funcs row
        assert result["total_samples"] == 4
        assert result["attributed_samples"] == 4
        assert result["rejected"] == 0

        async_rows, _ = parse_async_collapsed(ASYNC)
        result2 = import_samples(db, SHA, async_rows, create=True)
        # 0x140001000/0x140002000 attribute by address; the name-only rows match
        # no funcs row and stay unattributed -- never fabricated.
        assert result2["imported"] == 2
        assert result2["total_samples"] == 42 + 7 + 5 + 1500
        assert result2["attributed_samples"] == 7 + 5
        assert result2["unattributed"] == 3
        assert db.count_functions(SHA) == 2
        assert {f.name for f in db.get_functions(SHA)} == {
            "com.b.B.hot",
            "sub_140002000",
        }
