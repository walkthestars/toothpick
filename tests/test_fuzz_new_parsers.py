"""Fuzz tests for the new parser entry points: JVM, RTTI and JFR-async (FUZZ2).

Binding contract: ``docs/units/UNIT-FUZZ2-PLAN.md`` section 1 (Tests) and the
strict invariant carried by that plan and by ``docs/execution-plan.md`` section 4
(parser fuzz): every malformed, truncated or hostile input handed to

    ``toothpick.jvm_parse.parse_class_file``
    ``toothpick.jvm_parse.parse_jar``
    ``toothpick.rtti_parse.scan_vtables``
    ``toothpick.profile_jfr.parse_async_collapsed``
    ``toothpick.profile_jfr.parse_jfr_json``

must either return normally or raise one of the *structured* errors
(:class:`~toothpick.jvm_parse.JvmError`, :class:`~toothpick.rtti_parse.RttiError`,
:class:`~toothpick.profile.ProfileError`).  Any other exception --
``IndexError``, ``struct.error``, ``UnicodeDecodeError``, ``OverflowError``,
``MemoryError``, ... -- is a leak and fails the test.

Everything here is self-contained and deterministic: no JDK, no real artifact and
no file on disk is read.  The seeds are assembled in memory with :mod:`struct`
(a minimal class file), :mod:`zipfile` (a JAR), explicit ``struct.pack_into``
offsets (an x86 MSVC RTTI image) and literal text (async-profiler collapsed and
``jfr print --json``).  :class:`ByteMutator` is a *seeded* byte-level mutator
(:data:`MUTATOR_SEED`, :data:`ITERATIONS_PER_TARGET` per target), so a failure is
reproducible from the seed and the iteration label alone.

Hang guard: every seed is well under :data:`MAX_SEED_SIZE` (4KB) and the battery
is bounded (5 targets x a few hundred cases), so no timeout beyond the suite's
own is needed.

Trap 11: this suite never imports ``faulthandler``; ``pyproject.toml`` keeps
``-p no:faulthandler`` pinned.
"""

from __future__ import annotations

import io
import json
import random
import struct
import zipfile
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest

from toothpick.jvm_parse import JvmError, parse_class_file, parse_jar
from toothpick.profile import ProfileError
from toothpick.profile_jfr import parse_async_collapsed, parse_jfr_json
from toothpick.rtti_parse import RttiError, scan_vtables

#: Structured refusals -- the *only* exceptions a fuzzed input may raise.
STRUCTURED_ERRORS = (JvmError, RttiError, ProfileError)

#: Fixed mutator seed: the corpus is a pure function of it (reproducibility).
MUTATOR_SEED = 0x2F0220

#: Mutation iterations per target (plan: "bounded count, seeded RNG").
ITERATIONS_PER_TARGET = 200

#: Longest seed in this module (hang guard: inputs stay small).
MAX_SEED_SIZE = 4096

#: Image base / vtable candidate of the synthetic x86 RTTI fixture.
RTTI_BASE = 0x400000
#: Candidate vtable address; its ``vftable[-1]`` slot sits 4 bytes before it.
RTTI_VTABLE = RTTI_BASE + 0x104


# --------------------------------------------------------------------------- #
# Seeds (self-contained, in memory -- no file, no JDK, no real artifact)
# --------------------------------------------------------------------------- #


def _seed_class_file() -> bytes:
    """A minimal, structurally valid class file assembled with :mod:`struct`.

    One ``Foo`` class (super ``java/lang/Object``) with one method ``run()V``
    whose ``Code`` holds a single ``return`` (``0xB1``), plus exactly the
    constant-pool entries those structures reference.  Indices are derived from
    a slot counter rather than hand-guessed, and nothing is fabricated: the
    parser can only recover what is packed here.
    """
    slots: List[bytes] = []

    def add(entry: bytes) -> int:
        slots.append(entry)
        return len(slots)  # 1-based CP index

    def add_utf8(text: str) -> int:
        raw = text.encode("utf-8")
        return add(b"\x01" + struct.pack(">H", len(raw)) + raw)

    def add_class(name: str) -> int:
        return add(b"\x07" + struct.pack(">H", add_utf8(name)))

    this_index = add_class("Foo")
    super_index = add_class("java/lang/Object")
    run_name = add_utf8("run")
    run_descriptor = add_utf8("()V")
    code_name = add_utf8("Code")

    body = bytearray()
    body.extend(struct.pack(">HHH", 0x0021, this_index, super_index))  # access/this/super
    body.extend(struct.pack(">H", 0))  # interfaces: none
    body.extend(struct.pack(">H", 0))  # fields: none
    body.extend(struct.pack(">H", 1))  # methods: one
    body.extend(struct.pack(">HHHH", 0x0001, run_name, run_descriptor, 1))
    code = b"\xb1"  # return
    code_attribute = (
        struct.pack(">HHI", 4, 4, len(code))  # max_stack, max_locals, code_length
        + code
        + struct.pack(">HH", 0, 0)  # no exception table, no sub-attributes
    )
    body.extend(struct.pack(">HI", code_name, len(code_attribute)) + code_attribute)
    body.extend(struct.pack(">H", 0))  # class attributes: none

    header = (
        b"\xca\xfe\xba\xbe"
        + struct.pack(">HH", 0, 52)  # minor 0, major 52 (Java 8)
        + struct.pack(">H", len(slots) + 1)  # constant_pool_count
        + b"".join(slots)
    )
    return bytes(header + bytes(body))


def _seed_jar() -> bytes:
    """A minimal in-memory JAR (stored entries, so the bytes are deterministic)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("app/Main.class", _seed_class_file())
        archive.writestr("META-INF/MANIFEST.MF", b"Manifest-Version: 1.0\n")
        archive.writestr("README.txt", b"not bytecode\n")
    return buf.getvalue()


def _seed_rtti_x86() -> bytes:
    """A synthetic x86 MSVC RTTI image: class ``Foo`` whose base is ``Bar``.

    Every COL/CHD/BCD link is a 4-byte absolute virtual address (the x86 rule)
    and the ``vftable[-1]`` slot points at the COL, so ``scan_vtables`` at
    :data:`RTTI_VTABLE` can walk it.  Offsets are named so the layout is
    auditable: slot ``0x100``, COL ``0x140``, CHD ``0x1A0``, base-class array
    ``0x1C0``, BCD ``0x1E0``, type descriptors ``0x240`` (``Foo``) and ``0x280``
    (``Bar``, read through an 8-byte x86 header).
    """
    image = bytearray(0x300)

    def u32(offset: int, value: int) -> None:
        struct.pack_into("<I", image, offset, value)

    def s32(offset: int, value: int) -> None:
        struct.pack_into("<i", image, offset, value)

    def cstr(offset: int, text: str) -> None:
        raw = text.encode("ascii") + b"\x00"
        struct.pack_into(f"<{len(raw)}s", image, offset, raw)

    u32(0x100, RTTI_BASE + 0x140)  # vftable[-1]: absolute COL pointer (x86)
    u32(0x140, 0)  # COL.signature: absolute-pointer form
    u32(0x140 + 4, 0)  # COL.offset
    u32(0x140 + 8, 0)  # COL.cd_offset
    u32(0x140 + 12, RTTI_BASE + 0x240)  # COL.type_descriptor (absolute VA)
    u32(0x140 + 16, RTTI_BASE + 0x1A0)  # COL.class_hierarchy (absolute VA)
    u32(0x1A0, 0)  # CHD.signature
    u32(0x1A0 + 4, 0)  # CHD.attributes
    u32(0x1A0 + 8, 1)  # CHD.num_base_classes
    u32(0x1A0 + 12, RTTI_BASE + 0x1C0)  # CHD.base_class_array link
    u32(0x1C0, RTTI_BASE + 0x1E0)  # base_class_array[0] -> BCD
    u32(0x1E0, RTTI_BASE + 0x280)  # BCD.type_descriptor -> Bar
    u32(0x1E4, 1)  # BCD.num_contained_bases
    s32(0x1E8, 0)  # BCD.mdisp
    s32(0x1EC, -1)  # BCD.pdisp
    s32(0x1F0, 0)  # BCD.vdisp
    u32(0x1F4, 0)  # BCD.attributes
    u32(0x1F8, RTTI_BASE + 0x1A0)  # BCD.class_hierarchy
    cstr(0x248, ".?AVFoo@@")  # TypeDescriptor name (8-byte x86 header at 0x240)
    cstr(0x288, ".?AUBar@@")  # base TypeDescriptor name (header at 0x280)
    return bytes(image)


#: async-profiler collapsed text: a banner (skipped), a symbols-only stack and a
#: PC-only stack carrying both weight columns.
ASYNC_SEED_TEXT = "\n".join(
    [
        "# async-profiler collapsed",
        "--- Execution profile ---",
        "com/a.A;com/b.B.hot 42",
        "com/a.A;0x140001000 7 1024",
        "",
    ]
)

#: ``jfr print --json`` shaped projection: one execution sample and one TLAB
#: allocation sample, both attributing ``com/b/B.hot``.
JFR_SEED_TEXT = json.dumps(
    {
        "recording": {
            "events": [
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
            ]
        }
    }
)


# --------------------------------------------------------------------------- #
# The five entry points under fuzz
# --------------------------------------------------------------------------- #


def _call_class(raw: bytes) -> Any:
    return parse_class_file(raw)


def _call_jar(raw: bytes) -> Any:
    return parse_jar(raw)


def _call_rtti(raw: bytes) -> Any:
    return scan_vtables(raw, RTTI_BASE, "x86", [RTTI_VTABLE])


def _call_async(raw: bytes) -> Any:
    # Bytes are decoded the way hostile non-UTF-8 input arrives; the decode is
    # deliberately lenient so the *parser* sees the text, never a decode error.
    return parse_async_collapsed(raw.decode("utf-8", errors="replace"))


def _call_jfr(raw: bytes) -> Any:
    return parse_jfr_json(raw.decode("utf-8", errors="replace"))


#: ``name -> (seed_factory, entry_point)``; the factory is called per test so no
#: mutable buffer is shared between tests and no module-level state leaks.
TARGETS: Dict[str, Tuple[Callable[[], bytes], Callable[[bytes], Any]]] = {
    "class_file": (_seed_class_file, _call_class),
    "jar": (_seed_jar, _call_jar),
    "rtti_vtables": (_seed_rtti_x86, _call_rtti),
    "async_collapsed": (lambda: ASYNC_SEED_TEXT.encode("utf-8"), _call_async),
    "jfr_json": (lambda: JFR_SEED_TEXT.encode("utf-8"), _call_jfr),
}

TARGET_NAMES = sorted(TARGETS)


# --------------------------------------------------------------------------- #
# Seeded ByteMutator
# --------------------------------------------------------------------------- #


class ByteMutator:
    """Deterministic, seeded byte-level mutator (the FUZZ2 mutation battery).

    Seven mutation kinds; one to three are applied to a fresh copy of the seed
    per :meth:`next_bytes` call, all choices drawn from a single
    :class:`random.Random` seeded with :data:`MUTATOR_SEED`:

    ``truncate``  cut the buffer at a random offset (the truncation class);
    ``set``       overwrite one byte with an arbitrary value;
    ``flip``      XOR one byte with a single random bit;
    ``insert``    splice one random byte in (grows the buffer / shifts fields);
    ``delete``    drop one byte;
    ``length``    overwrite a 2- or 4-byte big-endian field (the length-prefix /
                  count / offset corruption class of JVMS and ZIP headers);
    ``splice``    copy a random span over another random span.

    The corpus is a pure function of the token sequence: the same seed yields the
    same cases on every run and every platform, so a failure is reproducible from
    the seed plus the yielded label.
    """

    _KINDS = ("truncate", "set", "flip", "insert", "delete", "length", "splice")

    #: Length-field values the ``length`` kind picks from (0 / 1 / all-ones /
    #: off-by-one / oversized-prefix).
    _LENGTH_VALUES = (0, 1, 0xFF, 0xFFFF, 0x100, 0x4000)

    def __init__(self, seed: int = MUTATOR_SEED) -> None:
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def next_bytes(self, data: bytes) -> Tuple[str, bytes]:
        """One mutated buffer plus the ``+``-joined labels that produced it."""
        buf = bytearray(data)
        kinds: List[str] = []
        for _ in range(self._rng.randint(1, 3)):
            if not buf:
                # An empty buffer can only be grown; keep mutating rather than
                # emitting the same empty input again.
                buf.insert(0, self._rng.randrange(256))
                kinds.append("insert")
                continue
            kind = self._KINDS[self._rng.randrange(len(self._KINDS))]
            kinds.append(kind)
            if kind == "truncate":
                del buf[self._rng.randrange(len(buf)) :]
            elif kind == "set":
                buf[self._rng.randrange(len(buf))] = self._rng.randrange(256)
            elif kind == "flip":
                pos = self._rng.randrange(len(buf))
                buf[pos] ^= 1 << self._rng.randrange(8)
            elif kind == "insert":
                buf.insert(self._rng.randrange(len(buf) + 1), self._rng.randrange(256))
            elif kind == "delete":
                del buf[self._rng.randrange(len(buf))]
            elif kind == "length":
                width = 2 if self._rng.randrange(2) else 4
                if len(buf) >= width:
                    pos = self._rng.randrange(len(buf) - width + 1)
                    value = self._rng.choice(self._LENGTH_VALUES) & (
                        0xFFFF if width == 2 else 0xFFFFFFFF
                    )
                    fmt = ">H" if width == 2 else ">I"
                    struct.pack_into(fmt, buf, pos, value)
            else:  # splice
                span = min(self._rng.randint(1, 8), len(buf))
                src = self._rng.randrange(len(buf) - span + 1)
                dst = self._rng.randrange(len(buf) - span + 1)
                buf[dst : dst + span] = bytes(buf[src : src + span])
        return "+".join(kinds), bytes(buf)

    def corpus(
        self, data: bytes, iterations: int = ITERATIONS_PER_TARGET
    ) -> Iterator[Tuple[str, bytes]]:
        """``(label, blob)`` cases: the seed, the empty input, then ``iterations``."""
        yield "seed", data
        yield "empty", b""
        for index in range(iterations):
            label, blob = self.next_bytes(data)
            yield f"iter{index}:{label}", blob


#: Hostile "garbage" byte cases every target must survive, per the plan.
GARBAGE_BLOBS: Tuple[Tuple[str, bytes], ...] = (
    ("zero", b"\x00"),
    ("zeros", b"\x00" * 64),
    ("ff", b"\xff" * 64),
    ("alternating", bytes(range(256))),
    ("ascii-garbage", b"this is not a class file, a JAR, an image or a profile\n"),
    ("pk-header-only", b"PK\x03\x04"),
    ("eocd-empty", b"PK\x05\x06" + b"\x00" * 18),
    ("eocd-hostile", b"PK\x05\x06" + b"\xff" * 18),
    ("cafebabe-only", b"\xca\xfe\xba\xbe" + b"\xff" * 12),
    ("utf16-surrogates", b"\xed\xa0\x80\xed\xb0\x80" * 8),
)


def _seed_and_target(name: str) -> Tuple[bytes, Callable[[bytes], Any]]:
    factory, entry = TARGETS[name]
    return factory(), entry


def _fuzz(entry: Callable[[bytes], Any], cases: Iterator[Tuple[str, bytes]]) -> List[str]:
    """Run every case; return one message per unstructured-exception leak."""
    leaks: List[str] = []
    for label, blob in cases:
        try:
            entry(blob)
        except STRUCTURED_ERRORS:
            continue
        except Exception as exc:  # noqa: BLE001 - catching everything is the point
            leaks.append(f"{label}: {type(exc).__name__}: {exc} (blob[{len(blob)}])")
    return leaks


# --------------------------------------------------------------------------- #
# Seed sanity: the generators must produce fixtures their parsers accept
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_seed_is_accepted_by_its_target(name: str) -> None:
    """A seed its own parser refuses would fuzz nothing but the refusal path."""
    seed, entry = _seed_and_target(name)
    assert 0 < len(seed) <= MAX_SEED_SIZE, f"{name}: seed size {len(seed)} out of bounds"
    try:
        result = entry(seed)
    except STRUCTURED_ERRORS as exc:  # pragma: no cover - seed bug, not parser bug
        pytest.fail(
            f"{name}: seed was refused by its own parser: {type(exc).__name__}: {exc}"
        )
    assert result is not None


def test_seed_fixtures_are_structurally_meaningful() -> None:
    """The seeds are built, not stubbed: each carries the fields the plan names."""
    cls = parse_class_file(_seed_class_file())
    assert cls.name == "Foo"
    assert cls.super_name == "java/lang/Object"
    assert [(method.name, method.code_len) for method in cls.methods] == [("run", 1)]

    classes, diagnostics = parse_jar(_seed_jar())
    assert [record.name for record in classes] == ["Foo"]
    assert diagnostics["entries"] == 3
    assert diagnostics["skipped"] == 2  # META-INF/ manifest + README.txt
    assert diagnostics["classes"] == 1

    scan = scan_vtables(_seed_rtti_x86(), RTTI_BASE, "x86", [RTTI_VTABLE])
    assert [record.name for record in scan.classes] == ["Foo"]
    assert [base.name for base in scan.classes[0].bases] == ["Bar"]
    assert scan.rejected == []

    samples, _diag = parse_async_collapsed(ASYNC_SEED_TEXT)
    # The leaf frame text is the symbol verbatim (the P1 epistemic rule).
    assert {sample.symbol for sample in samples} == {"com/b.B.hot", "0x140001000"}
    jfr_samples, _jfr_diag = parse_jfr_json(JFR_SEED_TEXT)
    assert [sample.symbol for sample in jfr_samples] == ["com.b.B.hot"]


# --------------------------------------------------------------------------- #
# The mutator itself must be deterministic and non-trivial
# --------------------------------------------------------------------------- #


def test_mutator_is_deterministic_for_a_fixed_seed() -> None:
    """Two runs of the same seed produce the identical corpus (reproducibility)."""
    seed = _seed_class_file()
    first = [blob for _label, blob in ByteMutator(MUTATOR_SEED).corpus(seed, 50)]
    second = [blob for _label, blob in ByteMutator(MUTATOR_SEED).corpus(seed, 50)]
    assert first == second
    other = [blob for _label, blob in ByteMutator(MUTATOR_SEED + 1).corpus(seed, 50)]
    assert other != first, "a different seed must produce a different corpus"


def test_mutator_actually_mutates_and_covers_kinds() -> None:
    """The corpus is not a no-op: it changes bytes and exercises every kind."""
    seed = _seed_class_file()
    cases = list(ByteMutator(MUTATOR_SEED).corpus(seed, ITERATIONS_PER_TARGET))
    assert len(cases) == ITERATIONS_PER_TARGET + 2
    kinds = {
        kind
        for label, _blob in cases
        if ":" in label
        for kind in label.split(":", 1)[1].split("+")
    }
    assert set(ByteMutator._KINDS) <= kinds
    blobs = [blob for _label, blob in cases]
    assert any(blob != seed for blob in blobs)
    assert any(len(blob) != len(seed) for blob in blobs)
    assert any(len(blob) < len(seed) for blob in blobs)
    assert any(len(blob) > len(seed) for blob in blobs)


# --------------------------------------------------------------------------- #
# The fuzz invariant: structured error or clean return, never anything else
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_seeded_mutations_leak_no_unstructured_exception(name: str) -> None:
    """Seeded ByteMutator battery: only structured refusals or clean returns."""
    seed, entry = _seed_and_target(name)
    leaks = _fuzz(entry, ByteMutator(MUTATOR_SEED).corpus(seed))
    assert not leaks, (
        f"{name}: {len(leaks)} unstructured exception leak(s) at seed "
        f"{MUTATOR_SEED:#x}: " + " | ".join(leaks[:5])
    )


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_empty_and_garbage_inputs_are_structurally_refused(name: str) -> None:
    """Empty, garbage and truncation-prefix inputs: no unstructured exception."""
    seed, entry = _seed_and_target(name)
    cases: List[Tuple[str, bytes]] = [("empty", b""), *GARBAGE_BLOBS]
    cases += [(f"truncate@{cut}", seed[:cut]) for cut in range(len(seed) + 1)]
    leaks = _fuzz(entry, iter(cases))
    assert not leaks, f"{name}: unstructured leaks on empty/garbage/truncation: {leaks[:5]}"


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_every_prefix_truncation_is_structurally_refused(name: str) -> None:
    """A truncated prefix is the highest-yield hostile input; assert it directly."""
    seed, entry = _seed_and_target(name)
    for cut in range(len(seed)):
        try:
            entry(seed[:cut])
        except STRUCTURED_ERRORS:
            continue
        except Exception as exc:  # noqa: BLE001 - catching everything is the point
            pytest.fail(
                f"{name}: truncation at {cut}/{len(seed)} raised "
                f"{type(exc).__name__}: {exc}"
            )


def test_text_target_boundaries_are_structured() -> None:
    """Text-level hostility for both text parsers: only ProfileError or a clean return."""
    async_cases = [
        "",
        "\x00",
        "\x00" * 100,
        "stack 5",
        "no-count-column",
        "a;" * 100 + " 1",
        ";" * 50 + " 1",
        "a;;b 1",
        "a;b; 1",
        "a;b;c 1 2 3 4 5",
        "a;b 0",
        "a;b 1 1",
        "a;b 1_024",
        "a;b 1.2K",
        "a;b 1e2",
        "a;b 1e-2",
        "a;b 0x10",
        "a;b " + "9" * 48,
        "\ufeffa;b 1",
        "a;b 1\r\na;c 2",
        "a;b 1\t7",
        "a b c d e f 1",
        "---\n+++\n",
    ]
    jfr_cases = [
        "",
        " ",
        "\x00",
        "{}",
        "[]",
        "null",
        "true",
        "42",
        '"str"',
        "not json at all",
        '{"events": "x"}',
        '{"events": 5}',
        '{"events": [1, 2, 3]}',
        '{"events": [{"type": "jdk.ExecutionSample"}]}',
        '{"events": [{"type": "jdk.ExecutionSample", "values": {"stackTrace": {}}}]}',
        '{"events": [{"type": "jdk.ExecutionSample", "values": {"stackTrace": {"frames": []}}}]}',
        '{"events": [{"type": "jdk.ExecutionSample", "values": {"stackTrace": {"frames": [[[[[1]]]]]}}}]}',
        '{"events": [{"type": "jdk.ExecutionSample", "values": {"stackTrace": {"frames": [{}]}}}]}',
        '{"events": {"jdk.ExecutionSample": "x"}}',
        '{"events": {"jdk.ExecutionSample": [{"values": {"stackTrace": {"frames": ["a/b.C.m"]}}}]}}',
        '{"recording": {"events": []}}',
        '{"recording": {"recording": {"recording": {"recording": {"recording": {"events": []}}}}}}',
        "\ufeff" + JFR_SEED_TEXT,
        '{"events": [{"type": "jdk.ObjectAllocationInNewTLAB", "values": {"tlabSize": "x", "stackTrace": {"frames": [{"method": "m"}]}}}]}',
        '{"events": [{"type": "jdk.ObjectAllocationInNewTLAB", "values": {"tlabSize": true, "stackTrace": {"frames": [{"method": "m"}]}}}]}',
        '{"events": [{"type": "jdk.ObjectAllocationInNewTLAB", "values": {"tlabSize": -1, "stackTrace": {"frames": [{"method": "m"}]}}}]}',
        '{"events": [{"type": "jdk.ExecutionSample", "values": {"stackTrace": {"frames": [{"address": "zzz", "method": {"name": "m"}}]}}}]}',
    ]
    leaks = _fuzz(
        parse_async_collapsed,
        iter((f"async:{text[:24]!r}", text.encode("utf-8")) for text in async_cases),
    )
    leaks += _fuzz(
        parse_jfr_json,
        iter((f"jfr:{text[:24]!r}", text.encode("utf-8")) for text in jfr_cases),
    )
    assert not leaks, f"text-boundary leaks: {leaks[:5]}"


def test_rtti_argument_boundaries_are_structured() -> None:
    """Hostile ``vtable_addrs`` / ``max_vtables`` / ``arch`` inputs: RttiError only."""
    image = _seed_rtti_x86()
    cases: List[Tuple[str, Dict[str, Any]]] = [
        ("negative-candidate", {"vtable_addrs": [-1]}),
        ("zero-candidate", {"vtable_addrs": [0]}),
        ("huge-candidate", {"vtable_addrs": [1 << 60]}),
        ("bool-candidate", {"vtable_addrs": [True]}),
        ("text-candidate", {"vtable_addrs": ["0x400104"]}),
        ("bytes-candidate", {"vtable_addrs": b"\x04\x01\x40"}),
        ("none-candidate", {"vtable_addrs": None}),
        ("empty-candidates", {"vtable_addrs": []}),
        ("duplicate-candidates", {"vtable_addrs": [RTTI_VTABLE] * 8}),
        ("many-candidates", {"vtable_addrs": list(range(RTTI_BASE, RTTI_BASE + 4000, 4))}),
        ("zero-budget", {"max_vtables": 0}),
        ("negative-budget", {"max_vtables": -3}),
        ("bool-budget", {"max_vtables": True}),
        ("text-budget", {"max_vtables": "many"}),
        ("huge-budget", {"max_vtables": 10_000}),
        ("bad-arch", {"arch": "arm"}),
        ("none-arch", {"arch": None}),
        ("negative-base", {"base_addr": -1}),
        ("bool-base", {"base_addr": True}),
    ]
    leaks: List[str] = []
    for label, overrides in cases:
        kwargs: Dict[str, Any] = {
            "data": image,
            "base_addr": RTTI_BASE,
            "arch": "x86",
            "vtable_addrs": [RTTI_VTABLE],
        }
        kwargs.update(overrides)
        try:
            scan_vtables(**kwargs)
        except RttiError:
            continue
        except TypeError as exc:  # pragma: no cover - signature drift, not a leak
            pytest.fail(f"rtti case {label}: scan_vtables signature changed: {exc}")
        except Exception as exc:  # noqa: BLE001 - catching everything is the point
            leaks.append(f"{label}: {type(exc).__name__}: {exc}")
    assert not leaks, f"rtti argument-boundary leaks: {leaks}"


def test_jvm_argument_boundaries_are_structured() -> None:
    """Non-bytes / hostile-typed input to the JVM entry points: JvmError only."""
    cases: List[Tuple[str, Callable[[], Any]]] = [
        ("class-none", lambda: parse_class_file(None)),
        ("class-text", lambda: parse_class_file("not bytes")),
        ("class-int", lambda: parse_class_file(1234)),
        ("class-list", lambda: parse_class_file([0xCA, 0xFE, 0xBA, 0xBE])),
        ("class-bytearray-empty", lambda: parse_class_file(bytearray())),
        ("jar-none", lambda: parse_jar(None)),
        ("jar-int", lambda: parse_jar(7)),
        ("jar-missing-path", lambda: parse_jar("definitely/not/a/jar.jar")),
    ]
    leaks: List[str] = []
    for label, call in cases:
        try:
            call()
        except (JvmError, ProfileError):
            continue
        except Exception as exc:  # noqa: BLE001 - catching everything is the point
            leaks.append(f"{label}: {type(exc).__name__}: {exc}")
    assert not leaks, f"jvm argument-boundary leaks: {leaks}"
