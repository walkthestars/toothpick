"""Toothpick IDA consumer (CONS-01): merge icepick extract docs into toothpick.db.

The IDA side is owned by icepick; this module only *consumes* its versioned
JSON documents. The boundary rule is absolute: JSON documents cross it, never
code, never IDA bits.

Schema functions are VENDORED below (not imported): a cross-repo import would
couple Toothpick's runtime to an icepick checkout, which is never guaranteed.
Sync rule: when icepick bumps ``SCHEMA_VERSION``, re-vendor this block from
``icepick/src/icepick/schema.py`` and update ``VENDORED_FROM``; the pin in
:func:`ingest_extract_doc` rejects drift loudly instead of ingesting it.

Name discipline: every IDA-proposed name goes through
:mod:`toothpick.name_conflicts` (``report_conflict``), which is the single
decider. This module never writes ``funcs.name`` directly with an IDA hint.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from toothpick import name_conflicts
from toothpick.storage import (
    Database,
    EdgeRecord,
    FunctionRecord,
    StorageError,
)

# ---------------------------------------------------------------------------
# vendored schema block — source: icepick/src/icepick/schema.py, SCHEMA v1.
# Vendored surface is the EXTRACT side only (validate + canonicalise +
# digest); the labels side never crosses this boundary. Re-vendor on any
# icepick schema bump; SCHEMA_VERSION pin below rejects drift at ingest.
# ---------------------------------------------------------------------------

VENDORED_FROM = "icepick/src/icepick/schema.py @ SCHEMA_VERSION v1"

SCHEMA_VERSION = "v1"
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (SCHEMA_VERSION,)

_BACKENDS: tuple[str, ...] = ("idalib", "idat")
_EDGE_KINDS: tuple[str, ...] = (
    "call",
    "tail_call",
    "jump",
    "data_ref",
    "vtable",
    "other",
)
_NAME_SOURCES: tuple[str, ...] = ("ida", "flirt", "manual", "dyn")
_DECOMPILE_STATUSES: tuple[str, ...] = ("ok", "too_big", "timeout", "failed")

_MAX_EA = (1 << 64) - 1
_NAME_MAX_LEN = 256

_DECOMPILE_FAILED_RE = re.compile(r"^DECOMPILE_FAILED\([a-z0-9_]+\)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[^\s\x00-\x1f\x7f]+$")
_HEX_RE = re.compile(r"^0x[0-9a-fA-F]{1,16}$")


def _parse_ea(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= _MAX_EA else None
    if isinstance(value, str) and _HEX_RE.match(value.strip()):
        parsed = int(value.strip(), 16)
        return parsed if 0 <= parsed <= _MAX_EA else None
    return None


def _check_ea(value: Any, path: str, errors: List[str]) -> Optional[int]:
    ea = _parse_ea(value)
    if ea is None:
        errors.append(f"{path}: not a valid 64-bit effective address: {value!r}")
    return ea


def _check_name(value: Any, path: str) -> List[str]:
    if not isinstance(value, str) or not value:
        return [f"{path}: expected a non-empty string"]
    errors: List[str] = []
    if len(value) > _NAME_MAX_LEN:
        errors.append(f"{path}: longer than {_NAME_MAX_LEN} characters")
    if not _NAME_RE.match(value):
        errors.append(f"{path}: must not contain whitespace or control characters")
    return errors


def _check_schema_version(obj: Mapping, path: str) -> List[str]:
    if "schema_version" not in obj:
        return [f"{path}.schema_version: missing (expected {SCHEMA_VERSION!r})"]
    version = obj["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        return [
            f"{path}.schema_version: expected {SCHEMA_VERSION!r}, got {version!r}"
        ]
    return []


def _check_provenance(prov: Any, path: str) -> List[str]:
    if not isinstance(prov, Mapping):
        return [
            f"{path}: expected a JSON object with "
            "ida_version/idb_sha256/auto_analysis/backend"
        ]
    errors: List[str] = []
    for key in ("ida_version", "idb_sha256", "auto_analysis", "backend"):
        if key not in prov:
            errors.append(f"{path}.{key}: missing required provenance field")
    ida_version = prov.get("ida_version")
    if ida_version is not None and (
        not isinstance(ida_version, str) or not ida_version.strip()
    ):
        errors.append(f"{path}.ida_version: expected a non-empty string")
    idb_sha = prov.get("idb_sha256")
    if idb_sha is not None and (
        not isinstance(idb_sha, str) or not _SHA256_RE.match(idb_sha)
    ):
        errors.append(f"{path}.idb_sha256: expected a lowercase 64-char hex sha256")
    auto = prov.get("auto_analysis")
    if auto is not None and not isinstance(auto, bool):
        errors.append(f"{path}.auto_analysis: expected true or false")
    backend = prov.get("backend")
    if backend is not None and backend not in _BACKENDS:
        errors.append(
            f"{path}.backend: expected one of 'idalib', 'idat'; got {backend!r}"
        )
    generator = prov.get("generator")
    if generator is not None and (
        not isinstance(generator, str) or not generator.strip()
    ):
        errors.append(f"{path}.generator: expected a non-empty string")
    return errors


def _check_func(func: Any, path: str, seen_eas: Dict[int, str]) -> List[str]:
    if not isinstance(func, Mapping):
        return [f"{path}: expected a JSON object, got {type(func).__name__}"]
    errors: List[str] = []
    ea = _check_ea(func.get("ea"), f"{path}.ea", errors)
    if ea is not None:
        if ea in seen_eas:
            errors.append(
                f"{path}.ea: duplicate effective address {ea:#x} "
                f"(first seen at {seen_eas[ea]})"
            )
        else:
            seen_eas[ea] = path
    errors.extend(_check_name(func.get("name"), f"{path}.name"))
    name_source = func.get("name_source", "ida")
    if name_source not in _NAME_SOURCES:
        errors.append(f"{path}.name_source: unexpected value {name_source!r}")
    flags = func.get("flags", [])
    if not isinstance(flags, list) or not all(
        isinstance(flag, str) and flag for flag in flags
    ):
        errors.append(f"{path}.flags: expected a list of non-empty strings")
    for key in ("size", "blocks"):
        value = func.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{path}.{key}: expected a non-negative integer or null")
    for key in ("callers", "callees"):
        value = func.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            errors.append(f"{path}.{key}: expected a list of effective addresses")
            continue
        for index, item in enumerate(value):
            _check_ea(item, f"{path}.{key}[{index}]", errors)
    decompiled = func.get("decompiled")
    if decompiled is not None and not isinstance(decompiled, str):
        errors.append(f"{path}.decompiled: expected a string or null")
    status = func.get("decompile_status")
    if status is not None and status not in _DECOMPILE_STATUSES:
        errors.append(f"{path}.decompile_status: unexpected value {status!r}")
    if isinstance(decompiled, str):
        if status == "failed":
            errors.append(
                f"{path}: decompiled text is present but decompile_status is 'failed'"
            )
    else:
        if status == "ok":
            errors.append(f"{path}.decompile_status: 'ok' requires decompiled text")
        error = func.get("decompile_error")
        if not (isinstance(error, str) and _DECOMPILE_FAILED_RE.match(error)):
            errors.append(
                f"{path}.decompile_error: expected a DECOMPILE_FAILED(<reason>) "
                "sentinel when decompiled is null"
            )
    return errors


def _check_edge(edge: Any, path: str, seen: set) -> List[str]:
    if not isinstance(edge, Mapping):
        return [f"{path}: expected a JSON object, got {type(edge).__name__}"]
    errors: List[str] = []
    src = _check_ea(edge.get("src"), f"{path}.src", errors)
    dst = _check_ea(edge.get("dst"), f"{path}.dst", errors)
    kind = edge.get("kind", "call")
    if kind not in _EDGE_KINDS:
        errors.append(f"{path}.kind: unexpected value {kind!r}")
    if src is not None and dst is not None and kind in _EDGE_KINDS:
        key = (src, dst, kind)
        if key in seen:
            errors.append(f"{path}: duplicate edge {src:#x} -> {dst:#x} ({kind})")
        else:
            seen.add(key)
    return errors


def validate_extract(obj: Any) -> List[str]:
    """Vendored: validate an extract document; ``[]`` when valid."""
    if not isinstance(obj, Mapping):
        return [f"extract: expected a JSON object, got {type(obj).__name__}"]
    errors: List[str] = []
    errors.extend(_check_schema_version(obj, "extract"))
    errors.extend(_check_provenance(obj.get("provenance"), "extract.provenance"))
    funcs = obj.get("funcs")
    if not isinstance(funcs, list):
        errors.append("extract.funcs: expected a list")
    elif not funcs:
        errors.append("extract.funcs: expected at least one function")
    else:
        seen_eas: Dict[int, str] = {}
        for index, func in enumerate(funcs):
            errors.extend(_check_func(func, f"extract.funcs[{index}]", seen_eas))
    edges = obj.get("edges")
    if edges is not None:
        if not isinstance(edges, list):
            errors.append("extract.edges: expected a list")
        else:
            seen_edges: set = set()
            for index, edge in enumerate(edges):
                errors.extend(_check_edge(edge, f"extract.edges[{index}]", seen_edges))
    return errors


def _canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _content_digest(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def _sorted_eas(values: Any) -> List[int]:
    parsed: set = set()
    for value in values or []:
        ea = _parse_ea(value)
        if ea is not None:
            parsed.add(ea)
    return sorted(parsed)


def _norm_func(func: Mapping) -> dict:
    decompiled = func.get("decompiled")
    if isinstance(decompiled, str):
        status = func.get("decompile_status") or "ok"
    else:
        status = func.get("decompile_status") or "failed"
    return {
        "ea": _parse_ea(func.get("ea")),
        "name": func.get("name"),
        "name_source": func.get("name_source", "ida"),
        "size": func.get("size"),
        "decompiled": decompiled if isinstance(decompiled, str) else None,
        "decompile_status": status,
    }


def _norm_provenance(prov: Mapping) -> dict:
    return {
        "ida_version": prov.get("ida_version"),
        "idb_sha256": prov.get("idb_sha256"),
        "auto_analysis": bool(prov.get("auto_analysis")),
        "backend": prov.get("backend"),
        "generator": prov.get("generator"),
    }


def collect_edges(obj: Mapping) -> List[dict]:
    """Vendored: canonical sorted edge list; derives from callees if absent."""
    raw = obj.get("edges")
    pairs: set = set()
    if raw is not None:
        for index, edge in enumerate(raw):
            src = _parse_ea(edge.get("src"))
            dst = _parse_ea(edge.get("dst"))
            kind = edge.get("kind", "call")
            if src is None or dst is None or kind not in _EDGE_KINDS:
                raise ValueError(f"extract.edges[{index}]: unvalidated edge")
            pairs.add((src, dst, kind))
    else:
        for func in obj.get("funcs", []):
            src = _parse_ea(func.get("ea"))
            if src is None:
                raise ValueError("extract.funcs: unvalidated function address")
            for callee in func.get("callees") or []:
                dst = _parse_ea(callee)
                if dst is None:
                    raise ValueError("extract.funcs.callees: unvalidated address")
                pairs.add((src, dst, "call"))
    return [
        {"src": src, "dst": dst, "kind": kind} for src, dst, kind in sorted(pairs)
    ]


def normalize_extract(obj: Mapping) -> dict:
    """Vendored: canonical form of a validated extract document."""
    funcs = [_norm_func(func) for func in obj.get("funcs", [])]
    funcs.sort(key=lambda func: func["ea"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "extract",
        "doc_source": "ida",
        "provenance": _norm_provenance(obj.get("provenance") or {}),
        "funcs": funcs,
        "edges": collect_edges(obj),
    }


def extract_digest(obj: Mapping) -> str:
    """Vendored: digest of the canonical extract (formatting-noise stable)."""
    return _content_digest(normalize_extract(obj))


# ---------------------------------------------------------------------------
# consumer
# ---------------------------------------------------------------------------


class IdaConsumerError(Exception):
    """A document was refused, or the store could not be written."""


#: icepick edge kinds mapped into Toothpick's ``call/xref/cfg`` taxonomy.
#: ``tail_call`` is still a call; reference-like kinds land as xrefs.
TOOTHPICK_EDGE_MAP: Dict[str, str] = {
    "call": "call",
    "tail_call": "call",
    "jump": "xref",
    "data_ref": "xref",
    "vtable": "xref",
    "other": "xref",
}

#: Warn (never refuse) when a doc of at least this many funcs has zero edges
#: (the Teardown-file shape: confident-looking output over encrypted bytes).
_ZERO_EDGE_SCALE = 20

#: Warn (never refuse) when the non-ok decompile share exceeds this.
_FAIL_RATE_WARN = 0.5

_COVERAGE_ADDR = 0  # func_addr sentinel for document-level coverage receipts.


def _first_line(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    return text.strip().splitlines()[0].strip()


def _open_db(target: Union[Database, str, Path]) -> tuple[Database, bool]:
    if isinstance(target, Database):
        return target, False
    try:
        return Database(Path(target), create=True), True
    except StorageError as exc:
        raise IdaConsumerError(f"cannot open database {target}: {exc}") from exc


def ingest_extract_doc(
    target: Union[Database, str, Path],
    doc: Any,
    *,
    binary_sha256: Optional[str] = None,
    name_source: str = "ida-hint",
) -> Dict[str, Any]:
    """Merge one validated schema-v1 extract document into the store.

    Raises :class:`IdaConsumerError` with reasons for schema-invalid input
    (nothing is written in that case). Suspicious-but-valid documents ingest
    with warn-and-continue plus a coverage receipt, never silent success.
    Re-ingest is idempotent: upserts plus deterministic conflict/receipt ids
    mean no duplicate rows and identical query answers.
    """
    errors = validate_extract(doc)
    if errors:
        raise IdaConsumerError(
            "refusing schema-invalid extract document: " + "; ".join(errors[:8])
        )

    digest = extract_digest(doc)
    normalized = normalize_extract(doc)
    prov = normalized["provenance"]
    warnings: List[str] = []

    sha: str
    sha_source: str
    if binary_sha256:
        sha = str(binary_sha256)
        sha_source = "explicit-sha"
    else:
        sha = str(prov.get("idb_sha256") or "")
        sha_source = "idb_sha256-fallback"
        warnings.append(
            "no --sha given: filing rows under provenance idb_sha256; "
            "pass --sha to key by binary hash"
        )
    if not sha:
        raise IdaConsumerError("refusing document with no binary identity")

    db, owned = _open_db(target)
    try:
        name_conflicts.ensure_schema(db)
        filled = 0
        ok_count = 0
        for func in normalized["funcs"]:
            addr = int(func["ea"])
            proposed = func.get("name") if isinstance(func.get("name"), str) else ""
            status = str(func.get("decompile_status") or "failed")
            if status == "ok":
                ok_count += 1
            # The queue is the single decider for the name: FILLED writes the
            # row's name itself (creating the row when missing); PENDING keeps
            # the grounded incumbent; IGNORED leaves ungrounded labels alone.
            result = name_conflicts.report_conflict(
                db, sha, addr, proposed, source=name_source
            )
            if result.outcome is name_conflicts.ConflictOutcome.FILLED:
                filled += 1
            if result.outcome is name_conflicts.ConflictOutcome.IGNORED:
                eff_name = proposed if proposed else ""
            else:
                eff_name = result.name
            decompiled = func.get("decompiled")
            size = func.get("size")
            record = FunctionRecord(
                binary_sha256=sha,
                func_addr=addr,
                name=eff_name,
                size=int(size) if isinstance(size, int) and size >= 0 else 0,
                calling_convention="",
                prototype=_first_line(decompiled),
                ir_pcode=None,  # IDA rows carry no SSA facts; def_use degrades
                decompiled_c=decompiled,
            )
            try:
                db.upsert_function(record, phase="ida_extract", write_receipt=False)
                db.add_receipt(
                    sha,
                    addr,
                    "ida_extract",
                    {
                        "name": eff_name,
                        "size": int(record.size),
                        "decompile_status": status,
                        "verifier_status": "UNGROUNDED",
                        "backend": prov.get("backend"),
                        "idb_sha256": prov.get("idb_sha256"),
                        "generator": prov.get("generator"),
                        "digest": digest,
                    },
                )
            except (StorageError, sqlite3.Error) as exc:
                raise IdaConsumerError(
                    f"failed writing function {addr:#x}: {exc}"
                ) from exc

        edge_records: List[EdgeRecord] = []
        for edge in normalized["edges"]:
            mapped = TOOTHPICK_EDGE_MAP.get(str(edge.get("kind", "call")), "xref")
            edge_records.append(
                EdgeRecord(
                    binary_sha256=sha,
                    src_addr=int(edge["src"]),
                    dst_addr=int(edge["dst"]),
                    kind=mapped,
                )
            )
        try:
            db.add_edges(edge_records)
        except (StorageError, sqlite3.Error) as exc:
            raise IdaConsumerError(f"failed writing edges: {exc}") from exc

        total = len(normalized["funcs"])
        failed = total - ok_count
        if total >= _ZERO_EDGE_SCALE and not edge_records:
            warnings.append(
                f"suspicious document: {total} funcs but zero call edges "
                "(Teardown-file shape: encrypted bulk extracts as ghosts)"
            )
        if total and failed / total > _FAIL_RATE_WARN:
            warnings.append(
                f"high decompile-failure rate: {failed}/{total} funcs "
                "without usable pseudocode"
            )

        try:
            db.add_receipt(
                sha,
                _COVERAGE_ADDR,
                "ida_extract_coverage",
                {
                    "digest": digest,
                    "generator": prov.get("generator"),
                    "backend": prov.get("backend"),
                    "idb_sha256": prov.get("idb_sha256"),
                    "sha_source": sha_source,
                    "funcs": total,
                    "edges": len(edge_records),
                    "decompile_ok": ok_count,
                    "decompile_failed": failed,
                    "warnings": warnings,
                },
            )
        except (StorageError, sqlite3.Error) as exc:
            raise IdaConsumerError(f"failed writing coverage receipt: {exc}") from exc

        pending = name_conflicts.list_conflicts(
            db, status="pending", binary_sha256=sha
        )
        return {
            "binary_sha256": sha,
            "digest": digest,
            "funcs_ingested": total,
            "edges_ingested": len(edge_records),
            "conflicts_pending": len(pending),
            "conflicts_filled": filled,
            "warnings": warnings,
        }
    finally:
        if owned:
            db.close()


def ingest_extract_file(
    db_path: Union[str, Path],
    doc_path: Union[str, Path],
    *,
    binary_sha256: Optional[str] = None,
    name_source: str = "ida-hint",
) -> Dict[str, Any]:
    """Load a JSON extract document from disk and merge it (refuse-first)."""
    path = Path(doc_path)
    if not path.is_file():
        raise IdaConsumerError(f"document not found: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IdaConsumerError(f"cannot read {path}: {exc}") from exc
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdaConsumerError(f"document is not valid UTF-8 JSON: {exc}") from exc
    return ingest_extract_doc(
        db_path, doc, binary_sha256=binary_sha256, name_source=name_source
    )
