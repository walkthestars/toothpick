"""
Toothpick MSVC RTTI recovery (unit P3) -- compatibility facade.

Binding contract: ``docs/units/UNIT-P3-RTTI-PLAN.md`` section 1 (RTTI module) and
section 2 (interface specification).  Design anchors: ``docs/design-document.md``
section 1 ("no fabricated structs -- every class row traces to parsed bytes") and
section 19 / Risk R1 (bounded emission: default 50 scanned vtables, ceiling 1000).

This module holds no RTTI logic of its own: it is the pre-split import site for
the two halves the MSVC RTTI module was moved into.

* :mod:`toothpick.rtti_parse` -- the *parsing* half: constants
  (``DEFAULT_MAX_VTABLES``, ``MAX_VTABLES``, ``MAX_BASE_CLASSES``,
  ``MAX_TYPE_NAME``), data contracts (:class:`ClassRecord`, :class:`BaseRef`,
  :class:`ScanResult`, :class:`RttiError`, :class:`RttiArch`), identity
  (:func:`class_type_id`), name recovery (:func:`demangle_msvc_type`), the
  bounds-checked structure readers (:func:`read_complete_object_locator`,
  :func:`read_type_descriptor`, :func:`read_class_hierarchy_descriptor`,
  :func:`read_base_class_descriptor`) and :func:`scan_vtables`.
* :mod:`toothpick.rtti_store` -- the *persistence* half: :func:`store`,
  :func:`get_class` and :data:`TargetLike` (``Database`` / path / connection
  targets, replace-per-sha onto the ``types`` table).

Recovered shape -- MSVC ABI, 32- and 64-bit:

    VFTable[-1]          -> RTTICompleteObjectLocator (COL)
    COL.type_descriptor  -> RTTITypeDescriptor (mangled name, e.g. ``.?AVFoo@@``)
    COL.class_hierarchy  -> RTTIClassHierarchyDescriptor (CHD)
    CHD.base_class_array[0 .. n-1] -> RTTIBaseClassDescriptor[] (one per base)

Everything exported below is *imported, never redefined*: exactly one
:class:`ClassRecord` / :class:`BaseRef` / :class:`RttiError` exists in the
package, so ``isinstance`` checks made against either half agree.  The epistemic
rules of the parser (every dereference bounds-checked against the caller-supplied
buffer, a truncated structure rejected rather than emitted as a partial class
row, a name only ever the exact mangled string or a positively demangled form of
it) and the persistence discipline (every SQLite handle closed in a ``finally``,
Trap 7) are documented and enforced in those two halves.
"""

from __future__ import annotations

from toothpick.rtti_parse import (
    DEFAULT_MAX_VTABLES,
    MAX_BASE_CLASSES,
    MAX_TYPE_NAME,
    MAX_VTABLES,
    BaseRef,
    ClassRecord,
    RttiArch,
    RttiError,
    ScanResult,
    class_type_id,
    demangle_msvc_type,
    read_base_class_descriptor,
    read_class_hierarchy_descriptor,
    read_complete_object_locator,
    read_type_descriptor,
    resolve_max_vtables,
    scan_vtables,
)
from toothpick.rtti_store import TargetLike, get_class, store

#: The public surface of the pre-split RTTI module, re-exported unchanged: the
#: parsing half first (constants, taxonomy, readers, scan), then the persistence
#: half (``store`` / ``get_class`` / ``TargetLike``).
__all__ = [
    "DEFAULT_MAX_VTABLES",
    "MAX_BASE_CLASSES",
    "MAX_TYPE_NAME",
    "MAX_VTABLES",
    "BaseRef",
    "ClassRecord",
    "RttiArch",
    "RttiError",
    "ScanResult",
    "TargetLike",
    "class_type_id",
    "demangle_msvc_type",
    "get_class",
    "read_base_class_descriptor",
    "read_class_hierarchy_descriptor",
    "read_complete_object_locator",
    "read_type_descriptor",
    "resolve_max_vtables",
    "scan_vtables",
    "store",
]
