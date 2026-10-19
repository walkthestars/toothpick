"""
Toothpick SMT reachability & branch constraint solver (unit W05).

Implements the W05 solver contract per UNIT-W05-SOLVER-PLAN.md, design-document.md §11,
and ADR-07:
- Direct Z3 SMT solver bindings over bitvector ASTs.
- Instruction semantic lifter translating dominating ALU/CMP/TEST/Jcc instructions
  into Z3 BitVec formulas.
- CFG basic block recovery and path constraint extraction.
- Hard timeout fallback (default 5.0 seconds per function) to prevent solver cascades (Risk R4).
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from capstone import Cs
import z3

from toothpick.static_engine import Instruction, arch_spec
from toothpick.storage import Database

__all__ = [
    "SolverError",
    "UnsupportedInstructionError",
    "SolverTimeoutError",
    "SolveResult",
    "ComparisonFact",
    "SymbolicState",
    "BasicBlock",
    "SMTBranchLifter",
    "SMTSolver",
    "parse_ir_pcode",
    "disassemble_bytes",
    "build_basic_blocks",
]

PathLike = Union[str, Path]

# Mnemonics whose predicate comes from a register tested by the branch itself
# (ARM64 cbz/cbnz/tbz/tbnz) rather than from a previously recorded comparison.
_REGISTER_TEST_MNEMONICS = frozenset({"cbz", "cbnz", "tbz", "tbnz"})


def _split_operands(op_str: str) -> List[str]:
    """Split a Capstone operand string, ignoring commas nested inside brackets."""
    if not op_str:
        return []
    tokens: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in op_str:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            tokens.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        tokens.append(tail)
    return [token for token in tokens if token]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SolverError(Exception):
    """Base exception for SMT solver failures."""


class UnsupportedInstructionError(SolverError):
    """Raised when an instruction cannot be lifted into SMT formulas."""


class SolverTimeoutError(SolverError):
    """Raised when solving exceeds the timeout budget."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    """Result of an SMT reachability solve attempt."""

    status: str  # "SAT" | "UNSAT" | "TIMEOUT" | "UNSUPPORTED"
    model: Optional[Dict[str, int]] = None
    constraints: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "constraints": list(self.constraints),
            "elapsed_sec": round(self.elapsed_sec, 4),
        }


# ---------------------------------------------------------------------------
# Architecture Register Layouts
# ---------------------------------------------------------------------------

# Register mapping: reg_name -> (canonical_base_reg, bit_width, bit_offset)
_X86_64_REGS: Dict[str, Tuple[str, int, int]] = {
    # 64-bit base registers
    "rax": ("rax", 64, 0), "rbx": ("rbx", 64, 0), "rcx": ("rcx", 64, 0), "rdx": ("rdx", 64, 0),
    "rsi": ("rsi", 64, 0), "rdi": ("rdi", 64, 0), "rbp": ("rbp", 64, 0), "rsp": ("rsp", 64, 0),
    "r8":  ("r8", 64, 0),  "r9":  ("r9", 64, 0),  "r10": ("r10", 64, 0), "r11": ("r11", 64, 0),
    "r12": ("r12", 64, 0), "r13": ("r13", 64, 0), "r14": ("r14", 64, 0), "r15": ("r15", 64, 0),
    # 32-bit sub-registers
    "eax": ("rax", 32, 0), "ebx": ("rbx", 32, 0), "ecx": ("rcx", 32, 0), "edx": ("rdx", 32, 0),
    "esi": ("rsi", 32, 0), "edi": ("rdi", 32, 0), "ebp": ("rbp", 32, 0), "esp": ("rsp", 32, 0),
    "r8d": ("r8", 32, 0),  "r9d": ("r9", 32, 0),  "r10d": ("r10", 32, 0), "r11d": ("r11", 32, 0),
    "r12d": ("r12", 32, 0), "r13d": ("r13", 32, 0), "r14d": ("r14", 32, 0), "r15d": ("r15", 32, 0),
    # 16-bit sub-registers
    "ax": ("rax", 16, 0), "bx": ("rbx", 16, 0), "cx": ("rcx", 16, 0), "dx": ("rdx", 16, 0),
    "si": ("rsi", 16, 0), "di": ("rdi", 16, 0), "bp": ("rbp", 16, 0), "sp": ("rsp", 16, 0),
    "r8w": ("r8", 16, 0), "r9w": ("r9", 16, 0), "r10w": ("r10", 16, 0), "r11w": ("r11", 16, 0),
    "r12w": ("r12", 16, 0), "r13w": ("r13", 16, 0), "r14w": ("r14", 16, 0), "r15w": ("r15", 16, 0),
    # 8-bit low sub-registers
    "al": ("rax", 8, 0), "bl": ("rbx", 8, 0), "cl": ("rcx", 8, 0), "dl": ("rdx", 8, 0),
    "sil": ("rsi", 8, 0), "dil": ("rdi", 8, 0), "bpl": ("rbp", 8, 0), "spl": ("rsp", 8, 0),
    "r8b": ("r8", 8, 0), "r9b": ("r9", 8, 0), "r10b": ("r10", 8, 0), "r11b": ("r11", 8, 0),
    "r12b": ("r12", 8, 0), "r13b": ("r13", 8, 0), "r14b": ("r14", 8, 0), "r15b": ("r15", 8, 0),
    # 8-bit high sub-registers
    "ah": ("rax", 8, 8), "bh": ("rbx", 8, 8), "ch": ("rcx", 8, 8), "dh": ("rdx", 8, 8),
}

_X86_32_REGS: Dict[str, Tuple[str, int, int]] = {
    # 32-bit base registers
    "eax": ("eax", 32, 0), "ebx": ("ebx", 32, 0), "ecx": ("ecx", 32, 0), "edx": ("edx", 32, 0),
    "esi": ("esi", 32, 0), "edi": ("edi", 32, 0), "ebp": ("ebp", 32, 0), "esp": ("esp", 32, 0),
    # 16-bit sub-registers
    "ax": ("eax", 16, 0), "bx": ("ebx", 16, 0), "cx": ("ecx", 16, 0), "dx": ("edx", 16, 0),
    "si": ("esi", 16, 0), "di": ("edi", 16, 0), "bp": ("ebp", 16, 0), "sp": ("esp", 16, 0),
    # 8-bit sub-registers
    "al": ("eax", 8, 0), "bl": ("ebx", 8, 0), "cl": ("ecx", 8, 0), "dl": ("edx", 8, 0),
    "ah": ("eax", 8, 8), "bh": ("ebx", 8, 8), "ch": ("ecx", 8, 8), "dh": ("edx", 8, 8),
}

_ARM64_REGS: Dict[str, Tuple[str, int, int]] = {
    **{f"x{i}": (f"x{i}", 64, 0) for i in range(31)},
    **{f"w{i}": (f"x{i}", 32, 0) for i in range(31)},
    "sp": ("sp", 64, 0), "wsp": ("sp", 32, 0),
    "xzr": ("xzr", 64, 0), "wzr": ("xzr", 32, 0),
}


# ---------------------------------------------------------------------------
# Symbolic State & Flag Tracking
# ---------------------------------------------------------------------------


@dataclass
class ComparisonFact:
    """Record of a comparison or ALU flag-setting operation."""

    op: str  # "cmp", "test", "sub", "add", "xor", "cmn", etc.
    left: z3.BitVecRef
    right: z3.BitVecRef
    width: int
    left_str: str = ""
    right_str: str = ""


class SymbolicState:
    """State of registers and path constraints during symbolic traversal."""

    def __init__(self, arch: str = "x86_64", pointer_size: int = 8) -> None:
        self.arch = arch
        self.pointer_size = pointer_size
        if arch in ("arm64", "aarch64"):
            self._regs_map = _ARM64_REGS
            self._base_width = 64
        elif arch == "x86":
            self._regs_map = _X86_32_REGS
            self._base_width = 32
        else:
            self._regs_map = _X86_64_REGS
            self._base_width = 64

        self._regs: Dict[str, z3.BitVecRef] = {}
        self._mem: Dict[str, z3.BitVecRef] = {}
        self.last_cmp: Optional[ComparisonFact] = None
        self.constraints: List[z3.BoolRef] = []
        self.constraint_strs: List[str] = []

    def clone(self) -> SymbolicState:
        """Create an isolated fork of this symbolic state for branching paths."""
        fork = SymbolicState(self.arch, self.pointer_size)
        fork._regs = dict(self._regs)
        fork._mem = dict(self._mem)
        fork.last_cmp = self.last_cmp
        fork.constraints = list(self.constraints)
        fork.constraint_strs = list(self.constraint_strs)
        return fork

    def get_reg(self, name: str) -> z3.BitVecRef:
        """Read a register symbolically, initializing as unconstrained BitVec on first use."""
        clean = name.lower().strip()
        if clean in ("xzr", "wzr"):
            width = 32 if clean == "wzr" else 64
            return z3.BitVecVal(0, width)

        if clean in self._regs_map:
            base, width, offset = self._regs_map[clean]
            if base not in self._regs:
                self._regs[base] = z3.BitVec(base, self._base_width)
            base_val = self._regs[base]
            if width == self._base_width and offset == 0:
                return base_val
            return z3.Extract(offset + width - 1, offset, base_val)

        # Fallback for unknown / generic register names
        if clean not in self._regs:
            self._regs[clean] = z3.BitVec(clean, self._base_width)
        return self._regs[clean]

    def set_reg(self, name: str, val: z3.BitVecRef) -> None:
        """Write a value to a register, handling sub-register zero-extension or slice merge."""
        clean = name.lower().strip()
        if clean in ("xzr", "wzr"):
            return

        if clean in self._regs_map:
            base, width, offset = self._regs_map[clean]
            if base not in self._regs:
                self._regs[base] = z3.BitVec(base, self._base_width)

            # Match bit widths
            val_width = val.size()
            if val_width < width:
                val = z3.ZeroExt(width - val_width, val)
            elif val_width > width:
                val = z3.Extract(width - 1, 0, val)

            if width == self._base_width and offset == 0:
                self._regs[base] = val
            elif self.arch in ("x86_64", "arm64", "aarch64") and width == 32 and offset == 0:
                # 32-bit writes in 64-bit mode zero-extend to 64 bits
                self._regs[base] = z3.ZeroExt(32, val)
            else:
                cur = self._regs[base]
                pieces: List[z3.BitVecRef] = []
                if offset + width < self._base_width:
                    pieces.append(z3.Extract(self._base_width - 1, offset + width, cur))
                pieces.append(val)
                if offset > 0:
                    pieces.append(z3.Extract(offset - 1, 0, cur))
                self._regs[base] = z3.Concat(*pieces) if len(pieces) > 1 else pieces[0]
        else:
            self._regs[clean] = val

    def eval_operand(self, token: str, default_width: int = 32) -> z3.BitVecRef:
        """Evaluate an operand string (immediate, register, or memory reference)."""
        s = token.strip()
        if s.startswith("#"):
            s = s[1:].strip()

        imm = self._parse_immediate(s)
        if imm is not None:
            mask = (1 << default_width) - 1
            return z3.BitVecVal(imm & mask, default_width)

        if "[" in s and "]" in s:
            mem_name = self._sanitize_mem_name(s)
            if mem_name not in self._mem:
                self._mem[mem_name] = z3.BitVec(mem_name, default_width)
            return self._mem[mem_name]

        return self.get_reg(s)

    def write_operand(self, token: str, val: z3.BitVecRef) -> None:
        """Store a value into a destination operand (register or memory location)."""
        s = token.strip()
        if "[" in s and "]" in s:
            mem_name = self._sanitize_mem_name(s)
            self._mem[mem_name] = val
        else:
            self.set_reg(s, val)

    @staticmethod
    def _parse_immediate(token: str) -> Optional[int]:
        s = token.strip()
        if not s:
            return None
        neg = s.startswith("-")
        body = s[1:] if neg else s
        try:
            if body.lower().startswith("0x"):
                val = int(body, 16)
            elif any(c in "abcdefABCDEF" for c in body):
                val = int(body, 16)
            else:
                val = int(body, 10)
            return -val if neg else val
        except ValueError:
            return None

    @staticmethod
    def _sanitize_mem_name(token: str) -> str:
        start = token.find("[")
        end = token.find("]")
        inner = token[start + 1 : end].strip() if 0 <= start < end else token
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", inner)
        return f"mem_{clean}"

    def set_cmp(
        self,
        op: str,
        left: z3.BitVecRef,
        right: z3.BitVecRef,
        width: int,
        left_str: str = "",
        right_str: str = "",
    ) -> None:
        """Record the active comparison / flag-setting operation."""
        lw, rw = left.size(), right.size()
        if lw != rw:
            target_w = max(lw, rw)
            if lw < target_w:
                left = z3.ZeroExt(target_w - lw, left)
            if rw < target_w:
                right = z3.ZeroExt(target_w - rw, right)
            width = target_w

        self.last_cmp = ComparisonFact(
            op=op,
            left=left,
            right=right,
            width=width,
            left_str=left_str or str(left),
            right_str=right_str or str(right),
        )

    @staticmethod
    def _branch_operand(
        operands: Optional[Union[str, Sequence[str]]], index: int
    ) -> Optional[str]:
        """Fetch the index-th operand of a branch, from raw op_str or a token list."""
        if operands is None:
            return None
        if isinstance(operands, str):
            tokens = _split_operands(operands)
        else:
            tokens = [str(token).strip() for token in operands]
        if not 0 <= index < len(tokens):
            return None
        token = tokens[index].strip()
        return token or None

    def _register_test_condition(
        self, mnemonic: str, operands: Optional[Union[str, Sequence[str]]]
    ) -> Optional[z3.BoolRef]:
        """Lift an ARM64 register test (cbz/cbnz/tbz/tbnz) from its own operands."""
        m = mnemonic.lower().strip()
        tested = self._branch_operand(operands, 0)
        if tested is None:
            return None
        val = self.eval_operand(tested, self._base_width)
        if m in ("cbz", "cbnz"):
            zero = z3.BitVecVal(0, val.size())
            return (val == zero) if m == "cbz" else (val != zero)

        bit_token = self._branch_operand(operands, 1)
        if bit_token is None:
            return None
        bit_no = self._parse_immediate(bit_token.lstrip("#").strip())
        if bit_no is None or not 0 <= bit_no < val.size():
            return None
        bit = z3.Extract(int(bit_no), int(bit_no), val)
        is_set = bit == z3.BitVecVal(1, 1)
        return z3.Not(is_set) if m == "tbz" else is_set

    def get_branch_condition(
        self,
        mnemonic: str,
        taken: bool = True,
        operands: Optional[Union[str, Sequence[str]]] = None,
    ) -> Optional[z3.BoolRef]:
        """Translate a conditional branch instruction into a Z3 boolean formula.

        ``operands`` carries the branch instruction's own operand list (a token
        sequence or the raw Capstone op_str). Single-register tests take their
        tested register from it instead of inheriting a comparison recorded
        earlier in the state, which is either stale or absent.
        """
        m = mnemonic.lower().strip()
        if m in ("jmp", "ljmp", "b", "br"):
            return z3.BoolVal(True) if taken else z3.BoolVal(False)

        # ARM64 register tests (cbz/cbnz/tbz/tbnz): compare the register named
        # by this very instruction against zero / a single bit.
        if m in _REGISTER_TEST_MNEMONICS:
            cond = self._register_test_condition(m, operands)
            if cond is None:
                return None
            return cond if taken else z3.Not(cond)

        if self.last_cmp is None:
            return None

        op = self.last_cmp.op
        left = self.last_cmp.left
        right = self.last_cmp.right

        cond: Optional[z3.BoolRef] = None
        if op in ("cmp", "sub", "cmn"):
            if op == "cmn":
                comp_val = left + right
                zero_val = z3.BitVecVal(0, self.last_cmp.width)
                left, right = comp_val, zero_val

            if m in ("je", "jz", "b.eq"):
                cond = (left == right)
            elif m in ("jne", "jnz", "b.ne"):
                cond = (left != right)
            elif m in ("jl", "jnge", "b.lt"):
                cond = (left < right)  # signed in Z3
            elif m in ("jle", "jng", "b.le"):
                cond = (left <= right)  # signed
            elif m in ("jg", "jnle", "b.gt"):
                cond = (left > right)  # signed
            elif m in ("jge", "jnl", "b.ge"):
                cond = (left >= right)  # signed
            elif m in ("jb", "jc", "jnae", "b.lo", "b.cc"):
                cond = z3.ULT(left, right)
            elif m in ("jbe", "jna", "b.ls"):
                cond = z3.ULE(left, right)
            elif m in ("ja", "jnbe", "b.hi"):
                cond = z3.UGT(left, right)
            elif m in ("jae", "jnc", "jnb", "b.hs", "b.cs"):
                cond = z3.UGE(left, right)
            elif m == "js":
                cond = ((left - right) < 0)
            elif m == "jns":
                cond = ((left - right) >= 0)

        elif op in ("test", "and", "tst"):
            and_val = left & right
            if m in ("je", "jz", "b.eq"):
                cond = (and_val == 0)
            elif m in ("jne", "jnz", "b.ne"):
                cond = (and_val != 0)
            elif m in ("jl", "jnge", "b.lt"):
                cond = (and_val < 0)
            elif m in ("jle", "jng", "b.le"):
                cond = (and_val <= 0)
            elif m in ("jg", "jnle", "b.gt"):
                cond = (and_val > 0)
            elif m in ("jge", "jnl", "b.ge"):
                cond = (and_val >= 0)
            elif m == "js":
                cond = (and_val < 0)
            elif m == "jns":
                cond = (and_val >= 0)

        elif op in ("xor", "inc", "dec", "add"):
            if m in ("je", "jz", "b.eq"):
                cond = (left == 0)
            elif m in ("jne", "jnz", "b.ne"):
                cond = (left != 0)
            elif m in ("jl", "b.lt", "js"):
                cond = (left < 0)
            elif m in ("jle", "b.le"):
                cond = (left <= 0)
            elif m in ("jg", "b.gt"):
                cond = (left > 0)
            elif m in ("jge", "b.ge", "jns"):
                cond = (left >= 0)

        if cond is None:
            return None
        return cond if taken else z3.Not(cond)

    def add_constraint(self, cond: z3.BoolRef, desc: Optional[str] = None) -> None:
        """Add a path condition to this state's constraint accumulator."""
        self.constraints.append(cond)
        self.constraint_strs.append(desc or str(cond))


# ---------------------------------------------------------------------------
# Basic Block & CFG Reconstruction
# ---------------------------------------------------------------------------


@dataclass
class BasicBlock:
    """A straight-line sequence of instructions terminating at a branch or fallthrough."""

    start_addr: int
    end_addr: int
    instructions: List[Instruction] = field(default_factory=list)
    # Successor edges: (dst_addr, edge_kind: "taken" | "fallthrough" | "uncond", branch_insn)
    successors: List[Tuple[int, str, Optional[Instruction]]] = field(default_factory=list)


def build_basic_blocks(instructions: Sequence[Instruction]) -> Dict[int, BasicBlock]:
    """Partition a sequence of instructions into basic blocks and build intra-CFG edges."""
    if not instructions:
        return {}

    insn_map = {insn.address: insn for insn in instructions}
    leaders: Set[int] = {instructions[0].address}

    # Pass 1: Identify all block entry points (leaders)
    for i, insn in enumerate(instructions):
        if insn.is_jump or _is_jump_mnemonic(insn.mnemonic):
            if insn.target is not None and insn.target in insn_map:
                leaders.add(insn.target)
            if i + 1 < len(instructions):
                leaders.add(instructions[i + 1].address)
        elif insn.is_ret or insn.mnemonic.lower() in ("ret", "retf"):
            if i + 1 < len(instructions):
                leaders.add(instructions[i + 1].address)

    sorted_leaders = sorted(leaders)
    blocks: Dict[int, BasicBlock] = {}

    # Pass 2: Group instructions into blocks
    for idx, start in enumerate(sorted_leaders):
        next_leader = sorted_leaders[idx + 1] if idx + 1 < len(sorted_leaders) else None
        block_insns: List[Instruction] = []
        for insn in instructions:
            if insn.address < start:
                continue
            if next_leader is not None and insn.address >= next_leader:
                break
            block_insns.append(insn)

        if not block_insns:
            continue

        end = block_insns[-1].end if block_insns[-1].size else (block_insns[-1].address + 4)
        blocks[start] = BasicBlock(start_addr=start, end_addr=end, instructions=block_insns)

    # Pass 3: Construct CFG edges between blocks
    for idx, start in enumerate(sorted_leaders):
        block = blocks.get(start)
        if not block or not block.instructions:
            continue
        last = block.instructions[-1]
        m = last.mnemonic.lower().strip()
        next_leader = sorted_leaders[idx + 1] if idx + 1 < len(sorted_leaders) else None

        if last.is_uncond_jump or m in ("jmp", "ljmp", "b", "br"):
            if last.target is not None and last.target in blocks:
                block.successors.append((last.target, "uncond", last))
        elif last.is_jump or _is_jump_mnemonic(m):
            if last.target is not None and last.target in blocks:
                block.successors.append((last.target, "taken", last))
            if next_leader is not None:
                block.successors.append((next_leader, "fallthrough", last))
        elif last.is_ret or m in ("ret", "retf"):
            pass  # function exit: no successors
        else:
            if next_leader is not None:
                block.successors.append((next_leader, "fallthrough", None))

    return blocks


def _is_jump_mnemonic(mnemonic: str) -> bool:
    m = mnemonic.lower().strip()
    return (
        m in (
            "jmp", "ljmp", "je", "jne", "jz", "jnz", "jl", "jle", "jg", "jge",
            "jb", "jbe", "ja", "jae", "jc", "jnc", "js", "jns", "cbz", "cbnz",
            "b", "br", "tbz", "tbnz",
        )
        or m.startswith("b.")
    )


# ---------------------------------------------------------------------------
# Instruction Semantic Lifter (SMTBranchLifter)
# ---------------------------------------------------------------------------


class SMTBranchLifter:
    """Translates disassembly instructions and branch chains into Z3 bitvector AST constraints."""

    def __init__(self, arch: str = "x86_64", pointer_size: int = 8) -> None:
        self.arch = arch
        self.pointer_size = pointer_size

    def lift_instruction(self, insn: Instruction, state: SymbolicState) -> None:
        """Update symbolic state by executing one instruction symbolically."""
        mnemonic = insn.mnemonic.lower().strip()
        op_str = insn.op_str.strip()
        operands = _split_operands(op_str)

        # Jumps and returns do not update general register values
        if insn.is_jump or insn.is_ret or _is_jump_mnemonic(mnemonic) or mnemonic in ("ret", "retf"):
            if mnemonic in ("cbz", "cbnz") and operands:
                reg_name = operands[0]
                val = state.get_reg(reg_name)
                state.set_cmp("cmp", val, z3.BitVecVal(0, val.size()), val.size(), reg_name, "0")
            return

        if mnemonic == "cmp" and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            left = state.eval_operand(op1, width)
            right = state.eval_operand(op2, width)
            state.set_cmp("cmp", left, right, width, op1, op2)

        elif mnemonic in ("test", "tst") and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            left = state.eval_operand(op1, width)
            right = state.eval_operand(op2, width)
            state.set_cmp("test", left, right, width, op1, op2)

        elif mnemonic == "sub" and len(operands) >= 2:
            if len(operands) == 3:  # ARM64: sub dst, src1, src2
                dst, src1, src2 = operands[0], operands[1], operands[2]
                width = self._op_width(dst)
                s1 = state.eval_operand(src1, width)
                s2 = state.eval_operand(src2, width)
                res = s1 - s2
                state.write_operand(dst, res)
                state.set_cmp("sub", s1, s2, width, src1, src2)
            else:
                op1, op2 = operands[0], operands[1]
                width = self._op_width(op1)
                left = state.eval_operand(op1, width)
                right = state.eval_operand(op2, width)
                res = left - right
                state.write_operand(op1, res)
                state.set_cmp("sub", left, right, width, op1, op2)

        elif mnemonic == "subs" and len(operands) >= 2:  # ARM64
            dst = operands[0]
            src1, src2 = (operands[1], operands[2]) if len(operands) == 3 else (operands[0], operands[1])
            width = self._op_width(src1)
            s1 = state.eval_operand(src1, width)
            s2 = state.eval_operand(src2, width)
            res = s1 - s2
            if dst not in ("xzr", "wzr"):
                state.write_operand(dst, res)
            state.set_cmp("cmp", s1, s2, width, src1, src2)

        elif mnemonic == "cmn" and len(operands) >= 2:  # ARM64
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            left = state.eval_operand(op1, width)
            right = state.eval_operand(op2, width)
            state.set_cmp("cmn", left, right, width, op1, op2)

        elif mnemonic == "add" and len(operands) >= 2:
            if len(operands) == 3:  # ARM64: add dst, src1, src2
                dst, src1, src2 = operands[0], operands[1], operands[2]
                width = self._op_width(dst)
                s1 = state.eval_operand(src1, width)
                s2 = state.eval_operand(src2, width)
                state.write_operand(dst, s1 + s2)
            else:
                op1, op2 = operands[0], operands[1]
                width = self._op_width(op1)
                left = state.eval_operand(op1, width)
                right = state.eval_operand(op2, width)
                state.write_operand(op1, left + right)

        elif mnemonic == "xor" and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            if op1 == op2:
                zero = z3.BitVecVal(0, width)
                state.write_operand(op1, zero)
                state.set_cmp("xor", zero, zero, width, op1, "0")
            else:
                left = state.eval_operand(op1, width)
                right = state.eval_operand(op2, width)
                res = left ^ right
                state.write_operand(op1, res)
                state.set_cmp("xor", res, z3.BitVecVal(0, width), width, op1, "0")

        elif mnemonic == "and" and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            left = state.eval_operand(op1, width)
            right = state.eval_operand(op2, width)
            res = left & right
            state.write_operand(op1, res)
            state.set_cmp("test", left, right, width, op1, op2)

        elif mnemonic in ("or", "orr") and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            left = state.eval_operand(op1, width)
            right = state.eval_operand(op2, width)
            state.write_operand(op1, left | right)

        elif mnemonic == "mov" and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            val = state.eval_operand(op2, width)
            state.write_operand(op1, val)

        elif mnemonic == "movzx" and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            dst_w = self._op_width(op1)
            src_w = self._op_width(op2) or 8
            val = state.eval_operand(op2, src_w)
            state.write_operand(op1, z3.ZeroExt(dst_w - src_w, val))

        elif mnemonic in ("movsx", "movsxd") and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            dst_w = self._op_width(op1)
            src_w = self._op_width(op2) or 32
            val = state.eval_operand(op2, src_w)
            state.write_operand(op1, z3.SignExt(dst_w - src_w, val))

        elif mnemonic in ("inc", "dec") and len(operands) >= 1:
            op1 = operands[0]
            width = self._op_width(op1)
            cur = state.eval_operand(op1, width)
            new_val = (cur + 1) if mnemonic == "inc" else (cur - 1)
            state.write_operand(op1, new_val)
            state.set_cmp(mnemonic, new_val, z3.BitVecVal(0, width), width)

        elif mnemonic in ("shl", "sal") and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            cur = state.eval_operand(op1, width)
            shift = state.eval_operand(op2, width)
            state.write_operand(op1, cur << shift)

        elif mnemonic == "shr" and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            cur = state.eval_operand(op1, width)
            shift = state.eval_operand(op2, width)
            state.write_operand(op1, z3.LShR(cur, shift))

        elif mnemonic == "sar" and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            cur = state.eval_operand(op1, width)
            shift = state.eval_operand(op2, width)
            state.write_operand(op1, cur >> shift)

        elif mnemonic == "lea" and len(operands) >= 2:
            op1, op2 = operands[0], operands[1]
            width = self._op_width(op1)
            addr_expr = self._eval_lea(op2, state, width)
            if addr_expr is not None:
                state.write_operand(op1, addr_expr)

    def lift_block(self, instructions: Sequence[Instruction], state: SymbolicState) -> None:
        """Sequentially execute instructions within a basic block."""
        for insn in instructions:
            self.lift_instruction(insn, state)

    def lift_branch_condition(
        self, branch_insn: Instruction, state: SymbolicState, taken: bool = True
    ) -> Optional[z3.BoolRef]:
        """Extract the boolean branch formula for one CFG edge.

        The branch instruction's own operands are threaded into the state so a
        register test (cbz/cbnz/tbz/tbnz) lifts the register the branch itself
        names rather than a stale or absent earlier comparison.
        """
        mnemonic = branch_insn.mnemonic
        cond = state.get_branch_condition(
            mnemonic, taken=taken, operands=branch_insn.op_str
        )
        if cond is None and mnemonic.lower().strip() in _REGISTER_TEST_MNEMONICS:
            raise UnsupportedInstructionError(
                f"Cannot lift '{mnemonic} {branch_insn.op_str}' at "
                f"0x{branch_insn.address:x}: tested register operand missing"
            )
        return cond

    def _op_width(self, token: str) -> int:
        clean = token.lower().strip()
        if "qword" in clean:
            return 64
        if "dword" in clean:
            return 32
        if "word" in clean and "dword" not in clean and "qword" not in clean:
            return 16
        if "byte" in clean:
            return 8
        reg_map = _ARM64_REGS if self.arch in ("arm64", "aarch64") else (_X86_32_REGS if self.arch == "x86" else _X86_64_REGS)
        if clean in reg_map:
            return reg_map[clean][1]
        return 32 if self.arch == "x86" else 64

    def _eval_lea(self, op: str, state: SymbolicState, width: int) -> Optional[z3.BitVecRef]:
        """Evaluate a simple LEA address expression: [base + disp] or [base + index*scale + disp]."""
        start = op.find("[")
        end = op.find("]")
        if start < 0 or end < 0 or start >= end:
            return None
        inner = op[start + 1 : end].strip()
        # Parse [reg + imm] or [reg - imm]
        m = re.match(r"^([a-zA-Z0-9_]+)\s*([\+\-])\s*(0x[0-9a-fA-F]+|[0-9]+)$", inner)
        if m:
            reg_name, sign, disp_str = m.groups()
            base_val = state.get_reg(reg_name)
            disp_val = int(disp_str, 0)
            if sign == "-":
                disp_val = -disp_val
            # Extend or extract base_val to target width
            if base_val.size() < width:
                base_val = z3.ZeroExt(width - base_val.size(), base_val)
            elif base_val.size() > width:
                base_val = z3.Extract(width - 1, 0, base_val)
            return base_val + z3.BitVecVal(disp_val, width)

        # Simple [reg]
        if re.match(r"^[a-zA-Z0-9_]+$", inner):
            val = state.get_reg(inner)
            if val.size() < width:
                return z3.ZeroExt(width - val.size(), val)
            elif val.size() > width:
                return z3.Extract(width - 1, 0, val)
            return val

        return None


# ---------------------------------------------------------------------------
# SMT Solver Engine (SMTSolver)
# ---------------------------------------------------------------------------


class SMTSolver:
    """SMT-based path reachability and constraint solver over disassembled control flow."""

    def __init__(
        self,
        db: Optional[Union[Database, PathLike]] = None,
        arch: str = "x86_64",
    ) -> None:
        self.arch = arch
        self.db = None
        self._owned_db = False
        if db is not None:
            if isinstance(db, Database):
                self.db = db
            else:
                self.db = Database(Path(db), create=False)
                self._owned_db = True

    def close(self) -> None:
        if self._owned_db and self.db is not None:
            self.db.close()
            self.db = None

    def __enter__(self) -> SMTSolver:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def solve_path(
        self,
        func_addr: int,
        target_block_addr: Optional[int] = None,
        timeout_sec: float = 5.0,
        binary_sha256: Optional[str] = None,
    ) -> SolveResult:
        """Solve for inputs required to reach target_block_addr within func_addr in toothpick.db."""
        if self.db is None:
            raise SolverError("Database required to solve by function address")

        sha = self._resolve_sha(binary_sha256)
        func = self.db.get_function(sha, func_addr)
        if func is None:
            raise SolverError(f"Function 0x{func_addr:x} not found in database")

        if not func.ir_pcode:
            raise SolverError(f"No IR/disassembly (ir_pcode) found for function 0x{func_addr:x}")

        meta = self.db.get_binary_metadata(sha)
        arch = meta.get("arch", self.arch)

        instructions = parse_ir_pcode(func.ir_pcode, arch=arch)
        return self.solve_instructions(
            instructions,
            target_addr=target_block_addr,
            timeout_sec=timeout_sec,
            arch=arch,
        )

    def solve_code(
        self,
        code_bytes: bytes,
        target_addr: Optional[int] = None,
        base_addr: int = 0x1000,
        timeout_sec: float = 5.0,
        arch: str = "x86_64",
    ) -> SolveResult:
        """Disassemble raw byte stream and solve for input values reaching target_addr."""
        instructions = disassemble_bytes(code_bytes, base_addr=base_addr, arch=arch)
        return self.solve_instructions(
            instructions,
            target_addr=target_addr,
            timeout_sec=timeout_sec,
            arch=arch,
        )

    def solve_instructions(
        self,
        instructions: Sequence[Instruction],
        target_addr: Optional[int] = None,
        timeout_sec: float = 5.0,
        arch: Optional[str] = None,
    ) -> SolveResult:
        """Solve path constraints over a sequence of Capstone instructions."""
        t0 = time.perf_counter()
        active_arch = arch or self.arch

        if not instructions:
            return SolveResult(status="UNSAT", elapsed_sec=0.0)

        blocks = build_basic_blocks(instructions)
        if not blocks:
            return SolveResult(status="UNSAT", elapsed_sec=0.0)

        entry_addr = instructions[0].address

        # Default target_addr if omitted
        if target_addr is None:
            target_addr = self._find_default_target(blocks, instructions)

        # Locate target block containing target_addr
        target_block = self._find_block_for_addr(blocks, target_addr)
        if target_block is None:
            return SolveResult(
                status="UNSAT",
                constraints=[f"Target address 0x{target_addr:x} not found in CFG"],
                elapsed_sec=time.perf_counter() - t0,
            )

        ptr_size = 4 if active_arch == "x86" else 8
        lifter = SMTBranchLifter(arch=active_arch, pointer_size=ptr_size)

        # BFS path exploration with depth and loop guards
        initial_state = SymbolicState(arch=active_arch, pointer_size=ptr_size)
        queue: deque[Tuple[int, List[Tuple[int, int, str]], SymbolicState]] = deque([
            (entry_addr, [], initial_state)
        ])

        max_paths = 100
        paths_explored = 0
        last_constraints: List[str] = []

        # Set when a branch predicate could not be lifted: the solve is then
        # reported as UNSUPPORTED instead of silently leaving the edge free.
        unsupported_reason: Optional[str] = None

        while queue and paths_explored < max_paths:
            # Enforce strict wall-clock timeout gate
            elapsed_now = time.perf_counter() - t0
            if elapsed_now >= timeout_sec:
                return SolveResult(status="TIMEOUT", elapsed_sec=elapsed_now)

            curr_addr, path_edges, curr_state = queue.popleft()
            curr_block = blocks.get(curr_addr)
            if curr_block is None:
                continue

            paths_explored += 1
            # Execute block instructions
            block_state = curr_state.clone()
            lifter.lift_block(curr_block.instructions, block_state)

            if curr_addr == target_block.start_addr:
                # Target basic block reached! Check SMT satisfiability
                rem_ms = max(1, int((timeout_sec - (time.perf_counter() - t0)) * 1000))
                solve_outcome = self._check_smt(block_state.constraints, timeout_ms=rem_ms)

                if solve_outcome == "TIMEOUT":
                    return SolveResult(
                        status="TIMEOUT",
                        constraints=block_state.constraint_strs,
                        elapsed_sec=time.perf_counter() - t0,
                    )
                elif solve_outcome[0] == "SAT":
                    model_dict = solve_outcome[1]
                    return SolveResult(
                        status="SAT",
                        model=model_dict,
                        constraints=block_state.constraint_strs,
                        elapsed_sec=time.perf_counter() - t0,
                    )
                else:
                    # UNSAT on this path: record constraints and search other branches
                    last_constraints = block_state.constraint_strs
                    continue

            # Check loop depth in current path
            visits = sum(1 for edge in path_edges if edge[0] == curr_addr)
            if visits >= 2 or len(path_edges) >= 32:
                continue

            # Expand successors
            for dst_addr, edge_kind, branch_insn in curr_block.successors:
                next_state = block_state.clone()
                if branch_insn is not None:
                    # An unconditional edge is always taken; only a fallthrough
                    # edge negates the branch predicate.
                    taken = edge_kind != "fallthrough"
                    try:
                        cond = lifter.lift_branch_condition(branch_insn, next_state, taken=taken)
                    except UnsupportedInstructionError as exc:
                        unsupported_reason = str(exc)
                        continue
                    if cond is None:
                        unsupported_reason = (
                            f"Cannot lift branch condition for "
                            f"'{branch_insn.mnemonic} {branch_insn.op_str}' at "
                            f"0x{branch_insn.address:x}"
                        )
                        continue
                    desc = f"{branch_insn.mnemonic} {branch_insn.op_str} ({'taken' if taken else 'fallthrough'}) -> {cond}"
                    next_state.add_constraint(cond, desc=desc)

                queue.append((dst_addr, path_edges + [(curr_addr, dst_addr, edge_kind)], next_state))

        elapsed_total = time.perf_counter() - t0
        if elapsed_total >= timeout_sec:
            return SolveResult(status="TIMEOUT", elapsed_sec=elapsed_total)

        if unsupported_reason is not None:
            return SolveResult(
                status="UNSUPPORTED",
                model=None,
                constraints=last_constraints + [unsupported_reason],
                elapsed_sec=elapsed_total,
            )

        return SolveResult(
            status="UNSAT",
            model=None,
            constraints=last_constraints,
            elapsed_sec=elapsed_total,
        )

    def _check_smt(
        self, constraints: Sequence[z3.BoolRef], timeout_ms: int
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        """Submit constraint batch to Z3 with strict timeout."""
        solver = z3.Solver()
        solver.set("timeout", max(1, timeout_ms))
        for c in constraints:
            solver.add(c)

        check_status = solver.check()
        if check_status == z3.sat:
            m = solver.model()
            model_dict = self._extract_model(m)
            return ("SAT", model_dict)
        elif check_status == z3.unsat:
            return ("UNSAT", {})
        else:
            reason = str(solver.reason_unknown()).lower()
            if "timeout" in reason or "canceled" in reason:
                return "TIMEOUT"
            return ("UNSAT", {})

    def _extract_model(self, m: z3.ModelRef) -> Dict[str, int]:
        """Extract model values and populate sub-register aliases."""
        result: Dict[str, int] = {}
        for d in m.decls():
            name = d.name()
            val = m[d]
            try:
                int_val = val.as_long()
                result[name] = int_val
            except Exception:
                pass

        # Populate sub-registers for x86_64
        for base64 in ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"):
            if base64 in result:
                val = result[base64]
                # 32-bit sub-reg
                reg32 = f"{base64}d" if base64.startswith("r") and base64[1:].isdigit() else ("e" + base64[1:] if base64.startswith("r") else base64)
                if reg32 not in result:
                    result[reg32] = val & 0xFFFFFFFF
                # 16-bit sub-reg
                reg16 = f"{base64}w" if base64.startswith("r") and base64[1:].isdigit() else base64[1:]
                if reg16 not in result:
                    result[reg16] = val & 0xFFFF
                # 8-bit sub-reg
                reg8 = f"{base64}b" if base64.startswith("r") and base64[1:].isdigit() else (base64[1] + "l" if len(base64) >= 2 else base64)
                if reg8 not in result:
                    result[reg8] = val & 0xFF

        # Populate sub-registers for ARM64
        for i in range(31):
            xi = f"x{i}"
            wi = f"w{i}"
            if xi in result and wi not in result:
                result[wi] = result[xi] & 0xFFFFFFFF

        return result

    def _resolve_sha(self, binary_sha256: Optional[str]) -> str:
        if self.db is None:
            raise SolverError("No database attached")
        if binary_sha256:
            return binary_sha256
        # Infer SHA if database contains exactly one binary
        rows = self.db._conn.execute("SELECT DISTINCT binary_sha256 FROM funcs").fetchall()
        shas = [r["binary_sha256"] for r in rows if r["binary_sha256"]]
        if len(shas) == 1:
            return shas[0]
        if not shas:
            raise SolverError("Database contains no analysed functions")
        raise SolverError(
            f"Ambiguous binary_sha256: database contains {len(shas)} binaries; pass --sha"
        )

    def _find_default_target(
        self, blocks: Dict[int, BasicBlock], instructions: Sequence[Instruction]
    ) -> int:
        # Search for first conditional jump's taken target
        for block in blocks.values():
            for dst_addr, edge_kind, _ in block.successors:
                if edge_kind == "taken":
                    return dst_addr
        # Fallback to last block or return instruction
        for block in reversed(list(blocks.values())):
            for insn in block.instructions:
                if insn.is_ret or insn.mnemonic.lower() in ("ret", "retf"):
                    return block.start_addr
        return instructions[-1].address

    def _find_block_for_addr(
        self, blocks: Dict[int, BasicBlock], addr: int
    ) -> Optional[BasicBlock]:
        if addr in blocks:
            return blocks[addr]
        for block in blocks.values():
            if block.start_addr <= addr < block.end_addr:
                return block
        return None


# ---------------------------------------------------------------------------
# Helpers: Disassembly & IR Parser
# ---------------------------------------------------------------------------


def disassemble_bytes(
    code_bytes: bytes, base_addr: int = 0x1000, arch: str = "x86_64"
) -> List[Instruction]:
    """Disassemble raw byte stream into Instruction objects using Capstone."""
    spec = arch_spec(arch)
    md = Cs(spec.capstone_arch, spec.capstone_mode)
    md.detail = True
    insns: List[Instruction] = []
    for cs_insn in md.disasm(code_bytes, base_addr):
        m = (cs_insn.mnemonic or "").lower()
        op_str = cs_insn.op_str or ""
        is_jump = _is_jump_mnemonic(m)
        is_uncond = m in ("jmp", "ljmp", "b", "br")
        is_ret = m in ("ret", "retf", "iret", "blr", "eret")
        is_call = m in ("call", "bl")

        target: Optional[int] = None
        if is_jump or is_call:
            target = _extract_target(op_str)

        insns.append(
            Instruction(
                address=int(cs_insn.address),
                size=int(cs_insn.size),
                mnemonic=m,
                op_str=op_str,
                opcode=bytes(cs_insn.bytes),
                is_call=is_call,
                is_jump=is_jump,
                is_ret=is_ret,
                is_uncond_jump=is_uncond,
                target=target,
            )
        )
    return insns


def parse_ir_pcode(ir_pcode: str, arch: str = "x86_64") -> List[Instruction]:
    """Parse textual IR snapshot from funcs.ir_pcode back into Instruction records."""
    instructions: List[Instruction] = []
    lines = ir_pcode.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        addr_str, mnemonic = parts[0], parts[1].lower()
        op_str = parts[2] if len(parts) > 2 else ""

        try:
            addr = int(addr_str, 16) if addr_str.startswith("0x") else int(addr_str)
        except ValueError:
            continue

        is_jump = _is_jump_mnemonic(mnemonic)
        is_uncond = mnemonic in ("jmp", "ljmp", "b", "br")
        is_ret = mnemonic in ("ret", "retf", "iret", "blr", "eret")
        is_call = mnemonic in ("call", "bl")

        target = _extract_target(op_str) if (is_jump or is_call) else None

        instructions.append(
            Instruction(
                address=addr,
                size=0,
                mnemonic=mnemonic,
                op_str=op_str,
                is_call=is_call,
                is_jump=is_jump,
                is_ret=is_ret,
                is_uncond_jump=is_uncond,
                target=target,
            )
        )
    return instructions


def _extract_target(op_str: str) -> Optional[int]:
    """Extract jump or call target address from an operand string."""
    tokens = [t.strip() for t in op_str.split(",")]
    if not tokens:
        return None
    target_token = tokens[-1]  # Target is usually the last operand
    if target_token.startswith("#"):
        target_token = target_token[1:]
    try:
        if target_token.startswith("0x") or any(c in "abcdefABCDEF" for c in target_token):
            return int(target_token, 16)
        return int(target_token, 10)
    except ValueError:
        return None
