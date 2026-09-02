"""Deterministic Low-Level Register Virtual Machine (VM) and Instruction Set Architecture (ISA).

Provides a deterministic microarchitecture environment for evaluating machine-level
assembly code, measuring execution clock cycles, tracking register pressure, and
enforcing hardware-level bounds.
"""
from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


class Opcode(str, enum.Enum):
    """Instruction Set Architecture (ISA) Opcodes."""
    # Data Movement
    MOV = "MOV"      # MOV dst, src/imm
    LOAD = "LOAD"    # LOAD dst, [addr_reg]
    STORE = "STORE"  # STORE [addr_reg], src
    PUSH = "PUSH"    # PUSH src
    POP = "POP"      # POP dst
    SWAP = "SWAP"    # SWAP reg1, reg2

    # Arithmetic
    ADD = "ADD"      # ADD dst, src/imm
    SUB = "SUB"      # SUB dst, src/imm
    MUL = "MUL"      # MUL dst, src/imm
    DIV = "DIV"      # DIV dst, src/imm (safe division)
    MOD = "MOD"      # MOD dst, src/imm (safe modulo)
    INC = "INC"      # INC dst
    DEC = "DEC"      # DEC dst
    NEG = "NEG"      # NEG dst

    # Bitwise & Hardware Intrinsics
    AND = "AND"      # AND dst, src/imm
    OR = "OR"        # OR dst, src/imm
    XOR = "XOR"      # XOR dst, src/imm
    NOT = "NOT"      # NOT dst
    SHL = "SHL"      # SHL dst, imm
    SHR = "SHR"      # SHR dst, imm
    POPCNT = "POPCNT"# POPCNT dst, src (Hardware population count)
    CLZ = "CLZ"      # CLZ dst, src (Count leading zeros)

    # Control Flow
    CMP = "CMP"      # CMP reg1, reg2/imm
    JMP = "JMP"      # JMP target_offset
    JZ = "JZ"        # JZ target_offset (jump if zero / equal)
    JNZ = "JNZ"      # JNZ target_offset (jump if not zero / not equal)
    JL = "JL"        # JL target_offset (jump if less)
    JG = "JG"        # JG target_offset (jump if greater)
    NOP = "NOP"      # NOP
    RET = "RET"      # RET (return from routine)
    HALT = "HALT"    # HALT (terminate execution)


# Execution cost in clock cycles per opcode
OPCODE_CYCLES: dict[Opcode, int] = {
    Opcode.MOV: 1,
    Opcode.LOAD: 2,
    Opcode.STORE: 2,
    Opcode.PUSH: 2,
    Opcode.POP: 2,
    Opcode.SWAP: 1,
    Opcode.ADD: 1,
    Opcode.SUB: 1,
    Opcode.MUL: 3,
    Opcode.DIV: 8,
    Opcode.MOD: 8,
    Opcode.INC: 1,
    Opcode.DEC: 1,
    Opcode.NEG: 1,
    Opcode.AND: 1,
    Opcode.OR: 1,
    Opcode.XOR: 1,
    Opcode.NOT: 1,
    Opcode.SHL: 1,
    Opcode.SHR: 1,
    Opcode.POPCNT: 1,
    Opcode.CLZ: 1,
    Opcode.CMP: 1,
    Opcode.JMP: 2,
    Opcode.JZ: 2,
    Opcode.JNZ: 2,
    Opcode.JL: 2,
    Opcode.JG: 2,
    Opcode.NOP: 1,
    Opcode.RET: 1,
    Opcode.HALT: 1,
}

# General purpose registers
REGISTERS: tuple[str, ...] = ("R0", "R1", "R2", "R3", "ACC")


@dataclass(frozen=True)
class Instruction:
    """A single low-level assembly instruction."""
    op: Opcode
    dst: str | None = None
    src: str | int | float | None = None

    def __init__(
        self,
        op: Opcode,
        dst: str | None = None,
        src: str | float | None = None,
        imm: str | float | None = None,
    ):
        if imm is not None:
            src = imm
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "dst", dst)
        object.__setattr__(self, "src", src)

    @property
    def imm(self) -> float | int | None:
        """Immediate value alias if operand is numeric."""
        if isinstance(self.src, (int, float)):
            return self.src
        return None

    def to_asm(self) -> str:
        """Returns human-readable assembly representation."""
        if self.dst is None and self.src is None:
            return self.op.value
        if self.src is None:
            return f"{self.op.value} {self.dst}"
        return f"{self.op.value} {self.dst}, {self.src}"

    def byte_size(self) -> int:
        """Calculates binary encoding byte size (1 byte op + operands)."""
        size = 1
        if self.dst is not None:
            size += 1
        if self.src is not None:
            size += 2 if isinstance(self.src, int) else 1
        return size

    def serialize(self) -> dict[str, Any]:
        return {"op": self.op.value, "dst": self.dst, "src": self.src}

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Instruction:
        return cls(
            op=Opcode(data["op"]),
            dst=data.get("dst"),
            src=data.get("src"),
        )


@dataclass
class VMExecutionResult:
    """Outcome of running an assembly program inside the VM."""
    success: bool
    return_value: int = 0
    registers: dict[str, int] = field(default_factory=dict)
    clock_cycles: int = 0
    instructions_executed: int = 0
    code_size_bytes: int = 0
    registers_used: set[str] = field(default_factory=set)
    memory_access_count: int = 0
    error: str | None = None
    timeout_triggered: bool = False


class VirtualMachine:
    """Deterministic, cycle-accurate low-level Register Machine emulator."""

    def __init__(self, memory_size: int = 256, max_cycles: int = 2000):
        self.memory_size = memory_size
        self.max_cycles = max_cycles
        self.memory: list[int] = [0] * memory_size
        self.stack: list[int] = []
        self.registers: dict[str, int] = {r: 0 for r in REGISTERS}
        self.flags: dict[str, bool] = {"zero": False, "sign": False}
        self.pc: int = 0
        self.clock_cycles: int = 0
        self.instructions_executed: int = 0
        self.active_registers_used: set[str] = set()
        self.memory_accesses: int = 0

    def reset(self, initial_args: Sequence[int] | None = None) -> None:
        """Resets VM state and initializes input registers."""
        self.memory = [0] * self.memory_size
        self.stack = []
        self.registers = {r: 0 for r in REGISTERS}
        self.flags = {"zero": False, "sign": False}
        self.pc = 0
        self.clock_cycles = 0
        self.instructions_executed = 0
        self.active_registers_used = set()
        self.memory_accesses = 0

        if initial_args:
            # Map input args into R0, R1, R2, R3
            for i, val in enumerate(initial_args[:4]):
                reg_name = f"R{i}"
                self.registers[reg_name] = int(val)
                self.active_registers_used.add(reg_name)

    def _resolve_src(self, src: str | int | None) -> int:
        if src is None:
            return 0
        if isinstance(src, int):
            return src
        if src in self.registers:
            self.active_registers_used.add(src)
            return self.registers[src]
        return 0

    def execute(
        self,
        program: Sequence[Instruction],
        initial_args: Sequence[int] | None = None,
    ) -> VMExecutionResult:
        """Executes the given program until HALT, RET, or cycle budget exhaustion."""
        self.reset(initial_args)
        prog_len = len(program)
        total_code_size = sum(ins.byte_size() for ins in program)

        while 0 <= self.pc < prog_len:
            if self.clock_cycles >= self.max_cycles:
                return VMExecutionResult(
                    success=False,
                    return_value=self.registers["ACC"],
                    registers=dict(self.registers),
                    clock_cycles=self.clock_cycles,
                    instructions_executed=self.instructions_executed,
                    code_size_bytes=total_code_size,
                    registers_used=set(self.active_registers_used),
                    memory_access_count=self.memory_accesses,
                    error=f"Execution exceeded cycle limit ({self.max_cycles} cycles)",
                    timeout_triggered=True,
                )

            ins = program[self.pc]
            op = ins.op
            cost = OPCODE_CYCLES.get(op, 1)
            self.clock_cycles += cost
            self.instructions_executed += 1

            if ins.dst in self.registers:
                self.active_registers_used.add(ins.dst)

            # Execution Dispatch
            if op == Opcode.HALT or op == Opcode.RET:
                break

            elif op == Opcode.NOP:
                self.pc += 1

            elif op == Opcode.MOV:
                if ins.dst in self.registers:
                    self.registers[ins.dst] = self._resolve_src(ins.src)
                self.pc += 1

            elif op == Opcode.SWAP:
                r1, r2 = ins.dst, str(ins.src)
                if r1 in self.registers and r2 in self.registers:
                    self.active_registers_used.add(r2)
                    self.registers[r1], self.registers[r2] = self.registers[r2], self.registers[r1]
                self.pc += 1

            elif op == Opcode.ADD:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src)
                    self.registers[ins.dst] = (self.registers[ins.dst] + val) & 0xFFFFFFFF
                self.pc += 1

            elif op == Opcode.SUB:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src)
                    self.registers[ins.dst] = (self.registers[ins.dst] - val) & 0xFFFFFFFF
                self.pc += 1

            elif op == Opcode.MUL:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src)
                    self.registers[ins.dst] = (self.registers[ins.dst] * val) & 0xFFFFFFFF
                self.pc += 1

            elif op == Opcode.DIV:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src)
                    if val != 0:
                        self.registers[ins.dst] = int(self.registers[ins.dst] // val) & 0xFFFFFFFF
                    else:
                        self.registers[ins.dst] = 0
                self.pc += 1

            elif op == Opcode.MOD:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src)
                    if val != 0:
                        self.registers[ins.dst] = int(self.registers[ins.dst] % val) & 0xFFFFFFFF
                    else:
                        self.registers[ins.dst] = 0
                self.pc += 1

            elif op == Opcode.INC:
                if ins.dst in self.registers:
                    self.registers[ins.dst] = (self.registers[ins.dst] + 1) & 0xFFFFFFFF
                self.pc += 1

            elif op == Opcode.DEC:
                if ins.dst in self.registers:
                    self.registers[ins.dst] = (self.registers[ins.dst] - 1) & 0xFFFFFFFF
                self.pc += 1

            elif op == Opcode.NEG:
                if ins.dst in self.registers:
                    self.registers[ins.dst] = (-self.registers[ins.dst]) & 0xFFFFFFFF
                self.pc += 1

            elif op == Opcode.AND:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src)
                    self.registers[ins.dst] &= val
                self.pc += 1

            elif op == Opcode.OR:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src)
                    self.registers[ins.dst] |= val
                self.pc += 1

            elif op == Opcode.XOR:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src)
                    self.registers[ins.dst] ^= val
                self.pc += 1

            elif op == Opcode.NOT:
                if ins.dst in self.registers:
                    self.registers[ins.dst] = (~self.registers[ins.dst]) & 0xFFFFFFFF
                self.pc += 1

            elif op == Opcode.SHL:
                if ins.dst in self.registers:
                    shift = self._resolve_src(ins.src) % 32
                    self.registers[ins.dst] = (self.registers[ins.dst] << shift) & 0xFFFFFFFF
                self.pc += 1

            elif op == Opcode.SHR:
                if ins.dst in self.registers:
                    shift = self._resolve_src(ins.src) % 32
                    self.registers[ins.dst] = (self.registers[ins.dst] >> shift) & 0xFFFFFFFF
                self.pc += 1

            elif op == Opcode.POPCNT:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src)
                    self.registers[ins.dst] = bin(val & 0xFFFFFFFF).count("1")
                self.pc += 1

            elif op == Opcode.CLZ:
                if ins.dst in self.registers:
                    val = self._resolve_src(ins.src) & 0xFFFFFFFF
                    binary = bin(val)[2:].zfill(32)
                    self.registers[ins.dst] = len(binary) - len(binary.lstrip("0"))
                self.pc += 1

            elif op == Opcode.LOAD:
                if ins.dst in self.registers:
                    addr = self._resolve_src(ins.src) % self.memory_size
                    self.registers[ins.dst] = self.memory[addr]
                    self.memory_accesses += 1
                self.pc += 1

            elif op == Opcode.STORE:
                addr = self._resolve_src(ins.dst) % self.memory_size
                val = self._resolve_src(ins.src)
                self.memory[addr] = val
                self.memory_accesses += 1
                self.pc += 1

            elif op == Opcode.PUSH:
                val = self._resolve_src(ins.dst or ins.src)
                if len(self.stack) < 64:
                    self.stack.append(val)
                self.pc += 1

            elif op == Opcode.POP:
                if ins.dst in self.registers and self.stack:
                    self.registers[ins.dst] = self.stack.pop()
                self.pc += 1

            elif op == Opcode.CMP:
                val1 = self.registers.get(ins.dst or "ACC", 0)
                val2 = self._resolve_src(ins.src)
                diff = val1 - val2
                self.flags["zero"] = (diff == 0)
                self.flags["sign"] = (diff < 0)
                self.pc += 1

            elif op == Opcode.JMP:
                target = self._resolve_src(ins.dst or ins.src)
                self.pc = target if 0 <= target < prog_len else self.pc + 1

            elif op == Opcode.JZ:
                target = self._resolve_src(ins.dst or ins.src)
                if self.flags["zero"]:
                    self.pc = target if 0 <= target < prog_len else self.pc + 1
                else:
                    self.pc += 1

            elif op == Opcode.JNZ:
                target = self._resolve_src(ins.dst or ins.src)
                if not self.flags["zero"]:
                    self.pc = target if 0 <= target < prog_len else self.pc + 1
                else:
                    self.pc += 1

            elif op == Opcode.JL:
                target = self._resolve_src(ins.dst or ins.src)
                if self.flags["sign"] and not self.flags["zero"]:
                    self.pc = target if 0 <= target < prog_len else self.pc + 1
                else:
                    self.pc += 1

            elif op == Opcode.JG:
                target = self._resolve_src(ins.dst or ins.src)
                if not self.flags["sign"] and not self.flags["zero"]:
                    self.pc = target if 0 <= target < prog_len else self.pc + 1
                else:
                    self.pc += 1

            else:
                self.pc += 1

        # Return value convention: ACC register, fallback to R0
        ret = self.registers.get("ACC", 0)
        if ret == 0 and "R0" in self.registers:
            ret = self.registers["R0"]

        return VMExecutionResult(
            success=True,
            return_value=ret,
            registers=dict(self.registers),
            clock_cycles=self.clock_cycles,
            instructions_executed=self.instructions_executed,
            code_size_bytes=total_code_size,
            registers_used=set(self.active_registers_used),
            memory_access_count=self.memory_accesses,
            error=None,
            timeout_triggered=False,
        )
