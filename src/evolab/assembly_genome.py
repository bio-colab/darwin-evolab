"""Low-Level Assembly Genome representation, metric edit distance, and peephole mutators."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

from .asm_vm import REGISTERS, Instruction, Opcode
from .genome import EvolabGenome

OPCODE_CATEGORIES: dict[Opcode, str] = {
    Opcode.ADD: "alu_arith", Opcode.SUB: "alu_arith", Opcode.MUL: "alu_arith",
    Opcode.DIV: "alu_arith", Opcode.MOD: "alu_arith", Opcode.INC: "alu_arith",
    Opcode.DEC: "alu_arith", Opcode.NEG: "alu_arith",
    Opcode.AND: "alu_bit", Opcode.OR: "alu_bit", Opcode.XOR: "alu_bit",
    Opcode.NOT: "alu_bit", Opcode.SHL: "alu_bit", Opcode.SHR: "alu_bit",
    Opcode.POPCNT: "alu_bit", Opcode.CLZ: "alu_bit",
    Opcode.MOV: "data", Opcode.LOAD: "data", Opcode.STORE: "data",
    Opcode.PUSH: "data", Opcode.POP: "data", Opcode.SWAP: "data", Opcode.NOP: "data",
    Opcode.CMP: "flow", Opcode.JMP: "flow", Opcode.JZ: "flow",
    Opcode.JNZ: "flow", Opcode.JL: "flow", Opcode.JG: "flow",
    Opcode.RET: "flow", Opcode.HALT: "flow",
}


def instruction_cost(ins1: Instruction, ins2: Instruction) -> float:
    """Calculates granular semantic difference between two instructions."""
    if ins1 == ins2:
        return 0.0
    if ins1.op == ins2.op:
        cost = 0.0
        if ins1.dst != ins2.dst:
            cost += 0.20
        if ins1.src != ins2.src:
            cost += 0.20
        return max(0.1, cost)
    cat1 = OPCODE_CATEGORIES.get(ins1.op, "unknown")
    cat2 = OPCODE_CATEGORIES.get(ins2.op, "unknown")
    if cat1 == cat2:
        # Same semantic category (e.g. ADD <-> SUB, MUL <-> DIV)
        return 0.50
    # Cross-category structural change (e.g. ADD <-> JMP)
    return 1.0


def asm_distance(g1: AssemblyGenome, g2: AssemblyGenome) -> float:
    """Computes normalized Damerau-Levenshtein Metric Distance between two instruction sequences."""
    seq1, seq2 = g1.instructions, g2.instructions
    len1, len2 = len(seq1), len(seq2)
    if len1 == 0 and len2 == 0:
        return 0.0
    if len1 == 0 or len2 == 0:
        return 1.0

    # Dynamic programming Damerau-Levenshtein table
    dp = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = float(i)
    for j in range(len2 + 1):
        dp[0][j] = float(j)

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            sub_cost = instruction_cost(seq1[i - 1], seq2[j - 1])
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,           # Deletion
                dp[i][j - 1] + 1.0,           # Insertion
                dp[i - 1][j - 1] + sub_cost,  # Substitution
            )
            # Damerau adjacent instruction transposition
            if i > 1 and j > 1 and seq1[i - 1] == seq2[j - 2] and seq1[i - 2] == seq2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 0.3)

    max_len = max(len1, len2)
    return round(min(1.0, dp[len1][len2] / max_len), 6)


@dataclass
class AssemblyGenome(EvolabGenome):
    """Genome representation consisting of a linear sequence of machine instructions."""

    instructions: list[Instruction] = field(default_factory=list)

    def clone(self) -> AssemblyGenome:
        return AssemblyGenome(instructions=list(self.instructions))

    def copy(self) -> AssemblyGenome:
        """Alias for clone() to support standard copy protocol."""
        return self.clone()

    def __len__(self) -> int:
        return len(self.instructions)

    def fingerprint(self) -> str:
        raw = json.dumps([ins.serialize() for ins in self.instructions], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def distance_to(self, other: EvolabGenome) -> float:
        if not isinstance(other, AssemblyGenome):
            raise TypeError(f"Cannot calculate distance between AssemblyGenome and {type(other)}")
        return asm_distance(self, other)

    def serialize(self) -> list[dict[str, Any]]:
        return [ins.serialize() for ins in self.instructions]

    def to_assembly_text(self) -> str:
        """Returns disassembled assembly code listing."""
        return "\n".join(f"{i:03d}:  {ins.to_asm()}" for i, ins in enumerate(self.instructions))

    def describe(self) -> dict[str, float | int]:
        total_ins = len(self.instructions)
        if total_ins == 0:
            return {
                "instruction_count": 0,
                "code_size_bytes": 0,
                "branch_ratio": 0.0,
                "arithmetic_ratio": 0.0,
                "register_pressure": 0,
            }

        code_size = sum(ins.byte_size() for ins in self.instructions)
        branch_ops = {Opcode.JMP, Opcode.JZ, Opcode.JNZ, Opcode.JL, Opcode.JG}
        arithmetic_ops = {
            Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
            Opcode.INC, Opcode.DEC, Opcode.NEG, Opcode.AND, Opcode.OR,
            Opcode.XOR, Opcode.NOT, Opcode.SHL, Opcode.SHR, Opcode.POPCNT,
        }

        branches = sum(1 for ins in self.instructions if ins.op in branch_ops)
        arithmetics = sum(1 for ins in self.instructions if ins.op in arithmetic_ops)

        unique_regs = set()
        for ins in self.instructions:
            if ins.dst in REGISTERS:
                unique_regs.add(ins.dst)
            if isinstance(ins.src, str) and ins.src in REGISTERS:
                unique_regs.add(ins.src)

        return {
            "instruction_count": total_ins,
            "code_size_bytes": code_size,
            "branch_ratio": round(branches / total_ins, 4),
            "arithmetic_ratio": round(arithmetics / total_ins, 4),
            "register_pressure": len(unique_regs),
        }

    def mutate(
        self,
        rng: random.Random | None = None,
        mutation_rate: float = 0.25,
        max_instructions: int = 40,
        **kwargs: Any,
    ) -> AssemblyGenome:
        """Applies domain-specific low-level instruction mutations."""
        r = rng or random
        ins_list = [Instruction(ins.op, ins.dst, ins.src, imm=ins.imm) for ins in self.instructions]
        if not ins_list:
            return create_random_assembly_genome(size=4, rng=r)

        # 1. Peephole Dead Instruction Elimination (Optimization pass)
        if r.random() < 0.35 and len(ins_list) > 2:
            cleaned = []
            for ins in ins_list:
                # Remove redundant MOV R0, R0 or NOP
                if ins.op == Opcode.MOV and ins.dst == ins.src:
                    continue
                if ins.op == Opcode.NOP and len(ins_list) > 3:
                    continue
                cleaned.append(ins)
            if cleaned:
                ins_list = cleaned

        # 2. Instruction-level mutations
        for idx in range(len(ins_list)):
            if r.random() < mutation_rate:
                action = r.choice(["opcode", "operand", "replace"])
                current = ins_list[idx]

                if action == "opcode":
                    # Mutate to a compatible opcode
                    arithmetic_pool = [
                        Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.AND,
                        Opcode.OR, Opcode.XOR, Opcode.SHL, Opcode.SHR,
                    ]
                    if current.op in arithmetic_pool:
                        new_op = r.choice(arithmetic_pool)
                        ins_list[idx] = Instruction(new_op, current.dst, current.src)

                elif action == "operand":
                    # Mutate source operand or immediate value
                    if isinstance(current.src, int):
                        delta = r.choice([-1, 1, -2, 2, 8])
                        new_src = (current.src + delta) & 0xFFFF
                        ins_list[idx] = Instruction(current.op, current.dst, new_src)
                    elif current.dst in REGISTERS:
                        new_dst = r.choice(REGISTERS)
                        ins_list[idx] = Instruction(current.op, new_dst, current.src)

                elif action == "replace":
                    ins_list[idx] = _random_instruction(r, max_target=len(ins_list))

        # 3. Structural insertion / deletion
        if r.random() < 0.20 and len(ins_list) < max_instructions:
            insert_pos = r.randint(0, len(ins_list))
            ins_list.insert(insert_pos, _random_instruction(r, max_target=len(ins_list)))

        if r.random() < 0.20 and len(ins_list) > 3:
            del_pos = r.randint(0, len(ins_list) - 1)
            # Avoid deleting terminal RET if it's the last instruction
            if del_pos != len(ins_list) - 1 or ins_list[del_pos].op != Opcode.RET:
                del ins_list[del_pos]

        # Ensure valid termination
        if not any(ins.op in (Opcode.RET, Opcode.HALT) for ins in ins_list):
            ins_list.append(Instruction(Opcode.RET))

        return AssemblyGenome(instructions=ins_list)

    def crossover(
        self, other: EvolabGenome, rng: random.Random | None = None, **kwargs: Any
    ) -> AssemblyGenome:
        """Performs two-point block crossover on instruction tapes."""
        if not isinstance(other, AssemblyGenome):
            raise TypeError(f"Cannot crossover AssemblyGenome with {type(other)}")
        r = rng or random
        ins1, ins2 = self.instructions, other.instructions
        if len(ins1) < 2 or len(ins2) < 2:
            return self.clone() if r.random() < 0.5 else other.clone()

        cut1 = r.randint(1, len(ins1) - 1)
        cut2 = r.randint(1, len(ins2) - 1)

        child_ins = list(ins1[:cut1]) + list(ins2[cut2:])
        # Ensure routine termination
        if not any(ins.op in (Opcode.RET, Opcode.HALT) for ins in child_ins):
            child_ins.append(Instruction(Opcode.RET))

        return AssemblyGenome(instructions=child_ins[:40])


def _random_instruction(rng: random.Random, max_target: int = 10) -> Instruction:
    """Generates a syntactically valid random assembly instruction."""
    category = rng.choice(["arithmetic", "mov", "control", "bitwise"])

    if category == "arithmetic":
        op = rng.choice([Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.INC, Opcode.DEC])
        dst = rng.choice(REGISTERS)
        src = rng.choice(list(REGISTERS) + [rng.randint(0, 16)])
        return Instruction(op=op, dst=dst, src=src)

    elif category == "mov":
        dst = rng.choice(REGISTERS)
        src = rng.choice(list(REGISTERS) + [rng.randint(0, 32)])
        return Instruction(op=Opcode.MOV, dst=dst, src=src)

    elif category == "bitwise":
        op = rng.choice([Opcode.AND, Opcode.OR, Opcode.XOR, Opcode.SHL, Opcode.SHR, Opcode.POPCNT])
        dst = rng.choice(REGISTERS)
        src = rng.choice(list(REGISTERS) + [rng.randint(1, 8)])
        return Instruction(op=op, dst=dst, src=src)

    else:
        op = rng.choice([Opcode.CMP, Opcode.JZ, Opcode.JNZ, Opcode.NOP])
        if op == Opcode.CMP:
            return Instruction(op=op, dst=rng.choice(REGISTERS), src=rng.choice(list(REGISTERS) + [0, 1]))
        target = rng.randint(0, max(1, max_target))
        return Instruction(op=op, dst=None, src=target)


def create_random_assembly_genome(size: int = 8, rng: random.Random | None = None) -> AssemblyGenome:
    """Creates a random initialized AssemblyGenome."""
    r = rng or random
    instructions = [_random_instruction(r, max_target=size) for _ in range(size - 1)]
    instructions.append(Instruction(Opcode.RET))
    return AssemblyGenome(instructions=instructions)
