"""Unit and integration tests for LLM Semantic Mutator, cost accounting, and syntax safety gates."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evolab.ast_genome import ASTGenome
from evolab.patch import PatchGenome
from evolab.llm_mutator import (
    LLMConfig,
    LLMResponse,
    MockLLMClient,
    GroqClient,
    LLMCostTracker,
    LLMSemanticMutator,
    extract_python_code,
)


def test_extract_python_code():
    raw_markdown = (
        "Here is the fixed code:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n"
        "Hope this helps!"
    )
    extracted = extract_python_code(raw_markdown)
    assert extracted.strip() == "def add(a, b):\n    return a + b"

    raw_plain = "def sub(a, b):\n    return a - b"
    assert extract_python_code(raw_plain).strip() == raw_plain


def test_cost_tracker_accounting():
    tracker = LLMCostTracker(cost_per_1m_prompt=0.05, cost_per_1m_completion=0.08)
    resp1 = LLMResponse(
        content="code",
        prompt_tokens=100_000,
        completion_tokens=50_000,
        total_tokens=150_000,
        latency_sec=0.5,
        success=True,
    )
    resp2 = LLMResponse(
        content="",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        latency_sec=0.1,
        success=False,
        error_message="timeout",
    )

    tracker.record(resp1)
    tracker.record(resp2)

    summary = tracker.summary()
    assert summary["total_calls"] == 2
    assert summary["successful_calls"] == 1
    assert summary["failed_calls"] == 1
    assert summary["prompt_tokens"] == 100_000
    assert summary["completion_tokens"] == 50_000
    # Cost = (100k / 1M * 0.05) + (50k / 1M * 0.08) = 0.005 + 0.004 = 0.009
    assert abs(summary["estimated_cost_usd"] - 0.009) < 1e-5


def test_mock_llm_client_mutation():
    mutator = LLMSemanticMutator(config=LLMConfig(provider="mock"))
    code = "def compute(x):\n    return x - 10\n"

    mutated, resp = mutator.mutate_code(code, error_context="AssertionError: 0 != 20")
    assert resp.success is True
    assert "return x + 10" in mutated
    compile(mutated, "<test>", "exec")


def test_ast_mutation_via_llm():
    mutator = LLMSemanticMutator(config=LLMConfig(provider="mock"))
    code = "def calc(a):\n    return a - 50\n"
    genome = ASTGenome.from_code(code)

    mutated_genome, resp = mutator.mutate_ast(genome, error_context="Test failure on calc(50)")
    assert resp.success is True
    assert isinstance(mutated_genome, ASTGenome)
    assert "return a + 50" in mutated_genome.to_code()


def test_patch_mutation_via_llm():
    mutator = LLMSemanticMutator(config=LLMConfig(provider="mock"))
    sources = {
        "algo.py": "def step(x):\n    return x - 1\n"
    }
    patch = PatchGenome()

    new_patch, resp = mutator.mutate_patch(patch, sources, target_file="algo.py")
    assert resp.success is True
    assert isinstance(new_patch, PatchGenome)
    applied = new_patch.apply_to(sources)
    assert "return x + 1" in applied["algo.py"]


def test_lethal_gate_invalid_syntax_recovery():
    class BrokenLLMClient:
        def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
            return LLMResponse(
                content="```python\ndef broken_code(:\n    return !!!\n```",
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
                latency_sec=0.01,
                success=True,
            )

    mutator = LLMSemanticMutator(client=BrokenLLMClient())
    original_code = "def safe_code(x):\n    return x * 2\n"

    mutated, resp = mutator.mutate_code(original_code)
    # The lethal gate must catch syntax error, reject broken code, and preserve original
    assert mutated == original_code
    assert resp.success is False
    assert "Lethal syntax error" in resp.error_message


def test_groq_client_execution():
    api_key = os.environ.get("GROQ_API_KEY", "")
    client = GroqClient(api_key=api_key)

    if not api_key:
        resp = client.complete("Hello")
        assert resp.success is False
        assert "GROQ_API_KEY is not set" in resp.error_message
    else:
        resp = client.complete("Return the single word: OK")
        assert resp.success is True
        assert len(resp.content) > 0


def test_reflection_loop_compilation_recovery():
    """Verify that reflection loop recovers from lethal syntax errors by self-correcting."""
    class FailingFirstClient:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                # Step 1: Broken syntax
                return LLMResponse(
                    content="```python\ndef broken_code(:\n    return !!!\n```",
                    prompt_tokens=15,
                    completion_tokens=10,
                    total_tokens=25,
                    latency_sec=0.01,
                    success=True,
                )
            else:
                # Step 2: Repaired syntax in response to reflection prompt
                assert "Failed validation with error" in prompt
                return LLMResponse(
                    content="```python\ndef broken_code():\n    return 42\n```",
                    prompt_tokens=25,
                    completion_tokens=10,
                    total_tokens=35,
                    latency_sec=0.01,
                    success=True,
                )

    client = FailingFirstClient()
    mutator = LLMSemanticMutator(client=client, config=LLMConfig(max_reflection_steps=2))
    original = "def broken_code():\n    return 0\n"

    mutated, resp = mutator.mutate_code(original)
    assert resp.success is True
    assert resp.reflection_steps_used == 1
    assert "return 42" in mutated
    assert client.calls == 2


def test_reflection_loop_verifier_recovery():
    """Verify reflection loop corrects logical failures flagged by a verifier function."""
    mutator = LLMSemanticMutator(config=LLMConfig(provider="mock", max_reflection_steps=2))
    original = "def compute(x):\n    return x - 10\n"

    def strict_verifier(code_str: str) -> tuple[bool, str]:
        if "return x + 10" not in code_str:
            return False, "Function must perform addition (return x + 10)"
        return True, "OK"

    mutated, resp = mutator.mutate_code(original, verifier=strict_verifier)
    assert resp.success is True
    assert "return x + 10" in mutated


def test_evolutionary_context_prompting():
    """Verify that evolutionary context (fitness, elite code, optimization vector) is passed and used."""
    class CapturingClient:
        def __init__(self):
            self.captured_prompt = ""

        def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
            self.captured_prompt = prompt
            return LLMResponse(
                content="```python\ndef solve(x):\n    return x * 10\n```",
                prompt_tokens=30,
                completion_tokens=15,
                total_tokens=45,
                latency_sec=0.01,
                success=True,
            )

    client = CapturingClient()
    mutator = LLMSemanticMutator(client=client)

    mutator.mutate_code(
        code="def solve(x):\n    return x\n",
        current_fitness=45.2,
        best_competitor_code="def solve(x):\n    return x * 8\n",
        best_competitor_fitness=88.7,
        optimization_direction="exploit_vectorized_arithmetic",
    )

    p = client.captured_prompt
    assert "Current Candidate Fitness: 45.2" in p
    assert "Leading Population Elite (Fitness: 88.7)" in p
    assert "return x * 8" in p
    assert "Optimization Trajectory: exploit_vectorized_arithmetic" in p


def test_structured_json_patch_mutation():
    """Verify surgical structured JSON hunk mutation via PatchGenome."""
    mutator = LLMSemanticMutator(config=LLMConfig(provider="mock"))
    original_code = (
        "def compute(x):\n"
        "    return x - 10\n"
    )

    patched_code, patch, resp = mutator.mutate_structured(
        code=original_code,
        file_path="algo.py",
        objective="Change subtraction to addition",
    )

    assert resp.success is True
    assert resp.mode == "structured_json"
    assert len(patch.hunks) == 1
    assert "return x + 10" in patched_code
    # Ensure syntax is valid
    compile(patched_code, "<patched_test>", "exec")


def test_property_driven_behavioral_mutation():
    """Verify behavioral property-driven mutation for MAP-Elites niches."""
    mutator = LLMSemanticMutator(config=LLMConfig(provider="mock"))
    code = (
        "def decide_action():\n"
        "    strategy = \"defensive\"\n"
        "    return strategy\n"
    )

    mutated, resp = mutator.mutate_behavioral(
        code=code,
        target_property="aggressive_attack_vector",
        current_behavior="defensive",
        target_niche="high_aggression_cluster",
    )

    assert resp.success is True
    assert "aggressive" in mutated
