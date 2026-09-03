"""LLM Semantic Rewrite Mutator module for evolutionary code optimization.

Provides client abstractions for Groq, Gemini, OpenAI, and offline Mocks,
coupled with syntax-safe AST extraction, lethal error handling, and token/cost accounting.
"""
from __future__ import annotations

import ast
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .ast_genome import ASTGenome
from .patch import Hunk, PatchGenome, apply_patch, create_patch_from_diff


@dataclass
class LLMConfig:
    provider: str = "mock"  # "groq", "gemini", "openai", "mock"
    api_key: str | None = field(default=None, repr=False)
    model_name: str = "qwen/qwen3.8-27b"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_sec: float = 10.0
    max_reflection_steps: int = 2
    system_prompt: str = (
        "You are an expert Python compiler and algorithm optimization specialist. "
        "You will be given a Python code snippet and optional evolutionary context. "
        "Rewrite the code to fix the bug, satisfy test requirements, or optimize the algorithm. "
        "Output ONLY the complete, syntactically valid Python code enclosed in a ```python ... ``` block."
    )
    structured_system_prompt: str = (
        "You are a precision code transformation agent. "
        "You will be given source code with line numbers and optimization/behavioral goals. "
        "Output ONLY a valid JSON object specifying the minimal surgical hunks to modify the code. "
        "Do not output markdown explanations outside of a ```json ... ``` block."
    )

    def get_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.provider == "groq":
            return os.environ.get("GROQ_API_KEY")
        if self.provider == "gemini":
            return os.environ.get("GEMINI_API_KEY")
        if self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return None


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_sec: float = 0.0
    model: str = ""
    success: bool = True
    error_message: str | None = None
    reflection_steps_used: int = 0
    mode: str = "code"


class LLMClient(Protocol):
    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        ...


class LLMCostTracker:
    """Tracks token usage, latencies, and estimated costs across LLM mutation calls."""

    def __init__(self, cost_per_1m_prompt: float = 0.05, cost_per_1m_completion: float = 0.08) -> None:
        self.cost_per_1m_prompt = cost_per_1m_prompt
        self.cost_per_1m_completion = cost_per_1m_completion
        self.total_calls: int = 0
        self.successful_calls: int = 0
        self.failed_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_latency_sec: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        prompt_cost = (self.total_prompt_tokens / 1_000_000.0) * self.cost_per_1m_prompt
        comp_cost = (self.total_completion_tokens / 1_000_000.0) * self.cost_per_1m_completion
        return round(prompt_cost + comp_cost, 6)

    def record(self, response: LLMResponse) -> None:
        self.total_calls += 1
        self.total_latency_sec += response.latency_sec
        if response.success:
            self.successful_calls += 1
            self.total_prompt_tokens += response.prompt_tokens
            self.total_completion_tokens += response.completion_tokens
        else:
            self.failed_calls += 1

    def summary(self) -> dict[str, Any]:
        avg_lat = round(self.total_latency_sec / max(self.total_calls, 1), 4)
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_sec": avg_lat,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


class MockLLMClient:
    """Deterministic offline mock client for CI, unit tests, reflection, and structured outputs."""

    def __init__(self, model_name: str = "mock-coder-v1") -> None:
        self.model_name = model_name

    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        t0 = time.perf_counter()

        # 1. Check if structured JSON is requested
        if "Digital Hardware Circuit Synthesis" in prompt or "Available IC Packages" in prompt:
            circuit_data = {
                "ic_packages": ["74HC86", "74HC08"],
                "connections": [
                    {"src_ic": -1, "src_pin": 0, "dst_ic": 0, "dst_pin": 1},
                    {"src_ic": -1, "src_pin": 1, "dst_ic": 0, "dst_pin": 2},
                    {"src_ic": -1, "src_pin": 0, "dst_ic": 1, "dst_pin": 1},
                    {"src_ic": -1, "src_pin": 1, "dst_ic": 1, "dst_pin": 2},
                    {"src_ic": 0, "src_pin": 3, "dst_ic": -1, "dst_pin": 100},
                    {"src_ic": 1, "src_pin": 3, "dst_ic": -1, "dst_pin": 101},
                ],
            }
            output = f"```json\n{json.dumps(circuit_data, indent=2)}\n```"
            latency = time.perf_counter() - t0
            return LLMResponse(
                content=output,
                prompt_tokens=max(len(prompt.split()), 10),
                completion_tokens=max(len(output.split()), 10),
                total_tokens=max(len(prompt.split()) + len(output.split()), 20),
                latency_sec=latency,
                model=self.model_name,
                success=True,
                mode="json",
            )

        if (system_prompt and "JSON" in system_prompt) or "Return ONLY a JSON object" in prompt:
            if "return x - 10" in prompt:
                hunk_data = {
                    "intent": "Optimize arithmetic subtraction to addition",
                    "file_path": "target.py",
                    "hunks": [
                        {
                            "start_line": 1,
                            "num_lines": 1,
                            "old_text": "    return x - 10\n",
                            "new_text": "    return x + 10\n",
                        }
                    ],
                }
            elif "strategy = \"defensive\"" in prompt or "defensive" in prompt:
                hunk_data = {
                    "intent": "Switch strategy to aggressive",
                    "file_path": "target.py",
                    "hunks": [
                        {
                            "start_line": 1,
                            "num_lines": 1,
                            "old_text": "    strategy = \"defensive\"\n",
                            "new_text": "    strategy = \"aggressive\"\n",
                        }
                    ],
                }
            else:
                hunk_data = {
                    "intent": "Mock structured patch",
                    "file_path": "target.py",
                    "hunks": [
                        {
                            "start_line": 0,
                            "num_lines": 0,
                            "old_text": "",
                            "new_text": "# structured patch applied\n",
                        }
                    ],
                }
            json_str = json.dumps(hunk_data, indent=2)
            output = f"```json\n{json_str}\n```"
            latency = time.perf_counter() - t0
            return LLMResponse(
                content=output,
                prompt_tokens=max(len(prompt.split()), 10),
                completion_tokens=max(len(output.split()), 10),
                total_tokens=max(len(prompt.split()) + len(output.split()), 20),
                latency_sec=latency,
                model=self.model_name,
                success=True,
                mode="structured_json",
            )

        # 2. Check for Reflection prompt
        if "Failed validation with error" in prompt or "analyze the error, fix the bug" in prompt:
            code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", prompt)
            raw_code = code_match.group(1) if code_match else prompt
            mutated = raw_code
            if "broken_code(:" in mutated:
                mutated = mutated.replace("broken_code(:", "broken_code():")
            if "return !!!" in mutated:
                mutated = mutated.replace("return !!!", "return 42")
            if "syntax_error" in mutated:
                mutated = mutated.replace("syntax_error", "valid_syntax = 1")
            if "Verification failure" in prompt or "AssertionError" in prompt:
                if "return x - 10" in mutated:
                    mutated = mutated.replace("return x - 10", "return x + 10")
                else:
                    mutated += "\n    # verified and repaired via reflection\n"

            latency = time.perf_counter() - t0
            output = f"```python\n{mutated.strip()}\n```"
            return LLMResponse(
                content=output,
                prompt_tokens=max(len(prompt.split()), 10),
                completion_tokens=max(len(output.split()), 10),
                total_tokens=max(len(prompt.split()) + len(output.split()), 20),
                latency_sec=latency,
                model=self.model_name,
                success=True,
                mode="reflection",
            )

        # 3. Standard code generation
        code_match = re.search(r"```(?:python)?\s*([\s\S]*?)```", prompt)
        if code_match:
            raw_code = code_match.group(1)
        else:
            raw_code = prompt

        mutated = raw_code
        if " - 10" in mutated:
            mutated = mutated.replace(" - 10", " + 10")
        elif " - 50" in mutated:
            mutated = mutated.replace(" - 50", " + 50")
        elif "x - 1" in mutated:
            mutated = mutated.replace("x - 1", "x + 1")
        elif "return 0" in mutated:
            mutated = mutated.replace("return 0", "return 1")
        elif "Behavioral Niche" in prompt or "Shift decision policy" in prompt or "Target Behavioral Property" in prompt:
            if "strategy = \"defensive\"" in mutated:
                mutated = mutated.replace("strategy = \"defensive\"", "strategy = \"aggressive\"")
            else:
                mutated = mutated + "\n    # adapted to behavioral niche\n"
        else:
            mutated = mutated + "\n# optimized by mock LLM\n"

        if "Leading Population Elite" in prompt:
            mutated += "    # inspired by elite competitor\n"

        latency = time.perf_counter() - t0
        output = f"```python\n{mutated.strip()}\n```"
        return LLMResponse(
            content=output,
            prompt_tokens=max(len(prompt.split()), 10),
            completion_tokens=max(len(output.split()), 10),
            total_tokens=max(len(prompt.split()) + len(output.split()), 20),
            latency_sec=latency,
            model=self.model_name,
            success=True,
            mode="code",
        )


class GroqClient:
    """Direct HTTP client for Groq API using standard library with WAF protection."""

    def __init__(self, api_key: str | None = None, model_name: str = "qwen/qwen3.8-27b", timeout_sec: float = 10.0) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                content="",
                success=False,
                error_message="GROQ_API_KEY is not set",
                model=self.model_name,
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (darwin-evolab)",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1024,
        }

        # Exponential retry loop for transient network glitches or rate limits
        for attempt in range(3):
            t0 = time.perf_counter()
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:  # nosec B310  # Endpoint is hardcoded HTTPS Groq API URL, not user input.
                    data = json.loads(response.read().decode("utf-8"))
                    latency = time.perf_counter() - t0
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return LLMResponse(
                        content=content,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        latency_sec=latency,
                        model=data.get("model", self.model_name),
                        success=True,
                    )
            except Exception as e:
                latency = time.perf_counter() - t0
                if attempt < 2 and "429" in str(e):
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return LLMResponse(
                    content="",
                    latency_sec=latency,
                    model=self.model_name,
                    success=False,
                    error_message=str(e),
                )
        return LLMResponse(content="", latency_sec=0.0, model=self.model_name, success=False)


def extract_python_code(raw_text: str) -> str:
    """Extract clean Python code from markdown code fences, stripping reasoning tags."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    pattern = r"```(?:python)?\s*\n([\s\S]*?)\n```"
    matches = re.findall(pattern, cleaned, re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip() + "\n"

    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        cleaned = cleaned.removesuffix("```")
        return cleaned.strip() + "\n"

    return cleaned.strip() + "\n"


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract clean JSON object from LLM response, handling markdown fences and think tags."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned, re.IGNORECASE)
    if json_match:
        return json.loads(json_match.group(1))

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    return json.loads(cleaned)


def build_mutation_prompt(
    code: str,
    error_context: str | None = None,
    objective: str | None = None,
    current_fitness: float | None = None,
    best_competitor_code: str | None = None,
    best_competitor_fitness: float | None = None,
    optimization_direction: str | None = None,
    behavioral_objective: str | None = None,
) -> str:
    """Constructs an evolution-aware prompt embedding population state, competitors, and goals."""
    prompt_parts = ["Here is the current Python code:\n```python\n" + code.strip() + "\n```"]

    evo_parts = []
    if current_fitness is not None:
        evo_parts.append(f"- Current Candidate Fitness: {current_fitness}")
    if optimization_direction:
        evo_parts.append(f"- Optimization Trajectory: {optimization_direction}")
    if behavioral_objective:
        evo_parts.append(f"- Behavioral Niche / Property Goal: {behavioral_objective}")
    if best_competitor_code is not None:
        elite_fit = f" (Fitness: {best_competitor_fitness})" if best_competitor_fitness is not None else ""
        evo_parts.append(
            f"- Leading Population Elite{elite_fit} for reference / cross-inspiration:\n"
            f"```python\n{best_competitor_code.strip()}\n```"
        )

    if evo_parts:
        prompt_parts.append("\nEvolutionary Context:\n" + "\n".join(evo_parts))

    if error_context:
        prompt_parts.append(f"\nExecution/Test Failure Context:\n{error_context.strip()}")
    if objective:
        prompt_parts.append(f"\nGoal:\n{objective.strip()}")

    prompt_parts.append("\nPlease rewrite and optimize this code. Return only valid Python in a ```python ... ``` block.")
    return "\n".join(prompt_parts)


class LLMSemanticMutator:
    """High-level semantic code mutator with lethal error protection, reflection loops, and structured JSON outputs."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        client: LLMClient | None = None,
        cost_tracker: LLMCostTracker | None = None,
    ) -> None:
        self.config = config or LLMConfig()
        if client is not None:
            self.client = client
        elif self.config.provider == "groq":
            self.client = GroqClient(
                api_key=self.config.api_key,
                model_name=self.config.model_name,
                timeout_sec=self.config.timeout_sec,
            )
        else:
            self.client = MockLLMClient(model_name=self.config.model_name)

        self.cost_tracker = cost_tracker or LLMCostTracker()

    def mutate_code(
        self,
        code: str,
        error_context: str | None = None,
        objective: str | None = None,
        current_fitness: float | None = None,
        best_competitor_code: str | None = None,
        best_competitor_fitness: float | None = None,
        optimization_direction: str | None = None,
        behavioral_objective: str | None = None,
        verifier: Any | None = None,
        max_reflection_steps: int | None = None,
    ) -> tuple[str, LLMResponse]:
        """Mutates Python code semantically using LLM with evolutionary context and self-reflection loops."""
        max_reflections = (
            max_reflection_steps
            if max_reflection_steps is not None
            else self.config.max_reflection_steps
        )

        prompt = build_mutation_prompt(
            code=code,
            error_context=error_context,
            objective=objective,
            current_fitness=current_fitness,
            best_competitor_code=best_competitor_code,
            best_competitor_fitness=best_competitor_fitness,
            optimization_direction=optimization_direction,
            behavioral_objective=behavioral_objective,
        )

        resp = self.client.complete(prompt, system_prompt=self.config.system_prompt)
        self.cost_tracker.record(resp)

        if not resp.success or not resp.content.strip():
            return code, resp

        extracted = extract_python_code(resp.content)
        reflection_steps = 0

        # Self-Reflection Loop: Catch syntax errors and test failures, feed back to LLM
        while True:
            validation_error = None
            try:
                ast.parse(extracted)
                compile(extracted, "<llm_mutated>", "exec")
            except (SyntaxError, ValueError) as e:
                validation_error = f"Lethal syntax error: {e}"

            if not validation_error and verifier is not None:
                try:
                    passed, v_msg = verifier(extracted)
                    if not passed:
                        validation_error = f"Verification failure: {v_msg}"
                except Exception as e:
                    validation_error = f"Verification exception: {e}"

            if not validation_error:
                resp.reflection_steps_used = reflection_steps
                return extracted, resp

            # Exhausted reflection budget: safe lethal fallback
            if reflection_steps >= max_reflections:
                resp.success = False
                resp.error_message = validation_error
                resp.reflection_steps_used = reflection_steps
                return code, resp

            # Trigger reflection step
            reflection_steps += 1
            reflection_prompt = (
                f"Your previously generated Python code:\n```python\n{extracted.strip()}\n```\n"
                f"Failed validation with error:\n{validation_error}\n"
                f"\nPlease analyze the error, fix the bug, and return ONLY the corrected, syntactically valid Python code "
                f"enclosed in a ```python ... ``` block."
            )
            resp = self.client.complete(reflection_prompt, system_prompt=self.config.system_prompt)
            self.cost_tracker.record(resp)

            if not resp.success or not resp.content.strip():
                resp.reflection_steps_used = reflection_steps
                return code, resp

            extracted = extract_python_code(resp.content)

    def mutate_structured(
        self,
        code: str,
        file_path: str = "target.py",
        objective: str | None = None,
        current_fitness: float | None = None,
        behavioral_objective: str | None = None,
        max_reflection_steps: int | None = None,
    ) -> tuple[str, PatchGenome, LLMResponse]:
        """Surgically mutates source code using structured JSON hunks to prevent full-file hallucinations."""
        max_reflections = (
            max_reflection_steps
            if max_reflection_steps is not None
            else self.config.max_reflection_steps
        )

        lines = code.splitlines(keepends=True)
        numbered_code = "".join(f"{i:3d} | {line}" for i, line in enumerate(lines))

        prompt_parts = [
            f"Target File: {file_path}",
            f"Code with 0-indexed line numbers:\n```python\n{numbered_code}\n```",
        ]
        if objective:
            prompt_parts.append(f"Objective: {objective}")
        if behavioral_objective:
            prompt_parts.append(f"Behavioral Strategy: {behavioral_objective}")
        if current_fitness is not None:
            prompt_parts.append(f"Current Fitness: {current_fitness}")

        prompt_parts.append(
            "\nReturn ONLY a JSON object specifying the minimal surgical hunks to modify:\n"
            "```json\n"
            "{\n"
            '  "intent": "Short summary",\n'
            f'  "file_path": "{file_path}",\n'
            '  "hunks": [\n'
            "    {\n"
            '      "start_line": <0-indexed start line integer>,\n'
            '      "num_lines": <number of lines to replace or 0 for insert>,\n'
            '      "old_text": "<exact old lines>",\n'
            '      "new_text": "<replacement lines>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )

        prompt = "\n".join(prompt_parts)
        resp = self.client.complete(prompt, system_prompt=self.config.structured_system_prompt)
        self.cost_tracker.record(resp)

        if not resp.success or not resp.content.strip():
            return code, PatchGenome(), resp

        reflection_steps = 0
        raw_content = resp.content

        while True:
            parse_error = None
            hunk_objs: list[Hunk] = []
            try:
                data = extract_json_payload(raw_content)
                for h in data.get("hunks", []):
                    hunk_objs.append(
                        Hunk(
                            file_path=h.get("file_path", file_path),
                            start_line=int(h["start_line"]),
                            num_lines=int(h["num_lines"]),
                            old_text=str(h.get("old_text", "")),
                            new_text=str(h.get("new_text", "")),
                        )
                    )
                patch = PatchGenome(hunks=hunk_objs)
                sources = {file_path: code}
                patched_sources = apply_patch(sources, patch)
                patched_code = patched_sources[file_path]
                ast.parse(patched_code)
                compile(patched_code, "<patched_code>", "exec")
                resp.reflection_steps_used = reflection_steps
                resp.mode = "structured_json"
                return patched_code, patch, resp
            except Exception as e:
                parse_error = f"Structured patch error: {e}"

            if reflection_steps >= max_reflections:
                resp.success = False
                resp.error_message = parse_error
                resp.reflection_steps_used = reflection_steps
                return code, PatchGenome(), resp

            reflection_steps += 1
            refl_prompt = (
                f"Your previously generated JSON patch:\n```json\n{raw_content}\n```\n"
                f"Failed with error: {parse_error}\n"
                f"Please fix the JSON hunk coordinates and return ONLY a valid, corrected JSON object in a ```json ... ``` block."
            )
            resp = self.client.complete(refl_prompt, system_prompt=self.config.structured_system_prompt)
            self.cost_tracker.record(resp)
            if not resp.success or not resp.content.strip():
                resp.reflection_steps_used = reflection_steps
                return code, PatchGenome(), resp
            raw_content = resp.content

    def mutate_behavioral(
        self,
        code: str,
        target_property: str,
        current_behavior: str | None = None,
        target_niche: str | None = None,
        objective: str | None = None,
        verifier: Any | None = None,
    ) -> tuple[str, LLMResponse]:
        """Directs the LLM to alter decision policies and behavioral properties for MAP-Elites niches."""
        behavioral_direction = (
            f"Shift decision policy from '{current_behavior or 'default'}' to '{target_niche or target_property}'. "
            f"Focus on altering branching logic, execution modes, or algorithmic strategy rather than mere micro-optimizations."
        )
        return self.mutate_code(
            code=code,
            objective=objective,
            behavioral_objective=behavioral_direction,
            optimization_direction=f"Target Behavioral Property: {target_property}",
            verifier=verifier,
        )

    def mutate_ast(
        self,
        genome: ASTGenome,
        error_context: str | None = None,
        current_fitness: float | None = None,
        best_competitor_code: str | None = None,
        optimization_direction: str | None = None,
        verifier: Any | None = None,
    ) -> tuple[ASTGenome, LLMResponse]:
        """Mutates an ASTGenome via LLM semantic rewrite with evolutionary context."""
        original_code = genome.to_code()
        mutated_code, resp = self.mutate_code(
            original_code,
            error_context=error_context,
            current_fitness=current_fitness,
            best_competitor_code=best_competitor_code,
            optimization_direction=optimization_direction,
            verifier=verifier,
        )
        if resp.success and mutated_code != original_code:
            try:
                new_genome = ASTGenome.from_code(mutated_code)
                return new_genome, resp
            except Exception:
                pass
        return genome.clone(), resp

    def mutate_patch(
        self,
        genome: PatchGenome,
        sources: dict[str, str],
        target_file: str,
        error_context: str | None = None,
        current_fitness: float | None = None,
        best_competitor_code: str | None = None,
        optimization_direction: str | None = None,
    ) -> tuple[PatchGenome, LLMResponse]:
        """Mutates a source file via LLM and computes a new PatchGenome."""
        current_sources = genome.apply_to(sources)
        if target_file not in current_sources:
            return genome.clone(), LLMResponse(
                content="", success=False, error_message=f"Target file {target_file} not found"
            )

        original_code = current_sources[target_file]
        mutated_code, resp = self.mutate_code(
            original_code,
            error_context=error_context,
            current_fitness=current_fitness,
            best_competitor_code=best_competitor_code,
            optimization_direction=optimization_direction,
        )
        if resp.success and mutated_code != original_code:
            try:
                new_patch = create_patch_from_diff(
                    target_file,
                    sources[target_file],
                    mutated_code,
                )
                return new_patch, resp
            except Exception as e:
                resp.success = False
                resp.error_message = f"Failed to create patch from diff: {e}"
        return genome.clone(), resp

    def mutate_circuit_netlist(
        self,
        current_topology: str,
        truth_table_specs: str,
        current_fitness: float | None = None,
        available_parts: list[str] | None = None,
    ) -> tuple[dict[str, Any] | None, LLMResponse]:
        """Surgically mutates or redesigns circuit IC connections using domain-aware LLM prompts."""
        parts_str = ", ".join(available_parts) if available_parts else "74HC00, 74HC08, 74HC32, 74HC86"
        prompt_parts = [
            "Domain: Digital Hardware Circuit Synthesis (74xx CMOS Logic ICs)",
            f"Available IC Packages: {parts_str}",
            f"Target Truth Table / Requirements:\n{truth_table_specs}",
            f"Current Circuit Topology / Netlist:\n```\n{current_topology}\n```",
        ]
        if current_fitness is not None:
            prompt_parts.append(f"Current Fitness Score: {current_fitness:.2f}%")

        prompt_parts.append(
            "\nAnalyze why the circuit fails the truth table or timing. Propose an updated wiring topology.\n"
            "Return ONLY a JSON object in this format:\n"
            "```json\n"
            "{\n"
            '  "ic_packages": ["74HC86", "74HC08"],\n'
            '  "connections": [\n'
            '    {"src_ic": -1, "src_pin": 0, "dst_ic": 0, "dst_pin": 1},\n'
            '    {"src_ic": 0, "src_pin": 3, "dst_ic": -1, "dst_pin": 100}\n'
            "  ]\n"
            "}\n"
            "```\n"
            "Note: src_ic=-1 means primary circuit input pins (0, 1...).\n"
            "dst_ic=-1 with dst_pin >= 100 means primary circuit output pins (100 is out0, 101 is out1)."
        )
        prompt = "\n".join(prompt_parts)
        system_prompt = (
            "You are an expert digital electronics and hardware logic synthesis engineer. "
            "Design digital circuits using standard 74HC ICs. Output ONLY the valid JSON circuit specification."
        )
        resp = self.client.complete(prompt, system_prompt=system_prompt)
        self.cost_tracker.record(resp)

        if not resp.success or not resp.content.strip():
            return None, resp

        try:
            raw = resp.content
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
            json_str = match.group(1) if match else raw.strip()
            data = json.loads(json_str)
            if "ic_packages" in data and "connections" in data:
                return data, resp
        except Exception as e:
            resp.error_message = f"Failed to parse circuit JSON: {e}"

        return None, resp
