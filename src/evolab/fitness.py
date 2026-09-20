"""Fitness landscapes: smooth proxy + ragged stress terrain."""
from __future__ import annotations

from .genome import Individual
from .landscapes import _clip


def default_fitness(ind: Individual) -> float:
    """Synthetic proxy landscape: per-gene closeness to a constant target.

    NOTE (post-audit): smooth, near-unimodal continuous problem.
    Validates mechanism dynamics, NOT transferability to rugged AST
    landscapes. Historical dead `speed_bonus` term removed with
    bit-identical outputs.
    """
    target = 3.0
    err = sum(abs(g - target) for g in ind.genome) / (len(ind.genome) * 8.0)
    return round(min(100.0, max(0.0, (1.0 - err)) * 100.0), 2)


def ragged_fitness(ind: Individual) -> float:
    """Multimodal stress landscape (Rastrigin-flavoured)."""
    n = len(ind.genome)
    base = sum((g - 3.0) ** 2 / (n * 12.5) for g in ind.genome)
    rp = (
        0.06
        * n
        * sum(__import__("math").cos(2.4 * (g - 3.0)) for g in ind.genome)
        / n
    )
    return _clip(100.0 - base * 100.0 - abs(rp) * 10.0)


class CurricularAnnealedEvaluator:
    """Dynamically anneals between continuous behavioral fitness (exploration)
    and strict formal multi-gate verification (rigor) over generations.

    Solves the Tabula Rasa dilemma:
    - Early generations (t=0 -> t_anneal * 0.5): High behavioral weight alpha_b.
      Allows pure primitive trees to climb gradients of numerical closeness,
      non-negativity invariants, and equilateral vanishing conditions.
    - Later generations (t -> t_anneal): Formal gates tighten, requiring 100%
      algebraic equivalence and constructive structural proof certification.
    """

    def __init__(
        self,
        behavioral_evaluator: Any,
        formal_evaluator: Any,
        total_generations: int = 40,
        annealing_schedule: str = "linear",  # "linear", "cosine", or "sigmoid"
    ) -> None:
        self.behavioral_evaluator = behavioral_evaluator
        self.formal_evaluator = formal_evaluator
        self.total_generations = max(1, total_generations)
        self.annealing_schedule = annealing_schedule
        self.current_generation = 1

    def set_generation(self, gen: int) -> None:
        self.current_generation = max(1, gen)

    @property
    def formal_weight(self) -> float:
        """Weight of the strict formal evaluation in [0.0, 1.0]."""
        progress = min(1.0, (self.current_generation - 1) / max(1, self.total_generations - 1))
        if self.annealing_schedule == "linear":
            return progress
        elif self.annealing_schedule == "cosine":
            import math
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.annealing_schedule == "sigmoid":
            import math
            k = 10.0
            return 1.0 / (1.0 + math.exp(-k * (progress - 0.5)))
        return progress

    def evaluate(self, target: Any, context: dict[str, Any] | None = None) -> Any:
        w_formal = self.formal_weight
        w_behavioral = 1.0 - w_formal

        res_beh = self.behavioral_evaluator.evaluate(target, context)
        if w_formal <= 1e-6:
            return res_beh

        res_form = self.formal_evaluator.evaluate(target, context)
        if w_behavioral <= 1e-6:
            return res_form

        blended_score = round(w_behavioral * res_beh.score + w_formal * res_form.score, 4)
        merged_sub_scores = dict(getattr(res_beh, "sub_scores", {}))
        for k, v in getattr(res_form, "sub_scores", {}).items():
            merged_sub_scores[f"formal_{k}"] = v
        merged_sub_scores["annealing_formal_weight"] = round(w_formal, 4)
        merged_sub_scores["annealing_generation"] = self.current_generation

        from .evaluators import FitnessResult
        return FitnessResult(score=blended_score, sub_scores=merged_sub_scores)


