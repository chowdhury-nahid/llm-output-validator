from __future__ import annotations

from abc import ABC, abstractmethod

from .models import EvalContext, EvalResult, LLMJudge, ThresholdConfig


class BaseEval(ABC):
    """Abstract base for non-deterministic evaluations.

    Subclasses implement ``evaluate`` which returns a scored ``EvalResult``.
    The decision gate (PASS/FLAG/BLOCK) is always rule-driven via
    ``ThresholdConfig`` — the model produces a score, the rules produce the
    decision.
    """

    eval_name: str

    def __init__(self, threshold: ThresholdConfig | None = None) -> None:
        self.threshold = threshold or ThresholdConfig()

    @abstractmethod
    def evaluate(
        self,
        ctx: EvalContext,
        *,
        judge: LLMJudge | None = None,
    ) -> EvalResult:
        """Run the evaluation and return a scored result. Never raises."""
        ...
