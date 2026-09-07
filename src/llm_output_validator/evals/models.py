from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class EvalDecision(StrEnum):
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"


@dataclass(frozen=True)
class ThresholdConfig:
    """Rule-driven decision gate for scored evaluations.

    Scores at or above ``pass_above`` → PASS.
    Scores below ``block_below`` → BLOCK.
    Scores in between → FLAG for human review.
    """

    pass_above: float = 0.8
    block_below: float = 0.4

    def __post_init__(self) -> None:
        if not (0.0 <= self.block_below <= self.pass_above <= 1.0):
            msg = (
                f"Thresholds must satisfy 0 <= block_below <= pass_above <= 1, "
                f"got block_below={self.block_below}, pass_above={self.pass_above}"
            )
            raise ValueError(msg)

    def decide(self, score: float) -> EvalDecision:
        if score >= self.pass_above:
            return EvalDecision.PASS
        if score < self.block_below:
            return EvalDecision.BLOCK
        return EvalDecision.FLAG


@dataclass(frozen=True)
class EvalScore:
    """A scored evaluation result with confidence interval."""

    value: float
    confidence: float = 1.0
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            msg = f"Score value must be in [0, 1], got {self.value}"
            raise ValueError(msg)
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"Confidence must be in [0, 1], got {self.confidence}"
            raise ValueError(msg)


@dataclass
class EvalResult:
    """Result of a non-deterministic evaluation."""

    eval_name: str
    score: EvalScore
    decision: EvalDecision
    threshold: ThresholdConfig
    reasoning: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "eval_name": self.eval_name,
            "score": self.score.value,
            "confidence": self.score.confidence,
            "decision": self.decision.value,
            "threshold": {
                "pass_above": self.threshold.pass_above,
                "block_below": self.threshold.block_below,
            },
            "reasoning": self.reasoning,
        }
        if self.claims:
            result["claims"] = self.claims
        if self.detail:
            result["detail"] = self.detail
        if self.score.lower_bound is not None:
            result["confidence_interval"] = {
                "lower": self.score.lower_bound,
                "upper": self.score.upper_bound,
            }
        return result


@dataclass(frozen=True)
class EvalContext:
    """Input context for non-deterministic evaluations.

    Bridges the deterministic ``LLMResponse`` world with the evaluation layer.
    The deterministic checks work on ``LLMResponse``; the non-deterministic
    evals work on ``EvalContext``; they compose when both are available.
    """

    question: str
    answer: str
    context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMJudge(Protocol):
    """Protocol for LLM-based evaluation.

    Any LLM client that can take a prompt and return text satisfies this.
    Keeps the eval layer provider-agnostic.
    """

    def evaluate(self, prompt: str) -> str: ...
