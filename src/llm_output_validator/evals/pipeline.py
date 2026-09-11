from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .base import BaseEval
from .models import EvalContext, EvalDecision, EvalResult, LLMJudge


@dataclass
class EvalReport:
    """Aggregated result from running the evaluation pipeline."""

    decision: EvalDecision
    results: list[EvalResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def passed(self) -> bool:
        return self.decision == EvalDecision.PASS

    def blocked(self) -> bool:
        return self.decision == EvalDecision.BLOCK

    def flagged_evals(self) -> list[EvalResult]:
        return [r for r in self.results if r.decision == EvalDecision.FLAG]

    def blocked_evals(self) -> list[EvalResult]:
        return [r for r in self.results if r.decision == EvalDecision.BLOCK]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "eval_count": len(self.results),
            "evals": [r.to_dict() for r in self.results],
        }


class EvalPipeline:
    """Orchestrates non-deterministic evaluations.

    Runs a sequence of evaluations and produces an aggregated report.
    Designed to compose with the deterministic ``OutputValidator``:
    run deterministic checks first (fast, cheap), then the eval pipeline
    (slower, may need LLM calls).

    The pipeline's decision is the worst result across all evals:
    any BLOCK → BLOCK, any FLAG → FLAG, all PASS → PASS.
    """

    def __init__(
        self,
        evals: list[BaseEval] | None = None,
        *,
        fail_fast: bool = False,
    ) -> None:
        self._evals: list[BaseEval] = evals or []
        self._fail_fast = fail_fast

    def add(self, evaluation: BaseEval) -> EvalPipeline:
        self._evals.append(evaluation)
        return self

    def run(
        self,
        ctx: EvalContext,
        *,
        judge: LLMJudge | None = None,
    ) -> EvalReport:
        start = time.monotonic()
        results: list[EvalResult] = []

        for evaluation in self._evals:
            result = evaluation.evaluate(ctx, judge=judge)
            results.append(result)

            if self._fail_fast and result.decision == EvalDecision.BLOCK:
                break

        decision = _aggregate_decision(results)
        elapsed = (time.monotonic() - start) * 1000

        return EvalReport(
            decision=decision,
            results=results,
            elapsed_ms=elapsed,
        )


def _aggregate_decision(results: list[EvalResult]) -> EvalDecision:
    if not results:
        return EvalDecision.PASS
    if any(r.decision == EvalDecision.BLOCK for r in results):
        return EvalDecision.BLOCK
    if any(r.decision == EvalDecision.FLAG for r in results):
        return EvalDecision.FLAG
    return EvalDecision.PASS
