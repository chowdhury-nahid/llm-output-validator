"""The Tier 1 (no-LLM) validation runner: ties the generic checks, the
scope profile, and the existing lexical eval pipeline into one call.

``validate(ctx, profile)`` is the single entry point both the MCP tool
(checkpoint 5) and any future CLI wiring call — kept independent of the MCP
SDK so it's testable without a server running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..evals.faithfulness import FaithfulnessEval
from ..evals.hallucination import HallucinationEval
from ..evals.models import EvalContext, EvalDecision, ThresholdConfig
from ..evals.pipeline import EvalPipeline
from ..evals.relevancy import AnswerRelevancyEval
from ..report import CheckResult, CheckStatus
from .content_rules_check import ContentRulesCheck
from .injection_check import InjectionCheck
from .json_schema_check import JsonSchemaCheck
from .pii_check import PiiCheck
from .profile import Profile

# Mirrors the answer/context size caps from the MCP best-practice review.
# Enforced here (not just at the future tool boundary) so the runner is
# safe to call directly, e.g. from a test or a future CLI.
MAX_ANSWER_CHARS = 100_000
MAX_CONTEXT_ITEMS = 50
MAX_CONTEXT_ITEM_CHARS = 20_000


@dataclass
class ValidationReport:
    """The single, structured result of a Tier 1 validation run."""

    decision: EvalDecision
    profile_name: str
    check_results: list[CheckResult] = field(default_factory=list)
    eval_results_dict: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    llm_tokens_used: int = 0  # always 0 in Tier 1; Tier 2 will populate this

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "profile": self.profile_name,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "llm_tokens_used": self.llm_tokens_used,
            "checks": [c.to_dict() for c in self.check_results],
            "evals": self.eval_results_dict,
        }

    def summary(self) -> str:
        """One-line human-readable summary, for a tool's text content."""
        failed_checks = [c.check_name for c in self.check_results if c.status == CheckStatus.FAIL]
        parts = [f"decision={self.decision.value}", f"profile={self.profile_name}"]
        if failed_checks:
            parts.append(f"failed_checks={failed_checks}")
        return ", ".join(parts)


def _oversized_input_report(profile_name: str, reason: str) -> ValidationReport:
    return ValidationReport(
        decision=EvalDecision.BLOCK,
        profile_name=profile_name,
        check_results=[
            CheckResult(
                pattern_id=100,
                check_name="input_size",
                status=CheckStatus.FAIL,
                severity="error",
                message=reason,
            )
        ],
    )


def validate(ctx: EvalContext, profile: Profile) -> ValidationReport:
    """Run every check and eval enabled in ``profile`` against ``ctx``.

    Never raises: an oversized input or an internal check failure both
    become a BLOCK-decision ``ValidationReport``, not an exception, so a
    calling MCP tool never has to catch anything from this function to stay
    within the "checks never raise" contract.
    """
    start = time.monotonic()

    answer = ctx.answer if isinstance(ctx.answer, str) else str(ctx.answer)
    if len(answer) > MAX_ANSWER_CHARS:
        return _oversized_input_report(
            profile.name,
            f"answer is {len(answer)} chars, over the {MAX_ANSWER_CHARS}-char limit",
        )
    if len(ctx.context) > MAX_CONTEXT_ITEMS:
        return _oversized_input_report(
            profile.name,
            f"context has {len(ctx.context)} items, over the {MAX_CONTEXT_ITEMS}-item limit",
        )
    for i, block in enumerate(ctx.context):
        block_len = len(block) if isinstance(block, str) else 0
        if block_len > MAX_CONTEXT_ITEM_CHARS:
            return _oversized_input_report(
                profile.name,
                f"context[{i}] is {block_len} chars, over the "
                f"{MAX_CONTEXT_ITEM_CHARS}-char per-item limit",
            )

    check_results: list[CheckResult] = []

    if profile.enable_json_schema:
        check_results.append(JsonSchemaCheck(schema=profile.json_schema.schema_).run(ctx))
    if profile.enable_injection:
        check_results.append(InjectionCheck().run(ctx))
    if profile.enable_pii:
        check_results.append(PiiCheck().run(ctx))
    if profile.enable_content_rules:
        check_results.append(
            ContentRulesCheck(
                required_terms=profile.content_rules.required_terms,
                banned_terms=profile.content_rules.banned_terms,
                case_sensitive=profile.content_rules.case_sensitive,
            ).run(ctx)
        )

    threshold = ThresholdConfig(
        pass_above=profile.eval_threshold.pass_above,
        block_below=profile.eval_threshold.block_below,
    )
    pipeline = EvalPipeline()
    if profile.enable_faithfulness:
        pipeline.add(FaithfulnessEval(threshold=threshold))
    if profile.enable_relevancy:
        pipeline.add(AnswerRelevancyEval(threshold=threshold))
    if profile.enable_hallucination:
        pipeline.add(HallucinationEval(threshold=threshold))

    # judge=None deliberately: Tier 1 never calls an LLM. This is the one
    # line in the runner that keeps this a no-LLM validator — Tier 2 will
    # be the only place a judge is ever passed.
    eval_report = pipeline.run(ctx, judge=None)

    check_decision = (
        EvalDecision.BLOCK
        if any(c.status == CheckStatus.FAIL for c in check_results)
        else EvalDecision.PASS
    )
    decision = (
        EvalDecision.BLOCK
        if check_decision == EvalDecision.BLOCK or eval_report.decision == EvalDecision.BLOCK
        else (EvalDecision.FLAG if eval_report.decision == EvalDecision.FLAG else EvalDecision.PASS)
    )

    elapsed = (time.monotonic() - start) * 1000
    return ValidationReport(
        decision=decision,
        profile_name=profile.name,
        check_results=check_results,
        eval_results_dict=eval_report.to_dict(),
        elapsed_ms=elapsed,
        llm_tokens_used=0,
    )
