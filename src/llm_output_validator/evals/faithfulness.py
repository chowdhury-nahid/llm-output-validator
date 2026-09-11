from __future__ import annotations

import json
import re
from collections.abc import Sequence

from .base import BaseEval
from .models import EvalContext, EvalDecision, EvalResult, EvalScore, LLMJudge, ThresholdConfig

_DECOMPOSE_PROMPT = """Extract every factual claim from the following answer.
Return a JSON array of strings, one claim per element. Include only statements
that assert a fact — skip hedging, greetings, and meta-commentary.

Answer:
{answer}

Return ONLY a JSON array, nothing else."""

_VERIFY_PROMPT = """Given the context below, determine whether the claim is
supported, contradicted, or not mentioned.

Context:
{context}

Claim: {claim}

Reply with exactly one word: SUPPORTED, CONTRADICTED, or NOT_MENTIONED."""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _lexical_overlap(claim: str, context: str) -> float:
    claim_tokens = set(claim.lower().split())
    context_tokens = set(context.lower().split())
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & context_tokens) / len(claim_tokens)


def _decompose_into_claims(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _parse_claims_from_llm(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(c) for c in parsed if c]
    except (json.JSONDecodeError, TypeError):
        pass
    return _decompose_into_claims(raw)


class FaithfulnessEval(BaseEval):
    """Measures what fraction of claims in the answer are supported by context.

    Implements the core RAGAS faithfulness metric: decompose the answer into
    atomic claims, then verify each claim against the provided context.

    Two strategies:
    - **lexical** (default): token-overlap heuristic. Fast, no LLM needed.
      Good for CI, smoke tests, and as a lower-bound estimate.
    - **llm_judge**: uses an LLM to decompose claims and verify each against
      context. More accurate but requires API calls.

    The decision gate is always rule-driven: the score maps to PASS/FLAG/BLOCK
    via ThresholdConfig regardless of strategy.
    """

    eval_name = "faithfulness"

    def __init__(
        self,
        threshold: ThresholdConfig | None = None,
        overlap_threshold: float = 0.3,
    ) -> None:
        super().__init__(threshold)
        self._overlap_threshold = overlap_threshold

    def evaluate(
        self,
        ctx: EvalContext,
        *,
        judge: LLMJudge | None = None,
    ) -> EvalResult:
        try:
            if judge:
                return self._evaluate_with_judge(ctx, judge)
            return self._evaluate_lexical(ctx)
        except Exception as exc:
            return EvalResult(
                eval_name=self.eval_name,
                score=EvalScore(value=0.0, confidence=0.0),
                decision=EvalDecision.FLAG,
                threshold=self.threshold,
                reasoning=f"Evaluation failed: {exc}",
            )

    def _evaluate_lexical(self, ctx: EvalContext) -> EvalResult:
        claims = _decompose_into_claims(ctx.answer)
        if not claims:
            return self._empty_result("No claims extracted from answer")

        combined_context = " ".join(ctx.context)
        if not combined_context.strip():
            return self._no_context_result(claims)

        verdicts = []
        for claim in claims:
            overlap = _lexical_overlap(claim, combined_context)
            supported = overlap >= self._overlap_threshold
            verdicts.append(
                {
                    "claim": claim,
                    "verdict": "supported" if supported else "unsupported",
                    "overlap_score": round(overlap, 3),
                }
            )

        score = self._compute_score(verdicts)
        decision = self.threshold.decide(score)
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.6),
            decision=decision,
            threshold=self.threshold,
            reasoning=self._build_reasoning(verdicts, "lexical"),
            claims=verdicts,
            detail={"strategy": "lexical", "claim_count": len(claims)},
        )

    def _evaluate_with_judge(self, ctx: EvalContext, judge: LLMJudge) -> EvalResult:
        raw_claims = judge.evaluate(
            _DECOMPOSE_PROMPT.format(answer=ctx.answer)
        )
        claims = _parse_claims_from_llm(raw_claims)
        if not claims:
            return self._empty_result("Judge extracted no claims from answer")

        combined_context = "\n".join(ctx.context)
        if not combined_context.strip():
            return self._no_context_result(claims)

        verdicts = []
        for claim in claims:
            raw_verdict = judge.evaluate(
                _VERIFY_PROMPT.format(context=combined_context, claim=claim)
            )
            verdict = self._parse_verdict(raw_verdict)
            verdicts.append({"claim": claim, "verdict": verdict})

        score = self._compute_score(verdicts)
        decision = self.threshold.decide(score)
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.85),
            decision=decision,
            threshold=self.threshold,
            reasoning=self._build_reasoning(verdicts, "llm_judge"),
            claims=verdicts,
            detail={"strategy": "llm_judge", "claim_count": len(claims)},
        )

    def _compute_score(self, verdicts: Sequence[dict]) -> float:
        if not verdicts:
            return 0.0
        supported = sum(1 for v in verdicts if v["verdict"] == "supported")
        return round(supported / len(verdicts), 4)

    def _parse_verdict(self, raw: str) -> str:
        cleaned = raw.strip().upper()
        if "SUPPORTED" in cleaned and "NOT" not in cleaned:
            return "supported"
        if "CONTRADICTED" in cleaned:
            return "contradicted"
        return "unsupported"

    def _build_reasoning(self, verdicts: Sequence[dict], strategy: str) -> str:
        total = len(verdicts)
        supported = sum(1 for v in verdicts if v["verdict"] == "supported")
        return f"{supported}/{total} claims supported by context (strategy: {strategy})"

    def _empty_result(self, reason: str) -> EvalResult:
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=1.0, confidence=0.3),
            decision=EvalDecision.FLAG,
            threshold=self.threshold,
            reasoning=reason,
            detail={"strategy": "none", "claim_count": 0},
        )

    def _no_context_result(self, claims: list[str]) -> EvalResult:
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=0.0, confidence=0.5),
            decision=self.threshold.decide(0.0),
            threshold=self.threshold,
            reasoning=f"No context provided to verify {len(claims)} claims against",
            claims=[{"claim": c, "verdict": "unverifiable"} for c in claims],
            detail={"strategy": "no_context", "claim_count": len(claims)},
        )
