from __future__ import annotations

import re
from collections.abc import Sequence

from .base import BaseEval
from .models import EvalContext, EvalDecision, EvalResult, EvalScore, LLMJudge, ThresholdConfig

_ATTRIBUTION_PROMPT = """Given a claim from an answer and the provided context,
determine whether the claim can be attributed to the context.

Context:
{context}

Claim: {claim}

Reply with exactly one word: ATTRIBUTABLE or NOT_ATTRIBUTABLE."""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "up", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "what", "which", "who",
})


def _decompose_answer(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _lexical_attribution(claim: str, context: str) -> float:
    claim_tokens = _tokenize(claim)
    context_tokens = _tokenize(context)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & context_tokens) / len(claim_tokens)


class ContextRecallEval(BaseEval):
    """Measures whether the context contains all information needed for the answer.

    Implements the core concept from RAGAS context_recall: decompose the answer
    (acting as a proxy for ground truth) into atomic claims, then check what
    fraction of those claims can be attributed to the provided context.

    A high score means the retrieval system found all the necessary documents.
    A low score means the context is missing information that the answer
    relies on — either the retrieval failed to find relevant documents, or
    the answer fabricated information not in any retrieved document.

    Two strategies:
    - **lexical** (default): token overlap between each answer claim and the
      combined context. Fast, deterministic, good for CI.
    - **llm_judge**: asks an LLM to judge whether each claim is attributable
      to the context. More accurate but requires API calls.
    """

    eval_name = "context_recall"

    def __init__(
        self,
        threshold: ThresholdConfig | None = None,
        attribution_threshold: float = 0.3,
    ) -> None:
        super().__init__(threshold)
        self._attribution_threshold = attribution_threshold

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
        claims = _decompose_answer(ctx.answer)
        if not claims:
            return self._empty_result("No claims extracted from answer")

        combined_context = " ".join(ctx.context)
        if not combined_context.strip():
            return self._no_context_result(claims)

        attributions = []
        for claim in claims:
            overlap = _lexical_attribution(claim, combined_context)
            attributed = overlap >= self._attribution_threshold
            attributions.append({
                "claim": claim,
                "verdict": "attributable" if attributed else "not_attributable",
                "overlap_score": round(overlap, 4),
            })

        score = self._compute_score(attributions)
        decision = self.threshold.decide(score)
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.6),
            decision=decision,
            threshold=self.threshold,
            reasoning=self._build_reasoning(attributions, "lexical"),
            claims=attributions,
            detail={
                "strategy": "lexical",
                "claim_count": len(claims),
                "attributed_count": sum(
                    1 for a in attributions if a["verdict"] == "attributable"
                ),
            },
        )

    def _evaluate_with_judge(self, ctx: EvalContext, judge: LLMJudge) -> EvalResult:
        claims = _decompose_answer(ctx.answer)
        if not claims:
            return self._empty_result("No claims extracted from answer")

        combined_context = "\n".join(ctx.context)
        if not combined_context.strip():
            return self._no_context_result(claims)

        attributions = []
        for claim in claims:
            raw = judge.evaluate(
                _ATTRIBUTION_PROMPT.format(context=combined_context, claim=claim)
            )
            verdict = self._parse_verdict(raw)
            attributions.append({"claim": claim, "verdict": verdict})

        score = self._compute_score(attributions)
        decision = self.threshold.decide(score)
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.85),
            decision=decision,
            threshold=self.threshold,
            reasoning=self._build_reasoning(attributions, "llm_judge"),
            claims=attributions,
            detail={
                "strategy": "llm_judge",
                "claim_count": len(claims),
                "attributed_count": sum(
                    1 for a in attributions if a["verdict"] == "attributable"
                ),
            },
        )

    def _compute_score(self, attributions: Sequence[dict]) -> float:
        if not attributions:
            return 0.0
        attributed = sum(1 for a in attributions if a["verdict"] == "attributable")
        return round(attributed / len(attributions), 4)

    def _parse_verdict(self, raw: str) -> str:
        cleaned = raw.strip().upper()
        if "NOT" in cleaned:
            return "not_attributable"
        if "ATTRIBUTABLE" in cleaned:
            return "attributable"
        return "not_attributable"

    def _build_reasoning(self, attributions: Sequence[dict], strategy: str) -> str:
        total = len(attributions)
        attributed = sum(1 for a in attributions if a["verdict"] == "attributable")
        return (
            f"{attributed}/{total} answer claims attributable to context "
            f"(strategy: {strategy})"
        )

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
