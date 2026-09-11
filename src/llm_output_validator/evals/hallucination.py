from __future__ import annotations

import json
import re
from collections.abc import Sequence

from .base import BaseEval
from .models import EvalContext, EvalDecision, EvalResult, EvalScore, LLMJudge, ThresholdConfig

_DETECT_PROMPT = """Analyze the answer for hallucinations — statements not supported by the
provided context. For each statement in the answer, classify it as:
- GROUNDED: directly supported by the context
- HALLUCINATED: asserts something not in or contradicted by the context
- AMBIGUOUS: partially supported or context is unclear

Context:
{context}

Answer:
{answer}

Return ONLY a JSON object:
{{
  "statements": [
    {{"text": "...", "verdict": "GROUNDED|HALLUCINATED|AMBIGUOUS", "reason": "..."}}
  ],
  "hallucination_rate": <float 0.0-1.0>
}}"""

_ENTITY_RE = re.compile(
    r"(?:"
    r"\d{1,2}[./]\d{1,2}[./]\d{2,4}"  # dates
    r"|\d+\.?\d*\s*%"                  # percentages
    r"|[$€£][\d,]+(?:\.\d+)?"         # currency amounts
    r"|\b\d+(?:\.\d+)?\b"             # plain numbers
    r")"
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "or", "but", "not", "no", "if", "so", "than", "too", "very", "just",
    "it", "its", "this", "that", "these", "those",
})


def _extract_entities(text: str) -> set[str]:
    return {m.group().strip() for m in _ENTITY_RE.finditer(text)}


def _extract_content_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _STOPWORDS


def _sentence_grounding_score(sentence: str, context: str) -> float:
    s_tokens = _extract_content_tokens(sentence)
    c_tokens = _extract_content_tokens(context)
    if not s_tokens:
        return 1.0
    return len(s_tokens & c_tokens) / len(s_tokens)


def _entity_grounding(answer: str, context: str) -> tuple[set[str], set[str]]:
    answer_entities = _extract_entities(answer)
    context_entities = _extract_entities(context)
    grounded = answer_entities & context_entities
    ungrounded = answer_entities - context_entities
    return grounded, ungrounded


class HallucinationEval(BaseEval):
    """Detects semantic hallucinations — claims in the answer not grounded in context.

    Unlike the deterministic injection_check (Pattern 7) which catches
    structural prompt injection, this eval catches *semantic* hallucination:
    plausible-sounding facts that are invented or contradict the source.

    Returns a score where 1.0 = no hallucination detected, 0.0 = fully
    hallucinated. This inverted scale matches the "quality" convention —
    higher is better — and maps cleanly to ThresholdConfig.

    Two strategies:
    - **lexical** (default): combines sentence-level token grounding with
      entity verification (numbers, dates, percentages). Fast, catches
      fabricated specifics. Good for CI and as a lower-bound estimate.
    - **llm_judge**: asks an LLM to classify each statement as grounded,
      hallucinated, or ambiguous. More accurate for semantic nuance.

    The score represents (1 - hallucination_rate) so that the PASS/FLAG/BLOCK
    gate works consistently with other evals: high score = good.
    """

    eval_name = "hallucination_detection"

    def __init__(
        self,
        threshold: ThresholdConfig | None = None,
        entity_weight: float = 0.4,
        sentence_weight: float = 0.6,
        grounding_cutoff: float = 0.3,
    ) -> None:
        super().__init__(threshold)
        self._entity_weight = entity_weight
        self._sentence_weight = sentence_weight
        self._grounding_cutoff = grounding_cutoff

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
        combined_context = " ".join(ctx.context)

        if not combined_context.strip():
            return self._no_context_result(ctx.answer)

        sentences = _SENTENCE_SPLIT.split(ctx.answer.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return EvalResult(
                eval_name=self.eval_name,
                score=EvalScore(value=1.0, confidence=0.3),
                decision=EvalDecision.FLAG,
                threshold=self.threshold,
                reasoning="No substantive sentences to evaluate",
                detail={"strategy": "lexical"},
            )

        statement_results = []
        for sentence in sentences:
            grounding = _sentence_grounding_score(sentence, combined_context)
            verdict = "grounded" if grounding >= self._grounding_cutoff else "hallucinated"
            statement_results.append({
                "text": sentence,
                "verdict": verdict,
                "grounding_score": round(grounding, 3),
            })

        grounded_ents, ungrounded_ents = _entity_grounding(ctx.answer, combined_context)
        total_entities = len(grounded_ents) + len(ungrounded_ents)
        entity_score = (
            len(grounded_ents) / total_entities if total_entities > 0 else 1.0
        )

        sentence_score = (
            sum(1 for s in statement_results if s["verdict"] == "grounded")
            / len(statement_results)
        )

        raw_score = (
            self._sentence_weight * sentence_score
            + self._entity_weight * entity_score
        )
        score = round(min(max(raw_score, 0.0), 1.0), 4)
        decision = self.threshold.decide(score)

        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.6),
            decision=decision,
            threshold=self.threshold,
            reasoning=self._build_reasoning(statement_results, entity_score, "lexical"),
            claims=statement_results,
            detail={
                "strategy": "lexical",
                "sentence_score": round(sentence_score, 4),
                "entity_score": round(entity_score, 4),
                "grounded_entities": sorted(grounded_ents),
                "ungrounded_entities": sorted(ungrounded_ents),
                "statement_count": len(sentences),
            },
        )

    def _evaluate_with_judge(self, ctx: EvalContext, judge: LLMJudge) -> EvalResult:
        combined_context = "\n".join(ctx.context)
        if not combined_context.strip():
            return self._no_context_result(ctx.answer)

        raw = judge.evaluate(
            _DETECT_PROMPT.format(context=combined_context, answer=ctx.answer)
        )
        statements, hallucination_rate = self._parse_judge_response(raw)
        score = round(1.0 - hallucination_rate, 4)
        decision = self.threshold.decide(score)

        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.85),
            decision=decision,
            threshold=self.threshold,
            reasoning=self._build_reasoning_judge(statements),
            claims=statements,
            detail={
                "strategy": "llm_judge",
                "hallucination_rate": round(hallucination_rate, 4),
                "statement_count": len(statements),
            },
        )

    def _parse_judge_response(
        self, raw: str
    ) -> tuple[list[dict], float]:
        try:
            parsed = json.loads(raw)
            statements = parsed.get("statements", [])
            rate = float(parsed.get("hallucination_rate", 0.0))
            rate = min(max(rate, 0.0), 1.0)
            return statements, rate
        except (json.JSONDecodeError, TypeError, ValueError):
            return [], 0.5

    def _build_reasoning(
        self,
        statements: Sequence[dict],
        entity_score: float,
        strategy: str,
    ) -> str:
        total = len(statements)
        hallucinated = sum(1 for s in statements if s["verdict"] == "hallucinated")
        return (
            f"{hallucinated}/{total} statements flagged as hallucinated, "
            f"entity grounding={entity_score:.2f} (strategy: {strategy})"
        )

    def _build_reasoning_judge(self, statements: list[dict]) -> str:
        if not statements:
            return "No statements analyzed by judge"
        total = len(statements)
        hallu = sum(
            1 for s in statements
            if str(s.get("verdict", "")).upper() == "HALLUCINATED"
        )
        return f"{hallu}/{total} statements classified as hallucinated (strategy: llm_judge)"

    def _no_context_result(self, answer: str) -> EvalResult:
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=0.0, confidence=0.5),
            decision=self.threshold.decide(0.0),
            threshold=self.threshold,
            reasoning="No context provided — cannot verify hallucination",
            detail={"strategy": "no_context"},
        )
