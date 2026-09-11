from __future__ import annotations

import math
import re
from collections import Counter

from .base import BaseEval
from .models import EvalContext, EvalDecision, EvalResult, EvalScore, LLMJudge, ThresholdConfig

_PRECISION_PROMPT = """Given the question and a piece of context, determine whether
the context is useful for answering the question.

Question: {question}

Context:
{context_item}

Reply with exactly one word: RELEVANT or IRRELEVANT."""

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


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _relevance_score(question: str, context_item: str) -> float:
    q_tokens = _tokenize(question)
    c_tokens = _tokenize(context_item)
    if not q_tokens or not c_tokens:
        return 0.0
    q_counter = Counter(q_tokens)
    c_counter = Counter(c_tokens)
    terms = set(q_counter) | set(c_counter)
    dot = sum(q_counter.get(t, 0) * c_counter.get(t, 0) for t in terms)
    mag_q = math.sqrt(sum(v * v for v in q_counter.values()))
    mag_c = math.sqrt(sum(v * v for v in c_counter.values()))
    if mag_q == 0 or mag_c == 0:
        return 0.0
    return dot / (mag_q * mag_c)


def _average_precision(relevance_flags: list[bool]) -> float:
    """Compute average precision from ordered relevance judgments.

    AP = (1/R) * sum_{k=1}^{n} (P@k * rel_k)
    where R is total relevant items, P@k is precision at position k,
    and rel_k is 1 if item k is relevant.
    """
    if not relevance_flags:
        return 0.0
    total_relevant = sum(relevance_flags)
    if total_relevant == 0:
        return 0.0
    cumulative = 0
    ap_sum = 0.0
    for k, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            cumulative += 1
            ap_sum += cumulative / k
    return ap_sum / total_relevant


class ContextPrecisionEval(BaseEval):
    """Measures whether relevant context items are ranked above irrelevant ones.

    Implements the core concept from RAGAS context_precision: for each context
    item in order, judge whether it is relevant to the question, then compute
    average precision — a metric that rewards relevant items appearing earlier
    in the context list.

    A high score means the retrieval system put useful documents first. A low
    score means relevant information is buried under noise — the LLM must
    sift through irrelevant context, increasing hallucination risk.

    Two strategies:
    - **lexical** (default): cosine similarity between question and each
      context item. Items above a relevance threshold are marked relevant.
      Fast, deterministic, good for CI.
    - **llm_judge**: asks an LLM to judge relevance of each context item.
      More accurate but requires API calls.
    """

    eval_name = "context_precision"

    def __init__(
        self,
        threshold: ThresholdConfig | None = None,
        relevance_threshold: float = 0.15,
    ) -> None:
        super().__init__(threshold)
        self._relevance_threshold = relevance_threshold

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
        if not ctx.context:
            return self._no_context_result()

        judgments: list[dict] = []
        relevance_flags: list[bool] = []

        for i, item in enumerate(ctx.context):
            score = _relevance_score(ctx.question, item)
            is_relevant = score >= self._relevance_threshold
            relevance_flags.append(is_relevant)
            judgments.append({
                "index": i,
                "context_preview": item[:80],
                "relevance_score": round(score, 4),
                "verdict": "relevant" if is_relevant else "irrelevant",
            })

        ap = _average_precision(relevance_flags)
        score = round(min(max(ap, 0.0), 1.0), 4)
        decision = self.threshold.decide(score)

        relevant_count = sum(relevance_flags)
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.6),
            decision=decision,
            threshold=self.threshold,
            reasoning=(
                f"{relevant_count}/{len(ctx.context)} context items relevant, "
                f"average precision={score} (strategy: lexical)"
            ),
            claims=judgments,
            detail={
                "strategy": "lexical",
                "context_count": len(ctx.context),
                "relevant_count": relevant_count,
                "average_precision": score,
            },
        )

    def _evaluate_with_judge(self, ctx: EvalContext, judge: LLMJudge) -> EvalResult:
        if not ctx.context:
            return self._no_context_result()

        judgments: list[dict] = []
        relevance_flags: list[bool] = []

        for i, item in enumerate(ctx.context):
            raw = judge.evaluate(
                _PRECISION_PROMPT.format(
                    question=ctx.question, context_item=item,
                )
            )
            cleaned = raw.strip().upper()
            is_relevant = "RELEVANT" in cleaned and "IRRELEVANT" not in cleaned
            relevance_flags.append(is_relevant)
            judgments.append({
                "index": i,
                "context_preview": item[:80],
                "verdict": "relevant" if is_relevant else "irrelevant",
            })

        ap = _average_precision(relevance_flags)
        score = round(min(max(ap, 0.0), 1.0), 4)
        decision = self.threshold.decide(score)

        relevant_count = sum(relevance_flags)
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.85),
            decision=decision,
            threshold=self.threshold,
            reasoning=(
                f"{relevant_count}/{len(ctx.context)} context items relevant, "
                f"average precision={score} (strategy: llm_judge)"
            ),
            claims=judgments,
            detail={
                "strategy": "llm_judge",
                "context_count": len(ctx.context),
                "relevant_count": relevant_count,
                "average_precision": score,
            },
        )

    def _no_context_result(self) -> EvalResult:
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=0.0, confidence=0.5),
            decision=self.threshold.decide(0.0),
            threshold=self.threshold,
            reasoning="No context provided to evaluate precision",
            detail={"strategy": "none", "context_count": 0},
        )
