from __future__ import annotations

import json
import math
import re
from collections import Counter

from .base import BaseEval
from .models import EvalContext, EvalDecision, EvalResult, EvalScore, LLMJudge, ThresholdConfig

_RELEVANCY_PROMPT = """Score how relevant the answer is to the question on a scale of 0.0 to 1.0.

Question: {question}
Answer: {answer}

Consider:
1. Does the answer address what was asked?
2. Is the information directly useful for answering the question?
3. Is the answer focused (not padded with irrelevant information)?

Return ONLY a JSON object: {{"score": <float>, "reasoning": "<one sentence>"}}"""

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about", "up",
    "it", "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "what", "which", "who", "whom",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _cosine_similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    counter_a = Counter(a)
    counter_b = Counter(b)
    terms = set(counter_a) | set(counter_b)
    dot = sum(counter_a.get(t, 0) * counter_b.get(t, 0) for t in terms)
    mag_a = math.sqrt(sum(v * v for v in counter_a.values()))
    mag_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _question_term_coverage(question_tokens: list[str], answer_tokens: list[str]) -> float:
    if not question_tokens:
        return 0.0
    answer_set = set(answer_tokens)
    covered = sum(1 for t in question_tokens if t in answer_set)
    return covered / len(question_tokens)


class AnswerRelevancyEval(BaseEval):
    """Measures how relevant the answer is to the question asked.

    Implements the core concept from RAGAS answer_relevancy: an answer is
    relevant when it directly addresses the question without excessive
    padding or topic drift.

    Two strategies:
    - **lexical** (default): combines TF-IDF cosine similarity with
      question-term coverage. Fast, deterministic, good for CI.
    - **llm_judge**: asks an LLM to score relevancy directly. More
      nuanced but requires API calls.
    """

    eval_name = "answer_relevancy"

    def __init__(
        self,
        threshold: ThresholdConfig | None = None,
        similarity_weight: float = 0.6,
        coverage_weight: float = 0.4,
    ) -> None:
        super().__init__(threshold)
        self._sim_weight = similarity_weight
        self._cov_weight = coverage_weight

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
        q_tokens = _tokenize(ctx.question)
        a_tokens = _tokenize(ctx.answer)

        if not q_tokens:
            return EvalResult(
                eval_name=self.eval_name,
                score=EvalScore(value=0.0, confidence=0.3),
                decision=EvalDecision.FLAG,
                threshold=self.threshold,
                reasoning="No meaningful tokens in question",
                detail={"strategy": "lexical"},
            )

        if not a_tokens:
            return EvalResult(
                eval_name=self.eval_name,
                score=EvalScore(value=0.0, confidence=0.7),
                decision=self.threshold.decide(0.0),
                threshold=self.threshold,
                reasoning="Empty or trivial answer",
                detail={"strategy": "lexical"},
            )

        similarity = _cosine_similarity(q_tokens, a_tokens)
        coverage = _question_term_coverage(q_tokens, a_tokens)
        score = round(
            self._sim_weight * similarity + self._cov_weight * coverage,
            4,
        )
        score = min(max(score, 0.0), 1.0)
        decision = self.threshold.decide(score)

        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.6),
            decision=decision,
            threshold=self.threshold,
            reasoning=(
                f"Similarity={similarity:.3f}, coverage={coverage:.3f} "
                f"(weights: {self._sim_weight}/{self._cov_weight})"
            ),
            detail={
                "strategy": "lexical",
                "cosine_similarity": round(similarity, 4),
                "question_term_coverage": round(coverage, 4),
            },
        )

    def _evaluate_with_judge(self, ctx: EvalContext, judge: LLMJudge) -> EvalResult:
        raw = judge.evaluate(
            _RELEVANCY_PROMPT.format(question=ctx.question, answer=ctx.answer)
        )
        score, reasoning = self._parse_judge_response(raw)
        decision = self.threshold.decide(score)

        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=score, confidence=0.85),
            decision=decision,
            threshold=self.threshold,
            reasoning=reasoning,
            detail={"strategy": "llm_judge"},
        )

    def _parse_judge_response(self, raw: str) -> tuple[float, str]:
        try:
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0.0))
            score = min(max(score, 0.0), 1.0)
            reasoning = str(parsed.get("reasoning", ""))
            return round(score, 4), reasoning
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        match = re.search(r"(\d+\.?\d*)", raw)
        if match:
            score = min(max(float(match.group(1)), 0.0), 1.0)
            return round(score, 4), raw.strip()
        return 0.0, f"Could not parse judge response: {raw[:100]}"
