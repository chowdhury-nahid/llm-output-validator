"""Tests for the answer relevancy evaluation."""

from __future__ import annotations

import pytest

from llm_output_validator.evals import EvalContext, EvalDecision, ThresholdConfig
from llm_output_validator.evals.relevancy import (
    AnswerRelevancyEval,
    _cosine_similarity,
    _question_term_coverage,
    _tokenize,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _ctx(question: str, answer: str) -> EvalContext:
    return EvalContext(question=question, answer=answer)


class StubJudge:
    def __init__(self, response: str) -> None:
        self._response = response

    def evaluate(self, prompt: str) -> str:
        return self._response


# ── Unit tests: tokenization ────────────────────────────────────────────────


class TestTokenization:
    def test_removes_stopwords(self) -> None:
        tokens = _tokenize("What is the tax rate for Germany?")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "tax" in tokens
        assert "rate" in tokens
        assert "germany" in tokens

    def test_lowercases(self) -> None:
        tokens = _tokenize("GERMANY Tax RATE")
        assert all(t == t.lower() for t in tokens)

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


# ── Unit tests: similarity ───────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical(self) -> None:
        tokens = ["tax", "rate", "germany"]
        assert _cosine_similarity(tokens, tokens) == pytest.approx(1.0)

    def test_disjoint(self) -> None:
        assert _cosine_similarity(["alpha", "beta"], ["gamma", "delta"]) == 0.0

    def test_empty(self) -> None:
        assert _cosine_similarity([], ["tax"]) == 0.0
        assert _cosine_similarity(["tax"], []) == 0.0

    def test_partial(self) -> None:
        sim = _cosine_similarity(["tax", "rate"], ["tax", "law"])
        assert 0.0 < sim < 1.0


class TestQuestionTermCoverage:
    def test_full_coverage(self) -> None:
        q = ["tax", "rate", "germany"]
        a = ["tax", "rate", "germany", "corporate", "federal"]
        assert _question_term_coverage(q, a) == pytest.approx(1.0)

    def test_no_coverage(self) -> None:
        q = ["tax", "rate"]
        a = ["moon", "cheese"]
        assert _question_term_coverage(q, a) == 0.0

    def test_partial_coverage(self) -> None:
        q = ["tax", "rate", "germany"]
        a = ["tax", "law", "germany"]
        assert _question_term_coverage(q, a) == pytest.approx(2 / 3)

    def test_empty_question(self) -> None:
        assert _question_term_coverage([], ["tax"]) == 0.0


# ── Integration tests: lexical strategy ──────────────────────────────────────


class TestRelevancyLexical:
    def test_highly_relevant(self) -> None:
        ctx = _ctx(
            question="What is the corporate tax rate in Germany?",
            answer="The corporate tax rate in Germany is 15% at the federal level.",
        )
        result = AnswerRelevancyEval().evaluate(ctx)
        assert result.score.value > 0.5
        assert result.eval_name == "answer_relevancy"
        assert result.detail["strategy"] == "lexical"

    def test_irrelevant(self) -> None:
        ctx = _ctx(
            question="What is the corporate tax rate in Germany?",
            answer="Photosynthesis converts sunlight into chemical energy in plants.",
        )
        result = AnswerRelevancyEval().evaluate(ctx)
        assert result.score.value < 0.3

    def test_partial_relevance(self) -> None:
        ctx = _ctx(
            question="What is the corporate tax rate in Germany?",
            answer="Germany has a complex tax system with many brackets.",
        )
        result = AnswerRelevancyEval().evaluate(ctx)
        assert 0.1 < result.score.value < 0.9

    def test_empty_answer(self) -> None:
        ctx = _ctx(question="What is the rate?", answer="")
        result = AnswerRelevancyEval().evaluate(ctx)
        assert result.score.value == 0.0

    def test_trivial_question(self) -> None:
        ctx = _ctx(question="the", answer="The rate is 15%.")
        result = AnswerRelevancyEval().evaluate(ctx)
        assert result.decision == EvalDecision.FLAG

    def test_custom_weights(self) -> None:
        ctx = _ctx(
            question="What is the tax rate?",
            answer="The tax rate is 15%.",
        )
        eval_heavy_sim = AnswerRelevancyEval(similarity_weight=1.0, coverage_weight=0.0)
        eval_heavy_cov = AnswerRelevancyEval(similarity_weight=0.0, coverage_weight=1.0)
        r1 = eval_heavy_sim.evaluate(ctx)
        r2 = eval_heavy_cov.evaluate(ctx)
        assert r1.detail["cosine_similarity"] == r1.score.value
        assert r2.detail["question_term_coverage"] == r2.score.value

    def test_detail_fields(self) -> None:
        ctx = _ctx(
            question="What is the rate?",
            answer="The rate is 15%.",
        )
        result = AnswerRelevancyEval().evaluate(ctx)
        assert "cosine_similarity" in result.detail
        assert "question_term_coverage" in result.detail


# ── Integration tests: LLM judge strategy ───────────────────────────────────


class TestRelevancyJudge:
    def test_high_score(self) -> None:
        judge = StubJudge('{"score": 0.95, "reasoning": "Directly answers the question."}')
        ctx = _ctx(question="What is the rate?", answer="The rate is 15%.")
        result = AnswerRelevancyEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.95
        assert result.decision == EvalDecision.PASS
        assert result.detail["strategy"] == "llm_judge"

    def test_low_score(self) -> None:
        judge = StubJudge('{"score": 0.1, "reasoning": "Off-topic response."}')
        ctx = _ctx(question="What is the rate?", answer="Unrelated text.")
        result = AnswerRelevancyEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.1
        assert result.decision == EvalDecision.BLOCK

    def test_malformed_judge_response(self) -> None:
        judge = StubJudge("I think the score is about 0.7")
        ctx = _ctx(question="Q?", answer="A.")
        result = AnswerRelevancyEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.7

    def test_unparseable_judge_response(self) -> None:
        judge = StubJudge("no numbers here at all")
        ctx = _ctx(question="Q?", answer="A.")
        result = AnswerRelevancyEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.0

    def test_score_clamped(self) -> None:
        judge = StubJudge('{"score": 5.0, "reasoning": "Amazing."}')
        ctx = _ctx(question="Q?", answer="A.")
        result = AnswerRelevancyEval().evaluate(ctx, judge=judge)
        assert result.score.value == 1.0


# ── Threshold tests ──────────────────────────────────────────────────────────


class TestRelevancyThresholds:
    def test_strict_threshold(self) -> None:
        threshold = ThresholdConfig(pass_above=0.95, block_below=0.5)
        ctx = _ctx(
            question="What is the corporate tax rate in Germany?",
            answer="The corporate tax rate in Germany is 15%.",
        )
        result = AnswerRelevancyEval(threshold=threshold).evaluate(ctx)
        assert result.threshold.pass_above == 0.95

    def test_lenient_threshold(self) -> None:
        threshold = ThresholdConfig(pass_above=0.2, block_below=0.05)
        ctx = _ctx(
            question="What is the tax rate?",
            answer="The tax rate varies by jurisdiction.",
        )
        result = AnswerRelevancyEval(threshold=threshold).evaluate(ctx)
        assert result.decision == EvalDecision.PASS
