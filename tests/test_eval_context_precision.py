"""Tests for context precision evaluation (RAGAS-style retrieval quality)."""

from __future__ import annotations

from llm_output_validator.evals import EvalContext, EvalDecision, ThresholdConfig
from llm_output_validator.evals.context_precision import (
    ContextPrecisionEval,
    _average_precision,
    _relevance_score,
    _tokenize,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _ctx(
    question: str = "What is the corporate tax rate in Germany?",
    answer: str = "The corporate tax rate is 15%.",
    context: list[str] | None = None,
) -> EvalContext:
    return EvalContext(question=question, answer=answer, context=context or [])


class StubJudge:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def evaluate(self, prompt: str) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        result = self._responses[idx]
        self._call_count += 1
        return result


# ── Unit tests: tokenization ─────────────────────────────────────────────────


class TestTokenize:
    def test_removes_stopwords(self) -> None:
        tokens = _tokenize("the tax rate is very high")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "tax" in tokens
        assert "rate" in tokens

    def test_lowercases(self) -> None:
        tokens = _tokenize("Corporate TAX Rate")
        assert "corporate" in tokens
        assert "tax" in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


# ── Unit tests: relevance score ──────────────────────────────────────────────


class TestRelevanceScore:
    def test_identical_texts(self) -> None:
        score = _relevance_score("corporate tax rate", "corporate tax rate")
        assert abs(score - 1.0) < 1e-9

    def test_no_overlap(self) -> None:
        score = _relevance_score("corporate tax rate", "weather forecast tomorrow")
        assert score == 0.0

    def test_partial_overlap(self) -> None:
        score = _relevance_score(
            "corporate tax rate Germany",
            "The corporate tax rate in Germany is 15% at the federal level",
        )
        assert 0.0 < score < 1.0

    def test_empty_question(self) -> None:
        assert _relevance_score("", "some context") == 0.0

    def test_empty_context(self) -> None:
        assert _relevance_score("some question", "") == 0.0


# ── Unit tests: average precision ────────────────────────────────────────────


class TestAveragePrecision:
    def test_all_relevant(self) -> None:
        assert _average_precision([True, True, True]) == 1.0

    def test_none_relevant(self) -> None:
        assert _average_precision([False, False, False]) == 0.0

    def test_empty(self) -> None:
        assert _average_precision([]) == 0.0

    def test_relevant_first(self) -> None:
        ap = _average_precision([True, False, False])
        assert ap == 1.0

    def test_relevant_last(self) -> None:
        ap = _average_precision([False, False, True])
        assert abs(ap - 1 / 3) < 0.01

    def test_mixed_order(self) -> None:
        ap = _average_precision([True, False, True])
        assert 0.0 < ap < 1.0

    def test_rewards_early_relevance(self) -> None:
        ap_early = _average_precision([True, True, False, False])
        ap_late = _average_precision([False, False, True, True])
        assert ap_early > ap_late


# ── Integration tests: lexical strategy ──────────────────────────────────────


class TestContextPrecisionLexical:
    def test_all_relevant_context(self) -> None:
        ctx = _ctx(context=[
            "The corporate tax rate in Germany is 15% at the federal level.",
            "Germany applies a solidarity surcharge of 5.5% on corporate tax.",
        ])
        result = ContextPrecisionEval().evaluate(ctx)
        assert result.score.value > 0.5
        assert result.eval_name == "context_precision"
        assert result.detail["strategy"] == "lexical"

    def test_all_irrelevant_context(self) -> None:
        ctx = _ctx(context=[
            "The weather in Paris is sunny today.",
            "Python 3.12 introduces new syntax features.",
            "The recipe calls for two cups of flour.",
        ])
        result = ContextPrecisionEval().evaluate(ctx)
        assert result.score.value < 0.5
        assert result.decision in (EvalDecision.FLAG, EvalDecision.BLOCK)

    def test_relevant_first_scores_higher(self) -> None:
        relevant = "The corporate tax rate in Germany is 15% at the federal level."
        irrelevant = "The weather in Paris is sunny today."
        ctx_good = _ctx(context=[relevant, irrelevant])
        ctx_bad = _ctx(context=[irrelevant, relevant])
        good_result = ContextPrecisionEval().evaluate(ctx_good)
        bad_result = ContextPrecisionEval().evaluate(ctx_bad)
        assert good_result.score.value >= bad_result.score.value

    def test_no_context(self) -> None:
        ctx = _ctx(context=[])
        result = ContextPrecisionEval().evaluate(ctx)
        assert result.score.value == 0.0
        assert result.decision in (EvalDecision.FLAG, EvalDecision.BLOCK)

    def test_single_relevant_context(self) -> None:
        ctx = _ctx(context=[
            "The corporate tax rate in Germany is 15% at the federal level.",
        ])
        result = ContextPrecisionEval().evaluate(ctx)
        assert result.score.value == 1.0
        assert result.decision == EvalDecision.PASS

    def test_custom_threshold(self) -> None:
        threshold = ThresholdConfig(pass_above=0.95, block_below=0.1)
        ctx = _ctx(context=[
            "The corporate tax rate in Germany is 15%.",
            "Unrelated noise about cooking.",
        ])
        result = ContextPrecisionEval(threshold=threshold).evaluate(ctx)
        assert result.threshold.pass_above == 0.95

    def test_claims_contain_context_previews(self) -> None:
        ctx = _ctx(context=[
            "Germany tax rate information.",
            "Something unrelated entirely.",
        ])
        result = ContextPrecisionEval().evaluate(ctx)
        assert len(result.claims) == 2
        assert all("context_preview" in c for c in result.claims)
        assert all("verdict" in c for c in result.claims)

    def test_detail_contains_counts(self) -> None:
        ctx = _ctx(context=[
            "Corporate tax rate Germany 15%.",
            "Weather forecast for today.",
        ])
        result = ContextPrecisionEval().evaluate(ctx)
        assert "context_count" in result.detail
        assert "relevant_count" in result.detail
        assert result.detail["context_count"] == 2

    def test_custom_relevance_threshold(self) -> None:
        ctx = _ctx(context=[
            "Germany has a federal corporate tax.",
        ])
        strict = ContextPrecisionEval(relevance_threshold=0.9).evaluate(ctx)
        lenient = ContextPrecisionEval(relevance_threshold=0.01).evaluate(ctx)
        assert lenient.score.value >= strict.score.value


# ── Integration tests: LLM judge strategy ────────────────────────────────────


class TestContextPrecisionJudge:
    def test_all_relevant(self) -> None:
        judge = StubJudge(["RELEVANT", "RELEVANT"])
        ctx = _ctx(context=[
            "Germany corporate tax is 15%.",
            "Solidarity surcharge applies at 5.5%.",
        ])
        result = ContextPrecisionEval().evaluate(ctx, judge=judge)
        assert result.score.value == 1.0
        assert result.decision == EvalDecision.PASS
        assert result.detail["strategy"] == "llm_judge"

    def test_all_irrelevant(self) -> None:
        judge = StubJudge(["IRRELEVANT", "IRRELEVANT"])
        ctx = _ctx(context=[
            "Weather in Paris.",
            "Recipe for soup.",
        ])
        result = ContextPrecisionEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.0
        assert result.decision == EvalDecision.BLOCK

    def test_mixed_with_good_ordering(self) -> None:
        judge = StubJudge(["RELEVANT", "IRRELEVANT"])
        ctx = _ctx(context=["Tax info.", "Noise."])
        result = ContextPrecisionEval().evaluate(ctx, judge=judge)
        assert result.score.value == 1.0

    def test_mixed_with_bad_ordering(self) -> None:
        judge = StubJudge(["IRRELEVANT", "RELEVANT"])
        ctx = _ctx(context=["Noise.", "Tax info."])
        result = ContextPrecisionEval().evaluate(ctx, judge=judge)
        assert result.score.value < 1.0

    def test_no_context_with_judge(self) -> None:
        judge = StubJudge([])
        ctx = _ctx(context=[])
        result = ContextPrecisionEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.0


# ── Serialization ────────────────────────────────────────────────────────────


class TestContextPrecisionSerialization:
    def test_to_dict(self) -> None:
        ctx = _ctx(context=[
            "Corporate tax rate Germany is 15%.",
            "Unrelated noise text here.",
        ])
        result = ContextPrecisionEval().evaluate(ctx)
        d = result.to_dict()
        assert "score" in d
        assert "decision" in d
        assert d["eval_name"] == "context_precision"
        assert isinstance(d["score"], float)
