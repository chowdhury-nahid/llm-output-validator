"""Tests for context recall evaluation (RAGAS-style retrieval completeness)."""

from __future__ import annotations

from llm_output_validator.evals import EvalContext, EvalDecision, ThresholdConfig
from llm_output_validator.evals.context_recall import (
    ContextRecallEval,
    _decompose_answer,
    _lexical_attribution,
    _tokenize,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _ctx(
    answer: str,
    context: list[str] | None = None,
    question: str = "What is the corporate tax rate in Germany?",
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


# ── Unit tests: answer decomposition ─────────────────────────────────────────


class TestAnswerDecomposition:
    def test_splits_sentences(self) -> None:
        text = "The rate is 15%. A surcharge applies. It is effective since 2020."
        claims = _decompose_answer(text)
        assert len(claims) == 3

    def test_filters_short_fragments(self) -> None:
        text = "Yes. The rate is 15%."
        claims = _decompose_answer(text)
        assert len(claims) == 1
        assert "15%" in claims[0]

    def test_empty_input(self) -> None:
        assert _decompose_answer("") == []

    def test_single_sentence(self) -> None:
        text = "The corporate tax rate in Germany is 15% at the federal level."
        claims = _decompose_answer(text)
        assert len(claims) == 1


# ── Unit tests: tokenization ─────────────────────────────────────────────────


class TestTokenize:
    def test_removes_stopwords(self) -> None:
        tokens = _tokenize("the tax rate is very high")
        assert "the" not in tokens
        assert "tax" in tokens

    def test_returns_set(self) -> None:
        tokens = _tokenize("tax tax tax")
        assert isinstance(tokens, set)
        assert "tax" in tokens


# ── Unit tests: lexical attribution ──────────────────────────────────────────


class TestLexicalAttribution:
    def test_full_attribution(self) -> None:
        score = _lexical_attribution(
            "The corporate tax rate is 15%",
            "The corporate tax rate is 15% at the federal level",
        )
        assert score > 0.5

    def test_no_attribution(self) -> None:
        score = _lexical_attribution(
            "Quantum entanglement drives photosynthesis",
            "The corporate tax rate in Germany is 15%",
        )
        assert score == 0.0

    def test_partial_attribution(self) -> None:
        score = _lexical_attribution(
            "The tax rate is 15% with a solidarity surcharge",
            "The tax rate is 15% at the federal level",
        )
        assert 0.0 < score < 1.0

    def test_empty_claim(self) -> None:
        assert _lexical_attribution("", "some context") == 0.0


# ── Integration tests: lexical strategy ──────────────────────────────────────


class TestContextRecallLexical:
    def test_fully_attributable(self) -> None:
        ctx = _ctx(
            answer=(
                "The corporate tax rate in Germany is 15%. "
                "A solidarity surcharge of 5.5% applies."
            ),
            context=[
                "The corporate tax rate in Germany is 15% at the federal level. "
                "A solidarity surcharge of 5.5% is levied on corporate tax."
            ],
        )
        result = ContextRecallEval().evaluate(ctx)
        assert result.score.value > 0.5
        assert result.eval_name == "context_recall"
        assert result.detail["strategy"] == "lexical"

    def test_not_attributable(self) -> None:
        ctx = _ctx(
            answer=(
                "Quantum computing will revolutionize tax compliance. "
                "Neural networks process claims."
            ),
            context=["The corporate tax rate in Germany is 15%."],
        )
        result = ContextRecallEval().evaluate(ctx)
        assert result.score.value < 0.5
        assert result.decision in (EvalDecision.FLAG, EvalDecision.BLOCK)

    def test_empty_context(self) -> None:
        ctx = _ctx(answer="The rate is 15%.", context=[])
        result = ContextRecallEval().evaluate(ctx)
        assert result.decision in (EvalDecision.FLAG, EvalDecision.BLOCK)
        assert "No context" in result.reasoning

    def test_empty_answer(self) -> None:
        ctx = _ctx(answer="OK.", context=["Some context."])
        result = ContextRecallEval().evaluate(ctx)
        assert result.decision == EvalDecision.FLAG
        assert "No claims" in result.reasoning

    def test_mixed_attribution(self) -> None:
        ctx = _ctx(
            answer="The tax rate is 15%. Martian colonies pay zero tax.",
            context=["The corporate tax rate in Germany is 15%."],
        )
        result = ContextRecallEval().evaluate(ctx)
        attributed = sum(1 for c in result.claims if c["verdict"] == "attributable")
        not_attributed = sum(1 for c in result.claims if c["verdict"] == "not_attributable")
        assert attributed >= 1
        assert not_attributed >= 1
        assert 0.0 < result.score.value < 1.0

    def test_custom_threshold(self) -> None:
        threshold = ThresholdConfig(pass_above=0.95, block_below=0.1)
        ctx = _ctx(
            answer="The rate is 15%. It applies federally.",
            context=["The rate is 15% at the federal level."],
        )
        result = ContextRecallEval(threshold=threshold).evaluate(ctx)
        assert result.threshold.pass_above == 0.95

    def test_claims_included(self) -> None:
        ctx = _ctx(
            answer="The rate is 15%. It applies to all companies.",
            context=["The rate is 15% for companies."],
        )
        result = ContextRecallEval().evaluate(ctx)
        assert len(result.claims) > 0
        assert all("claim" in c and "verdict" in c for c in result.claims)

    def test_detail_contains_counts(self) -> None:
        ctx = _ctx(
            answer="The rate is 15%. Surcharge applies at 5.5%.",
            context=["The corporate tax rate is 15% with 5.5% surcharge."],
        )
        result = ContextRecallEval().evaluate(ctx)
        assert "claim_count" in result.detail
        assert "attributed_count" in result.detail

    def test_custom_attribution_threshold(self) -> None:
        ctx = _ctx(
            answer="Germany has a federal corporate tax.",
            context=["Germany applies corporate tax at federal level."],
        )
        strict = ContextRecallEval(attribution_threshold=0.9).evaluate(ctx)
        lenient = ContextRecallEval(attribution_threshold=0.01).evaluate(ctx)
        assert lenient.score.value >= strict.score.value

    def test_multiple_context_docs(self) -> None:
        ctx = _ctx(
            answer="The tax rate is 15%. The surcharge is 5.5%.",
            context=[
                "Germany's corporate tax rate is 15%.",
                "A solidarity surcharge of 5.5% is applied.",
            ],
        )
        result = ContextRecallEval().evaluate(ctx)
        assert result.score.value > 0.5


# ── Integration tests: LLM judge strategy ────────────────────────────────────


class TestContextRecallJudge:
    def test_all_attributable(self) -> None:
        judge = StubJudge(["ATTRIBUTABLE", "ATTRIBUTABLE"])
        ctx = _ctx(
            answer="The rate is 15%. Surcharge applies.",
            context=["Tax rate 15% with surcharge."],
        )
        result = ContextRecallEval().evaluate(ctx, judge=judge)
        assert result.score.value == 1.0
        assert result.decision == EvalDecision.PASS
        assert result.detail["strategy"] == "llm_judge"

    def test_none_attributable(self) -> None:
        judge = StubJudge(["NOT_ATTRIBUTABLE", "NOT_ATTRIBUTABLE"])
        ctx = _ctx(
            answer="Claim A is true. Claim B also holds.",
            context=["Unrelated context."],
        )
        result = ContextRecallEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.0
        assert result.decision == EvalDecision.BLOCK

    def test_mixed_attributions(self) -> None:
        judge = StubJudge(["ATTRIBUTABLE", "NOT_ATTRIBUTABLE"])
        ctx = _ctx(
            answer="The rate is 15%. It was enacted in 2099.",
            context=["Rate is 15%."],
        )
        result = ContextRecallEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.5
        assert result.decision == EvalDecision.FLAG

    def test_empty_context_with_judge(self) -> None:
        judge = StubJudge(["ATTRIBUTABLE"])
        ctx = _ctx(answer="Some claim here.", context=[])
        result = ContextRecallEval().evaluate(ctx, judge=judge)
        assert "No context" in result.reasoning

    def test_empty_answer_with_judge(self) -> None:
        judge = StubJudge([])
        ctx = _ctx(answer="OK.", context=["Some context."])
        result = ContextRecallEval().evaluate(ctx, judge=judge)
        assert "No claims" in result.reasoning


# ── Serialization ────────────────────────────────────────────────────────────


class TestContextRecallSerialization:
    def test_to_dict(self) -> None:
        ctx = _ctx(
            answer="The rate is 15%.",
            context=["The corporate tax rate is 15%."],
        )
        result = ContextRecallEval().evaluate(ctx)
        d = result.to_dict()
        assert "score" in d
        assert "decision" in d
        assert d["eval_name"] == "context_recall"
        assert isinstance(d["score"], float)
