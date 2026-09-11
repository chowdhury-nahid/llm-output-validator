"""Tests for the faithfulness evaluation (RAGAS-style claim verification)."""

from __future__ import annotations

from llm_output_validator.evals import EvalContext, EvalDecision, ThresholdConfig
from llm_output_validator.evals.faithfulness import (
    FaithfulnessEval,
    _decompose_into_claims,
    _lexical_overlap,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _ctx(
    answer: str,
    context: list[str] | None = None,
    question: str = "What is the tax rate?",
) -> EvalContext:
    return EvalContext(
        question=question,
        answer=answer,
        context=context or [],
    )


class StubJudge:
    """Deterministic LLM judge for testing."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def evaluate(self, prompt: str) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        result = self._responses[idx]
        self._call_count += 1
        return result


# ── Unit tests: claim decomposition ─────────────────────────────────────────


class TestClaimDecomposition:
    def test_splits_sentences(self) -> None:
        text = "The rate is 19%. It applies to all corporations. Effective since 2020."
        claims = _decompose_into_claims(text)
        assert len(claims) == 3

    def test_filters_short_fragments(self) -> None:
        text = "Yes. The rate is 19%."
        claims = _decompose_into_claims(text)
        assert len(claims) == 1
        assert "19%" in claims[0]

    def test_empty_input(self) -> None:
        assert _decompose_into_claims("") == []

    def test_single_long_sentence(self) -> None:
        text = "The corporate tax rate in Germany is 15% at the federal level."
        claims = _decompose_into_claims(text)
        assert len(claims) == 1


# ── Unit tests: lexical overlap ──────────────────────────────────────────────


class TestLexicalOverlap:
    def test_full_overlap(self) -> None:
        assert _lexical_overlap("the rate is high", "the rate is high") == 1.0

    def test_no_overlap(self) -> None:
        assert _lexical_overlap("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_partial_overlap(self) -> None:
        score = _lexical_overlap("the tax rate is 19%", "the tax rate applies here")
        assert 0.0 < score < 1.0

    def test_empty_claim(self) -> None:
        assert _lexical_overlap("", "some context") == 0.0


# ── Integration tests: lexical strategy ──────────────────────────────────────


class TestFaithfulnessLexical:
    def test_fully_grounded_answer(self) -> None:
        ctx = _ctx(
            answer="The corporate tax rate in Germany is 15%. It applies to all corporations.",
            context=["The corporate tax rate in Germany is 15% at the federal level. "
                      "It applies to all corporations registered in Germany."],
        )
        result = FaithfulnessEval().evaluate(ctx)
        assert result.score.value > 0.5
        assert result.eval_name == "faithfulness"
        assert result.detail["strategy"] == "lexical"

    def test_ungrounded_answer(self) -> None:
        ctx = _ctx(
            answer="Quantum entanglement drives photosynthesis. Martian soil contains platinum.",
            context=["The corporate tax rate in Germany is 15%."],
        )
        result = FaithfulnessEval().evaluate(ctx)
        assert result.score.value < 0.5
        assert result.decision in (EvalDecision.FLAG, EvalDecision.BLOCK)

    def test_empty_context_flags(self) -> None:
        ctx = _ctx(answer="The rate is 19%.", context=[])
        result = FaithfulnessEval().evaluate(ctx)
        assert result.decision in (EvalDecision.FLAG, EvalDecision.BLOCK)
        assert "No context" in result.reasoning

    def test_empty_answer(self) -> None:
        ctx = _ctx(answer="OK.", context=["Some context."])
        result = FaithfulnessEval().evaluate(ctx)
        assert result.decision == EvalDecision.FLAG
        assert "No claims" in result.reasoning

    def test_custom_threshold(self) -> None:
        threshold = ThresholdConfig(pass_above=0.9, block_below=0.1)
        ctx = _ctx(
            answer="The rate is 15%. It applies federally.",
            context=["The rate is 15% at the federal level."],
        )
        result = FaithfulnessEval(threshold=threshold).evaluate(ctx)
        assert result.threshold.pass_above == 0.9
        assert result.threshold.block_below == 0.1

    def test_claims_included_in_result(self) -> None:
        ctx = _ctx(
            answer="The rate is 15%. It applies to all companies.",
            context=["The rate is 15% for companies."],
        )
        result = FaithfulnessEval().evaluate(ctx)
        assert len(result.claims) > 0
        assert all("claim" in c and "verdict" in c for c in result.claims)

    def test_mixed_grounding(self) -> None:
        ctx = _ctx(
            answer="The tax rate is 19%. Unicorns pay double tax.",
            context=["Germany's VAT rate is 19%."],
        )
        result = FaithfulnessEval().evaluate(ctx)
        supported = sum(1 for c in result.claims if c["verdict"] == "supported")
        unsupported = sum(1 for c in result.claims if c["verdict"] == "unsupported")
        assert supported >= 1
        assert unsupported >= 1
        assert 0.0 < result.score.value < 1.0


# ── Integration tests: LLM judge strategy ───────────────────────────────────


class TestFaithfulnessJudge:
    def test_all_supported(self) -> None:
        judge = StubJudge([
            '["The rate is 15%.", "It applies to corporations."]',
            "SUPPORTED",
            "SUPPORTED",
        ])
        ctx = _ctx(
            answer="The rate is 15%. It applies to corporations.",
            context=["The corporate rate is 15% for all corporations."],
        )
        result = FaithfulnessEval().evaluate(ctx, judge=judge)
        assert result.score.value == 1.0
        assert result.decision == EvalDecision.PASS
        assert result.detail["strategy"] == "llm_judge"

    def test_one_contradicted(self) -> None:
        judge = StubJudge([
            '["The rate is 15%.", "It was enacted in 2024."]',
            "SUPPORTED",
            "CONTRADICTED",
        ])
        ctx = _ctx(
            answer="The rate is 15%. It was enacted in 2024.",
            context=["The rate is 15%, enacted in 2019."],
        )
        result = FaithfulnessEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.5
        assert result.decision == EvalDecision.FLAG

    def test_all_unsupported(self) -> None:
        judge = StubJudge([
            '["Claim A.", "Claim B."]',
            "NOT_MENTIONED",
            "NOT_MENTIONED",
        ])
        ctx = _ctx(
            answer="Claim A. Claim B.",
            context=["Unrelated context."],
        )
        result = FaithfulnessEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.0
        assert result.decision == EvalDecision.BLOCK

    def test_empty_context_with_judge(self) -> None:
        judge = StubJudge(['["Some claim."]'])
        ctx = _ctx(answer="Some claim.", context=[])
        result = FaithfulnessEval().evaluate(ctx, judge=judge)
        assert "No context" in result.reasoning


# ── Serialization ────────────────────────────────────────────────────────────


class TestEvalResultSerialization:
    def test_to_dict(self) -> None:
        ctx = _ctx(
            answer="The rate is 15%.",
            context=["The corporate tax rate is 15%."],
        )
        result = FaithfulnessEval().evaluate(ctx)
        d = result.to_dict()
        assert "score" in d
        assert "decision" in d
        assert "threshold" in d
        assert isinstance(d["score"], float)
        assert d["eval_name"] == "faithfulness"
