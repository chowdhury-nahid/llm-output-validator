"""Tests for semantic hallucination detection evaluation."""

from __future__ import annotations

import json

import pytest

from llm_output_validator.evals import EvalContext, EvalDecision, ThresholdConfig
from llm_output_validator.evals.hallucination import (
    HallucinationEval,
    _entity_grounding,
    _extract_entities,
    _sentence_grounding_score,
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
    def __init__(self, response: str) -> None:
        self._response = response

    def evaluate(self, prompt: str) -> str:
        return self._response


# ── Unit tests: entity extraction ────────────────────────────────────────────


class TestEntityExtraction:
    def test_percentages(self) -> None:
        entities = _extract_entities("The rate is 19% for Germany and 15% federal.")
        assert "19%" in entities
        assert "15%" in entities

    def test_numbers(self) -> None:
        entities = _extract_entities("There are 5 brackets starting at 10000.")
        assert "5" in entities
        assert "10000" in entities

    def test_currency(self) -> None:
        entities = _extract_entities("The threshold is $50,000 or €45,000.")
        assert "$50,000" in entities
        assert "€45,000" in entities

    def test_dates(self) -> None:
        entities = _extract_entities("Effective from 01/01/2024 to 12/31/2024.")
        assert "01/01/2024" in entities
        assert "12/31/2024" in entities

    def test_no_entities(self) -> None:
        entities = _extract_entities("Germany has a complex tax system.")
        assert len(entities) == 0


# ── Unit tests: sentence grounding ───────────────────────────────────────────


class TestSentenceGrounding:
    def test_fully_grounded(self) -> None:
        score = _sentence_grounding_score(
            "The corporate tax rate in Germany is 15 percent.",
            "The corporate tax rate in Germany is 15 percent at the federal level.",
        )
        assert score > 0.7

    def test_ungrounded(self) -> None:
        score = _sentence_grounding_score(
            "Quantum computing revolutionizes cryptography.",
            "The corporate tax rate in Germany is 15 percent.",
        )
        assert score < 0.3

    def test_empty_sentence(self) -> None:
        assert _sentence_grounding_score("", "Some context.") == 1.0


# ── Unit tests: entity grounding ─────────────────────────────────────────────


class TestEntityGrounding:
    def test_all_grounded(self) -> None:
        grounded, ungrounded = _entity_grounding(
            "The rate is 15% since 2020.",
            "Germany's 15% rate has been effective since 2020.",
        )
        assert "15%" in grounded
        assert "2020" in grounded
        assert len(ungrounded) == 0

    def test_fabricated_number(self) -> None:
        grounded, ungrounded = _entity_grounding(
            "The rate is 42% since 2020.",
            "Germany's 15% rate has been effective since 2020.",
        )
        assert "42%" in ungrounded
        assert "2020" in grounded

    def test_no_entities_in_either(self) -> None:
        grounded, ungrounded = _entity_grounding(
            "Germany has taxes.",
            "Germany has a tax system.",
        )
        assert len(grounded) == 0
        assert len(ungrounded) == 0


# ── Integration tests: lexical strategy ──────────────────────────────────────


class TestHallucinationLexical:
    def test_well_grounded_answer(self) -> None:
        ctx = _ctx(
            answer="The corporate tax rate in Germany is 15% at the federal level. "
                   "A solidarity surcharge of 5.5% also applies.",
            context=[
                "The corporate income tax rate in Germany is 15% at the federal level. "
                "A solidarity surcharge of 5.5% is levied on top of the corporate tax."
            ],
        )
        result = HallucinationEval().evaluate(ctx)
        assert result.score.value > 0.5
        assert result.eval_name == "hallucination_detection"
        assert result.detail["strategy"] == "lexical"

    def test_fabricated_facts(self) -> None:
        ctx = _ctx(
            answer="The corporate tax rate in Jupiter is 99%. "
                   "Mars charges a flat 50% on all interplanetary trade.",
            context=["The corporate tax rate in Germany is 15%."],
        )
        result = HallucinationEval().evaluate(ctx)
        assert result.score.value < 0.8
        assert len(result.detail.get("ungrounded_entities", [])) > 0

    def test_empty_context(self) -> None:
        ctx = _ctx(answer="The rate is 15%.", context=[])
        result = HallucinationEval().evaluate(ctx)
        assert result.decision in (EvalDecision.FLAG, EvalDecision.BLOCK)
        assert "No context" in result.reasoning

    def test_empty_answer(self) -> None:
        ctx = _ctx(answer="OK.", context=["Some context here."])
        result = HallucinationEval().evaluate(ctx)
        assert result.decision == EvalDecision.FLAG

    def test_entity_detail_populated(self) -> None:
        ctx = _ctx(
            answer="The rate is 15% and the threshold is $50,000.",
            context=["Germany's rate is 15%. The threshold is $100,000."],
        )
        result = HallucinationEval().evaluate(ctx)
        detail = result.detail
        assert "grounded_entities" in detail
        assert "ungrounded_entities" in detail
        assert "15%" in detail["grounded_entities"]

    def test_all_entities_fabricated(self) -> None:
        ctx = _ctx(
            answer="The rate is 42% with a $999 surcharge.",
            context=["Germany has a tax system with multiple brackets."],
        )
        result = HallucinationEval().evaluate(ctx)
        assert result.detail["entity_score"] == 0.0

    def test_claims_list(self) -> None:
        ctx = _ctx(
            answer="The rate is 15%. It applies to corporations.",
            context=["The rate is 15% for corporations."],
        )
        result = HallucinationEval().evaluate(ctx)
        assert len(result.claims) > 0
        assert all("text" in c and "verdict" in c for c in result.claims)

    def test_mixed_grounding(self) -> None:
        ctx = _ctx(
            answer="The corporate tax rate in Germany is 15%. "
                   "Unicorns pay a special levy of 200% on magical goods.",
            context=[
                "The corporate income tax rate in Germany is 15% at the federal level."
            ],
        )
        result = HallucinationEval().evaluate(ctx)
        grounded = sum(1 for c in result.claims if c["verdict"] == "grounded")
        hallucinated = sum(1 for c in result.claims if c["verdict"] == "hallucinated")
        assert grounded >= 1
        assert hallucinated >= 1


# ── Integration tests: LLM judge strategy ───────────────────────────────────


class TestHallucinationJudge:
    def test_no_hallucination(self) -> None:
        response = json.dumps({
            "statements": [
                {"text": "Rate is 15%.", "verdict": "GROUNDED", "reason": "Matches context."}
            ],
            "hallucination_rate": 0.0,
        })
        judge = StubJudge(response)
        ctx = _ctx(
            answer="Rate is 15%.",
            context=["The rate is 15%."],
        )
        result = HallucinationEval().evaluate(ctx, judge=judge)
        assert result.score.value == 1.0
        assert result.decision == EvalDecision.PASS
        assert result.detail["strategy"] == "llm_judge"

    def test_full_hallucination(self) -> None:
        response = json.dumps({
            "statements": [
                {"text": "Made up fact.", "verdict": "HALLUCINATED", "reason": "Not in context."}
            ],
            "hallucination_rate": 1.0,
        })
        judge = StubJudge(response)
        ctx = _ctx(
            answer="Made up fact.",
            context=["Real context."],
        )
        result = HallucinationEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.0
        assert result.decision == EvalDecision.BLOCK

    def test_partial_hallucination(self) -> None:
        response = json.dumps({
            "statements": [
                {"text": "Rate is 15%.", "verdict": "GROUNDED", "reason": "OK."},
                {"text": "Enacted in 2099.", "verdict": "HALLUCINATED", "reason": "Wrong."},
            ],
            "hallucination_rate": 0.5,
        })
        judge = StubJudge(response)
        ctx = _ctx(
            answer="Rate is 15%. Enacted in 2099.",
            context=["The rate is 15%, enacted in 2019."],
        )
        result = HallucinationEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.5
        assert result.decision == EvalDecision.FLAG

    def test_malformed_judge_response(self) -> None:
        judge = StubJudge("This is not JSON at all")
        ctx = _ctx(answer="Something.", context=["Context."])
        result = HallucinationEval().evaluate(ctx, judge=judge)
        assert result.score.value == 0.5
        assert result.decision == EvalDecision.FLAG

    def test_empty_context_with_judge(self) -> None:
        judge = StubJudge('{"statements": [], "hallucination_rate": 0.0}')
        ctx = _ctx(answer="Something.", context=[])
        result = HallucinationEval().evaluate(ctx, judge=judge)
        assert "No context" in result.reasoning


# ── Threshold tests ──────────────────────────────────────────────────────────


class TestHallucinationThresholds:
    def test_strict_threshold(self) -> None:
        threshold = ThresholdConfig(pass_above=0.95, block_below=0.5)
        ctx = _ctx(
            answer="The rate is 15%. A special surcharge applies.",
            context=["The rate is 15% for all corporations."],
        )
        result = HallucinationEval(threshold=threshold).evaluate(ctx)
        assert result.threshold.pass_above == 0.95

    def test_custom_weights(self) -> None:
        ctx = _ctx(
            answer="The rate is 42%. It applies everywhere.",
            context=["The rate is 15%. It applies to corporations."],
        )
        r_entity_heavy = HallucinationEval(entity_weight=0.9, sentence_weight=0.1).evaluate(ctx)
        r_sentence_heavy = HallucinationEval(entity_weight=0.1, sentence_weight=0.9).evaluate(ctx)
        assert r_entity_heavy.score.value != r_sentence_heavy.score.value


# ── Serialization ────────────────────────────────────────────────────────────


class TestHallucinationSerialization:
    def test_to_dict(self) -> None:
        ctx = _ctx(
            answer="The rate is 15%.",
            context=["The corporate tax rate is 15%."],
        )
        result = HallucinationEval().evaluate(ctx)
        d = result.to_dict()
        assert d["eval_name"] == "hallucination_detection"
        assert isinstance(d["score"], float)
        assert d["decision"] in ("pass", "flag", "block")
