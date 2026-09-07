"""Tests for the evaluation pipeline orchestrator."""

from __future__ import annotations

from llm_output_validator.evals import (
    BaseEval,
    EvalContext,
    EvalDecision,
    EvalResult,
    EvalScore,
    LLMJudge,
    ThresholdConfig,
)
from llm_output_validator.evals.faithfulness import FaithfulnessEval
from llm_output_validator.evals.hallucination import HallucinationEval
from llm_output_validator.evals.pipeline import EvalPipeline, EvalReport
from llm_output_validator.evals.relevancy import AnswerRelevancyEval


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ctx(
    question: str = "What is the tax rate?",
    answer: str = "The rate is 15%.",
    context: list[str] | None = None,
) -> EvalContext:
    return EvalContext(
        question=question,
        answer=answer,
        context=context or ["The corporate tax rate is 15%."],
    )


class AlwaysPassEval(BaseEval):
    eval_name = "always_pass"

    def evaluate(self, ctx: EvalContext, *, judge: LLMJudge | None = None) -> EvalResult:
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=1.0),
            decision=EvalDecision.PASS,
            threshold=self.threshold,
            reasoning="Always passes",
        )


class AlwaysBlockEval(BaseEval):
    eval_name = "always_block"

    def evaluate(self, ctx: EvalContext, *, judge: LLMJudge | None = None) -> EvalResult:
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=0.0),
            decision=EvalDecision.BLOCK,
            threshold=self.threshold,
            reasoning="Always blocks",
        )


class AlwaysFlagEval(BaseEval):
    eval_name = "always_flag"

    def evaluate(self, ctx: EvalContext, *, judge: LLMJudge | None = None) -> EvalResult:
        return EvalResult(
            eval_name=self.eval_name,
            score=EvalScore(value=0.5),
            decision=EvalDecision.FLAG,
            threshold=self.threshold,
            reasoning="Always flags",
        )


# ── Pipeline construction ───────────────────────────────────────────────────


class TestPipelineConstruction:
    def test_empty_pipeline(self) -> None:
        pipeline = EvalPipeline()
        report = pipeline.run(_ctx())
        assert report.decision == EvalDecision.PASS
        assert len(report.results) == 0

    def test_add_returns_self(self) -> None:
        pipeline = EvalPipeline()
        result = pipeline.add(AlwaysPassEval())
        assert result is pipeline

    def test_chained_construction(self) -> None:
        pipeline = (
            EvalPipeline()
            .add(AlwaysPassEval())
            .add(AlwaysFlagEval())
        )
        report = pipeline.run(_ctx())
        assert len(report.results) == 2

    def test_list_construction(self) -> None:
        pipeline = EvalPipeline([AlwaysPassEval(), AlwaysPassEval()])
        report = pipeline.run(_ctx())
        assert len(report.results) == 2


# ── Decision aggregation ────────────────────────────────────────────────────


class TestDecisionAggregation:
    def test_all_pass(self) -> None:
        pipeline = EvalPipeline([AlwaysPassEval(), AlwaysPassEval()])
        report = pipeline.run(_ctx())
        assert report.decision == EvalDecision.PASS
        assert report.passed()

    def test_any_block_overrides(self) -> None:
        pipeline = EvalPipeline([AlwaysPassEval(), AlwaysBlockEval(), AlwaysPassEval()])
        report = pipeline.run(_ctx())
        assert report.decision == EvalDecision.BLOCK
        assert report.blocked()

    def test_flag_without_block(self) -> None:
        pipeline = EvalPipeline([AlwaysPassEval(), AlwaysFlagEval()])
        report = pipeline.run(_ctx())
        assert report.decision == EvalDecision.FLAG
        assert not report.passed()
        assert not report.blocked()

    def test_block_overrides_flag(self) -> None:
        pipeline = EvalPipeline([AlwaysFlagEval(), AlwaysBlockEval()])
        report = pipeline.run(_ctx())
        assert report.decision == EvalDecision.BLOCK

    def test_flagged_evals(self) -> None:
        pipeline = EvalPipeline([AlwaysPassEval(), AlwaysFlagEval()])
        report = pipeline.run(_ctx())
        flagged = report.flagged_evals()
        assert len(flagged) == 1
        assert flagged[0].eval_name == "always_flag"

    def test_blocked_evals(self) -> None:
        pipeline = EvalPipeline([AlwaysPassEval(), AlwaysBlockEval()])
        report = pipeline.run(_ctx())
        blocked = report.blocked_evals()
        assert len(blocked) == 1
        assert blocked[0].eval_name == "always_block"


# ── Fail-fast mode ──────────────────────────────────────────────────────────


class TestFailFast:
    def test_stops_on_block(self) -> None:
        pipeline = EvalPipeline(
            [AlwaysBlockEval(), AlwaysPassEval()],
            fail_fast=True,
        )
        report = pipeline.run(_ctx())
        assert report.decision == EvalDecision.BLOCK
        assert len(report.results) == 1

    def test_continues_on_flag(self) -> None:
        pipeline = EvalPipeline(
            [AlwaysFlagEval(), AlwaysPassEval()],
            fail_fast=True,
        )
        report = pipeline.run(_ctx())
        assert len(report.results) == 2

    def test_all_run_without_fail_fast(self) -> None:
        pipeline = EvalPipeline(
            [AlwaysBlockEval(), AlwaysPassEval(), AlwaysFlagEval()],
            fail_fast=False,
        )
        report = pipeline.run(_ctx())
        assert len(report.results) == 3


# ── Real eval composition ───────────────────────────────────────────────────


class TestRealEvalComposition:
    def test_full_pipeline_grounded_answer(self) -> None:
        pipeline = EvalPipeline([
            FaithfulnessEval(),
            AnswerRelevancyEval(),
            HallucinationEval(),
        ])
        ctx = _ctx(
            question="What is the corporate tax rate in Germany?",
            answer="The corporate tax rate in Germany is 15% at the federal level.",
            context=[
                "The corporate income tax rate in Germany is 15% at the federal level. "
                "A solidarity surcharge of 5.5% is also applied."
            ],
        )
        report = pipeline.run(ctx)
        assert report.decision in (EvalDecision.PASS, EvalDecision.FLAG)
        assert len(report.results) == 3
        for r in report.results:
            assert r.score.value > 0.3

    def test_full_pipeline_hallucinated_answer(self) -> None:
        pipeline = EvalPipeline([
            FaithfulnessEval(),
            AnswerRelevancyEval(),
            HallucinationEval(),
        ])
        ctx = _ctx(
            question="What is the corporate tax rate in Germany?",
            answer="Quantum tunneling enables faster photon decay in nebulae.",
            context=["The corporate income tax rate in Germany is 15%."],
        )
        report = pipeline.run(ctx)
        assert report.decision in (EvalDecision.FLAG, EvalDecision.BLOCK)

    def test_pipeline_with_custom_thresholds(self) -> None:
        strict = ThresholdConfig(pass_above=0.95, block_below=0.5)
        pipeline = EvalPipeline([
            FaithfulnessEval(threshold=strict),
            HallucinationEval(threshold=strict),
        ])
        ctx = _ctx(
            question="What is the rate?",
            answer="The rate is 15%.",
            context=["The rate is 15%."],
        )
        report = pipeline.run(ctx)
        assert all(r.threshold.pass_above == 0.95 for r in report.results)


# ── Report serialization ────────────────────────────────────────────────────


class TestReportSerialization:
    def test_to_dict(self) -> None:
        pipeline = EvalPipeline([AlwaysPassEval(), AlwaysFlagEval()])
        report = pipeline.run(_ctx())
        d = report.to_dict()
        assert "decision" in d
        assert "elapsed_ms" in d
        assert "eval_count" in d
        assert d["eval_count"] == 2
        assert len(d["evals"]) == 2

    def test_elapsed_ms(self) -> None:
        pipeline = EvalPipeline([AlwaysPassEval()])
        report = pipeline.run(_ctx())
        assert report.elapsed_ms >= 0
