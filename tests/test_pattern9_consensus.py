from datetime import date
from pathlib import Path

import pytest

from llm_output_validator import Citation, LLMResponse, OutputValidator, TaxRateResponse
from llm_output_validator.checks.consensus_check import CrossModelConsensusCheck
from llm_output_validator.consensus import ConsensusClaim, ConsensusReference
from llm_output_validator.report import CheckStatus

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "consensus"


def _make_response(jurisdiction: str, rate: float) -> LLMResponse:
    return LLMResponse(
        response=TaxRateResponse(
            jurisdiction=jurisdiction,
            rate=rate,
            effective_date=date(2023, 7, 1),
            source_id="IRS-2023-001",
            confidence="high",
        ),
        citations=[Citation(document_id="IRS-2023-001")],
        raw_prompt="test",
    )


def _make_reference(
    jurisdiction: str,
    claims: dict[str, ConsensusClaim],
) -> ConsensusReference:
    return ConsensusReference(
        question_id=f"test_{jurisdiction.lower()}",
        jurisdiction=jurisdiction,
        sources=[],
        consensus=claims,
    )


def test_consensus_fixtures_load_from_directory():
    refs = ConsensusReference.load_from_dir(FIXTURES_DIR)
    assert len(refs) == 5
    question_ids = {r.question_id for r in refs}
    assert "germany_corporate_tax" in question_ids
    assert "ireland_corporate_tax" in question_ids
    assert "nyc_sales_tax" in question_ids
    assert "uk_corporation_tax" in question_ids
    assert "us_federal_brackets_2025" in question_ids


def test_consensus_pass_matching_rate():
    ref = _make_reference("DE", {
        "rate": ConsensusClaim(consensus_value=0.15, agreement_count=5, total_sources=5),
    })
    check = CrossModelConsensusCheck(references=[ref])
    result = check.run(_make_response("DE", 0.15))
    assert result.status == CheckStatus.PASS
    assert len(result.detail["matched"]) == 1


def test_consensus_fail_critical_divergence():
    ref = _make_reference("DE", {
        "rate": ConsensusClaim(
            consensus_value=0.15, agreement_count=5, total_sources=5, critical=True,
        ),
    })
    check = CrossModelConsensusCheck(references=[ref])
    result = check.run(_make_response("DE", 0.25))
    assert result.status == CheckStatus.FAIL
    assert result.severity == "error"
    assert len(result.detail["diverged"]) == 1
    assert result.detail["diverged"][0]["claim_key"] == "rate"


def test_consensus_warn_noncritical_divergence():
    ref = _make_reference("DE", {
        "rate": ConsensusClaim(
            consensus_value=0.15, agreement_count=5, total_sources=5, critical=True,
        ),
        "confidence": ConsensusClaim(
            consensus_value="medium", agreement_count=3, total_sources=5, critical=False,
        ),
    })
    check = CrossModelConsensusCheck(references=[ref])
    # Rate matches (critical), confidence diverges (non-critical)
    result = check.run(_make_response("DE", 0.15))
    assert result.status == CheckStatus.WARN
    assert result.severity == "warning"


def test_consensus_warn_no_reference():
    check = CrossModelConsensusCheck(references=[])
    result = check.run(_make_response("ZZ", 0.10))
    assert result.status == CheckStatus.WARN
    assert "No consensus reference data" in result.message


def test_consensus_tolerance_pass():
    ref = _make_reference("DE", {
        "rate": ConsensusClaim(
            consensus_value=0.15, agreement_count=5, total_sources=5, tolerance=0.005,
        ),
    })
    check = CrossModelConsensusCheck(references=[ref])
    result = check.run(_make_response("DE", 0.1504))
    assert result.status == CheckStatus.PASS


def test_consensus_tolerance_fail():
    ref = _make_reference("DE", {
        "rate": ConsensusClaim(
            consensus_value=0.15, agreement_count=5, total_sources=5, tolerance=0.001,
        ),
    })
    check = CrossModelConsensusCheck(references=[ref])
    result = check.run(_make_response("DE", 0.16))
    assert result.status == CheckStatus.FAIL


def test_consensus_detail_structure():
    ref = _make_reference("DE", {
        "rate": ConsensusClaim(consensus_value=0.15, agreement_count=5, total_sources=5),
    })
    check = CrossModelConsensusCheck(references=[ref])
    result = check.run(_make_response("DE", 0.15))
    assert "jurisdiction" in result.detail
    assert "question_id" in result.detail
    assert "matched" in result.detail
    assert "diverged" in result.detail
    assert "coverage" in result.detail
    assert result.detail["matched"][0]["agreement"] == "5/5"


def test_validator_includes_consensus_check(base_corpus, base_range_table):
    refs = ConsensusReference.load_from_dir(FIXTURES_DIR)
    validator = OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        include_golden_check=False,
        consensus_references=refs,
        reference_date=date(2024, 6, 1),
    )
    response = _make_response("US-CA", 0.0925)
    report = validator.validate(response)
    check_names = [c.check_name for c in report.checks]
    assert "cross_model_consensus" in check_names


@pytest.mark.parametrize("fixture_path", sorted(FIXTURES_DIR.glob("*.json")))
def test_bundled_fixtures_are_valid(fixture_path):
    ref = ConsensusReference.from_json_file(fixture_path)
    assert ref.question_id
    assert ref.jurisdiction
    assert len(ref.consensus) > 0
    for claim in ref.consensus.values():
        assert claim.agreement_count <= claim.total_sources
        assert claim.total_sources > 0
