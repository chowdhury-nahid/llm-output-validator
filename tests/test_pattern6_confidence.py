from datetime import date

import pytest

from llm_output_validator import Citation, DocumentCorpus, LLMResponse, OutputValidator, TaxRateResponse
from llm_output_validator.checks.confidence_check import THRESHOLDS
from llm_output_validator.report import CheckStatus


def _make_response(confidence: str, citations: list[Citation]) -> LLMResponse:
    return LLMResponse(
        response=TaxRateResponse(
            jurisdiction="US-CA",
            rate=0.0925,
            effective_date=date(2023, 7, 1),
            source_id="IRS-2023-001",
            confidence=confidence,  # type: ignore[arg-type]
        ),
        citations=citations,
        raw_prompt="test",
    )


def test_high_confidence_with_strong_evidence_passes(validator):
    report = validator.validate(
        _make_response("high", [
            Citation(document_id="IRS-2023-001"),
            Citation(document_id="CA-FTB-2023-TAX"),
            Citation(document_id="USC-TITLE26-SEC11"),
        ])
    )
    check = next(c for c in report.checks if c.check_name == "confidence_calibration")
    assert check.status == CheckStatus.PASS


def test_high_confidence_with_single_low_authority_warns(base_range_table):
    low_corpus = DocumentCorpus.from_dict({"COMMENTARY-2023": 0.3})
    validator = OutputValidator(
        corpus=low_corpus,
        range_table=base_range_table,
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    report = validator.validate(
        _make_response("high", [Citation(document_id="COMMENTARY-2023")])
    )
    check = next(c for c in report.checks if c.check_name == "confidence_calibration")
    assert check.status == CheckStatus.WARN
    assert check.severity == "warning"


def test_calibration_is_warning_not_fail(base_range_table):
    low_corpus = DocumentCorpus.from_dict({"COMMENTARY-2023": 0.3})
    validator = OutputValidator(
        corpus=low_corpus,
        range_table=base_range_table,
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    report = validator.validate(
        _make_response("high", [Citation(document_id="COMMENTARY-2023")])
    )
    check = next(c for c in report.checks if c.check_name == "confidence_calibration")
    assert check.status != CheckStatus.FAIL


def test_medium_confidence_with_one_grounded_citation_passes(base_corpus, base_range_table):
    validator = OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    report = validator.validate(
        _make_response("medium", [Citation(document_id="IRS-2023-001")])
    )
    check = next(c for c in report.checks if c.check_name == "confidence_calibration")
    assert check.status == CheckStatus.PASS


def test_low_confidence_always_passes_calibration(base_corpus, base_range_table):
    validator = OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    report = validator.validate(_make_response("low", []))
    check = next(c for c in report.checks if c.check_name == "confidence_calibration")
    # Low confidence makes no claim; citation check fails but calibration should pass
    assert check.status == CheckStatus.PASS


def test_thresholds_match_declared_constants():
    assert THRESHOLDS["high"]["min_authority"] == 0.7
    assert THRESHOLDS["medium"]["min_citations"] == 1
    assert THRESHOLDS["low"]["min_authority"] == 0.0
