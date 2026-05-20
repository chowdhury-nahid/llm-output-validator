"""Property-based tests using Hypothesis to fuzz the verification layer."""
from datetime import date

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from llm_output_validator import (
    Citation,
    DocumentCorpus,
    LLMResponse,
    OutputValidator,
    RangeTable,
    TaxRateResponse,
)
from llm_output_validator.checks.confidence_check import THRESHOLDS
from llm_output_validator.report import CheckStatus, ReportStatus

# Shared fixtures for property tests
_CORPUS = DocumentCorpus.from_dict({"DOC-A": 1.0, "DOC-B": 0.8, "DOC-C": 0.4})
_RANGE_TABLE = RangeTable.from_dict({"US-CA": {"lo": 0.07, "hi": 0.145}})
_VALIDATOR = OutputValidator(
    corpus=_CORPUS,
    range_table=_RANGE_TABLE,
    include_golden_check=False,
    reference_date=date(2024, 6, 1),
)

_valid_jurisdiction = st.just("US-CA")
_valid_rate = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_valid_date = st.dates(min_value=date(2000, 1, 1), max_value=date(2024, 5, 31))
_confidence = st.sampled_from(["high", "medium", "low"])
_doc_ids = st.sampled_from(["DOC-A", "DOC-B", "DOC-C"])


@given(rate=_valid_rate, effective_date=_valid_date)
@settings(max_examples=100)
def test_valid_rates_never_raise(rate: float, effective_date: date) -> None:
    response = LLMResponse(
        response=TaxRateResponse(
            jurisdiction="US-CA",
            rate=rate,
            effective_date=effective_date,
            source_id="DOC-A",
            confidence="medium",
        ),
        citations=[Citation(document_id="DOC-A"), Citation(document_id="DOC-B")],
        raw_prompt="test",
    )
    # Validation must always return a report, never raise
    report = _VALIDATOR.validate(response)
    assert report.status in (ReportStatus.PASS, ReportStatus.WARN, ReportStatus.FAIL)


@given(
    confidence=_confidence,
    doc_ids=st.lists(_doc_ids, min_size=0, max_size=5),
)
@settings(max_examples=100)
def test_calibration_invariant(confidence: str, doc_ids: list[str]) -> None:
    """If citation_count >= min_citations AND authority >= min_authority, calibration must PASS."""
    required = THRESHOLDS[confidence]
    citations = [Citation(document_id=d) for d in doc_ids]
    authority = _CORPUS.authority_score(citations)

    assume(len(citations) >= required["min_citations"])
    assume(authority >= required["min_authority"])

    response = LLMResponse(
        response=TaxRateResponse(
            jurisdiction="US-CA",
            rate=0.09,
            effective_date=date(2023, 1, 1),
            source_id="DOC-A",
            confidence=confidence,  # type: ignore[arg-type]
        ),
        citations=citations,
        raw_prompt="safe query",
    )
    report = _VALIDATOR.validate(response)
    cal_check = next(c for c in report.checks if c.check_name == "confidence_calibration")
    assert cal_check.status == CheckStatus.PASS


@given(
    rate=st.floats(min_value=0.146, max_value=1.0, allow_nan=False, allow_infinity=False),
    effective_date=_valid_date,
)
@settings(max_examples=50)
def test_out_of_range_rate_always_fails(rate: float, effective_date: date) -> None:
    response = LLMResponse(
        response=TaxRateResponse(
            jurisdiction="US-CA",
            rate=rate,
            effective_date=effective_date,
            source_id="DOC-A",
            confidence="low",
        ),
        citations=[Citation(document_id="DOC-A")],
        raw_prompt="test",
    )
    report = _VALIDATOR.validate(response)
    boundary_check = next((c for c in report.checks if c.check_name == "numeric_boundary"), None)
    if boundary_check:
        assert boundary_check.status == CheckStatus.FAIL
