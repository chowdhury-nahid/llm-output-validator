from datetime import date

import pytest

from llm_output_validator import Citation, LLMResponse, OutputValidator, TaxRateResponse
from llm_output_validator.report import CheckStatus


def _make_response(jurisdiction: str, effective_date: date) -> LLMResponse:
    return LLMResponse(
        response=TaxRateResponse(
            jurisdiction=jurisdiction,
            rate=0.0925,
            effective_date=effective_date,
            source_id="IRS-2023-001",
            confidence="medium",
        ),
        citations=[Citation(document_id="IRS-2023-001"), Citation(document_id="CA-FTB-2023-TAX"), Citation(document_id="USC-TITLE26-SEC11")],
        raw_prompt="test",
    )


def test_current_effective_date_passes(validator):
    report = validator.validate(_make_response("US-CA", date(2023, 7, 1)))
    check = next(c for c in report.checks if c.check_name == "temporal_consistency")
    assert check.status == CheckStatus.PASS


def test_future_effective_date_caught_by_schema(validator):
    # Schema check (Pattern 1) catches this first — temporal check never runs
    report = validator.validate(_make_response("US-CA", date(2099, 1, 1)))
    assert report.status.value == "fail"
    schema_check = next(c for c in report.checks if c.check_name == "schema_validation")
    assert schema_check.status == CheckStatus.FAIL


def test_superseded_rule_detected(base_corpus, base_range_table):
    # US-CA-OLD was superseded after 2023-01-01; using it with reference 2024-01-01 should fail
    validator = OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    response = _make_response("US-CA-OLD", date(2022, 6, 1))
    report = validator.validate(response)
    check = next(c for c in report.checks if c.check_name == "temporal_consistency")
    assert check.status == CheckStatus.FAIL
    assert check.detail["superseded"] is True


def test_effective_date_after_supersession_passes(base_corpus, base_range_table):
    validator = OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        include_golden_check=False,
        reference_date=date(2024, 6, 1),
    )
    # effective_date 2023-06-01 is after the supersession cutoff of 2023-01-01
    response = _make_response("US-CA-OLD", date(2023, 6, 1))
    report = validator.validate(response)
    check = next(c for c in report.checks if c.check_name == "temporal_consistency")
    assert check.status == CheckStatus.PASS


def test_reference_date_injectable(base_corpus, base_range_table):
    v1 = OutputValidator(corpus=base_corpus, range_table=base_range_table, include_golden_check=False, reference_date=date(2022, 1, 1))
    v2 = OutputValidator(corpus=base_corpus, range_table=base_range_table, include_golden_check=False, reference_date=date(2024, 1, 1))
    response = _make_response("US-CA-OLD", date(2021, 1, 1))

    r1 = v1.validate(response)
    r2 = v2.validate(response)
    c1 = next(c for c in r1.checks if c.check_name == "temporal_consistency")
    c2 = next(c for c in r2.checks if c.check_name == "temporal_consistency")
    # With reference 2022-01-01, no supersession yet (superseded_after=2023-01-01)
    assert c1.status == CheckStatus.PASS
    # With reference 2024-01-01, supersession is detected
    assert c2.status == CheckStatus.FAIL


def test_jurisdiction_without_supersession_passes(validator):
    report = validator.validate(_make_response("US-NY", date(2023, 1, 1)))
    check = next(c for c in report.checks if c.check_name == "temporal_consistency")
    assert check.status == CheckStatus.PASS
