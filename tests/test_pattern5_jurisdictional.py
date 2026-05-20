from datetime import date

import pytest

from llm_output_validator import Citation, LLMResponse, OutputValidator, RegulatoryCllaim, TaxRateResponse
from llm_output_validator.report import CheckStatus


def _make_response(primary: str, claim_jurisdictions: list[str]) -> LLMResponse:
    return LLMResponse(
        response=TaxRateResponse(
            jurisdiction=primary,
            rate=0.0925,
            effective_date=date(2023, 7, 1),
            source_id="IRS-2023-001",
            confidence="medium",
        ),
        citations=[Citation(document_id="IRS-2023-001"), Citation(document_id="CA-FTB-2023-TAX"), Citation(document_id="USC-TITLE26-SEC11")],
        regulatory_claims=[
            RegulatoryCllaim(jurisdiction=j, claim_text=f"Claim for {j}")
            for j in claim_jurisdictions
        ],
        raw_prompt="test",
    )


def test_claim_matches_primary_passes(validator):
    report = validator.validate(_make_response("US-CA", ["US-CA"]))
    check = next(c for c in report.checks if c.check_name == "jurisdictional_scoping")
    assert check.status == CheckStatus.PASS


def test_no_claims_passes(validator):
    report = validator.validate(_make_response("US-CA", []))
    check = next(c for c in report.checks if c.check_name == "jurisdictional_scoping")
    assert check.status == CheckStatus.PASS


def test_out_of_scope_claim_fails(validator):
    report = validator.validate(_make_response("US-CA", ["US-NY"]))
    check = next(c for c in report.checks if c.check_name == "jurisdictional_scoping")
    assert check.status == CheckStatus.FAIL
    assert "US-NY" in check.detail["out_of_scope"]


def test_multiple_out_of_scope_claims_all_reported(validator):
    report = validator.validate(_make_response("US-CA", ["US-NY", "US-TX"]))
    check = next(c for c in report.checks if c.check_name == "jurisdictional_scoping")
    assert check.status == CheckStatus.FAIL
    assert set(check.detail["out_of_scope"]) == {"US-NY", "US-TX"}


def test_related_jurisdictions_allow_list(base_corpus, base_range_table):
    validator = OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        related_jurisdictions={"US-CA": {"US-CA", "US-FED"}},
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    report = validator.validate(_make_response("US-CA", ["US-CA", "US-FED"]))
    check = next(c for c in report.checks if c.check_name == "jurisdictional_scoping")
    assert check.status == CheckStatus.PASS


def test_related_jurisdiction_outside_allow_list_still_fails(base_corpus, base_range_table):
    validator = OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        related_jurisdictions={"US-CA": {"US-CA", "US-FED"}},
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    report = validator.validate(_make_response("US-CA", ["US-NY"]))
    check = next(c for c in report.checks if c.check_name == "jurisdictional_scoping")
    assert check.status == CheckStatus.FAIL
