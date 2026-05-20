from datetime import date

from llm_output_validator import Citation, LLMResponse, TaxRateResponse
from llm_output_validator.report import CheckStatus


def _make_response(jurisdiction: str, rate: float) -> LLMResponse:
    return LLMResponse(
        response=TaxRateResponse(
            jurisdiction=jurisdiction,
            rate=rate,
            effective_date=date(2023, 7, 1),
            source_id="IRS-2023-001",
            confidence="medium",
        ),
        citations=[
            Citation(document_id="IRS-2023-001"),
            Citation(document_id="CA-FTB-2023-TAX"),
            Citation(document_id="USC-TITLE26-SEC11"),
        ],
        raw_prompt="test",
    )


def test_rate_within_bounds_passes(validator):
    report = validator.validate(_make_response("US-CA", 0.0925))
    check = next(c for c in report.checks if c.check_name == "numeric_boundary")
    assert check.status == CheckStatus.PASS


def test_rate_at_lower_bound_passes(validator):
    report = validator.validate(_make_response("US-CA", 0.07))
    check = next(c for c in report.checks if c.check_name == "numeric_boundary")
    assert check.status == CheckStatus.PASS


def test_rate_at_upper_bound_passes(validator):
    report = validator.validate(_make_response("US-CA", 0.145))
    check = next(c for c in report.checks if c.check_name == "numeric_boundary")
    assert check.status == CheckStatus.PASS


def test_rate_below_lower_bound_fails(validator):
    report = validator.validate(_make_response("US-CA", 0.05))
    check = next(c for c in report.checks if c.check_name == "numeric_boundary")
    assert check.status == CheckStatus.FAIL
    assert check.detail["allowed_lo"] == 0.07


def test_rate_above_upper_bound_fails(validator):
    report = validator.validate(_make_response("US-CA", 0.20))
    check = next(c for c in report.checks if c.check_name == "numeric_boundary")
    assert check.status == CheckStatus.FAIL
    assert check.detail["allowed_hi"] == 0.145


def test_unknown_jurisdiction_warns(validator):
    report = validator.validate(_make_response("US-ZZ", 0.09))
    check = next(c for c in report.checks if c.check_name == "numeric_boundary")
    assert check.status == CheckStatus.WARN
    assert check.severity == "warning"


def test_zero_rate_valid_for_texas(validator):
    report = validator.validate(_make_response("US-TX", 0.0))
    check = next(c for c in report.checks if c.check_name == "numeric_boundary")
    assert check.status == CheckStatus.PASS


def test_detail_contains_bounds(validator):
    report = validator.validate(_make_response("US-CA", 0.20))
    check = next(c for c in report.checks if c.check_name == "numeric_boundary")
    assert "allowed_lo" in check.detail
    assert "allowed_hi" in check.detail
    assert check.detail["rate"] == 0.20
