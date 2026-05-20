from datetime import date

import pytest
from pydantic import ValidationError

from llm_output_validator import LLMResponse, TaxRateResponse
from llm_output_validator.checks.schema_check import SchemaValidationCheck
from llm_output_validator.report import CheckStatus


def test_valid_response_passes_schema(validator, valid_response):
    report = validator.validate(valid_response)
    schema_check = next(c for c in report.checks if c.check_name == "schema_validation")
    assert schema_check.status == CheckStatus.PASS


def test_future_effective_date_fails(validator, valid_response):
    future_response = LLMResponse(
        response=TaxRateResponse(
            jurisdiction="US-CA",
            rate=0.0925,
            effective_date=date(2099, 1, 1),
            source_id="IRS-2023-001",
            confidence="high",
        ),
        citations=valid_response.citations,
        regulatory_claims=valid_response.regulatory_claims,
    )
    report = validator.validate(future_response)
    schema_check = next(c for c in report.checks if c.check_name == "schema_validation")
    assert schema_check.status == CheckStatus.FAIL
    assert schema_check.severity == "error"


def test_rate_above_1_rejected_by_pydantic():
    with pytest.raises(ValidationError):
        TaxRateResponse(
            jurisdiction="US-CA",
            rate=1.5,
            effective_date=date(2023, 1, 1),
            source_id="IRS-2023-001",
            confidence="high",
        )


def test_rate_below_0_rejected_by_pydantic():
    with pytest.raises(ValidationError):
        TaxRateResponse(
            jurisdiction="US-CA",
            rate=-0.01,
            effective_date=date(2023, 1, 1),
            source_id="IRS-2023-001",
            confidence="high",
        )


def test_unknown_confidence_rejected_by_pydantic():
    with pytest.raises(ValidationError):
        TaxRateResponse(
            jurisdiction="US-CA",
            rate=0.09,
            effective_date=date(2023, 1, 1),
            source_id="IRS-2023-001",
            confidence="very_high",  # type: ignore[arg-type]
        )


def test_empty_source_id_rejected_by_pydantic():
    with pytest.raises(ValidationError):
        TaxRateResponse(
            jurisdiction="US-CA",
            rate=0.09,
            effective_date=date(2023, 1, 1),
            source_id="",
            confidence="high",
        )


def test_validate_raw_valid_dict_passes(base_corpus, base_range_table):
    from llm_output_validator import OutputValidator

    validator = OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    raw = {
        "response": {
            "jurisdiction": "US-CA",
            "rate": 0.0925,
            "effective_date": "2023-07-01",
            "source_id": "IRS-2023-001",
            "confidence": "high",
        },
        "citations": [
            {"document_id": "IRS-2023-001"},
            {"document_id": "CA-FTB-2023-TAX"},
            {"document_id": "USC-TITLE26-SEC11"},
        ],
        "regulatory_claims": [{"jurisdiction": "US-CA", "claim_text": "CA tax applies"}],
    }
    report = validator.validate_raw(raw)
    schema_check = next(c for c in report.checks if c.check_name == "schema_validation")
    assert schema_check.status == CheckStatus.PASS


def test_validate_raw_missing_required_field_fails(base_corpus, base_range_table):
    from llm_output_validator import OutputValidator

    validator = OutputValidator(
        corpus=base_corpus, range_table=base_range_table, include_golden_check=False
    )
    raw = {
        "response": {
            "rate": 0.09,
            "effective_date": "2023-07-01",
            "source_id": "IRS-2023-001",
            "confidence": "high",
            # missing "jurisdiction"
        }
    }
    report = validator.validate_raw(raw)
    assert report.status.value == "fail"


def test_validate_raw_returns_structured_errors(base_corpus, base_range_table):
    raw = {
        "response": {
            "rate": 2.0,
            "effective_date": "2023-01-01",
            "source_id": "x",
            "confidence": "high",
        }
    }
    _, errors = SchemaValidationCheck.validate_raw(raw)
    assert isinstance(errors, list)
    assert len(errors) > 0
