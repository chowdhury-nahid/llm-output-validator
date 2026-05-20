from datetime import date
from pathlib import Path

import pytest

from llm_output_validator import Citation, LLMResponse, OutputValidator, TaxRateResponse
from llm_output_validator.checks.golden_check import GoldenFixture, GoldenRegressionCheck
from llm_output_validator.report import CheckStatus

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "golden"


def test_fixtures_load_from_directory():
    fixtures = GoldenFixture.load_fixtures_from_dir(FIXTURES_DIR)
    assert len(fixtures) >= 2
    names = [f.name for f in fixtures]
    assert "us_ca_standard_rate" in names
    assert "eu_vat_de_standard" in names


@pytest.mark.parametrize("fixture_path", sorted(FIXTURES_DIR.glob("*.json")))
def test_bundled_fixtures_pass(fixture_path):
    fixtures = GoldenFixture.load_fixtures_from_dir(fixture_path.parent)
    fixture = next(f for f in fixtures if f.name == fixture_path.stem)

    sub_validator = OutputValidator(
        corpus=fixture.input_corpus,
        range_table=fixture.input_range_table,
        include_golden_check=False,
        reference_date=fixture.reference_date,
    )
    report = sub_validator.validate(fixture.input_response)
    check_summary = "\n".join(
        f"  {c.check_name}: {c.status.value} — {c.message}" for c in report.checks
    )
    assert report.status.value == fixture.expected_status, (
        f"Fixture '{fixture.name}' expected status={fixture.expected_status},"
        f" got={report.status.value}\n{check_summary}"
    )


def test_golden_check_detects_drift(base_corpus, base_range_table):
    # Create a fixture that expects "fail" but will actually produce "pass"
    good_response = LLMResponse(
        response=TaxRateResponse(
            jurisdiction="US-CA",
            rate=0.0925,
            effective_date=date(2023, 7, 1),
            source_id="IRS-2023-001",
            confidence="high",
        ),
        citations=[
            Citation(document_id="IRS-2023-001"),
            Citation(document_id="CA-FTB-2023-TAX"),
            Citation(document_id="USC-TITLE26-SEC11"),
        ],
        raw_prompt="test",
    )
    fixture = GoldenFixture(
        name="drift_test",
        pattern_tag="test",
        created=date(2024, 1, 1),
        reference_date=date(2024, 6, 1),
        input_response=good_response,
        input_corpus=base_corpus,
        input_range_table=base_range_table,
        expected_status="fail",  # intentionally wrong
        expected_checks=[],
    )
    check = GoldenRegressionCheck(fixtures=[fixture])
    result = check.run(good_response)
    assert result.status == CheckStatus.WARN
    assert len(result.detail["drifted"]) > 0


def test_golden_check_does_not_recurse(base_corpus, base_range_table):
    fixtures = GoldenFixture.load_fixtures_from_dir(FIXTURES_DIR)
    golden_check = GoldenRegressionCheck(fixtures=fixtures)

    # Run the check — if it recurses into itself, it would hang or fail
    response = fixtures[0].input_response
    result = golden_check.run(response)
    # Should complete without error
    assert result.check_name == "golden_regression"


def test_golden_regression_pass_with_all_fixtures(base_corpus, base_range_table):
    fixtures = GoldenFixture.load_fixtures_from_dir(FIXTURES_DIR)
    # Use a response just to satisfy the interface; golden check is self-contained
    dummy_response = fixtures[0].input_response
    check = GoldenRegressionCheck(fixtures=fixtures)
    result = check.run(dummy_response)
    assert result.status == CheckStatus.PASS
    assert result.detail["drifted"] == []
