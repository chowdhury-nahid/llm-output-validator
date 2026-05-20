import json
from datetime import date

from llm_output_validator import Citation, LLMResponse, OutputValidator, TaxRateResponse
from llm_output_validator.exporters import JsonReportExporter, TextReportExporter


def _make_report(base_corpus, base_range_table):
    validator = OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        include_golden_check=False,
        reference_date=date(2024, 6, 1),
    )
    response = LLMResponse(
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
        raw_prompt="What is the CA tax rate?",
    )
    return validator.validate(response)


def test_json_exporter_produces_valid_json(base_corpus, base_range_table):
    report = _make_report(base_corpus, base_range_table)
    output = JsonReportExporter.export(report)
    parsed = json.loads(output)
    assert parsed["status"] == "pass"
    assert len(parsed["checks"]) > 0


def test_json_exporter_includes_all_check_fields(base_corpus, base_range_table):
    report = _make_report(base_corpus, base_range_table)
    parsed = json.loads(JsonReportExporter.export(report))
    first_check = parsed["checks"][0]
    assert "pattern_id" in first_check
    assert "check_name" in first_check
    assert "status" in first_check
    assert "message" in first_check


def test_text_exporter_contains_status(base_corpus, base_range_table):
    report = _make_report(base_corpus, base_range_table)
    output = TextReportExporter.export(report)
    assert "PASS" in output
    assert "US-CA" in output


def test_text_exporter_lists_all_checks(base_corpus, base_range_table):
    report = _make_report(base_corpus, base_range_table)
    output = TextReportExporter.export(report)
    assert "schema_validation" in output
    assert "citation_grounding" in output
    assert "numeric_boundary" in output


def test_text_exporter_shows_check_symbols(base_corpus, base_range_table):
    report = _make_report(base_corpus, base_range_table)
    output = TextReportExporter.export(report)
    assert "✓" in output
