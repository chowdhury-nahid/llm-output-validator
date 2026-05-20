from datetime import date

import pytest

from llm_output_validator import Citation, DocumentCorpus, LLMResponse, TaxRateResponse
from llm_output_validator.report import CheckStatus


def _make_response(citations: list[Citation], jurisdiction: str = "US-CA") -> LLMResponse:
    return LLMResponse(
        response=TaxRateResponse(
            jurisdiction=jurisdiction,
            rate=0.0925,
            effective_date=date(2023, 7, 1),
            source_id="IRS-2023-001",
            confidence="high",
        ),
        citations=citations,
        raw_prompt="test",
    )


def test_all_citations_grounded_passes(validator, valid_response):
    report = validator.validate(valid_response)
    check = next(c for c in report.checks if c.check_name == "citation_grounding")
    assert check.status == CheckStatus.PASS


def test_missing_citation_fails(validator):
    response = _make_response([
        Citation(document_id="IRS-2023-001"),
        Citation(document_id="HALLUCINATED-DOC-999"),
    ])
    report = validator.validate(response)
    check = next(c for c in report.checks if c.check_name == "citation_grounding")
    assert check.status == CheckStatus.FAIL
    assert "HALLUCINATED-DOC-999" in check.detail["missing_doc_ids"]


def test_all_missing_citations_fail(validator):
    response = _make_response([
        Citation(document_id="FAKE-1"),
        Citation(document_id="FAKE-2"),
    ])
    report = validator.validate(response)
    check = next(c for c in report.checks if c.check_name == "citation_grounding")
    assert check.status == CheckStatus.FAIL


def test_empty_citations_list_fails(validator):
    response = _make_response([])
    report = validator.validate(response)
    check = next(c for c in report.checks if c.check_name == "citation_grounding")
    assert check.status == CheckStatus.FAIL


def test_low_authority_score_warns(base_range_table):
    from llm_output_validator import OutputValidator
    low_corpus = DocumentCorpus.from_dict({"COMMENTARY-2023": 0.2})
    validator = OutputValidator(
        corpus=low_corpus,
        range_table=base_range_table,
        authority_threshold=0.5,
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    response = _make_response([Citation(document_id="COMMENTARY-2023")])
    report = validator.validate(response)
    check = next(c for c in report.checks if c.check_name == "citation_grounding")
    assert check.status == CheckStatus.WARN
    assert check.severity == "warning"


def test_authority_score_calculation():
    corpus = DocumentCorpus.from_dict({"A": 1.0, "B": 0.5})
    citations = [Citation(document_id="A"), Citation(document_id="B")]
    score = corpus.authority_score(citations)
    assert abs(score - 0.75) < 1e-9


def test_authority_threshold_configurable(base_range_table):
    from llm_output_validator import OutputValidator
    low_corpus = DocumentCorpus.from_dict({"COMMENTARY-2023": 0.2})
    validator = OutputValidator(
        corpus=low_corpus,
        range_table=base_range_table,
        authority_threshold=0.0,  # no threshold
        include_golden_check=False,
        reference_date=date(2024, 1, 1),
    )
    response = _make_response([Citation(document_id="COMMENTARY-2023")])
    report = validator.validate(response)
    check = next(c for c in report.checks if c.check_name == "citation_grounding")
    assert check.status == CheckStatus.PASS
