from datetime import date

import pytest

from llm_output_validator import (
    Citation,
    DocumentCorpus,
    LLMResponse,
    OutputValidator,
    RangeTable,
    RegulatoryCllaim,
    TaxRateResponse,
)


@pytest.fixture
def base_corpus() -> DocumentCorpus:
    return DocumentCorpus.from_dict({
        "IRS-2023-001": 1.0,
        "CA-FTB-2023-TAX": 0.9,
        "USC-TITLE26-SEC11": 1.0,
        "COMMENTARY-2023": 0.3,
    })


@pytest.fixture
def base_range_table() -> RangeTable:
    return RangeTable.from_dict({
        "US-CA": {"lo": 0.07, "hi": 0.145},
        "US-NY": {"lo": 0.06, "hi": 0.109},
        "US-TX": {"lo": 0.0,  "hi": 0.0},
        "DE":    {"lo": 0.19, "hi": 0.19},
        "US-CA-OLD": {
            "lo": 0.06, "hi": 0.12,
            "superseded_after": "2023-01-01",
            "superseded_by": "US-CA",
        },
    })


@pytest.fixture
def valid_response() -> LLMResponse:
    return LLMResponse(
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
        regulatory_claims=[
            RegulatoryCllaim(jurisdiction="US-CA", claim_text="California state income tax applies."),
        ],
        raw_prompt="What is the tax rate for California in 2023?",
    )


@pytest.fixture
def validator(base_corpus: DocumentCorpus, base_range_table: RangeTable) -> OutputValidator:
    return OutputValidator(
        corpus=base_corpus,
        range_table=base_range_table,
        authority_threshold=0.5,
        include_golden_check=False,
        reference_date=date(2024, 6, 1),
    )
