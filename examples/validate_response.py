"""End-to-end example: validate a simulated LLM response."""

from datetime import date

from llm_output_validator import (
    Citation,
    DocumentCorpus,
    LLMResponse,
    OutputValidator,
    RangeTable,
    RegulatoryCllaim,
    TaxRateResponse,
)
from llm_output_validator.exporters import TextReportExporter

corpus = DocumentCorpus.from_dict(
    {
        "IRS-2023-001": 1.0,
        "CA-FTB-2023-TAX": 0.9,
        "USC-TITLE26-SEC11": 1.0,
    }
)

range_table = RangeTable.from_dict(
    {
        "US-CA": {"lo": 0.07, "hi": 0.145},
    }
)

validator = OutputValidator(
    corpus=corpus,
    range_table=range_table,
    reference_date=date(2024, 6, 1),
    include_golden_check=False,
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
    regulatory_claims=[
        RegulatoryCllaim(jurisdiction="US-CA", claim_text="California state income tax applies."),
    ],
    raw_prompt="What is the California state income tax rate for 2023?",
)

report = validator.validate(response)
print(TextReportExporter.export(report))
