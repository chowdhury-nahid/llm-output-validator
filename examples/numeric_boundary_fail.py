"""Example: catch an out-of-range tax rate from the wrong jurisdiction."""
from datetime import date

from llm_output_validator import (
    Citation,
    DocumentCorpus,
    LLMResponse,
    OutputValidator,
    RangeTable,
    TaxRateResponse,
)
from llm_output_validator.exporters import TextReportExporter

corpus = DocumentCorpus.from_dict({
    "IRS-2023-001": 1.0,
})

range_table = RangeTable.from_dict({
    "US-CA": {"lo": 0.07, "hi": 0.145},
})

validator = OutputValidator(
    corpus=corpus,
    range_table=range_table,
    reference_date=date(2024, 6, 1),
    include_golden_check=False,
)

response = LLMResponse(
    response=TaxRateResponse(
        jurisdiction="US-CA",
        rate=0.25,  # way outside the valid range [0.07, 0.145]
        effective_date=date(2023, 7, 1),
        source_id="IRS-2023-001",
        confidence="high",
    ),
    citations=[Citation(document_id="IRS-2023-001")],
    raw_prompt="What is the California state income tax rate for 2023?",
)

report = validator.validate(response)
print(TextReportExporter.export(report))
# numeric_boundary check will FAIL — 0.25 is outside [0.07, 0.145]
