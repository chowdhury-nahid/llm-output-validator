"""Example: validate a response against cross-model consensus data."""

from datetime import date
from pathlib import Path

from llm_output_validator import (
    Citation,
    ConsensusReference,
    DocumentCorpus,
    LLMResponse,
    OutputValidator,
    RangeTable,
    TaxRateResponse,
)
from llm_output_validator.exporters import TextReportExporter

consensus_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "consensus"
consensus_refs = ConsensusReference.load_from_dir(consensus_dir)
print(f"Loaded {len(consensus_refs)} consensus reference(s)\n")

corpus = DocumentCorpus.from_dict({"BMF-2024-001": 1.0})
range_table = RangeTable.from_dict({"DE": {"lo": 0.14, "hi": 0.35}})

validator = OutputValidator(
    corpus=corpus,
    range_table=range_table,
    reference_date=date(2024, 6, 1),
    include_golden_check=False,
    consensus_references=consensus_refs,
)

response = LLMResponse(
    response=TaxRateResponse(
        jurisdiction="DE",
        rate=0.15825,
        effective_date=date(2024, 1, 1),
        source_id="BMF-2024-001",
        confidence="high",
    ),
    citations=[Citation(document_id="BMF-2024-001")],
    raw_prompt="What is the German corporate tax rate?",
)

report = validator.validate(response)
print(TextReportExporter.export(report))
