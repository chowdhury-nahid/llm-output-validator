"""Example: run golden fixture regression checks against saved baselines."""
from pathlib import Path

from llm_output_validator import OutputValidator
from llm_output_validator.checks.golden_check import GoldenFixture
from llm_output_validator.exporters import TextReportExporter

fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "golden"

golden_fixtures = GoldenFixture.load_fixtures_from_dir(fixtures_dir)
print(f"Loaded {len(golden_fixtures)} golden fixture(s) from {fixtures_dir}\n")

for fixture in golden_fixtures:
    validator = OutputValidator(
        corpus=fixture.input_corpus,
        range_table=fixture.input_range_table,
        golden_fixtures=golden_fixtures,
        reference_date=fixture.reference_date,
    )
    report = validator.validate(fixture.input_response)
    print(f"--- {fixture.name} ---")
    print(TextReportExporter.export(report))
    print()
