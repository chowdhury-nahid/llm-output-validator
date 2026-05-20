from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-validate",
        description="Validate an LLM response JSON against the verification layer contracts",
    )
    parser.add_argument("input", help="Path to JSON file containing the LLM response (use - for stdin)")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSON file")
    parser.add_argument("--range-table", required=True, dest="range_table", help="Path to range table JSON file")
    parser.add_argument("--golden-dir", dest="golden_dir", help="Directory of golden fixture JSON files")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    if args.input == "-":
        raw = json.load(sys.stdin)
    else:
        raw = json.loads(Path(args.input).read_text())

    from .validator import OutputValidator
    from .exporters import JsonReportExporter, TextReportExporter

    validator = OutputValidator.build_default(
        corpus_path=Path(args.corpus),
        range_table_path=Path(args.range_table),
        golden_dir=Path(args.golden_dir) if args.golden_dir else None,
    )

    report = validator.validate_raw(raw)

    if args.format == "json":
        print(JsonReportExporter.export(report))
    else:
        print(TextReportExporter.export(report))

    sys.exit(0 if report.status.value in ("pass", "warn") else 1)
