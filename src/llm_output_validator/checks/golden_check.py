from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..corpus import DocumentCorpus
from ..models import LLMResponse
from ..range_table import RangeTable
from ..report import CheckResult, CheckStatus
from . import BaseCheck


@dataclass
class GoldenFixture:
    name: str
    pattern_tag: str
    created: date
    reference_date: date
    input_response: LLMResponse
    input_corpus: DocumentCorpus
    input_range_table: RangeTable
    expected_status: str
    expected_checks: list[dict]  # [{"pattern_id": int, "check_name": str, "status": str}]

    @staticmethod
    def load_fixtures_from_dir(directory: Path) -> list[GoldenFixture]:
        fixtures: list[GoldenFixture] = []
        for path in sorted(Path(directory).glob("*.json")):
            raw = json.loads(path.read_text())
            meta = raw["meta"]
            inp = raw["input"]

            response = LLMResponse.model_validate(inp["response"])
            corpus = DocumentCorpus.from_dict(inp.get("corpus_docs", {}))
            range_table = RangeTable.from_dict(inp.get("range_table", {}))

            fixtures.append(
                GoldenFixture(
                    name=meta["name"],
                    pattern_tag=meta["pattern_tag"],
                    created=date.fromisoformat(meta["created"]),
                    reference_date=date.fromisoformat(meta.get("reference_date", meta["created"])),
                    input_response=response,
                    input_corpus=corpus,
                    input_range_table=range_table,
                    expected_status=raw["expected"]["status"],
                    expected_checks=raw["expected"]["checks"],
                )
            )
        return fixtures


class GoldenRegressionCheck(BaseCheck):
    pattern_id = 8
    check_name = "golden_regression"

    def __init__(self, fixtures: list[GoldenFixture]) -> None:
        self.fixtures = fixtures

    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        # Lazy import to break circular dependency with validator
        from ..validator import OutputValidator

        drifted: list[str] = []

        for fixture in self.fixtures:
            sub_validator = OutputValidator(
                corpus=fixture.input_corpus,
                range_table=fixture.input_range_table,
                include_golden_check=False,
                reference_date=fixture.reference_date,
            )
            report = sub_validator.validate(fixture.input_response)

            if report.status.value != fixture.expected_status:
                drifted.append(
                    f"{fixture.name}: expected status={fixture.expected_status},"
                    f" got={report.status.value}"
                )
                continue

            for expected_check in fixture.expected_checks:
                actual = next(
                    (c for c in report.checks if c.check_name == expected_check["check_name"]),
                    None,
                )
                if actual is None:
                    drifted.append(
                        f"{fixture.name}: check "
                        f"'{expected_check['check_name']}' missing from report"
                    )
                elif actual.status.value != expected_check["status"]:
                    drifted.append(
                        f"{fixture.name}/{expected_check['check_name']}: "
                        f"expected={expected_check['status']}, got={actual.status.value}"
                    )

        if drifted:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.WARN,
                severity="warning",
                message=f"{len(drifted)} golden fixture(s) drifted",
                detail={"drifted": drifted},
            )

        return CheckResult(
            pattern_id=self.pattern_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message=f"All {len(self.fixtures)} golden fixture(s) matched expected reports",
            detail={"drifted": []},
        )
