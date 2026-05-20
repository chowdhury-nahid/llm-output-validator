from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any

from .checks import BaseCheck
from .checks.citation_check import CitationGroundingCheck
from .checks.confidence_check import ConfidenceCalibrationCheck
from .checks.golden_check import GoldenFixture, GoldenRegressionCheck
from .checks.injection_check import PromptInjectionCheck
from .checks.jurisdictional_check import JurisdictionalScopingCheck
from .checks.numeric_boundary_check import NumericBoundaryCheck
from .checks.schema_check import SchemaValidationCheck
from .checks.temporal_check import TemporalConsistencyCheck
from .corpus import DocumentCorpus
from .models import LLMResponse
from .range_table import RangeTable
from .report import CheckResult, CheckStatus, ReportStatus, VerificationReport


class OutputValidator:
    def __init__(
        self,
        corpus: DocumentCorpus,
        range_table: RangeTable,
        authority_threshold: float = 0.5,
        related_jurisdictions: dict[str, set[str]] | None = None,
        golden_fixtures: list[GoldenFixture] | None = None,
        extra_injection_patterns: list | None = None,
        reference_date: date | None = None,
        include_golden_check: bool = True,
    ) -> None:
        self.corpus = corpus
        self.range_table = range_table
        self.reference_date = reference_date

        self._checks: list[BaseCheck] = [
            SchemaValidationCheck(),
            CitationGroundingCheck(authority_threshold=authority_threshold),
            NumericBoundaryCheck(),
            TemporalConsistencyCheck(),
            JurisdictionalScopingCheck(related_jurisdictions=related_jurisdictions),
            ConfidenceCalibrationCheck(),
            PromptInjectionCheck(extra_patterns=extra_injection_patterns),
        ]
        if include_golden_check and golden_fixtures:
            self._checks.append(GoldenRegressionCheck(fixtures=golden_fixtures))

    def validate(self, response: LLMResponse) -> VerificationReport:
        start = time.monotonic()
        collected: list[CheckResult] = []

        kwargs = {
            "corpus": self.corpus,
            "range_table": self.range_table,
            "reference_date": self.reference_date,
        }

        # Pattern 1: fail-fast schema gate
        schema_result = self._checks[0].run(response, **kwargs)
        collected.append(schema_result)
        if schema_result.status == CheckStatus.FAIL:
            return VerificationReport(
                status=ReportStatus.FAIL,
                response_id=None,
                checks=collected,
                elapsed_ms=(time.monotonic() - start) * 1000,
            )

        # Patterns 2–N: collect all
        for check in self._checks[1:]:
            collected.append(check.run(response, **kwargs))

        return VerificationReport(
            status=_compute_status(collected),
            response_id=response.response.jurisdiction,
            checks=collected,
            elapsed_ms=(time.monotonic() - start) * 1000,
        )

    def validate_raw(self, raw: dict[str, Any]) -> VerificationReport:
        valid, errors = SchemaValidationCheck.validate_raw(raw)
        if not valid:
            result = CheckResult(
                pattern_id=1,
                check_name="schema_validation",
                status=CheckStatus.FAIL,
                severity="error",
                message=f"JSON Schema validation failed: {errors[0]}",
                detail={"errors": errors},
            )
            return VerificationReport(
                status=ReportStatus.FAIL,
                response_id=None,
                checks=[result],
            )

        from pydantic import ValidationError

        try:
            response = LLMResponse.model_validate(raw)
        except ValidationError as exc:
            result = CheckResult(
                pattern_id=1,
                check_name="schema_validation",
                status=CheckStatus.FAIL,
                severity="error",
                message=f"Pydantic validation failed: {exc.error_count()} error(s)",
                detail={"errors": [str(e) for e in exc.errors()]},
            )
            return VerificationReport(
                status=ReportStatus.FAIL,
                response_id=None,
                checks=[result],
            )

        return self.validate(response)

    @classmethod
    def build_default(
        cls,
        corpus_path: Path,
        range_table_path: Path,
        golden_dir: Path | None = None,
        **kwargs: Any,
    ) -> OutputValidator:
        corpus = DocumentCorpus.from_json_file(corpus_path)
        range_table = RangeTable.from_json_file(range_table_path)
        golden_fixtures: list[GoldenFixture] | None = None
        if golden_dir:
            golden_fixtures = GoldenFixture.load_fixtures_from_dir(golden_dir)
        return cls(corpus=corpus, range_table=range_table, golden_fixtures=golden_fixtures, **kwargs)


def _compute_status(checks: list[CheckResult]) -> ReportStatus:
    if any(c.status == CheckStatus.FAIL for c in checks):
        return ReportStatus.FAIL
    if any(c.status == CheckStatus.WARN for c in checks):
        return ReportStatus.WARN
    return ReportStatus.PASS
