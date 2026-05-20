from __future__ import annotations

from datetime import date

from ..models import LLMResponse
from ..range_table import RangeTable
from ..report import CheckResult, CheckStatus
from . import BaseCheck


class TemporalConsistencyCheck(BaseCheck):
    pattern_id = 4
    check_name = "temporal_consistency"

    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        range_table: RangeTable = kwargs["range_table"]  # type: ignore[assignment]
        reference_date: date | None = kwargs.get("reference_date")  # type: ignore[assignment]
        today = reference_date or date.today()

        jurisdiction = response.response.jurisdiction
        effective_date = response.response.effective_date

        if effective_date > today:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=f"Response cites future effective date {effective_date} (reference: {today})",
                detail={"effective_date": str(effective_date), "reference_date": str(today), "jurisdiction": jurisdiction, "superseded": False},
            )

        superseded = range_table.has_superseding_rule(jurisdiction, after=effective_date, before=today)
        if superseded:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=f"A more recent rule exists for {jurisdiction} between {effective_date} and {today}",
                detail={"effective_date": str(effective_date), "reference_date": str(today), "jurisdiction": jurisdiction, "superseded": True},
            )

        return CheckResult(
            pattern_id=self.pattern_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message=f"Temporal consistency verified for {jurisdiction} (effective: {effective_date})",
            detail={"effective_date": str(effective_date), "reference_date": str(today), "jurisdiction": jurisdiction, "superseded": False},
        )
