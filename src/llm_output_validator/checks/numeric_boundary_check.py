from __future__ import annotations

from ..models import LLMResponse
from ..range_table import RangeTable
from ..report import CheckResult, CheckStatus
from . import BaseCheck


class NumericBoundaryCheck(BaseCheck):
    pattern_id = 3
    check_name = "numeric_boundary"

    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        range_table: RangeTable = kwargs["range_table"]  # type: ignore[assignment]
        jurisdiction = response.response.jurisdiction
        rate = response.response.rate

        try:
            lo, hi = range_table.get_range(jurisdiction)
        except KeyError:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.WARN,
                severity="warning",
                message=f"Jurisdiction '{jurisdiction}' not in reference range table; cannot validate rate",
                detail={"jurisdiction": jurisdiction, "rate": rate},
            )

        if rate < lo or rate > hi:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=f"Rate {rate} for {jurisdiction} outside reference range [{lo}, {hi}]",
                detail={"jurisdiction": jurisdiction, "rate": rate, "allowed_lo": lo, "allowed_hi": hi},
            )

        return CheckResult(
            pattern_id=self.pattern_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message=f"Rate {rate} within reference range [{lo}, {hi}] for {jurisdiction}",
            detail={"jurisdiction": jurisdiction, "rate": rate, "allowed_lo": lo, "allowed_hi": hi},
        )
