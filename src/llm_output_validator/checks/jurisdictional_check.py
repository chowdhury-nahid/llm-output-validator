from __future__ import annotations

from ..models import LLMResponse
from ..report import CheckResult, CheckStatus
from . import BaseCheck


class JurisdictionalScopingCheck(BaseCheck):
    pattern_id = 5
    check_name = "jurisdictional_scoping"

    def __init__(self, related_jurisdictions: dict[str, set[str]] | None = None) -> None:
        self.related_jurisdictions = related_jurisdictions or {}

    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        primary = response.response.jurisdiction
        allowed = self.related_jurisdictions.get(primary, {primary})

        out_of_scope = [
            rc.jurisdiction for rc in response.regulatory_claims if rc.jurisdiction not in allowed
        ]

        if out_of_scope:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=f"Out-of-scope regulatory claims for {primary}: {', '.join(out_of_scope)}",
                detail={
                    "primary_jurisdiction": primary,
                    "allowed": list(allowed),
                    "out_of_scope": out_of_scope,
                },
            )

        return CheckResult(
            pattern_id=self.pattern_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message=f"All regulatory claims scoped to {primary}",
            detail={"primary_jurisdiction": primary, "allowed": list(allowed), "out_of_scope": []},
        )
