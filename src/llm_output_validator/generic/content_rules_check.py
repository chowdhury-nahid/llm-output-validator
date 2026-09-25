"""Generic content-rules check: caller-supplied required/banned literal
terms.

Per the 2026-09-24 via-negativa cut, this is literal-only — no user-supplied
regex. That removes the ReDoS risk (a crafted regex hanging the server) at
its source rather than mitigating it with a timeout, so this check has no
extra dependency and cannot hang regardless of input.
"""

from __future__ import annotations

from ..evals.models import EvalContext
from ..report import CheckResult, CheckStatus
from . import BaseGenericCheck

# Mirrors the input size caps from the MCP best-practice review: a rules
# list this large would already be a misuse of the tool, not a real profile.
_MAX_RULES = 100


class ContentRulesCheck(BaseGenericCheck):
    check_name = "content_rules"

    def __init__(
        self,
        required_terms: list[str] | None = None,
        banned_terms: list[str] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self.required_terms = (required_terms or [])[:_MAX_RULES]
        self.banned_terms = (banned_terms or [])[:_MAX_RULES]
        self.case_sensitive = case_sensitive

    def run(self, ctx: EvalContext, **kwargs: object) -> CheckResult:
        answer = ctx.answer if isinstance(ctx.answer, str) else ""
        haystack = answer if self.case_sensitive else answer.lower()

        missing_required = [
            term
            for term in self.required_terms
            if (term if self.case_sensitive else term.lower()) not in haystack
        ]
        found_banned = [
            term
            for term in self.banned_terms
            if (term if self.case_sensitive else term.lower()) in haystack
        ]

        if missing_required or found_banned:
            problems = []
            if missing_required:
                problems.append(f"missing required term(s): {missing_required}")
            if found_banned:
                problems.append(f"contains banned term(s): {found_banned}")
            return CheckResult(
                pattern_id=104,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message="; ".join(problems),
                detail={"missing_required": missing_required, "found_banned": found_banned},
            )

        return CheckResult(
            pattern_id=104,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message="All required terms present, no banned terms found",
            detail={"missing_required": [], "found_banned": []},
        )
