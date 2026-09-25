"""Generic prompt-injection check: scans any answer/context text for
injection and jailbreak markers.

Reuses ``INJECTION_PATTERNS`` from ``checks/injection_check.py`` rather than
maintaining a second copy of the pattern list.
"""

from __future__ import annotations

import re

from ..checks.injection_check import INJECTION_PATTERNS
from ..evals.models import EvalContext
from ..report import CheckResult, CheckStatus
from . import BaseGenericCheck

# Evidence snippets are capped per the MCP best-practice review (2026-09-23):
# keep tool output bounded regardless of how large the matched field is.
_SNIPPET_CONTEXT_CHARS = 20
_MAX_SNIPPET_LEN = 200


class InjectionCheck(BaseGenericCheck):
    check_name = "injection"

    def __init__(self, extra_patterns: list[re.Pattern[str]] | None = None) -> None:
        self.patterns = list(INJECTION_PATTERNS) + (extra_patterns or [])

    def run(self, ctx: EvalContext, **kwargs: object) -> CheckResult:
        fields_to_scan: list[tuple[str, str]] = [("answer", ctx.answer)]
        for i, block in enumerate(ctx.context):
            fields_to_scan.append((f"context[{i}]", block))

        matches: list[dict[str, str]] = []
        for field_name, text in fields_to_scan:
            # Defensive: never let a non-string field crash the scan — a
            # malformed context entry becomes "no match found", not a raise.
            if not isinstance(text, str):
                continue
            for pattern in self.patterns:
                m = pattern.search(text)
                if m:
                    start = max(0, m.start() - _SNIPPET_CONTEXT_CHARS)
                    end = m.end() + _SNIPPET_CONTEXT_CHARS
                    snippet = text[start:end][:_MAX_SNIPPET_LEN]
                    matches.append(
                        {
                            "field": field_name,
                            "pattern": pattern.pattern,
                            "snippet": repr(snippet),
                        }
                    )

        if matches:
            return CheckResult(
                pattern_id=102,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=f"Prompt injection pattern(s) detected in {len(matches)} field(s)",
                detail={"matches": matches[:50]},
            )

        return CheckResult(
            pattern_id=102,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message="No injection patterns detected",
            detail={"matches": []},
        )
