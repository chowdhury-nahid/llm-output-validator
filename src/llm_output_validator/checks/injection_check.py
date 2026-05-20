from __future__ import annotations

import re

from ..models import LLMResponse
from ..report import CheckResult, CheckStatus
from . import BaseCheck

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*/?(?:system|user|assistant)\s*>", re.IGNORECASE),
    re.compile(r"\\u[0-9a-fA-F]{4}", re.IGNORECASE),
    re.compile(r"\x00"),
    re.compile(r"RLHF\s*override", re.IGNORECASE),
    re.compile(r"new\s+system\s+prompt", re.IGNORECASE),
]


class PromptInjectionCheck(BaseCheck):
    pattern_id = 7
    check_name = "prompt_injection_resilience"

    def __init__(self, extra_patterns: list[re.Pattern[str]] | None = None) -> None:
        self.patterns = INJECTION_PATTERNS + (extra_patterns or [])

    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        fields_to_scan: list[tuple[str, str]] = []

        if response.raw_prompt:
            fields_to_scan.append(("raw_prompt", response.raw_prompt))

        fields_to_scan.append(("jurisdiction", response.response.jurisdiction))
        fields_to_scan.append(("source_id", response.response.source_id))

        for claim in response.regulatory_claims:
            fields_to_scan.append(("claim_text", claim.claim_text))
            fields_to_scan.append(("claim_jurisdiction", claim.jurisdiction))

        for citation in response.citations:
            fields_to_scan.append(("document_id", citation.document_id))

        matches: list[dict[str, str]] = []
        for field_name, text in fields_to_scan:
            for pattern in self.patterns:
                m = pattern.search(text)
                if m:
                    snippet = text[max(0, m.start() - 20) : m.end() + 20]
                    matches.append(
                        {
                            "field": field_name,
                            "pattern": pattern.pattern,
                            "snippet": repr(snippet),
                        }
                    )

        if matches:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=f"Prompt injection pattern(s) detected in {len(matches)} field(s)",
                detail={"matches": matches},
            )

        return CheckResult(
            pattern_id=self.pattern_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message="No injection patterns detected",
            detail={"matches": []},
        )
