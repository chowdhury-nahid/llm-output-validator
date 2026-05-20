from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class ReportStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    pattern_id: int
    check_name: str
    status: CheckStatus
    severity: str  # "error" | "warning" | "info"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "check_name": self.check_name,
            "status": self.status.value,
            "severity": self.severity,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class VerificationReport:
    status: ReportStatus
    response_id: str | None
    checks: list[CheckResult] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def passed(self) -> bool:
        return self.status == ReportStatus.PASS

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    def warned_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.WARN]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "response_id": self.response_id,
            "elapsed_ms": self.elapsed_ms,
            "checks": [c.to_dict() for c in self.checks],
        }
