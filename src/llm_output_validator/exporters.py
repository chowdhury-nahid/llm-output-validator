from __future__ import annotations

import json

from .report import CheckStatus, VerificationReport

_STATUS_SYMBOL = {
    CheckStatus.PASS: "✓",
    CheckStatus.WARN: "⚠",
    CheckStatus.FAIL: "✗",
}


class JsonReportExporter:
    @staticmethod
    def export(report: VerificationReport) -> str:
        return json.dumps(report.to_dict(), indent=2)


class TextReportExporter:
    @staticmethod
    def export(report: VerificationReport) -> str:
        lines: list[str] = [
            f"Verification Report — {report.status.value.upper()}",
            f"Response ID : {report.response_id or 'N/A'}",
            f"Elapsed     : {report.elapsed_ms:.1f}ms",
            "",
            f"{'#':<4} {'Check':<30} {'Status':<8} Message",
            "-" * 80,
        ]
        for c in report.checks:
            symbol = _STATUS_SYMBOL.get(c.status, "?")
            lines.append(
                f"{c.pattern_id:<4} {c.check_name:<30} {symbol} {c.status.value:<6}  {c.message}"
            )
        return "\n".join(lines)
