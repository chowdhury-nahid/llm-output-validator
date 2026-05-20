from __future__ import annotations

from ..corpus import DocumentCorpus
from ..models import LLMResponse
from ..report import CheckResult, CheckStatus
from . import BaseCheck

THRESHOLDS: dict[str, dict[str, float | int]] = {
    "high": {"min_citations": 2, "min_authority": 0.7},
    "medium": {"min_citations": 1, "min_authority": 0.4},
    "low": {"min_citations": 0, "min_authority": 0.0},
}


class ConfidenceCalibrationCheck(BaseCheck):
    pattern_id = 6
    check_name = "confidence_calibration"

    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        corpus: DocumentCorpus = kwargs["corpus"]  # type: ignore[assignment]

        declared = response.response.confidence
        citation_count = len(response.citations)
        authority = corpus.authority_score(response.citations)
        required = THRESHOLDS[declared]

        detail = {
            "declared_confidence": declared,
            "citation_count": citation_count,
            "authority_score": round(authority, 4),
            "required": required,
        }

        reasons: list[str] = []
        if citation_count < required["min_citations"]:
            reasons.append(f"needs {required['min_citations']} citation(s), got {citation_count}")
        if authority < required["min_authority"]:
            reasons.append(
                f"authority score {authority:.2f} below required {required['min_authority']}"
            )

        if reasons:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.WARN,
                severity="warning",
                message=f"Confidence '{declared}' not supported by evidence: {'; '.join(reasons)}",
                detail=detail,
            )

        return CheckResult(
            pattern_id=self.pattern_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message=f"Confidence '{declared}' calibration verified",
            detail=detail,
        )
