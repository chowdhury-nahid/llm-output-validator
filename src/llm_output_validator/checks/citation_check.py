from __future__ import annotations

from ..corpus import DocumentCorpus
from ..models import LLMResponse
from ..report import CheckResult, CheckStatus
from . import BaseCheck


class CitationGroundingCheck(BaseCheck):
    pattern_id = 2
    check_name = "citation_grounding"

    def __init__(self, authority_threshold: float = 0.5) -> None:
        self.authority_threshold = authority_threshold

    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        corpus: DocumentCorpus = kwargs["corpus"]  # type: ignore[assignment]

        if not response.citations:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message="No citations provided; response is ungrounded",
                detail={"missing_doc_ids": [], "authority_score": 0.0, "threshold": self.authority_threshold},
            )

        missing = corpus.missing_citations(response.citations)
        authority = corpus.authority_score(response.citations)

        if missing:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=f"Hallucinated citation(s): {', '.join(missing)}",
                detail={"missing_doc_ids": missing, "authority_score": authority, "threshold": self.authority_threshold},
            )

        if authority < self.authority_threshold:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.WARN,
                severity="warning",
                message=f"All citations grounded but authority score {authority:.2f} below threshold {self.authority_threshold}",
                detail={"missing_doc_ids": [], "authority_score": authority, "threshold": self.authority_threshold},
            )

        return CheckResult(
            pattern_id=self.pattern_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message=f"All citations grounded (authority={authority:.2f})",
            detail={"missing_doc_ids": [], "authority_score": authority, "threshold": self.authority_threshold},
        )
