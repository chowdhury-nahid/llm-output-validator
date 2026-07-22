from __future__ import annotations

from ..consensus import ConsensusReference
from ..models import LLMResponse
from ..report import CheckResult, CheckStatus
from . import BaseCheck


class CrossModelConsensusCheck(BaseCheck):
    pattern_id = 9
    check_name = "cross_model_consensus"

    def __init__(self, references: list[ConsensusReference]) -> None:
        self._refs = {r.jurisdiction: r for r in references}

    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        jurisdiction = response.response.jurisdiction
        ref = self._refs.get(jurisdiction)

        if ref is None:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.WARN,
                severity="warning",
                message=f"No consensus reference data for jurisdiction '{jurisdiction}'",
                detail={"jurisdiction": jurisdiction},
            )

        response_claims = self._extract_claims(response)
        matched: list[dict[str, object]] = []
        diverged: list[dict[str, object]] = []

        for claim_key, consensus_claim in ref.consensus.items():
            if claim_key not in response_claims:
                continue

            response_value = response_claims[claim_key]
            if consensus_claim.matches(response_value):
                matched.append(
                    {
                        "claim_key": claim_key,
                        "value": response_value,
                        "consensus_value": consensus_claim.consensus_value,
                        "agreement": f"{consensus_claim.agreement_count}"
                        f"/{consensus_claim.total_sources}",
                    }
                )
            else:
                diverged.append(
                    {
                        "claim_key": claim_key,
                        "value": response_value,
                        "consensus_value": consensus_claim.consensus_value,
                        "agreement": f"{consensus_claim.agreement_count}"
                        f"/{consensus_claim.total_sources}",
                        "critical": consensus_claim.critical,
                    }
                )

        testable = len(matched) + len(diverged)
        total_consensus = len(ref.consensus)
        detail: dict[str, object] = {
            "jurisdiction": jurisdiction,
            "question_id": ref.question_id,
            "matched": matched,
            "diverged": diverged,
            "coverage": f"{testable}/{total_consensus}",
        }

        if not diverged:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                severity="info",
                message=(
                    f"All {testable} testable claim(s) match cross-model consensus "
                    f"for {jurisdiction}"
                ),
                detail=detail,
            )

        critical_diverged = [d for d in diverged if d["critical"]]
        if critical_diverged:
            labels = ", ".join(str(d["claim_key"]) for d in critical_diverged)
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=(f"Critical consensus divergence for {jurisdiction}: {labels}"),
                detail=detail,
            )

        labels = ", ".join(str(d["claim_key"]) for d in diverged)
        return CheckResult(
            pattern_id=self.pattern_id,
            check_name=self.check_name,
            status=CheckStatus.WARN,
            severity="warning",
            message=(f"Non-critical consensus divergence for {jurisdiction}: {labels}"),
            detail=detail,
        )

    @staticmethod
    def _extract_claims(response: LLMResponse) -> dict[str, float | str]:
        claims: dict[str, float | str] = {}
        r = response.response
        claims["rate"] = r.rate
        claims["jurisdiction"] = r.jurisdiction
        claims["confidence"] = r.confidence
        claims["source_id"] = r.source_id

        # Map common rate field names so consensus fixtures can match
        # against any of: kst_rate, trading_rate, main_rate, combined_rate, etc.
        # The response only has one rate — the consensus fixture determines
        # which claim key it maps to via the fixture's claim keys.
        return claims
