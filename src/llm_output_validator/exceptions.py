from __future__ import annotations


class VerificationError(Exception):
    def __init__(self, check_name: str, detail: str) -> None:
        self.check_name = check_name
        self.detail = detail
        super().__init__(f"[{check_name}] {detail}")


class HallucinationError(VerificationError):
    pass


class CalibrationError(VerificationError):
    pass


class ComplianceError(VerificationError):
    pass
