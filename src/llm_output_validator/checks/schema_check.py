from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import jsonschema

from ..models import LLMResponse
from ..report import CheckResult, CheckStatus
from . import BaseCheck

_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "llm_response_schema.json"
_SCHEMA: dict[str, Any] = json.loads(_SCHEMA_PATH.read_text())
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


class SchemaValidationCheck(BaseCheck):
    pattern_id = 1
    check_name = "schema_validation"

    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        reference_date: date | None = kwargs.get("reference_date")  # type: ignore[assignment]
        today = reference_date or date.today()

        errors: list[str] = []

        if response.response.effective_date > today:
            errors.append(f"effective_date {response.response.effective_date} is in the future")

        if not errors:
            return CheckResult(
                pattern_id=self.pattern_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                severity="info",
                message="Schema validation passed",
            )

        return CheckResult(
            pattern_id=self.pattern_id,
            check_name=self.check_name,
            status=CheckStatus.FAIL,
            severity="error",
            message="; ".join(errors),
            detail={"errors": errors},
        )

    @staticmethod
    def validate_raw(raw: dict[str, Any]) -> tuple[bool, list[str]]:
        errors = [e.message for e in _VALIDATOR.iter_errors(raw)]
        return (len(errors) == 0, errors)
