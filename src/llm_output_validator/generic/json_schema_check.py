"""Generic JSON Schema check: does the answer parse as JSON and validate
against a schema the caller supplies?

Reuses ``jsonschema`` (already a dependency of this package via
``checks/schema_check.py``) rather than adding a new dependency.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from ..evals.models import EvalContext
from ..report import CheckResult, CheckStatus
from . import BaseGenericCheck


class JsonSchemaCheck(BaseGenericCheck):
    check_name = "json_schema"

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        """``schema`` is optional: with none configured, the check only
        confirms the answer parses as JSON (still useful on its own), and
        never fails for lacking a schema to validate against.
        """
        self.schema = schema

    def run(self, ctx: EvalContext, **kwargs: object) -> CheckResult:
        # Bug-class check: a malformed configured schema must never raise —
        # it becomes a FAIL result the caller can act on, per BaseGenericCheck's
        # "never raises" contract.
        if self.schema is not None:
            try:
                jsonschema.Draft202012Validator.check_schema(self.schema)
            except jsonschema.SchemaError as exc:
                return CheckResult(
                    pattern_id=101,
                    check_name=self.check_name,
                    status=CheckStatus.FAIL,
                    severity="error",
                    message=f"Configured schema is itself invalid: {exc.message}",
                    detail={"schema_error_path": list(exc.path)},
                )

        try:
            parsed = json.loads(ctx.answer)
        except (TypeError, ValueError) as exc:
            # ctx.answer itself may not be a str/bytes (a caller could bypass
            # EvalContext's dataclass typing at runtime), so building the
            # preview must not raise either.
            preview = repr(ctx.answer)[:200]
            return CheckResult(
                pattern_id=101,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=f"Answer is not valid JSON: {exc}",
                detail={"answer_preview": preview},
            )

        if self.schema is None:
            return CheckResult(
                pattern_id=101,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                severity="info",
                message="Answer parses as JSON (no schema configured to validate against)",
            )

        validator = jsonschema.Draft202012Validator(self.schema)
        errors = [
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in validator.iter_errors(parsed)
        ]

        if errors:
            return CheckResult(
                pattern_id=101,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                severity="error",
                message=f"Answer does not match the configured schema ({len(errors)} error(s))",
                detail={"errors": errors[:50]},  # cap detail size, mirrors input size caps
            )

        return CheckResult(
            pattern_id=101,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            severity="info",
            message="Answer is valid JSON and matches the configured schema",
        )
