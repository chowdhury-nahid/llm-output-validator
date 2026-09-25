"""Pass/catch tests for the generic json_schema check.

Per the checkpoint bug-class list in CLAUDE.md's ship gates: every check
needs a catch test that is actually red before the check exists, and every
error path (malformed answer, malformed configured schema) must return a
CheckResult rather than raise.
"""

from __future__ import annotations

from llm_output_validator.evals.models import EvalContext
from llm_output_validator.generic.json_schema_check import JsonSchemaCheck
from llm_output_validator.report import CheckStatus


def _ctx(answer: str) -> EvalContext:
    return EvalContext(question="q", answer=answer)


def test_passes_on_valid_json_no_schema_configured() -> None:
    result = JsonSchemaCheck().run(_ctx('{"a": 1}'))
    assert result.status == CheckStatus.PASS


def test_catches_malformed_json() -> None:
    result = JsonSchemaCheck().run(_ctx("{not json"))
    assert result.status == CheckStatus.FAIL
    assert "not valid JSON" in result.message


def test_catches_json_that_is_not_an_object_or_array_but_still_valid_json() -> None:
    # A bare number or string is valid JSON; this should still pass when no
    # schema is configured (regression guard for over-tight parsing).
    result = JsonSchemaCheck().run(_ctx("42"))
    assert result.status == CheckStatus.PASS


def test_passes_when_answer_matches_configured_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    result = JsonSchemaCheck(schema=schema).run(_ctx('{"name": "Nahid"}'))
    assert result.status == CheckStatus.PASS


def test_catches_answer_that_violates_configured_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    result = JsonSchemaCheck(schema=schema).run(_ctx('{"age": 5}'))
    assert result.status == CheckStatus.FAIL
    assert result.detail["errors"]


def test_catches_invalid_configured_schema_without_raising() -> None:
    # The schema itself is broken (not the answer) — this must not raise,
    # per BaseGenericCheck's "never raises" contract.
    bad_schema = {"type": "not-a-real-type"}
    result = JsonSchemaCheck(schema=bad_schema).run(_ctx('{"a": 1}'))
    assert result.status == CheckStatus.FAIL
    assert "schema is itself invalid" in result.message


def test_does_not_raise_on_non_string_answer_type() -> None:
    # Defensive: EvalContext.answer is typed as str, but a caller could still
    # pass something else through duck typing (e.g. from untrusted JSON-RPC
    # input before Pydantic validation runs at the tool layer). The check
    # must degrade to a FAIL result, not crash the server.
    class NotAString:
        def __repr__(self) -> str:
            return "<NotAString>"

    result = JsonSchemaCheck().run(EvalContext(question="q", answer=NotAString()))  # type: ignore[arg-type]
    assert result.status == CheckStatus.FAIL
