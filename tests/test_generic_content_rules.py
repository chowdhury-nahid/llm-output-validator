"""Pass/catch tests for the generic content_rules check."""

from __future__ import annotations

from llm_output_validator.evals.models import EvalContext
from llm_output_validator.generic.content_rules_check import ContentRulesCheck
from llm_output_validator.report import CheckStatus


def _ctx(answer: str) -> EvalContext:
    return EvalContext(question="q", answer=answer)


def test_passes_with_no_rules_configured() -> None:
    result = ContentRulesCheck().run(_ctx("anything at all"))
    assert result.status == CheckStatus.PASS


def test_catches_missing_required_term() -> None:
    check = ContentRulesCheck(required_terms=["disclaimer"])
    result = check.run(_ctx("This is a plain answer with nothing extra."))
    assert result.status == CheckStatus.FAIL
    assert "disclaimer" in result.detail["missing_required"]


def test_passes_when_required_term_present() -> None:
    check = ContentRulesCheck(required_terms=["disclaimer"])
    result = check.run(_ctx("Please read this disclaimer before proceeding."))
    assert result.status == CheckStatus.PASS


def test_catches_banned_term() -> None:
    check = ContentRulesCheck(banned_terms=["guaranteed"])
    result = check.run(_ctx("This investment is guaranteed to succeed."))
    assert result.status == CheckStatus.FAIL
    assert "guaranteed" in result.detail["found_banned"]


def test_case_insensitive_by_default() -> None:
    check = ContentRulesCheck(banned_terms=["GUARANTEED"])
    result = check.run(_ctx("this is guaranteed."))
    assert result.status == CheckStatus.FAIL


def test_case_sensitive_when_configured() -> None:
    check = ContentRulesCheck(banned_terms=["GUARANTEED"], case_sensitive=True)
    result = check.run(_ctx("this is guaranteed."))
    assert result.status == CheckStatus.PASS  # lowercase doesn't match exact-case rule


def test_rules_list_is_capped_at_max_rules() -> None:
    many_terms = [f"term{i}" for i in range(150)]
    check = ContentRulesCheck(banned_terms=many_terms)
    assert len(check.banned_terms) == 100


def test_no_regex_metacharacters_are_interpreted() -> None:
    # Literal-only by design (2026-09-24 via-negativa cut): a banned term
    # containing regex metacharacters must be treated as a literal string,
    # not a pattern, and must not raise even if it would be invalid regex.
    check = ContentRulesCheck(banned_terms=["a(b["])
    result = check.run(_ctx("this contains a(b[ literally"))
    assert result.status == CheckStatus.FAIL


def test_does_not_raise_on_non_string_answer() -> None:
    result = ContentRulesCheck(required_terms=["x"]).run(
        EvalContext(question="q", answer=None)  # type: ignore[arg-type]
    )
    assert result.status == CheckStatus.FAIL
