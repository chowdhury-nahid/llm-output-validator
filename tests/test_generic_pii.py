"""Pass/catch tests for the generic pii check."""

from __future__ import annotations

from llm_output_validator.evals.models import EvalContext
from llm_output_validator.generic.pii_check import PiiCheck
from llm_output_validator.report import CheckStatus


def _ctx(answer: str, context: list[str] | None = None) -> EvalContext:
    return EvalContext(question="q", answer=answer, context=context or [])


def test_passes_on_clean_answer() -> None:
    result = PiiCheck().run(_ctx("The capital of France is Paris."))
    assert result.status == CheckStatus.PASS


def test_catches_email() -> None:
    result = PiiCheck().run(_ctx("Contact me at nahid@example.com for details."))
    assert result.status == CheckStatus.FAIL
    assert any(f["kind"] == "email" for f in result.detail["findings"])


def test_catches_valid_luhn_credit_card() -> None:
    # 4111111111111111 is a well-known Luhn-valid test card number.
    result = PiiCheck().run(_ctx("Card: 4111 1111 1111 1111"))
    assert result.status == CheckStatus.FAIL
    assert any(f["kind"] == "credit_card" for f in result.detail["findings"])


def test_does_not_flag_luhn_invalid_16_digit_number() -> None:
    # A 16-digit number that fails the Luhn checksum should not be flagged
    # as a credit card — this is the false-positive guard the Luhn check
    # exists for (e.g. an order/invoice number).
    non_card = "1234567812345678"
    assert not _luhn_valid_for_test(non_card)
    result = PiiCheck().run(_ctx(f"Order number: {non_card}"))
    assert not any(f["kind"] == "credit_card" for f in result.detail["findings"])


def _luhn_valid_for_test(digits: str) -> bool:
    from llm_output_validator.generic.pii_check import _luhn_valid

    return _luhn_valid(digits)


def test_catches_ipv4_address() -> None:
    result = PiiCheck().run(_ctx("The server is at 192.168.1.100."))
    assert result.status == CheckStatus.FAIL
    assert any(f["kind"] == "ipv4" for f in result.detail["findings"])


def test_catches_iban() -> None:
    result = PiiCheck().run(_ctx("Transfer to DE89370400440532013000."))
    assert result.status == CheckStatus.FAIL
    assert any(f["kind"] == "iban" for f in result.detail["findings"])


def test_catches_phone_number() -> None:
    result = PiiCheck().run(_ctx("Call me at +1-555-123-4567."))
    assert result.status == CheckStatus.FAIL
    assert any(f["kind"] == "phone" for f in result.detail["findings"])


def test_does_not_flag_short_numbers_as_phone() -> None:
    result = PiiCheck().run(_ctx("The answer is 42, and question 7 was skipped."))
    assert result.status == CheckStatus.PASS


def test_scans_context_blocks_too() -> None:
    result = PiiCheck().run(_ctx("Clean answer.", context=["Email: leak@example.com"]))
    assert result.status == CheckStatus.FAIL
    assert any(f["field"] == "context[0]" for f in result.detail["findings"])


def test_does_not_raise_on_non_string_context_entry() -> None:
    result = PiiCheck().run(EvalContext(question="q", answer="fine", context=[None]))  # type: ignore[list-item]
    assert result.status == CheckStatus.PASS


def test_luhn_helper_accepts_and_rejects_correctly() -> None:
    from llm_output_validator.generic.pii_check import _luhn_valid

    assert _luhn_valid("4111111111111111") is True
    assert _luhn_valid("4111111111111112") is False
