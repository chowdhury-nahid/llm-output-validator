"""Tests for scope profiles: the two built-in presets and inline construction."""

from __future__ import annotations

import pytest

from llm_output_validator.generic.profile import (
    Profile,
    get_preset,
    list_preset_names,
)


def test_list_preset_names_returns_minimal_and_rag() -> None:
    assert list_preset_names() == ["minimal", "rag"]


def test_minimal_preset_has_no_evals_enabled() -> None:
    profile = get_preset("minimal")
    # json_schema is opt-in, not on by default: most answers are plain
    # text, not JSON, and this was a real bug caught by the runner tests
    # (a clean plain-text answer was blocked because "minimal" used to
    # enable json_schema unconditionally).
    assert profile.enable_json_schema is False
    assert profile.enable_injection is True
    assert profile.enable_pii is True
    assert profile.enable_faithfulness is False


def test_rag_preset_enables_lexical_evals() -> None:
    profile = get_preset("rag")
    assert profile.enable_faithfulness is True
    assert profile.enable_relevancy is True
    assert profile.enable_hallucination is True


def test_unknown_preset_raises_value_error_listing_valid_names() -> None:
    with pytest.raises(ValueError, match="minimal.*rag|rag.*minimal"):
        get_preset("nonexistent")


def test_presets_are_independent_objects_not_shared_mutable_state() -> None:
    # Bug-class guard: mutating one call's result must never affect another.
    a = get_preset("minimal")
    b = get_preset("minimal")
    a.content_rules.required_terms.append("mutated")
    assert b.content_rules.required_terms == []


def test_inline_profile_construction_from_dict() -> None:
    profile = Profile(
        name="custom",
        enable_json_schema=True,
        enable_content_rules=True,
        content_rules={"required_terms": ["disclaimer"], "banned_terms": ["guaranteed"]},
    )
    assert profile.content_rules.required_terms == ["disclaimer"]


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(Exception):  # noqa: PT011 - Pydantic ValidationError
        Profile(unknown_field="not allowed")  # type: ignore[call-arg]


def test_rule_terms_list_is_capped() -> None:
    with pytest.raises(Exception):  # noqa: PT011 - Pydantic ValidationError
        Profile(content_rules={"required_terms": [f"t{i}" for i in range(150)]})
