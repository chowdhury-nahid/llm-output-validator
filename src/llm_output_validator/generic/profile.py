"""Scope profiles: what a validation run checks, and how strict it is.

Per the 2026-09-24 via-negativa cut, v1 ships exactly two built-in presets
(``minimal``, ``rag``) plus support for an inline profile object passed by
the caller. No YAML, no operator-configured profile directory — that's
v1.1 scope if users ask for it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Mirrors the input size caps from the MCP best-practice review.
_MAX_RULE_TERMS = 100


class ContentRulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_terms: list[str] = Field(default_factory=list, max_length=_MAX_RULE_TERMS)
    banned_terms: list[str] = Field(default_factory=list, max_length=_MAX_RULE_TERMS)
    case_sensitive: bool = False


class JsonSchemaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: dict | None = Field(default=None, alias="schema")


class ThresholdOverride(BaseModel):
    """Mirrors evals.models.ThresholdConfig but as a plain, JSON-friendly
    input shape — the runner converts this to a real ThresholdConfig."""

    model_config = ConfigDict(extra="forbid")

    pass_above: float = Field(default=0.8, ge=0.0, le=1.0)
    block_below: float = Field(default=0.4, ge=0.0, le=1.0)


class Profile(BaseModel):
    """A scope profile: which checks and evals run, and their configuration.

    Built with Pydantic (not a plain dataclass) so it can be constructed
    directly from an inline JSON object passed through the MCP tool's input
    schema, with validation and clear error messages for free.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "custom"

    # Which of the 4 frozen v1 checks are enabled.
    enable_json_schema: bool = False  # off by default: most answers aren't JSON
    enable_injection: bool = True
    enable_pii: bool = True
    enable_content_rules: bool = False  # off by default: needs rules configured to be useful

    json_schema: JsonSchemaConfig = Field(default_factory=JsonSchemaConfig)
    content_rules: ContentRulesConfig = Field(default_factory=ContentRulesConfig)

    # Which lexical evals are enabled (no-LLM strategy only in v1 — no judge
    # is ever passed by the Tier 1 runner).
    enable_faithfulness: bool = False
    enable_relevancy: bool = False
    enable_hallucination: bool = False

    eval_threshold: ThresholdOverride = Field(default_factory=ThresholdOverride)


def _minimal_profile() -> Profile:
    """The lightest useful profile: structural + safety checks only."""
    return Profile(
        name="minimal",
        enable_json_schema=False,
        enable_injection=True,
        enable_pii=True,
        enable_content_rules=False,
        enable_faithfulness=False,
        enable_relevancy=False,
        enable_hallucination=False,
    )


def _rag_profile() -> Profile:
    """For answers grounded in retrieved context: adds the lexical evals
    that need ``ctx.context`` to mean something."""
    return Profile(
        name="rag",
        enable_json_schema=False,
        enable_injection=True,
        enable_pii=True,
        enable_content_rules=False,
        enable_faithfulness=True,
        enable_relevancy=True,
        enable_hallucination=True,
    )


# Built as a function-keyed registry rather than a module-level dict of
# Profile instances, so each call gets its own object — a shared mutable
# default would let one caller's changes leak into another's, the classic
# mutable-default bug class this checkpoint's review is watching for.
_PRESET_BUILDERS = {
    "minimal": _minimal_profile,
    "rag": _rag_profile,
}


def get_preset(name: str) -> Profile:
    """Look up a built-in preset by name.

    Raises ValueError (not KeyError) with the valid names listed, since this
    message is meant to reach an end user or calling LLM through the MCP
    tool's error path, not just a Python traceback.
    """
    builder = _PRESET_BUILDERS.get(name)
    if builder is None:
        valid = ", ".join(sorted(_PRESET_BUILDERS))
        raise ValueError(f"Unknown profile {name!r}. Valid built-in profiles: {valid}")
    return builder()


def list_preset_names() -> list[str]:
    return sorted(_PRESET_BUILDERS)
