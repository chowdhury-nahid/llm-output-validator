"""MCP server exposing llm-output-validator as an AI response validator.

Exposes exactly one tool, ``llmval_validate_response``, per the 2026-09-24
via-negativa cut: the profile's own input schema already documents every
check and its parameters, so separate list_checks/list_profiles/get_profile
tools were removed rather than shipped. See ARCHITECTURE.md for the full
design and CLAUDE.md for the ship gates this file is built under.
"""

from __future__ import annotations

import sys
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover - exercised via missing-extra path
    raise ImportError(
        "The MCP server requires the 'mcp' extra. Install with: "
        "pip install 'llm-output-validator[mcp]'"
    ) from exc

from .evals.models import EvalContext
from .generic.profile import Profile, get_preset, list_preset_names
from .generic.runner import validate

mcp = FastMCP("llm_validator_mcp")

_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

# Mirrors the runner's own caps (generic/runner.py) — enforcing them again
# at the Pydantic input boundary means a malformed request is rejected by
# FastMCP's own schema validation before it ever reaches the runner, with a
# clearer error than the runner's internal BLOCK-report path produces.
_MAX_ANSWER_CHARS = 100_000
_MAX_CONTEXT_ITEMS = 50
_ContextItem = Annotated[str, Field(max_length=20_000)]


class ValidateResponseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_ANSWER_CHARS,
        description="The AI-generated text to validate, e.g. 'Paris is the capital of France.'",
    )
    question: str | None = Field(
        default=None,
        max_length=10_000,
        description="The prompt/question the answer responds to, if relevant to the checks "
        "(e.g. for the rag profile's faithfulness/relevancy evals).",
    )
    context: list[_ContextItem] = Field(
        default_factory=list,
        max_length=_MAX_CONTEXT_ITEMS,
        description="Retrieved source documents the answer should be grounded in, e.g. "
        "['Paris is the capital of France, with a population of over 2 million.']. "
        "Required for the rag profile's evals to mean anything.",
    )
    profile_name: str | None = Field(
        default=None,
        description="A built-in scope profile: 'minimal' (structural + safety checks: "
        "json_schema off by default, injection, pii) or 'rag' (adds faithfulness, "
        "relevancy and hallucination checks against the supplied context). "
        "Defaults to 'minimal' if neither this nor 'profile' is given.",
    )
    profile: Profile | None = Field(
        default=None,
        description="A fully custom inline scope profile, for callers who need more control "
        "than the built-in presets — e.g. {'enable_content_rules': true, "
        "'content_rules': {'banned_terms': ['guaranteed']}}. Takes precedence over "
        "profile_name if both are given.",
    )


class ValidateResponseOutput(BaseModel):
    """Typed mirror of ValidationReport.to_dict(), so FastMCP can generate
    an outputSchema and populate structuredContent automatically — a bare
    ``dict`` return annotation carries no shape for FastMCP to introspect."""

    decision: str
    profile: str
    elapsed_ms: float
    llm_tokens_used: int
    checks: list[dict]
    evals: dict


def _resolve_profile(input_: ValidateResponseInput) -> Profile:
    if input_.profile is not None:
        return input_.profile
    name = input_.profile_name or "minimal"
    try:
        return get_preset(name)
    except ValueError as exc:
        # Re-raised as ToolError so this reaches the caller as an isError
        # tool result (per MCP best practices), not a protocol-level error
        # or a raw traceback.
        valid = ", ".join(list_preset_names())
        raise ToolError(f"{exc} (valid: {valid})") from exc


@mcp.tool(name="llmval_validate_response", annotations=_READ_ONLY)
def llmval_validate_response(input: ValidateResponseInput) -> ValidateResponseOutput:
    """Validate an AI-generated response with no LLM calls (Tier 1, deterministic).

    Runs a scope profile's enabled checks (json_schema, injection, pii,
    content_rules) and lexical evals (faithfulness, relevancy, hallucination
    — word-overlap based, not model-judged) against the answer, and returns
    a structured pass/flag/block verdict with evidence for each failure.

    Use profile_name='minimal' for a plain answer, or 'rag' when the answer
    should be grounded in the supplied context. Pass a custom `profile` for
    finer control (e.g. enabling content_rules with your own banned terms).
    """
    profile = _resolve_profile(input)
    ctx = EvalContext(
        question=input.question or "",
        answer=input.answer,
        context=list(input.context),
    )
    report = validate(ctx, profile)
    return ValidateResponseOutput(**report.to_dict())


def main() -> None:
    """Entry point for the ``llm-validate-mcp`` console script.

    Runs over stdio only. Per the security guidance in ARCHITECTURE.md, this
    process must never write anything to stdout outside the MCP protocol
    itself — logging, if added later, goes to stderr.
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(0)
