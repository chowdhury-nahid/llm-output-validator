"""Generic, schema-agnostic checks that validate any AI response text.

Unlike ``checks/`` (which validates the tax-domain ``LLMResponse`` model),
these checks operate on ``evals.models.EvalContext`` — just a question,
answer, context list and metadata — so they work on any response, not one
tied to a specific domain schema. This is the Tier 1 (no-LLM) layer of the
validation MCP server; see ARCHITECTURE.md for the full three-tier design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..evals.models import EvalContext
from ..report import CheckResult


class BaseGenericCheck(ABC):
    """Abstract base for generic, domain-agnostic checks.

    Mirrors ``checks.BaseCheck``'s contract (a check never raises; it always
    returns a ``CheckResult``) but operates on ``EvalContext`` instead of the
    tax-domain ``LLMResponse``, so it composes with the eval layer and works
    on any response.
    """

    check_name: str

    @abstractmethod
    def run(self, ctx: EvalContext, **kwargs: object) -> CheckResult:
        """Execute the check and return a CheckResult. Never raises."""
        ...
