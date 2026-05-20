from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import LLMResponse
from ..report import CheckResult


class BaseCheck(ABC):
    pattern_id: int
    check_name: str

    @abstractmethod
    def run(self, response: LLMResponse, **kwargs: object) -> CheckResult:
        """Execute the check and return a CheckResult. Never raises."""
        ...
