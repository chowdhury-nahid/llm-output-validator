from __future__ import annotations

from llm_output_validator import LLMResponse


class MockLLMClient:
    """Simulates an LLM client returning pre-programmed responses for testing."""

    def __init__(self, responses: dict[str, LLMResponse]) -> None:
        self._responses = responses

    def query(self, jurisdiction: str, **kwargs: object) -> LLMResponse:
        if jurisdiction not in self._responses:
            raise KeyError(f"No mock response configured for jurisdiction '{jurisdiction}'")
        return self._responses[jurisdiction]

    def add_response(self, jurisdiction: str, response: LLMResponse) -> None:
        self._responses[jurisdiction] = response
