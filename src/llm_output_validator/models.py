from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., min_length=1)
    page: int | None = Field(default=None, ge=1)
    excerpt: str | None = None


# NOTE: 'RegulatoryCllaim' preserves the spelling from the architecture contract (double-l).
class RegulatoryCllaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    jurisdiction: str = Field(..., min_length=2, max_length=10)
    claim_text: str
    effective_from: date | None = None


class TaxRateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    jurisdiction: str = Field(..., min_length=2, max_length=10)
    rate: float = Field(..., ge=0.0, le=1.0)
    effective_date: date
    source_id: str = Field(..., min_length=1)
    confidence: Literal["high", "medium", "low"]


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    response: TaxRateResponse
    citations: list[Citation] = Field(default_factory=list)
    regulatory_claims: list[RegulatoryCllaim] = Field(default_factory=list)
    raw_prompt: str | None = Field(
        default=None,
        description="Original prompt text, used for injection resilience check",
    )
