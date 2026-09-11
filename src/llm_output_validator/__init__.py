from .consensus import ConsensusReference
from .corpus import DocumentCorpus
from .evals import (
    AnswerRelevancyEval,
    BaseEval,
    ContextPrecisionEval,
    ContextRecallEval,
    EvalContext,
    EvalDecision,
    EvalPipeline,
    EvalReport,
    EvalResult,
    EvalScore,
    FaithfulnessEval,
    HallucinationEval,
    LLMJudge,
    ThresholdConfig,
)
from .exceptions import CalibrationError, ComplianceError, HallucinationError, VerificationError
from .models import Citation, LLMResponse, RegulatoryCllaim, TaxRateResponse
from .range_table import RangeTable
from .report import CheckResult, CheckStatus, ReportStatus, VerificationReport
from .validator import OutputValidator

__version__ = "0.1.0"

__all__ = [
    # Deterministic verification
    "ConsensusReference",
    "Citation",
    "RegulatoryCllaim",
    "TaxRateResponse",
    "LLMResponse",
    "DocumentCorpus",
    "RangeTable",
    "CheckResult",
    "CheckStatus",
    "ReportStatus",
    "VerificationReport",
    "OutputValidator",
    "VerificationError",
    "HallucinationError",
    "CalibrationError",
    "ComplianceError",
    # Non-deterministic evaluation
    "AnswerRelevancyEval",
    "BaseEval",
    "ContextPrecisionEval",
    "ContextRecallEval",
    "EvalContext",
    "EvalDecision",
    "EvalPipeline",
    "EvalReport",
    "EvalResult",
    "EvalScore",
    "FaithfulnessEval",
    "HallucinationEval",
    "LLMJudge",
    "ThresholdConfig",
]
