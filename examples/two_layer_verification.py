"""Two-layer verification: deterministic checks + non-deterministic evals.

Demonstrates the composition pattern:
1. Run deterministic checks first (fast, cheap, no LLM calls)
2. If deterministic checks pass, run non-deterministic evals (semantic analysis)

The deterministic layer catches structural issues (missing fields, fabricated
citations, out-of-range numbers). The eval layer catches semantic issues
(unsupported claims, irrelevant answers, hallucinated facts).
"""

from llm_output_validator.evals import (
    AnswerRelevancyEval,
    EvalContext,
    EvalDecision,
    EvalPipeline,
    FaithfulnessEval,
    HallucinationEval,
    ThresholdConfig,
)

# Configure thresholds for a compliance-critical pipeline
strict_threshold = ThresholdConfig(pass_above=0.8, block_below=0.4)

# Build the evaluation pipeline
pipeline = EvalPipeline([
    FaithfulnessEval(threshold=strict_threshold),
    AnswerRelevancyEval(threshold=strict_threshold),
    HallucinationEval(threshold=strict_threshold),
])

# Simulate a well-grounded response
good_ctx = EvalContext(
    question="What is the corporate tax rate in Germany?",
    answer="The corporate tax rate in Germany is 15% at the federal level. "
           "A solidarity surcharge of 5.5% applies on top.",
    context=[
        "The corporate income tax rate in Germany is 15% at the federal level. "
        "A solidarity surcharge (Solidaritätszuschlag) of 5.5% is levied "
        "on top of the corporate income tax."
    ],
)

print("=== Well-grounded response ===")
report = pipeline.run(good_ctx)
print(f"Decision: {report.decision.value}")
print(f"Elapsed: {report.elapsed_ms:.1f}ms")
for r in report.results:
    print(f"  {r.eval_name}: score={r.score.value:.2f} → {r.decision.value}")

# Simulate a hallucinated response
bad_ctx = EvalContext(
    question="What is the corporate tax rate in Germany?",
    answer="The corporate tax rate in Germany is 42%. "
           "This was introduced by the Quantum Tax Reform of 2099.",
    context=[
        "The corporate income tax rate in Germany is 15% at the federal level."
    ],
)

print("\n=== Hallucinated response ===")
report = pipeline.run(bad_ctx)
print(f"Decision: {report.decision.value}")
for r in report.results:
    print(f"  {r.eval_name}: score={r.score.value:.2f} → {r.decision.value}")
    if r.claims:
        for claim in r.claims:
            verdict = claim.get("verdict", "unknown")
            text = claim.get("claim", claim.get("text", ""))
            print(f"    - [{verdict}] {text}")

# Show the LLMJudge protocol — any LLM client that implements evaluate(prompt) -> str works
print("\n=== LLMJudge protocol ===")
print("The LLMJudge protocol accepts any object with an evaluate(prompt: str) -> str method.")
print("Example with Anthropic:")
print("""
    import anthropic

    class ClaudeJudge:
        def __init__(self, model="claude-sonnet-4-6"):
            self.client = anthropic.Anthropic()
            self.model = model

        def evaluate(self, prompt: str) -> str:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

    # Use with any eval
    report = pipeline.run(ctx, judge=ClaudeJudge())
""")

if report.decision == EvalDecision.BLOCK:
    print("Result: Response BLOCKED — do not use in production.")
elif report.decision == EvalDecision.FLAG:
    print("Result: Response FLAGGED — human review recommended.")
else:
    print("Result: Response PASSED all evaluations.")
