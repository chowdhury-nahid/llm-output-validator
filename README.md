# llm-output-validator

[![CI](https://github.com/chowdhury-nahid/llm-output-validator/actions/workflows/ci.yml/badge.svg)](https://github.com/chowdhury-nahid/llm-output-validator/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

LLMs produce plausible text, not verified facts. For most applications that is acceptable. For systems where outputs drive financial, legal, or compliance decisions, it is not.

This library defines a **deterministic verification layer** — a set of composable checks that wrap LLM outputs in contracts a test suite can enforce, without making the test suite non-deterministic itself.

---

## The problem with string assertions

The obvious approach to testing LLM outputs is:

```python
assert "19%" in response.text
assert "Germany" in response.text
```

These pass. They also pass when the response is factually wrong in every other respect. And they fail when the model correctly renders `19%` as `0.19`, or describes Germany as an out-of-scope reference rather than the target jurisdiction.

String assertions on free-text LLM outputs check vocabulary, not correctness. This library checks contracts.

---

## How it works

Three verification pillars compose into a single pass:

```
┌─────────────────────────────────────────────────────┐
│                   LLM Response                      │
└────────────────────┬────────────────────────────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│   Schema   │ │Hallucination│ │ Compliance │
│ Validation │ │ Detection  │ │  Testing   │
└─────┬──────┘ └─────┬──────┘ └─────┬──────┘
      │               │              │
      └───────────────┼──────────────┘
                      ▼
           ┌─────────────────────┐
           │  VerificationReport │
           │  pass / warn / fail │
           └─────────────────────┘
```

Schema validation runs first and fails fast. If the response is structurally invalid, there is nothing to run semantic checks against. All remaining checks run and collect results — callers see every problem in one report, not just the first.

---

## Nine verification patterns

| Pattern | Check | Catches |
|---------|-------|---------|
| 1 | Structural contract validation | Missing fields, wrong types, values outside declared ranges |
| 2 | Citation grounding | Hallucinated document IDs, citations outside the retrieval corpus |
| 3 | Numeric boundary assertion | Rates from the wrong year, wrong jurisdiction applied to correct one |
| 4 | Temporal consistency | Stale rule citations, regulations amended before the query date |
| 5 | Jurisdictional claim scoping | Legislation from one jurisdiction cited in a response about another |
| 6 | Confidence calibration | High confidence asserted with low-authority or absent citations |
| 7 | Prompt injection resilience | Structural corruption from injected instructions, jurisdiction overrides |
| 8 | Golden fixture regression | Regressions from prompt changes, model upgrades, index updates |
| 9 | Cross-model consensus | Factual claims that contradict agreement across multiple independent AI models |

None of these checks require knowing the correct answer. All of them are deterministic.

---

## Quick start

```python
from llm_output_validator import (
    Citation, DocumentCorpus, LLMResponse,
    OutputValidator, RangeTable, TaxRateResponse,
)
from datetime import date

corpus = DocumentCorpus.from_dict({
    "IRS-2023-001": 1.0,       # primary legislation — highest authority
    "CA-FTB-2023-TAX": 0.9,
    "COMMENTARY-2023": 0.3,    # commentary — low authority
})

range_table = RangeTable.from_dict({
    "US-CA": {"lo": 0.07, "hi": 0.145},
})

validator = OutputValidator(corpus=corpus, range_table=range_table)

response = LLMResponse(
    response=TaxRateResponse(
        jurisdiction="US-CA",
        rate=0.0925,
        effective_date=date(2023, 7, 1),
        source_id="IRS-2023-001",
        confidence="high",
    ),
    citations=[
        Citation(document_id="IRS-2023-001"),
        Citation(document_id="CA-FTB-2023-TAX"),
    ],
    raw_prompt="What is the California state income tax rate for 2023?",
)

report = validator.validate(response)
print(report.status)          # ReportStatus.PASS
print(report.passed())        # True
```

### From raw JSON (e.g. directly from an LLM client)

```python
import json

raw = json.loads(llm_client.complete(prompt))
report = validator.validate_raw(raw)

if not report.passed():
    for check in report.failed_checks():
        print(f"[{check.check_name}] {check.message}")
```

### CLI

```bash
llm-validate response.json \
  --corpus corpus.json \
  --range-table ranges.json \
  --format text
```

Exit code `0` on pass or warn, `1` on fail — pipeline-friendly by default.

---

## Verification report

```
Verification Report — PASS
Response ID : US-CA
Elapsed     : 0.1ms

#    Check                          Status   Message
--------------------------------------------------------------------------------
1    schema_validation              ✓ pass    Schema validation passed
2    citation_grounding             ✓ pass    All citations grounded (authority=0.97)
3    numeric_boundary               ✓ pass    Rate 0.0925 within reference range [0.07, 0.145] for US-CA
4    temporal_consistency           ✓ pass    Temporal consistency verified for US-CA (effective: 2023-07-01)
5    jurisdictional_scoping         ✓ pass    All regulatory claims scoped to US-CA
6    confidence_calibration         ✓ pass    Confidence 'high' calibration verified
7    prompt_injection_resilience    ✓ pass    No injection patterns detected
```

---

## What this layer does not solve

A verification layer narrows the space of valid outputs. It does not guarantee correctness within that space.

- A response that passes all nine patterns may still give wrong advice
- If the reference corpus or range table is wrong, the layer will pass wrong outputs confidently
- The injection resilience pattern tests against known patterns — novel attacks are outside the tested envelope

The verification layer is a necessary condition for safe LLM output in compliance contexts. It is not sufficient on its own.

---

## Use as an MCP server

The checks above are tied to the tax-domain `TaxRateResponse` schema. For validating *any* AI response — not just this repo's own domain — install the `mcp` extra and run the bundled MCP server:

```bash
pip install "llm-output-validator[mcp]"
claude mcp add llm-validator -- llm-validate-mcp
```

It exposes one tool, `llmval_validate_response`, which runs a no-LLM (Tier 1) scope profile against any answer and returns a structured pass/flag/block verdict:

```python
{
    "answer": "It was completed in 1920, stands 500 meters tall, and cost 50000 francs.",
    "question": "When was it completed and how tall is it?",
    "context": ["The tower was completed in 1889 and stands 330 meters tall."],
    "profile_name": "rag",
}
```

```json
{
  "decision": "flag",
  "profile": "rag",
  "elapsed_ms": 0.3,
  "llm_tokens_used": 0,
  "checks": [
    {"check_name": "injection", "status": "pass", "detail": {"matches": []}},
    {"check_name": "pii", "status": "pass", "detail": {"findings": []}}
  ],
  "evals": {
    "decision": "flag",
    "eval_count": 3,
    "evals": [
      {
        "eval_name": "hallucination_detection",
        "score": 0.6,
        "decision": "flag",
        "reasoning": "0/1 statements flagged as hallucinated, entity grounding=0.00 (strategy: lexical)"
      }
    ]
  }
}
```

This is real output from `examples/mcp_validate.py`, not a fabricated sample — run the script yourself to reproduce it. Note the honest limit: the lexical strategy verifies *numbers, dates and currencies* against context (which is why the wrong year and height here get flagged), not place names or other prose facts — a plain factual swap like "Paris" → "Berlin" passes this tier undetected. That gap is exactly what an LLM-judge (Tier 2, not yet built) is for.

Two built-in scope profiles ship in v1:

| Profile | What it runs |
|---|---|
| `minimal` | Structural + safety checks: injection, pii (json_schema is opt-in, since most answers aren't JSON) |
| `rag` | Adds the lexical faithfulness/relevancy/hallucination evals (word-overlap based — no LLM call) against supplied context |

Pass a custom `profile` object instead of `profile_name` for finer control — see `examples/mcp_validate.py`. `llm_tokens_used` is always `0` in this Tier: this is the deterministic, no-LLM layer. An LLM-judge tier and plugin evaluators (Presidio, DeepEval) are planned but not yet built — see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Installation

```bash
pip install -e ".[dev]"   # development
pip install .              # library only
```

**Requires Python 3.11+**

Dependencies: `pydantic>=2.7`, `jsonschema>=4.22`

---

## Project structure

```
src/llm_output_validator/
├── checks/          # one module per verification pattern
├── schemas/         # JSON Schema for raw response validation
├── models.py        # Pydantic response models
├── corpus.py        # DocumentCorpus — citation grounding
├── range_table.py   # RangeTable — numeric boundary data
├── validator.py     # OutputValidator — orchestrates all checks
├── report.py        # VerificationReport, CheckResult
├── exporters.py     # JSON and text report formatters
└── cli.py           # llm-validate entrypoint

tests/
├── test_pattern{1-9}_*.py   # one test file per pattern
├── test_properties.py       # Hypothesis property-based fuzzing
└── fixtures/
    ├── golden/              # regression fixtures (US-CA, EU DE VAT)
    └── consensus/           # cross-model consensus reference data
```

---

## Architecture

The design rationale, full pattern descriptions, and discussion of failure modes are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Regulatory context

Deterministic verification has a specific role under the EU AI Act: it is compliance
infrastructure, not a regulated AI system.

- [`docs/eu-ai-act-context.md`](docs/eu-ai-act-context.md) — how this library maps to Art. 12, 14, and 15 (engineering perspective)
- [`docs/eu-ai-act-risk-classification.md`](docs/eu-ai-act-risk-classification.md) — full risk classification guide for enterprise tax and financial AI: enforcement timeline, Art. 5 elimination, Annex III Category 5 analysis, Art. 50 transparency, and GPAI deployer obligations (8 sections, practitioner reference)
