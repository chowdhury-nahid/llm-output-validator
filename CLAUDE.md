# CLAUDE.md — llm-output-validator

## What this is
Deterministic verification layer for LLM outputs in compliance contexts. Nine composable verification patterns plus a non-deterministic evaluation layer with five RAGAS-style metrics. Also exposes a general-purpose (schema-agnostic) AI response validator as an MCP server — see "MCP server" below.

## Tech stack
- Python 3.11+, Hatchling build system
- Pydantic v2 for data models
- jsonschema for schema validation
- pytest + hypothesis for testing
- ruff for linting, mypy for type checking

## Project structure
- `src/llm_output_validator/` — main package
  - `checks/` — the 9 tax-domain deterministic patterns
  - `evals/` — the 5 lexical/LLM-judge evals
  - `generic/` — schema-agnostic checks, scope profiles, runner (MCP server's Tier 1)
  - `server.py` — the MCP server itself (`llmval_validate_response`)
- `tests/` — test suite organized by verification pattern
- `examples/` — usage examples
- `docs/` — documentation
- `ARCHITECTURE.md` — system design

## MCP server
`pip install "llm-output-validator[mcp]"` adds the `llm-validate-mcp` entry point. One tool, `llmval_validate_response`, validates any AI response (not just this repo's tax domain) with no LLM calls (Tier 1). Two built-in scope profiles: `minimal`, `rag`. Full design, including the planned (not built) Tier 2/3, is in ARCHITECTURE.md's "Validation MCP server" section. Self-tests: `tests/test_mcp_server.py` (in-memory client), plus a pass/catch test pair per generic check in `tests/test_generic_*.py`.

## Verification patterns (test files map 1:1)
1. Schema validation (`test_pattern1_schema`)
2. Citation verification (`test_pattern2_citation`)
3. Numeric accuracy (`test_pattern3_numeric`)
4. Temporal consistency (`test_pattern4_temporal`)
5. Jurisdictional compliance (`test_pattern5_jurisdictional`)
6. Confidence scoring (`test_pattern6_confidence`)
7. Injection detection (`test_pattern7_injection`)
8. Golden output comparison (`test_pattern8_golden`)
9. Cross-model consensus (`test_pattern9_consensus`)

## Commands
```bash
pytest                    # run all tests
pytest --cov             # with coverage
ruff check .             # lint
mypy src/                # type check
pip install -e ".[dev]"  # install with dev dependencies
```

## Rules
- No fabricated metrics — if a number isn't measured, don't claim it
- Incremental commits — no single large dump (timestamps visible to reviewers)
- No claim without a working artifact — every public statement about this project must be backed by code
