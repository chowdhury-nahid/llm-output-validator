# CLAUDE.md — llm-output-validator

## What this is
Deterministic verification layer for LLM outputs in compliance contexts. Flagship portfolio repo for the AI Verification Engineer repositioning. Currently private — must be reviewed and made public (tracked in life-ops career/001).

## Tech stack
- Python 3.11+, Hatchling build system
- Pydantic v2 for data models
- jsonschema for schema validation
- pytest + hypothesis for testing
- ruff for linting, mypy for type checking

## Project structure
- `src/llm_output_validator/` — main package
- `tests/` — test suite organized by verification pattern
- `examples/` — usage examples
- `docs/` — documentation
- `ARCHITECTURE.md` — system design

## Verification patterns (test files map 1:1)
1. Schema validation (`test_pattern1_schema`)
2. Citation verification (`test_pattern2_citation`)
3. Numeric accuracy (`test_pattern3_numeric`)
4. Temporal consistency (`test_pattern4_temporal`)
5. Jurisdictional compliance (`test_pattern5_jurisdictional`)
6. Confidence scoring (`test_pattern6_confidence`)
7. Injection detection (`test_pattern7_injection`)
8. Golden output comparison (`test_pattern8_golden`)

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
- This repo anchors all AI verification claims in the CV/LinkedIn — artifact must exist before any public claim
