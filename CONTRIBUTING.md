# Contributing

Thanks for your interest in contributing to llm-output-validator.

## Development setup

```bash
git clone https://github.com/chowdhury-nahid/llm-output-validator.git
cd llm-output-validator
pip install -e ".[dev]"
```

## Running checks

```bash
pytest                # tests
pytest --cov          # tests with coverage
ruff check src/ tests/  # lint
ruff format --check src/ tests/  # format check
mypy src/             # type check
```

All checks must pass before submitting a PR. CI runs the same checks on Python 3.11 and 3.12.

## Adding a verification pattern

Each pattern lives in `src/llm_output_validator/checks/` with a corresponding test file in `tests/test_pattern{N}_*.py`. Follow the existing structure: one module per check, one test file per pattern, deterministic assertions only.

## Pull requests

- Keep PRs focused on a single change
- Include tests for new functionality
- Run the full check suite locally before opening a PR
