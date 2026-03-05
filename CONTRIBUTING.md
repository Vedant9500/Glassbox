# Contributing to Glassbox

Thanks for your interest in improving Glassbox.

## Ways to Contribute

- Report bugs and edge cases
- Suggest features and research directions
- Improve docs and examples
- Submit code changes with tests

## Development Setup

1. Fork the repository and create a feature branch from `master`.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional: build the C++ extension in place.

```bash
cd glassbox/sr/cpp
python setup.py build_ext --inplace
cd ../../..
```

## Running Tests and Checks

Run the test suite before opening a pull request.

```bash
pytest tests -q
```

Recommended smoke checks for symbolic regression behavior:

```bash
python scripts/benchmark_suite.py --tier 2 --device cpu --quiet
python scripts/benchmark_feynman_easy.py --device cpu --sample 500
```

## Coding Guidelines

- Keep pull requests focused and reviewable.
- Add or update tests for behavior changes.
- Update docs when CLI flags, workflows, or defaults change.
- Prefer clear names and small functions over deeply nested logic.
- Keep C++ and Python interfaces synchronized when changing bridge code.

## Pull Request Process

1. Open a pull request with a clear title and problem statement.
2. Link related issues using `Closes #<issue-number>` when applicable.
3. Include test evidence (commands and short results) in the PR description.
4. Call out risk areas, especially fast-path default changes or simplification behavior.

A maintainer may request changes before merge.

## Reporting Bugs

Use the bug issue template and include:

- Minimal reproducible example
- Expected vs actual behavior
- Environment details (OS, Python version, CPU or GPU)
- Logs or traceback output

## Code of Conduct

This project follows the guidelines in `CODE_OF_CONDUCT.md`.
By participating, you agree to uphold those standards.
