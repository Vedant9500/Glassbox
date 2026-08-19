# Security Policy

## Supported Versions

Security updates are currently provided for the latest state of `master`.

| Version | Supported |
| ------- | --------- |
| master  | yes       |

## Reporting a Vulnerability

Please do not open public issues for security vulnerabilities.

Use one of the following private channels:

- GitHub Security Advisories: `Security` tab in this repository
- Direct contact with the repository owner on GitHub for coordinated disclosure

When reporting, include:

- A clear description of the issue
- Reproduction steps or proof of concept
- Impact assessment
- Suggested mitigation, if available

You can expect an initial acknowledgement within 5 business days.

## Scope

Security reports are especially helpful for:

- Native C++ extension and pybind boundary handling
- Unsafe input handling in formula parsing, simplification, or evaluation paths
- Dependency vulnerabilities with practical impact on this project

## Trusted Local Artifacts

Glassbox model checkpoints are treated as trusted local artifacts. PyTorch
loaders first attempt weights-only loading. If an older pickle-backed checkpoint
is required, fallback loading requires BOTH an explicit opt-in via the
`GLASSBOX_ALLOW_PICKLE_CHECKPOINT=1` environment variable AND a repository-local
`models/` or `artifacts/` path. Do not load checkpoints from untrusted users or
remote locations without converting them to a weights-only format first.

Thank you for helping keep Glassbox and its users safe.
