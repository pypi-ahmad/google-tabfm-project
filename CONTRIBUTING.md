# Contributing to TabFM Local Research Workbench

Thank you for improving the workbench. Contributions must preserve its **local, research-only,
non-commercial** scope and the license boundary between this Apache-2.0 project and TabFM's
non-commercial pretrained weights.

## Before opening a change

1. Search existing issues to avoid duplicates.
2. Use a bug, feature, or documentation issue form for nontrivial work.
3. Keep changes narrowly scoped; discuss architecture or dependency changes before implementation.
4. Never attach real credentials, proprietary tables, model checkpoints, or prediction outputs.

Security vulnerabilities belong in a private report under the repository's **Security** tab, not a
public issue. See [SECURITY.md](SECURITY.md).

## Development environment

The repository requires Python 3.12.10 and uv. Do not introduce `requirements.txt`, system-level
`pip`, or an additional environment resolver.

```powershell
git clone https://github.com/pypi-ahmad/google-tabfm-project.git
Set-Location google-tabfm-project
uv python install 3.12.10
uv sync --locked
```

The default development sync excludes multi-gigabyte TabFM checkpoints. For manual runtime work,
choose exactly one backend:

```powershell
uv sync --locked --extra cpu --extra integrations
# OR
uv sync --locked --extra cu130 --extra integrations
```

Copy `.env.example` to `.env` only for local runs. `.env` must remain untracked.

## Change principles

| Principle | Expectation |
|---|---|
| Small scope | Change only what the issue requires |
| Existing boundaries | Keep ingestion, providers, remote security, prediction, and UI modular |
| Public behavior | Add tests for new contracts and update tutorials/README |
| Network inputs | Apply timeouts, size limits, path normalization, and actionable errors |
| Secrets and data | Never log or render tokens, full tables, labels, or PII payloads |
| TabFM semantics | Call `fit()` context preparation, never fine-tuning |
| Licensing | Do not weaken the non-commercial/non-production gate or language |

## Tests and quality gates

Run all checks before submitting:

```powershell
uv run ruff check .
uv run mypy
uv run pytest -p no:cacheprovider
```

Tests must use deterministic fakes at the TabFM/provider boundaries. Do not make CI download model
weights or require Kaggle/Hugging Face credentials.

Documentation changes must also pass Markdownlint and link validation. Keep code fences labeled,
use relative links for repository files, and cite first-party sources for changing APIs or model
claims.

## Pull requests

A pull request should explain:

- what changed and why;
- how it was verified, with exact commands;
- user-visible or security impact;
- residual limitations;
- related issue(s).

Complete the pull request template. Keep unrelated formatting/refactoring out of the diff. New
dependencies require a clear capability justification and synchronized `pyproject.toml`/`uv.lock`.

## Commit guidance

Use imperative, atomic messages. Conventional prefixes are encouraged:

```text
docs: add provider authentication troubleshooting
fix: reject duplicate test columns before prediction
test: cover numeric classification probabilities
chore: pin documentation actions by commit SHA
```

Do not commit downloaded datasets, model caches, `.env`, browser artifacts, or generated outputs.

## Documentation sources

Framework/API instructions must follow current official documentation. TabFM-specific claims should
cite the pinned [Google Research source](https://github.com/google-research/tabfm/tree/cb6ba46b7ebc9a6581a81827e14e9c246202afb9),
the [official model card](https://huggingface.co/google/tabfm-1.0.0-pytorch), or the
[official weight license](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
