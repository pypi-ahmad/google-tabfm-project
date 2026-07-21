# TabFM Workbench Instructions

Global Codex instructions apply unless this file overrides them.

## Required skills

- Always discover and automatically use every required or clearly applicable available skill for each task. Do not wait for the user to name a skill.

## Project constraints

- This is a local research workbench. TabFM model weights must not be used commercially or in production.
- Use Python 3.12 and `uv`; keep `pyproject.toml` and `uv.lock` synchronized.
- Never create `requirements.txt`.
- Keep credentials in environment variables or `.env`; never expose them in UI, logs, tests, or commits.
- Run `uv run pytest -p no:cacheprovider`, `uv run ruff check .`, and `uv run mypy` before completion.
