# Pull Request

## Summary

Describe the user-visible change and why it is needed.

## Verification

List exact commands and results.

```text
uv run ruff check .
uv run mypy
uv run pytest -p no:cacheprovider
```

## Checklist

- [ ] The change is focused and follows existing module boundaries.
- [ ] New/changed behavior has public-contract tests.
- [ ] README/tutorials match user-visible behavior.
- [ ] `pyproject.toml` and `uv.lock` remain synchronized when dependencies change.
- [ ] No secrets, private data, model checkpoints, or generated outputs are included.
- [ ] TabFM is described as frozen in-context learning, not task-specific fine-tuning.
- [ ] The research-only, non-commercial/non-production license boundary is preserved.
- [ ] Network/file inputs retain explicit errors, time/size limits, and safe paths.

## Risk and residual limitations

Describe security, compatibility, performance, or model-quality risks that remain.

## Related issue

Link the issue, for example `Closes #123`.
