# Security Policy

## Supported versions

This project is pre-1.0. Security fixes are applied to the current default branch only.

| Version | Supported |
|---|---:|
| Current default branch | Yes |
| Older snapshots and forks | No |

## Report a vulnerability privately

Use GitHub's private vulnerability-reporting flow:

1. Open the repository's **Security** tab.
2. Select **Advisories** → **Report a vulnerability**.
3. Describe the affected version/commit, impact, reproduction conditions, and suggested mitigation.
4. Remove credentials, private datasets, PII, and live exploit targets from the report.

Do not disclose the issue publicly until a fix and coordinated disclosure are ready. If private
reporting is unavailable, contact the maintainer through the
[pypi-ahmad GitHub profile](https://github.com/pypi-ahmad) without including exploit details in a
public issue.

## Response process

The maintainer will aim to acknowledge a report within seven days, validate severity and scope,
prepare a minimal fix, and coordinate disclosure. Timelines can vary because this is a research
project without a production support commitment.

## Security boundaries

The workbench is designed for one trusted local user:

- Streamlit binds to `127.0.0.1`.
- Credentials are read from local environment variables/`.env` and must not be entered into UI.
- Direct URL downloads reject local/private destinations, disable redirects, and apply time/size
  limits.
- Provider downloads stay under dedicated workspace directories; selected Hugging Face filenames
  are normalized before their workspace copy.
- MCP is limited to allowlisted read-only discovery tools.
- Downloaded datasets and model caches are gitignored.

It is **not** hardened for hostile multi-user hosting, public internet exposure, or production use.
TabFM's weight license independently prohibits production use.

## Operational guidance

- Use least-privilege read/fine-grained Hugging Face tokens.
- Rotate any token exposed in terminal output, screenshots, logs, commits, or issues.
- Do not process data you are not authorized to store locally.
- Treat provider/MCP metadata and remote table content as untrusted input.
- Keep uv dependencies and GitHub Actions pins reviewed and current.

## Out of scope

The following are not project vulnerabilities:

- model-quality, fairness, or calibration limitations documented by the model card;
- expected CPU slowness or GPU out-of-memory errors;
- attacks requiring a user to deliberately expose the loopback app through an external proxy;
- vulnerabilities in an unsupported modified fork without a reproducer against the current branch.

Responsible reports about dependency or upstream TabFM vulnerabilities are welcome when they show
impact on this workbench.
