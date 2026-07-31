# TabFM Local Research Workbench

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![uv](https://img.shields.io/badge/dependencies-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![CI](https://github.com/pypi-ahmad/google-tabfm-project/actions/workflows/ci.yml/badge.svg)](https://github.com/pypi-ahmad/google-tabfm-project/actions/workflows/ci.yml)
[![Source license: Apache-2.0](https://img.shields.io/badge/source-Apache--2.0-blue)](LICENSE)
[![Weights: non-commercial](https://img.shields.io/badge/weights-non--commercial-critical)](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE)

A local-first Streamlit interface for zero-shot tabular classification and regression with
[Google Research TabFM](https://github.com/google-research/tabfm). Load tables from local files,
URLs, Kaggle, or Hugging Face; prepare labeled rows as in-context examples; then run batch or
single-row predictions without dataset-specific weight training.

> [!WARNING]
> **Research and evaluation only.** This repository's original source and documentation are
> Apache-2.0 licensed, but TabFM pretrained weights use the
> [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).
> The weights prohibit commercial and production use. The app blocks model loading until you
> explicitly acknowledge those terms.

## Why this workbench?

| Capability | Behavior |
|---|---|
| Mixed tabular data | Numerical, categorical, Boolean, and datetime-aware preprocessing |
| Multiple sources | CSV, Parquet, XLSX, direct HTTPS, Kaggle, Hugging Face, and MCP discovery |
| Two tasks | Classification with 2–10 classes and numeric regression |
| In-context adaptation | `fit()` prepares encoders and labeled context; pretrained weights remain frozen |
| Prediction modes | Blank-target rows, separate test files, and typed manual cases |
| Evaluation | Full classification/regression diagnostics when held-out labels are available |
| EDA and reports | Bounded interactive EDA plus automatic HTML/PDF/CSV/JSON report bundles |
| Durable history | Permanent local run index with downloadable bundles, 10 runs per page |
| Local-first security | Loopback binding, environment-only secrets, bounded URL downloads, local artifacts |
| Reproducibility | Python 3.12.10, uv lockfile, deterministic eight-member ensemble |

## How it works

```mermaid
flowchart LR
    A[CSV / Parquet / XLSX] --> D[Active DataFrame]
    B[HTTPS / Kaggle / HF] --> D
    C[MCP discovery] -. provider reference .-> B
    D --> E[Labeled context + test rows]
    E --> F[TabFM preprocessing and frozen ICL model]
    F --> G[Class probabilities or regression values]
    G --> H[Metrics and bounded EDA]
    H --> I[Automatic report bundle and local history]
```

TabFM reframes tabular prediction as in-context learning: labeled rows and new test rows form one
inference problem for a frozen pretrained model. See the
[Google Research overview](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
and [official PyTorch model card](https://huggingface.co/google/tabfm-1.0.0-pytorch).

## Quick start

### 1. Clone into a local Windows 11 workspace

Install `uv` with its official Windows installer, then verify that the command is available. If
your environment requires script review, inspect the installer URL before piping it to PowerShell.

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

Clone and install the pinned Python runtime and dependencies:

```powershell
Set-Location D:\AI\Github
git clone https://github.com/pypi-ahmad/google-tabfm-project.git
Set-Location D:\AI\Github\google-tabfm-project
uv python install 3.12.10

# Choose exactly one runtime.
uv sync --locked --extra cu130 --extra integrations  # NVIDIA CUDA 13
# uv sync --locked --extra cpu --extra integrations  # CPU (much slower)
```

If you clone elsewhere, use that actual path in `Set-Location`. On macOS/Linux, use `cd` and the
CPU sync command; CUDA installation is platform/driver dependent. CPU inference can be very slow.

### 2. Configure local settings

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

Review the model-weight license. Only for a permitted use, change this local setting:

```dotenv
TABFM_ACCEPT_NON_COMMERCIAL_LICENSE=true
```

Optional provider credentials are read from `.env`/environment only. Leave unused values blank:

```dotenv
HF_TOKEN=
KAGGLE_API_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
```

Never commit `.env`; it is ignored by Git.

### 3. Run

```powershell
uv run --env-file .env streamlit run app.py
```

Open `http://127.0.0.1:8501` if the browser does not launch automatically.

### 4. Complete the five-page workflow

1. **Data** — load one or more tables and select the active dataset.
2. **Model** — choose the target/task and prepare labeled context. The first load downloads the
   appropriate checkpoint; weights remain frozen.
3. **Predictions** — use **Batch** for blank-target rows or a separate test file, or **Single** for
   one typed case. Submit a prediction and inspect results and held-out metrics when labels exist.
4. **EDA & Reports** — inspect data quality and bounded charts, then download the report bundle
   automatically generated after every successful prediction run.
5. **History** — revisit saved run metadata and download any still-available bundle.

The prediction ZIP contains exactly `report.html`, `report.pdf`, `predictions.csv`, and
`metrics.json`. History is local and persistent; see [Local state and privacy](#local-state-and-privacy).

## Guides

| Document | Best for |
|---|---|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Clone, install, configure, and run the app in five minutes, plus why you'd use it |
| [GUIDE.md](GUIDE.md) | The complete guide — business + technical: every feature explained in depth, the AI theory in plain language, the code implementation, and the security model, all in one file |

## Zero-to-Master tutorial

Open the [Interactive HTML edition](index.html) for searchable chapters, installation tracks,
rendered equations and diagrams, theme controls, and copyable code. Serve the repository with
`uv run python -m http.server` for the complete CDN-enhanced experience.

| Chapter | Outcome |
|---|---|
| [01 — Introduction](docs/tutorial/01_introduction.md) | Understand TabFM, GBDTs, ICL, architecture equations, limits, and feasibility |
| [02 — Installation](docs/tutorial/02_installation.md) | Configure uv/Conda, CPU/CUDA, `.env`, Kaggle, Hugging Face, and MCP safely |
| [03 — Data Handling](docs/tutorial/03_data_handling.md) | Import local/remote datasets, validate schemas, and separate context from test rows |
| [04 — TabFM Mastery](docs/tutorial/04_tabfm_mastery.md) | Prepare context, predict, interpret metrics, diagnose failures, and evaluate responsibly |

## Runtime limits

| Constraint | Workbench setting |
|---|---:|
| Classification classes | 2–10 |
| Feature columns | 1–500 |
| Context rows per ensemble member | Up to 5,000 |
| Ensemble members | 8 |
| Upload/download size | 500 MB by default |
| Checkpoint disk size | Approximately 6.6 GB per task |
| Model use | Non-commercial, non-production only |

Memory depends on context rows, features, test rows, ensemble state, and device. Start small and
increase deliberately. Changing the active data, target, task, or context invalidates prepared
prediction state.

## Security model

- Streamlit binds to `127.0.0.1`, not all network interfaces.
- Credentials stay in environment variables or `.env` and are never rendered or logged.
- Direct imports require HTTPS by default, reject private/local destinations and embedded
  credentials, disable redirects, and enforce time/byte limits.
- Provider downloads stay under dedicated workspace directories; selected Hugging Face filenames
  are normalized before their workspace copy.
- MCP endpoints are optional and restricted to allowlisted read-only discovery tools.
- Raw datasets, labels, predictions, and secrets are not written to application logs.

See [architecture and security decisions](docs/architecture.md) and
[the security policy](SECURITY.md).

## Local state and privacy

`TABFM_HISTORY_DIR` configures permanent local history storage and defaults to
`data/sessions/history`. It contains `history.sqlite3` metadata and external ZIP bundles under
`bundles/`. There is no automatic eviction: datasets, predictions, targets, metrics, and report
content can remain on disk until you clear history or remove the directory. Do not use sensitive
data unless that persistence is acceptable and the workstation is appropriately protected.

- **Clear loaded datasets** removes all loaded tables and their in-memory model/prediction state;
  it does not delete provider downloads or permanent history.
- **Start new task** resets target, prepared context, prediction inputs/results, and current report
  references while retaining loaded datasets and permanent history.
- **Clear history** requires confirmation, permanently deletes SQLite-indexed metadata first, then
  attempts best-effort report-bundle removal; it does not clear loaded datasets or provider
  downloads. A locked ZIP can remain orphaned. The UI reports cleanup warnings; after closing
  processes that hold the files, manually remove each warned ZIP or the history directory.

History is newest-first and paginated at 10 runs per page. A `failed` run means predictions
completed but report persistence failed. An `unavailable` run has malformed metadata or a missing,
corrupt, or unsafe bundle; its metadata remains visible but the bundle cannot be downloaded.

## Evaluation and reports

Classification diagnostics include evaluated rows, accuracy, balanced accuracy, macro
precision/recall/F1, MCC, majority-baseline accuracy and lift, confusion/per-class tables, and—when
compatible probabilities exist—log loss and ROC-AUC/ROC data. Regression diagnostics include
evaluated rows, MAE, RMSE, median absolute error, mean-baseline MAE/RMSE and MAE lift, R², explained
variance, residuals, actual-versus-predicted data, and error quantiles. Undefined metrics are
reported as warnings rather than fabricated values.

EDA summaries use the full table, while chart samples are deterministic and capped at 10,000 rows;
scatter plots use at most 5,000 rows, correlations at most 20 numeric columns, and categorical
charts show the top 20 values. Report generation depends on Altair/Vega conversion and ReportLab:
this adds local rendering dependencies but produces self-contained HTML and PDF artifacts.

## Repository structure

```text
.
├── app.py                       # Streamlit entry point
├── app_pages/                   # Data, Model, Predictions, EDA & Reports, History pages
├── src/tabfm_workbench/
│   ├── config.py                # Validated environment settings and license gate
│   ├── loader.py                # CSV/Parquet/XLSX loading and row partitioning
│   ├── remote.py                # Bounded HTTPS retrieval
│   ├── integrations.py          # Kaggle, Hugging Face, and MCP adapters
│   ├── predictor.py             # TabFM preparation, alignment, metrics, prediction
│   ├── analytics.py             # Bounded EDA and evaluation diagnostics
│   ├── reports.py               # Report bundles and durable local history
│   └── ui.py                    # Page views and session orchestration
├── docs/
│   ├── architecture.md
│   └── tutorial/                # Four-part Zero-to-Master guide
├── data/downloads/               # Local provider/manual datasets (gitignored)
├── tests/                       # Unit and Streamlit smoke tests using deterministic fakes
├── pyproject.toml
└── uv.lock
```

## Development

Model checkpoints are not downloaded by the automated suite.

```powershell
uv sync --locked
uv run ruff check .
uv run mypy
uv run pytest -p no:cacheprovider
```

Documentation checks run in CI with Markdownlint and Lychee. Contributions must not include
credentials, private datasets, generated predictions, or downloaded checkpoints.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete workflow.

## Status and limitations

- TabFM is not an officially supported Google product.
- This repository is a local research workbench, not a production service.
- Task-specific fine-tuning, checkpoint writing, and derivative-model distribution are unsupported.
- Direct URL DNS validation cannot fully prevent DNS rebinding; use trusted sources only.
- Full model integration tests are omitted because each task checkpoint is approximately 6.6 GB.
- Performance, calibration, and fairness must be evaluated on representative held-out data.

## Citation

Use [CITATION.cff](CITATION.cff) to cite this workbench. Cite TabFM separately using the attribution
provided in the [official model card](https://huggingface.co/google/tabfm-1.0.0-pytorch#citation).

## Contributing and conduct

Contributions are welcome within the research-only scope. Read
[CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and
[SECURITY.md](SECURITY.md) before opening an issue or pull request.

## License

Original workbench source and documentation: [Apache License 2.0](LICENSE).

Third-party TabFM source and pretrained weights retain their own terms. In particular, the weights
are governed by the TabFM Non-Commercial License v1.0. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Nothing in this repository relicenses or grants
additional rights to those weights.
