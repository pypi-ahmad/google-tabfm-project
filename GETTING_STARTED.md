# Getting Started

Clone it, set it up, run it — and why you'd want to.

## Why use this app?

You have a table (CSV, Parquet, or Excel) and you want predictions — a category
("will this customer churn?") or a number ("what will this cost?") — **without**
training a machine learning model yourself. This app runs Google's TabFM, a
model that makes predictions by reading your labeled examples as context at
prediction time, instead of being trained on them. That means:

- **No training pipeline to build.** Load data, pick a target column, get predictions in minutes.
- **Runs entirely on your own machine.** Nothing leaves your computer except the one-time model download and whatever data source you explicitly choose.
- **Free, non-commercial, research/evaluation use.** Good for prototyping, learning, and honest baselines — not for production decisions (see the license note below).

For the full concept explanation (in plain language and in theory) and a
feature-by-feature walkthrough, see [GUIDE.md](GUIDE.md). This file only covers
getting it running.

## 1. Clone it

```powershell
Set-Location D:\AI\Github          # or wherever you keep projects
git clone https://github.com/pypi-ahmad/google-tabfm-project.git
Set-Location google-tabfm-project
```

```bash
cd ~/projects
git clone https://github.com/pypi-ahmad/google-tabfm-project.git
cd google-tabfm-project
```

## 2. Set it up

This project uses [`uv`](https://docs.astral.sh/uv/) for everything — Python
version, virtual environment, and dependencies. Don't use system `pip` or a
manually created `venv`.

```powershell
# Install uv, if you don't already have it
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install the exact Python version this project is pinned to
uv python install 3.12.10

# Install dependencies — choose exactly ONE runtime:
uv sync --locked --extra cu130 --extra integrations   # NVIDIA GPU (CUDA 13)
# uv sync --locked --extra cpu --extra integrations   # CPU only (much slower, still works)
```

`integrations` adds optional Hugging Face / Kaggle / MCP dataset support — leave
it off if you only plan to use file uploads and direct URLs.

## 3. Configure

```powershell
Copy-Item .env.example .env
```

```bash
cp .env.example .env
```

Open `.env` and:

1. **Read the [TabFM weight license](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE)** — it's non-commercial, research/evaluation use only.
2. If your intended use is genuinely covered by that license, set:

   ```dotenv
   TABFM_ACCEPT_NON_COMMERCIAL_LICENSE=true
   ```
3. Leave `HF_TOKEN`, `KAGGLE_API_TOKEN`, etc. blank unless you plan to pull datasets from Hugging Face or Kaggle.

`.env` is gitignored — it never gets committed, and nothing in it is ever shown in the UI or logged.

## 4. Run it

```powershell
uv run --env-file .env streamlit run app.py
```

```bash
uv run --env-file .env streamlit run app.py
```

Open <http://127.0.0.1:8501>. That's it — the app binds only to your own
machine (`127.0.0.1`), so nothing else on your network can reach it.

## 5. Use it (the five-minute version)

1. **Data** — upload a CSV/Parquet/XLSX file.
2. **Model** — pick the column you want to predict, click **Load TabFM and prepare context**. First run downloads the model checkpoint (a few GB — be patient).
3. **Predictions** — click **Run batch prediction** (for rows with a blank target) or use **Single** to try one hand-typed row.
4. **EDA & Reports** — browse the data quality charts, download the auto-generated report bundle.
5. **History** — every run you've ever made is saved here, permanently, on your own disk.

For what each button actually does, and why, see the full
[feature-by-feature guide](GUIDE.md#5-the-full-feature-tour--every-page-every-button).

## Verify it's working (no GPU/model download needed)

```powershell
uv run pytest -p no:cacheprovider
uv run ruff check .
uv run mypy
```

These never download the TabFM checkpoint — they use lightweight fakes for
anything model-related.

## Something not working?

| Symptom | Likely fix |
|---|---|
| `uv sync` resolver conflict | You selected both `--extra cpu` and `--extra cu130` — pick exactly one |
| Model button stays disabled | Set `TABFM_ACCEPT_NON_COMMERCIAL_LICENSE=true` in `.env` and restart |
| `torch.cuda.is_available()` is `False` on a GPU machine | Re-sync `--extra cu130` and check `nvidia-smi` |
| First model load looks stuck | It's downloading a multi-gigabyte checkpoint — check your network, don't restart repeatedly |

Full troubleshooting tables: [docs/tutorial/02_installation.md](docs/tutorial/02_installation.md).

## Where to go next

| Want to... | Read |
|---|---|
| Understand what TabFM actually is and why `fit()` isn't training | [GUIDE.md § 2](GUIDE.md#2-the-idea-behind-tabfm-in-plain-language-then-in-theory) |
| Learn every feature in the app in depth | [GUIDE.md § 5](GUIDE.md#5-the-full-feature-tour--every-page-every-button) |
| See how the code implements each feature | [GUIDE.md § 6](GUIDE.md#6-under-the-hood-how-the-code-implements-each-feature) |
| Understand the security/privacy model | [GUIDE.md § 7](GUIDE.md#7-security-privacy-and-where-your-data-goes) / [SECURITY.md](SECURITY.md) |
| Get the deep math/theory (Fourier features, attention, RoPE, metrics) | [docs/tutorial/](docs/tutorial/01_introduction.md) |
| Contribute code | [CONTRIBUTING.md](CONTRIBUTING.md) |
