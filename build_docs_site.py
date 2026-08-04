"""Generate the static multipage HTMX documentation site from this repo's Markdown files.

Each page is a real, standalone HTML file (crawlable, linkable, works with
JS disabled) that fetches and renders its source .md file client-side.
htmx (`hx-boost`) upgrades navigation between pages to feel like a single-page
app, without a build step for content: edit a .md file, refresh the browser.

Usage:
    uv run python build_docs_site.py

Serve it with:
    uv run python -m http.server
    -> http://localhost:8000/site/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    description: str
    source: str  # path to the .md file, relative to ROOT


PAGES: list[Page] = [
    Page("index", "Home", "Project overview and quick start", "README.md"),
    Page(
        "getting-started",
        "Getting Started",
        "Clone, set up, and run the app",
        "GETTING_STARTED.md",
    ),
    Page(
        "guide",
        "Complete Guide",
        "Every feature, the theory, the code, and the security model",
        "GUIDE.md",
    ),
    Page(
        "architecture",
        "Architecture",
        "Package boundaries and security decisions",
        "docs/architecture.md",
    ),
    Page(
        "tutorial-01-introduction",
        "Tutorial 01 — Introduction",
        "TabFM foundations: from GBDTs to in-context learning",
        "docs/tutorial/01_introduction.md",
    ),
    Page(
        "tutorial-02-installation",
        "Tutorial 02 — Installation",
        "Environments and credentials",
        "docs/tutorial/02_installation.md",
    ),
    Page(
        "tutorial-03-data-handling",
        "Tutorial 03 — Data Handling",
        "Dataset ingestion and schema rules",
        "docs/tutorial/03_data_handling.md",
    ),
    Page(
        "tutorial-04-tabfm-mastery",
        "Tutorial 04 — TabFM Mastery",
        "Context, prediction, and evaluation",
        "docs/tutorial/04_tabfm_mastery.md",
    ),
    Page("security", "Security Policy", "Reporting vulnerabilities and security boundaries", "SECURITY.md"),
    Page("contributing", "Contributing", "How to contribute to this project", "CONTRIBUTING.md"),
    Page("code-of-conduct", "Code of Conduct", "Community standards", "CODE_OF_CONDUCT.md"),
    Page(
        "third-party-notices",
        "Third-Party Notices",
        "Licenses of bundled and referenced third-party work",
        "THIRD_PARTY_NOTICES.md",
    ),
    Page("agents", "Agent Instructions", "Instructions for AI coding agents working in this repo", "AGENTS.md"),
    Page("onboarding", "Team Onboarding", "Onboarding guide for new teammates", "ONBOARDING.md"),
]

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · TabFM Workbench Docs</title>
<meta name="description" content="{description}">
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.3.3" integrity="sha384-aJ9rL4k6lF+91guGvUFVSkpIcge7Zd9EiI4TQDLoK9kFaFJgKHgjEXVvG/qA5COj" crossorigin="anonymous"></script>
<script src="https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js" integrity="sha384-ujb1lZYygJmzgSwoxRggbCHcjc0rB2XoQrxeTUQyRjrOnlCoYta87iKBWq3EsdM2" crossorigin="anonymous"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js" integrity="sha384-RH2xi4eIQ/gjtbs9fUXM68sLSi99C7ZWBRX1vDrVv6GQXRibxXLbwO2NGZB74MbU" crossorigin="anonymous"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github-dark.min.css" integrity="sha384-wH75j6z1lH97ZOpMOInqhgKzFkAInZPPSPlZpYKYTOqsaizPvhQZmAtLcPKXpLyH" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.18.1/dist/katex.min.js" integrity="sha384-ycJ6GAwiS15LoUPipwJOrWTvkUHl/YqELValBwI5I4awP1EeEQJYarj+w85ntcz7" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.18.1/dist/contrib/auto-render.min.js" integrity="sha384-bjyGPfbij8/NDKJhSGZNP/khQVgtHUE5exjm4Ydllo42FwIgYsdLO2lXGmRBf5Mz" crossorigin="anonymous"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.18.1/dist/katex.min.css" integrity="sha384-1vdNCNel6Tx/NQa8IR1mGOGKsbGreCkOPfbtPPnUURJ5Tu2PRVfQ/7KLZC+Pi1p1" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js" integrity="sha384-/TQbtLCAerC3jgaim+N78RZSDYV7ryeoBCVqTuzRrFec2akfBkHS7ACQ3PQhvMVi" crossorigin="anonymous"></script>
<link rel="stylesheet" href="assets/site.css">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen" hx-boost="true">
<div class="flex min-h-screen">
  <nav id="site-nav" class="w-72 shrink-0 border-r border-slate-800 bg-slate-900/60 p-4 overflow-y-auto">
    <div class="mb-4">
      <a href="index.html" class="text-lg font-semibold text-emerald-400">TabFM Workbench Docs</a>
    </div>
    <ul class="space-y-1 text-sm">
{nav_items}
    </ul>
  </nav>
  <main class="flex-1 min-w-0">
    <article id="content" class="prose prose-invert max-w-3xl mx-auto px-6 py-10"
              data-md-src="{md_src}">
      <p class="text-slate-400">Loading {title}…</p>
    </article>
  </main>
</div>
<script src="assets/site.js"></script>
</body>
</html>
"""


def build_nav(active_slug: str) -> str:
    items = []
    for page in PAGES:
        classes = "block rounded px-2 py-1 hover:bg-slate-800"
        if page.slug == active_slug:
            classes += " bg-emerald-500/10 text-emerald-400 font-medium"
        href = f"{page.slug}.html"
        items.append(f'      <li><a class="{classes}" href="{href}">{page.title}</a></li>')
    return "\n".join(items)


def md_src_for(page: Page) -> str:
    # site/*.html sits one directory below the repo root.
    return f"../{page.source}"


SITE_CSS = """
:root { color-scheme: dark; }
body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
#content h1, #content h2, #content h3 { color: #34d399; }
#content h1 { font-size: 2rem; margin: 1.5rem 0 1rem; }
#content h2 { font-size: 1.5rem; margin: 2rem 0 0.75rem; border-top: 1px solid #1e293b; padding-top: 1.5rem; }
#content h3 { font-size: 1.15rem; margin: 1.5rem 0 0.5rem; }
#content p, #content li { line-height: 1.7; }
#content a { color: #38bdf8; text-decoration: underline; }
#content table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
#content th, #content td { border: 1px solid #334155; padding: 0.4rem 0.6rem; text-align: left; }
#content th { background: #1e293b; }
#content code { background: #1e293b; padding: 0.1rem 0.35rem; border-radius: 0.25rem; font-size: 0.9em; }
#content pre code { background: none; padding: 0; }
#content pre { padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }
#content blockquote { border-left: 3px solid #34d399; padding-left: 1rem; color: #cbd5e1;
  margin: 1rem 0; }
"""

SITE_JS = """
// Renders the Markdown file referenced by #content[data-md-src] into HTML.
// Runs on first load and again after every htmx-boosted page swap, since
// hx-boost replaces <body> via AJAX rather than a full page navigation.
async function renderMarkdownPage() {
  const el = document.getElementById("content");
  if (!el) return;
  const src = el.getAttribute("data-md-src");
  if (!src) return;
  try {
    const response = await fetch(src);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const raw = await response.text();
    el.innerHTML = marked.parse(raw, { gfm: true, breaks: false });
  } catch (error) {
    el.innerHTML = `<p class="text-red-400">Could not load ${src}: ${error.message}. ` +
      `Serve this site over HTTP (e.g. <code>uv run python -m http.server</code>) rather than
      opening it directly as a file.</p>`;
    return;
  }
  if (window.hljs) {
    el.querySelectorAll("pre code").forEach((block) => window.hljs.highlightElement(block));
  }
  if (window.renderMathInElement) {
    window.renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
    });
  }
}

document.addEventListener("DOMContentLoaded", renderMarkdownPage);
document.body.addEventListener("htmx:afterSettle", renderMarkdownPage);
"""


def main() -> None:
    SITE.mkdir(exist_ok=True)
    assets = SITE / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "site.css").write_text(SITE_CSS, encoding="utf-8")
    (assets / "site.js").write_text(SITE_JS, encoding="utf-8")

    for page in PAGES:
        source_path = ROOT / page.source
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source for page {page.slug!r}: {source_path}")
        html = PAGE_TEMPLATE.format(
            title=page.title,
            description=page.description,
            nav_items=build_nav(page.slug),
            md_src=md_src_for(page),
        )
        (SITE / f"{page.slug}.html").write_text(html, encoding="utf-8")

    print(f"Generated {len(PAGES)} pages in {SITE}")


if __name__ == "__main__":
    main()
"""Build the static documentation site."""

# Generated HTML/CSS/JavaScript literals intentionally exceed the source line
# limit; wrapping them would reduce readability without changing output.
# ruff: noqa: E501
