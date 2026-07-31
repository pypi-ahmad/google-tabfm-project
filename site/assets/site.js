
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
      `Serve this site over HTTP (e.g. <code>uv run python -m http.server</code>) rather than opening it directly as a file.</p>`;
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
