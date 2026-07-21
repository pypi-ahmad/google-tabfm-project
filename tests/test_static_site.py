from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.id_values: list[str] = []
        self.classes: list[set[str]] = []
        self.tags: list[str] = []
        self.scripts: list[str] = []
        self.links: list[str] = []
        self._script_parts: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        self.tags.append(tag)
        if element_id := attributes.get("id"):
            self.ids.add(element_id)
            self.id_values.append(element_id)
        self.classes.append(set((attributes.get("class") or "").split()))
        if tag == "script":
            source = attributes.get("src")
            if source:
                self.scripts.append(source)
            else:
                self._script_parts = []
        if tag == "link" and (href := attributes.get("href")):
            self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script_parts is not None:
            self.scripts.append("".join(self._script_parts))
            self._script_parts = None

    def handle_data(self, data: str) -> None:
        if self._script_parts is not None:
            self._script_parts.append(data)


def parse_site() -> tuple[str, SiteParser]:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    parser = SiteParser()
    parser.feed(html)
    return html, parser


def test_static_site_exposes_the_tutorial_contract() -> None:
    html, parser = parse_site()

    assert html.count("<h1") == 1
    assert {"introduction", "installation", "data-handling", "tabfm-mastery"} <= parser.ids
    assert len(parser.id_values) == len(parser.ids)
    assert {"theme-toggle", "mobile-menu-button", "heading-search", "install-tabs"} <= parser.ids
    assert sum("mermaid" in classes for classes in parser.classes) == 13
    assert parser.tags.count("table") >= 30
    assert parser.tags.count("code") >= 34
    assert "$body$" not in html


def test_static_site_pins_rendering_dependencies_and_interactions() -> None:
    html, parser = parse_site()
    assets = "\n".join(parser.scripts + parser.links)

    assert "@tailwindcss/browser@4.3.3" in assets
    assert "highlight.js/11.11.1" in assets
    assert "mermaid@11.16.0" in assets
    assert "katex@0.18.1" in assets
    assert "localStorage" in html
    assert "aria-live" in html
    assert "prefers-reduced-motion" in html
    assert "navigator.clipboard" in html


def test_readme_links_to_the_interactive_edition() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "[Interactive HTML edition](index.html)" in readme
