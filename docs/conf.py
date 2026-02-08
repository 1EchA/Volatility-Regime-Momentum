"""
Sphinx configuration for Volatility-Regime-Momentum.

This repo keeps docs as Markdown files in `docs/`. We use MyST to render them.
"""

from __future__ import annotations

import datetime

project = "Volatility-Regime-Momentum"
author = "1EchA"
copyright = f"{datetime.date.today().year}, {author}"

extensions = [
    "myst_parser",
]

# Allow both `.rst` and `.md` sources (Markdown is primary in this repo).
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "sphinx_rtd_theme"

# Keep it minimal: no custom templates/static assets required.
templates_path: list[str] = []
html_static_path: list[str] = []

# Better anchors for Markdown headings.
myst_heading_anchors = 3
