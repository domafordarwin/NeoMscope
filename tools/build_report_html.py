"""Render a Markdown progress report as a self-contained HTML file.

Wraps the rendered body in a single HTML5 document with embedded CSS
(no external assets) so it can be opened by double-click on Windows
or shared as a single file.

CLI:
    python tools/build_report_html.py docs/03-analysis/<report>.md
    # writes <report>.html next to the input

    python tools/build_report_html.py <input.md> --output <output.html>
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import sys
from datetime import datetime
from pathlib import Path

import markdown

CSS = """
:root {
    color-scheme: light dark;
    --bg: #ffffff;
    --fg: #1f2328;
    --fg-muted: #57606a;
    --border: #d0d7de;
    --code-bg: #f6f8fa;
    --link: #0969da;
    --accent: #2563eb;
    --table-stripe: #f6f8fa;
    --quote-border: #d0d7de;
    --quote-bg: #f6f8fa33;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0d1117;
        --fg: #e6edf3;
        --fg-muted: #9198a1;
        --border: #30363d;
        --code-bg: #161b22;
        --link: #4493f8;
        --accent: #58a6ff;
        --table-stripe: #161b22;
        --quote-border: #30363d;
        --quote-bg: #161b2299;
    }
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Pretendard",
                 "Noto Sans KR", system-ui, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}
.container {
    max-width: 980px;
    margin: 0 auto;
    padding: 32px 28px 96px;
}
h1, h2, h3, h4 {
    font-weight: 600;
    margin-top: 28px;
    margin-bottom: 14px;
    line-height: 1.25;
}
h1 {
    font-size: 2em;
    border-bottom: 2px solid var(--border);
    padding-bottom: 12px;
    margin-top: 0;
}
h2 {
    font-size: 1.5em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-top: 36px;
}
h3 { font-size: 1.2em; }
h4 { font-size: 1.05em; color: var(--fg-muted); }
p { margin: 12px 0; }
a {
    color: var(--link);
    text-decoration: none;
}
a:hover { text-decoration: underline; }
strong { font-weight: 600; }
ul, ol { padding-left: 28px; }
li { margin: 4px 0; }
blockquote {
    margin: 14px 0;
    padding: 8px 14px;
    border-left: 4px solid var(--quote-border);
    background: var(--quote-bg);
    color: var(--fg-muted);
}
code {
    font-family: ui-monospace, "SFMono-Regular", "JetBrains Mono", Consolas, monospace;
    font-size: 0.92em;
    padding: 2px 6px;
    background: var(--code-bg);
    border-radius: 6px;
}
pre {
    background: var(--code-bg);
    padding: 14px 16px;
    border-radius: 8px;
    overflow-x: auto;
    border: 1px solid var(--border);
    line-height: 1.45;
}
pre code {
    background: transparent;
    padding: 0;
    border-radius: 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    font-size: 0.95em;
}
th, td {
    border: 1px solid var(--border);
    padding: 8px 12px;
    text-align: left;
    vertical-align: top;
}
th { background: var(--table-stripe); font-weight: 600; }
tbody tr:nth-child(even) td { background: var(--table-stripe); }
hr {
    border: 0;
    border-top: 1px solid var(--border);
    margin: 28px 0;
}
.report-meta {
    color: var(--fg-muted);
    font-size: 0.9em;
    border: 1px solid var(--border);
    padding: 14px 18px;
    border-radius: 8px;
    margin: 0 0 28px;
    background: var(--code-bg);
}
.report-meta .label {
    font-weight: 600;
    color: var(--fg);
    margin-right: 8px;
}
@media print {
    body { background: white; color: black; }
    .container { max-width: none; padding: 0; }
    a { color: black; }
    pre, code { border: 1px solid #ddd; }
}
"""

EXTENSIONS = [
    "extra",          # tables + fenced code + footnotes + def_list
    "sane_lists",
    "smarty",
    "toc",
    "nl2br",
    "admonition",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}


def embed_local_images(html: str, base_dir: Path) -> str:
    """Replace `<img src="local/path">` with `<img src="data:image/...;base64,...">`.

    Skips http(s):// and data: URLs. Resolves relative paths against base_dir.
    Silently skips images that don't exist (so the HTML still renders).
    """
    pattern = re.compile(r'<img\s+([^>]*?)src="([^"]+)"([^>]*?)>', re.IGNORECASE)

    def replace(match: re.Match) -> str:
        before, src, after = match.group(1), match.group(2), match.group(3)
        if src.startswith(("http://", "https://", "data:", "//")):
            return match.group(0)

        img_path = (base_dir / src).resolve()
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            return match.group(0)

        mime = mimetypes.guess_type(str(img_path))[0] or "application/octet-stream"
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f'<img {before}src="data:{mime};base64,{b64}"{after}>'

    return pattern.sub(replace, html)


def render(
    md_text: str,
    title: str,
    source_path: Path,
    embed_images_from: Path | None = None,
) -> str:
    """Render markdown to a complete HTML document."""
    md = markdown.Markdown(
        extensions=EXTENSIONS,
        extension_configs={
            "toc": {"permalink": False, "anchorlink": True},
        },
        output_format="html5",
    )
    body = md.convert(md_text)
    if embed_images_from is not None:
        body = embed_local_images(body, embed_images_from)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="NeoMscope tools/build_report_html.py">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
<div class="report-meta">
    <div><span class="label">Source:</span><code>{source_path}</code></div>
    <div><span class="label">Generated:</span>{generated_at}</div>
    <div><span class="label">Tool:</span>tools/build_report_html.py</div>
</div>
{body}
</div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", type=Path, help="Markdown source file")
    p.add_argument("--output", type=Path, help="HTML output path (default: <input>.html)")
    p.add_argument("--title", default=None, help="Document title (default: derived from filename)")
    args = p.parse_args(argv)

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    md_text = args.input.read_text(encoding="utf-8")
    out_path = args.output or args.input.with_suffix(".html")
    title = args.title or args.input.stem.replace("-", " ").title()

    html = render(md_text, title=title, source_path=args.input.name)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
