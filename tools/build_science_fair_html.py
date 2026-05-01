"""Build a self-contained science-fair HTML report with embedded detection images.

Reads the science-fair markdown report and injects an image gallery
(base64-encoded JPEGs) so the resulting HTML can be opened anywhere
without the runs/ directory.

Usage:
    .venv/Scripts/python.exe tools/build_science_fair_html.py
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import markdown
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "docs" / "03-analysis" / "2026-05-01-science-fair-report.md"
OUT_PATH = ROOT / "docs" / "03-analysis" / "2026-05-01-science-fair-report.html"

# Figures: (path, caption-html, target-max-width-px)
FIGURES = [
    {
        "id": "fig-input-sample",
        "path": ROOT / "captured_raw_images" / "2021-10-07_10-48-58.jpg",
        "label": "그림 1",
        "title": "입력 데이터 예시 — 양파 표피세포 squash slide",
        "caption": (
            "광학현미경 400배 시야로 촬영한 양파 뿌리 끝 압착 표본 (600×450 px). "
            "한 시야에 50–150개의 세포가 관찰되며, 다양한 분열 단계가 혼재한다. "
            "본 연구의 학습·시연 입력의 표준 단위."
        ),
        "max_w": 900,
    },
    {
        "id": "fig-rois-overview",
        "path": ROOT / "runs" / "apex_rois_test" / "2020_10_07__17_31__0065-Image Export-01__rois_overview.jpg",
        "label": "그림 2",
        "title": "원본 whole-mount 이미지와 자동 추출된 메리스템 ROI",
        "caption": (
            "raws/JPEG_Export_Data 원본 이미지(8000×31000 px)에서 알고리즘이 자동 식별한 "
            "두 개의 양파 뿌리 끝(meristematic apex) ROI. 분열 활동이 집중된 메리스템 영역만 "
            "추출하여 후속 타일링·검출의 효율을 높인다."
        ),
        "max_w": 600,
    },
    {
        "id": "fig-tissue-mask",
        "path": ROOT / "runs" / "tiled_detect_filtered" / "2020_10_07__17_31__0065-Image Export-01" / "_tissue_mask.jpg",
        "label": "그림 3",
        "title": "조직 마스크 (Otsu thresholding)",
        "caption": (
            "양파 뿌리 끝 두 개의 실루엣을 깨끗하게 분리. 흰 배경의 점·노이즈는 모두 제거되어 "
            "후속 검출에서 ‘조직 외부 false positive 97개’를 사전에 차단한다."
        ),
        "max_w": 600,
    },
    {
        "id": "fig-detect-filtered",
        "path": ROOT / "runs" / "tiled_detect_filtered" / "2020_10_07__17_31__0065-Image Export-01" / "_overview.jpg",
        "label": "그림 4",
        "title": "3차 시도 결과 — 필터링 후 240개 검출 (평균 conf 0.78)",
        "caption": (
            "타일링(512×512, 1886타일) → 조직 마스크 → min bbox size 30 px → confidence 0.6 필터를 "
            "순차 적용한 최종 검출 결과. 메리스템 영역에 박스가 집중되며, 흰 배경의 false positive는 "
            "거의 보이지 않는다. 클래스별 분포: Pro 51, Meta 26, Ana 130, Telo 27, Inter 6."
        ),
        "max_w": 600,
    },
    {
        "id": "fig-detect-apex0",
        "path": ROOT / "runs" / "tiled_detect_apex" / "2020_10_07__17_31__0065-Image Export-01__apex0" / "_overview.jpg",
        "label": "그림 5",
        "title": "메리스템 ROI 0 — 검출 100개 (avg conf 0.76)",
        "caption": (
            "왼쪽 뿌리 끝 ROI를 단독 추출 후 타일 검출. 메리스템 영역에 집중된 검출과 분열 단계별 "
            "색상이 명확히 보인다. 빨강=Inter, 초록=Pro, 파랑=Meta, 시안=Ana, 노랑=Telo."
        ),
        "max_w": 500,
    },
    {
        "id": "fig-detect-apex1",
        "path": ROOT / "runs" / "tiled_detect_apex" / "2020_10_07__17_31__0065-Image Export-01__apex1" / "_overview.jpg",
        "label": "그림 6",
        "title": "메리스템 ROI 1 — 검출 59개 (avg conf 0.77)",
        "caption": (
            "오른쪽 뿌리 끝 ROI. 검출 수는 다르나 패턴은 일관되며, 두 ROI 모두 평균 신뢰도 ≥ 0.76으로 "
            "필터링이 안정적임을 보여준다."
        ),
        "max_w": 500,
    },
]


def encode_image(path: Path, max_width: int) -> tuple[str, int, int]:
    """Resize image keeping aspect, encode JPEG quality=85, return (base64, w, h)."""
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if w > max_width:
            new_h = int(h * max_width / w)
            im = im.resize((max_width, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return b64, im.size[0], im.size[1]


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
    --figure-bg: #f6f8fa;
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
        --figure-bg: #161b22;
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
a { color: var(--link); text-decoration: none; }
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
pre code { background: transparent; padding: 0; border-radius: 0; }
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
hr { border: 0; border-top: 1px solid var(--border); margin: 28px 0; }
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

/* === Figure gallery === */
.figure-gallery {
    margin: 24px 0;
    padding: 0;
}
.figure-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--figure-bg);
    padding: 16px 18px 18px;
    margin: 20px 0;
    page-break-inside: avoid;
}
.figure-card .fig-image-wrapper {
    text-align: center;
    background: #000;
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 12px;
}
.figure-card img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    display: inline-block;
}
.figure-card .fig-label {
    font-weight: 700;
    color: var(--accent);
    font-size: 0.95em;
    margin-right: 6px;
}
.figure-card .fig-title {
    font-weight: 600;
    font-size: 1.05em;
    margin-bottom: 8px;
}
.figure-card .fig-caption {
    color: var(--fg-muted);
    font-size: 0.95em;
    line-height: 1.55;
    margin: 0;
}

.figure-pair {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
@media (max-width: 720px) {
    .figure-pair { grid-template-columns: 1fr; }
}
.figure-pair .figure-card { margin: 0; }

/* Class-color legend */
.color-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 12px 0 20px;
    padding: 12px 14px;
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
}
.color-legend .swatch {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.9em;
}
.color-legend .swatch::before {
    content: "";
    width: 14px;
    height: 14px;
    border-radius: 3px;
    background: var(--c);
    display: inline-block;
    border: 1px solid #00000022;
}

@media print {
    body { background: white; color: black; }
    .container { max-width: none; padding: 0; }
    a { color: black; }
    pre, code { border: 1px solid #ddd; }
    .figure-card { background: white; border-color: #999; }
    .figure-card .fig-image-wrapper { background: #f5f5f5; }
}
"""


def build_figure_card(fig: dict) -> str:
    b64, w, h = encode_image(fig["path"], fig["max_w"])
    return f"""<figure class="figure-card" id="{fig['id']}">
  <div class="fig-image-wrapper">
    <img src="data:image/jpeg;base64,{b64}" width="{w}" height="{h}" alt="{fig['title']}">
  </div>
  <figcaption>
    <div class="fig-title"><span class="fig-label">{fig['label']}.</span>{fig['title']}</div>
    <p class="fig-caption">{fig['caption']}</p>
  </figcaption>
</figure>"""


def build_gallery_html() -> str:
    legend = """<div class="color-legend">
  <span class="swatch" style="--c:#ff3232;">Inter (간기)</span>
  <span class="swatch" style="--c:#00ff00;">Pro (전기)</span>
  <span class="swatch" style="--c:#0064ff;">Meta (중기)</span>
  <span class="swatch" style="--c:#00ffff;">Ana (후기)</span>
  <span class="swatch" style="--c:#ffff00;">Telo (말기)</span>
</div>"""

    cards = [build_figure_card(FIGURES[0])]  # 입력 샘플
    cards.append(build_figure_card(FIGURES[1]))  # whole-mount + ROI
    cards.append(
        '<div class="figure-pair">'
        + build_figure_card(FIGURES[2])
        + build_figure_card(FIGURES[3])
        + "</div>"
    )
    cards.append(legend)
    cards.append(
        '<div class="figure-pair">'
        + build_figure_card(FIGURES[4])
        + build_figure_card(FIGURES[5])
        + "</div>"
    )

    return (
        '<h2 id="visuals">시각 자료 — 검출 파이프라인 결과</h2>\n'
        "<p>본 절은 §5.5(3차 라벨링 결과) 및 §6(핵심 발견)을 시각적으로 보완한다. "
        "모든 이미지는 본 보고서에 base64로 직접 임베드되어 있어 외부 자원 없이 단독 열람 가능하다.</p>\n"
        '<div class="figure-gallery">\n' + "\n".join(cards) + "\n</div>\n"
    )


def render(md_text: str) -> str:
    md = markdown.Markdown(
        extensions=["extra", "sane_lists", "smarty", "toc", "nl2br", "admonition"],
        extension_configs={"toc": {"permalink": False, "anchorlink": True}},
        output_format="html5",
    )
    body = md.convert(md_text)

    gallery = build_gallery_html()
    # Inject the gallery right before "## 6. 핵심 발견" so visuals appear at
    # the natural pivot between progress narrative and key findings.
    # The toc extension strips Korean and keeps only the numeric prefix → id="6".
    marker = '<h2 id="6">'
    if marker in body:
        body = body.replace(marker, gallery + "\n" + marker, 1)
    else:
        body = body + "\n" + gallery

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = "NeoMscope: 엣지 AI 기반 양파 체세포 분열 단계 자동 검출 시스템 — 과학전람회 보고서"

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="NeoMscope tools/build_science_fair_html.py">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
<div class="report-meta">
    <div><span class="label">Source:</span><code>{MD_PATH.relative_to(ROOT).as_posix()}</code></div>
    <div><span class="label">Generated:</span>{generated_at}</div>
    <div><span class="label">Tool:</span>tools/build_science_fair_html.py (with embedded images)</div>
    <div><span class="label">Figures:</span>{len(FIGURES)} (base64-embedded JPEG)</div>
</div>
{body}
</div>
</body>
</html>
"""


def main() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")
    html = render(md_text)
    OUT_PATH.write_text(html, encoding="utf-8")
    size_kb = len(html.encode("utf-8")) / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:,.1f} KB, {len(FIGURES)} figures embedded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
