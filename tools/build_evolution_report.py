"""Generate the interface-evolution HTML report.

Bakes all PNG/JPG screenshots into a single self-contained HTML file
using base64 data URIs — no external image references, no broken links
when the file is shared via email or USB.

Output: docs/03-analysis/2026-05-10-interface-evolution.html
"""

from __future__ import annotations

import base64
import mimetypes
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "03-analysis" / "2026-05-10-interface-evolution.html"

SCREENSHOTS = REPO / "docs" / "03-analysis" / "screenshots"


def img(rel_path: str, alt: str = "", cls: str = "") -> str:
    """Embed an image at REPO-relative path as base64 data URI."""
    p = REPO / rel_path
    if not p.is_file():
        return f'<div class="missing">missing: {rel_path}</div>'
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    cls_attr = f' class="{cls}"' if cls else ""
    return f'<img{cls_attr} alt="{alt}" src="data:{mime};base64,{b64}">'


CSS = """
:root {
  --paper: #F6F4ED;
  --paper-dim: #EFECE2;
  --ink: #0F1611;
  --ink-soft: #41463F;
  --ink-mute: #797E76;
  --line: #DEDACA;
  --line-soft: #ECE8D8;
  --accent: #1D49C0;
  --accent-deep: #122F86;
  --accent-soft: #DEE7FB;
  --c-inter: #C42A2A;
  --c-pro: #1E8B47;
  --c-meta: #2256C9;
  --c-ana: #0B7F9C;
  --c-telo: #B07A07;
  --mono: 'IBM Plex Mono', 'JetBrains Mono', Consolas, monospace;
  --sans: 'Pretendard', 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.55;
  color: var(--ink);
  background: var(--paper);
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='320' height='320'><filter id='n'><feTurbulence baseFrequency='1.4' numOctaves='2' seed='3' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .04 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
  background-size: 320px 320px;
  -webkit-font-smoothing: antialiased;
}
.container {
  max-width: 980px;
  margin: 0 auto;
  padding: 56px 32px 96px;
}

/* ---------- masthead ---------- */
.masthead {
  border-bottom: 1px solid var(--line);
  padding-bottom: 28px;
  margin-bottom: 40px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 24px;
  align-items: end;
}
.eyebrow {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--ink-mute);
  margin-bottom: 12px;
}
h1 {
  font-size: 38px;
  font-weight: 700;
  letter-spacing: -0.025em;
  line-height: 1.05;
  margin: 0;
  color: var(--ink);
}
h1 .ko { display: block; font-size: 22px; font-weight: 500; color: var(--ink-soft); margin-top: 8px; letter-spacing: -0.01em; }
.colophon {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.04em;
  color: var(--ink-mute);
  text-align: right;
  line-height: 1.7;
}
.colophon b { color: var(--ink); font-weight: 500; }

/* ---------- abstract ---------- */
.abstract {
  font-size: 16px;
  line-height: 1.65;
  color: var(--ink-soft);
  border-left: 3px solid var(--accent);
  padding-left: 18px;
  margin: 0 0 56px;
  max-width: 720px;
}
.abstract::before {
  content: "초록 · ABSTRACT";
  display: block;
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.18em;
  color: var(--ink-mute);
  margin-bottom: 8px;
  text-transform: uppercase;
}

/* ---------- timeline ---------- */
.timeline {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-bottom: 64px;
}
.tl-step {
  position: relative;
  padding: 14px 12px 12px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,0.55);
  border-radius: 3px;
}
.tl-step .num {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--ink-mute);
}
.tl-step .name {
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.01em;
  margin: 4px 0 6px;
}
.tl-step .meta {
  font-family: var(--mono);
  font-size: 9.5px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
  line-height: 1.5;
}
.tl-step.active {
  background: white;
  border-color: var(--accent);
  border-width: 1.5px;
}
.tl-step.active .num { color: var(--accent); }

/* ---------- sections ---------- */
section {
  margin-bottom: 72px;
}
section.stage {
  border-top: 1px solid var(--line);
  padding-top: 36px;
}
.stage-label {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-mute);
  margin-bottom: 6px;
}
h2 {
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.15;
  margin: 0 0 8px;
  color: var(--ink);
}
h2 .v {
  font-family: var(--mono);
  font-size: 15px;
  font-weight: 500;
  color: var(--accent);
  letter-spacing: 0.02em;
  margin-right: 8px;
  vertical-align: middle;
}
h3 {
  font-size: 17px;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 28px 0 10px;
  color: var(--ink);
}
.lead {
  font-size: 16px;
  color: var(--ink-soft);
  margin: 0 0 24px;
  max-width: 640px;
  line-height: 1.6;
}

p { margin: 0 0 14px; color: var(--ink); }
p code, li code { font-family: var(--mono); font-size: 13px; color: var(--accent-deep); background: rgba(29,73,192,0.07); padding: 1px 6px; border-radius: 2px; }
ul, ol { margin: 0 0 16px; padding-left: 22px; }
li { margin-bottom: 6px; }
strong { color: var(--ink); font-weight: 600; }

/* ---------- figure ---------- */
figure {
  margin: 24px 0 32px;
}
figure img {
  display: block;
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 3px;
  box-shadow: 0 8px 24px -12px rgba(15,22,17,0.18);
}
figcaption {
  margin-top: 10px;
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.05em;
  color: var(--ink-mute);
  line-height: 1.55;
}
figcaption b { color: var(--ink); font-family: var(--sans); font-weight: 600; font-size: 11.5px; letter-spacing: 0; }

.gallery {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.gallery figure { margin: 0; }
.gallery img { border-radius: 2px; }

/* ---------- comparison table ---------- */
table.compare {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0 32px;
  font-size: 14px;
}
table.compare th, table.compare td {
  text-align: left;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}
table.compare th {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-mute);
  font-weight: 500;
  background: var(--paper-dim);
  border-bottom: 2px solid var(--line);
}
table.compare td.label {
  font-weight: 600;
  color: var(--ink);
  background: rgba(255,255,255,0.4);
}
table.compare td.win { color: var(--accent-deep); font-weight: 500; }
table.compare td code { font-size: 12px; }

/* ---------- callout ---------- */
.callout {
  background: rgba(29,73,192,0.05);
  border-left: 3px solid var(--accent);
  padding: 14px 18px;
  margin: 20px 0 24px;
  border-radius: 0 3px 3px 0;
}
.callout .label {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 6px;
  font-weight: 600;
}
.callout p:last-child { margin: 0; }

/* ---------- footer ---------- */
.colophon-bottom {
  margin-top: 96px;
  padding-top: 28px;
  border-top: 1px solid var(--line);
  font-family: var(--mono);
  font-size: 11px;
  color: var(--ink-mute);
  letter-spacing: 0.04em;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}

/* ---------- print ---------- */
@media print {
  body { background: white; background-image: none; }
  .container { max-width: none; padding: 24px 16px; }
  figure img { box-shadow: none; }
  section.stage { break-inside: avoid; }
  figure { break-inside: avoid; }
}

/* ---------- mobile ---------- */
@media (max-width: 720px) {
  .masthead { grid-template-columns: 1fr; }
  .colophon { text-align: left; }
  .timeline { grid-template-columns: 1fr 1fr; }
  .gallery { grid-template-columns: 1fr; }
  h1 { font-size: 28px; }
  .container { padding: 32px 18px 64px; }
}
"""


def build() -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NeoMscope 인터페이스 진화 — CLI에서 PySide6 GUI까지</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<div class="container">

<!-- =============== MASTHEAD =============== -->
<header class="masthead">
  <div>
    <div class="eyebrow">NeoMscope · Interface Evolution Report</div>
    <h1>CLI에서 PySide6 GUI까지<span class="ko">사용자 인터페이스 진화 과정</span></h1>
  </div>
  <div class="colophon">
    <div><b>Project</b>&nbsp;NeoMscope v2.0</div>
    <div><b>Hardware</b>&nbsp;Pi 5 + AI HAT+ 2 (Hailo-10H)</div>
    <div><b>Date</b>&nbsp;{generated}</div>
    <div><b>Author</b>&nbsp;domafordarwin</div>
    <div><b>Repo</b>&nbsp;github.com/domafordarwin/NeoMscope</div>
  </div>
</header>

<!-- =============== ABSTRACT =============== -->
<p class="abstract">
양파 뿌리 끝 체세포 분열 5단계(Inter / Pro / Meta / Ana / Telo)를
Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H NPU) 위에서 실시간 검출하는 NeoMscope의
사용자 인터페이스를, 2022년 OpenCV 단일 창에서 시작해 2026년 5월 PySide6
다중 탭 GUI 및 출판용 디자인 컴프까지 4단계로 발전시킨 과정을 정리한다.
각 단계의 시각적 결과물, 기술적 변경점, 그리고 다음 단계로 이행한 동기를
함께 기록한다.
</p>

<!-- =============== TIMELINE =============== -->
<div class="timeline">
  <div class="tl-step">
    <div class="num">v0 · 2022</div>
    <div class="name">레거시 CLI</div>
    <div class="meta">Mask R-CNN · cv2.imshow · 단일 창</div>
  </div>
  <div class="tl-step">
    <div class="num">v1 · 2026-05-09</div>
    <div class="name">신규 PySide6 초안</div>
    <div class="meta">YOLOv11 ready · 4탭 mockup · 1024×600</div>
  </div>
  <div class="tl-step">
    <div class="num">v2 · 2026-05-09</div>
    <div class="name">프로덕션 GUI</div>
    <div class="meta">실 카메라 · QThread 워커 · 적용 가능</div>
  </div>
  <div class="tl-step active">
    <div class="num">v3 · 2026-05-09</div>
    <div class="name">에디토리얼 컴프</div>
    <div class="meta">출판/포스터용 · base64 self-contained</div>
  </div>
</div>

<!-- =============== STAGE 0 — CLI =============== -->
<section class="stage">
  <div class="stage-label">Stage 0 · 2022</div>
  <h2><span class="v">v0</span>레거시 CLI — OpenCV 단일 창</h2>
  <p class="lead">
    학교 생물 시간에 학생 한 명이 현미경 옆에 노트북을 두고 USB 웹캠으로 슬라이드를
    들여다본다. 명령창에 한 줄을 치면 검출 박스가 그려진 비디오 한 화면이 뜬다.
  </p>

  <figure>
    {img("docs/03-analysis/screenshots/v0/cli_live_simulation.png", "v0 cli", "")}
    <figcaption>
      <b>그림 1.</b>
      <code>python -m inference.live_detect --camera /dev/video0</code>이 띄우는
      cv2.imshow 창. 좌측 상단에 클래스별 카운트 패널, 좌측 하단에 FPS 카운터,
      이미지 위에 5색 detection bbox. 종료는 q 키.
    </figcaption>
  </figure>

  <h3>특성</h3>
  <ul>
    <li><b>단일 창</b>: cv2.imshow 한 개. 풀스크린 친화, 즉시 시작, 종료도 한 번의 키.</li>
    <li><b>스크립트 분리</b>: <code>live_detect.py</code> / <code>batch_detect.py</code> / <code>capture_and_detect.py</code> — 작업별 별도 진입점.</li>
    <li><b>의존성 최소</b>: opencv-python + numpy만으로 동작 (Qt/GTK 불요).</li>
  </ul>

  <h3>한계</h3>
  <ul>
    <li>임계값·카메라·모델 변경은 <em>다시 시작 + 인자 변경</em>이 유일한 방법.</li>
    <li>현재 검출 결과를 곧장 다른 작업(Roboflow 업로드, 일괄 평가)으로 넘기기 번거로움.</li>
    <li>저장된 결과물 검색·비교 기능 없음.</li>
    <li>전람회 시연 시 명령창이 노출되어 비전문가에게 위협적.</li>
  </ul>
</section>

<!-- =============== STAGE 1 — V1 MOCKUP =============== -->
<section class="stage">
  <div class="stage-label">Stage 1 · 2026-05-09 (오전)</div>
  <h2><span class="v">v1</span>첫 PySide6 mockup — 4탭 구조 도입</h2>
  <p class="lead">
    "RAIM Scope 교육용 GUI" 참고 디자인을 받아 PySide6로 처음 옮긴 단계. 동작은 정적 모형이고
    레이아웃의 골격만 잡았다. 이후 v2에서 실 카메라·추론을 붙였다.
  </p>

  <figure>
    {img("docs/03-analysis/screenshots/2026-05-09_ui_mockup_v1.png", "v1 mockup", "")}
    <figcaption>
      <b>그림 2.</b>
      4탭(Live · Batch · Archive · Settings) 구조와 우측 패널(검출 결과 · 도구),
      하단 3분할 (촬영 제어 · 검출 모드 · 이미지 조정)이 처음 자리잡은 모습.
      <code>captured_raw_images/2022-07-18_18-02-29.jpg</code>에 모의 박스 12개를 그려 넣었다.
    </figcaption>
  </figure>

  <h3>v0 → v1 변경점</h3>
  <ul>
    <li><b>창 한 개 → 4탭 단일 윈도우</b>: Live/Batch/Archive/Settings 워크플로 분리.</li>
    <li><b>설정 인라인화</b>: 임계값 슬라이더·검출 모드 라디오·이미지 조정 슬라이더가 화면 안.</li>
    <li><b>클래스 시각화</b>: 5색 chip이 Inter/Pro/Meta/Ana/Telo의 카운터 역할.</li>
    <li><b>한국어 1차 시민</b>: 모든 라벨 한국어 + 영어 보조.</li>
  </ul>

  <div class="callout">
    <div class="label">의도된 한계</div>
    <p>
      이 단계의 PNG는 정적 모형이다. 클릭·드래그·실 카메라가 동작하지 않는다.
      구조 합의가 목적이었고, v2에서 진짜 동작을 붙였다.
    </p>
  </div>
</section>

<!-- =============== STAGE 2 — V2 PRODUCTION =============== -->
<section class="stage">
  <div class="stage-label">Stage 2 · 2026-05-09 (오후)</div>
  <h2><span class="v">v2</span>프로덕션 PySide6 GUI — 실 카메라 + 4탭 동작</h2>
  <p class="lead">
    v1을 모듈로 분해(<code>inference/ui/tabs/*.py</code>)하고 모든 탭에 진짜 기능을 붙였다.
    실제 USB 웹캠을 열고, mock 파이프라인으로 검출하고, QThread로 일괄 처리하고,
    저장된 결과를 썸네일로 검색한다. 화면 4종 모두 단일 1024×600 안에 들어간다.
  </p>

  <div class="gallery">
    <figure>{img("docs/03-analysis/screenshots/v2/Live.png", "v2 Live")}<figcaption><b>Live 탭.</b> 실 웹캠 1280×720 + mock 검출(중앙 빨강) + chip 카운터 + FPS 17.0.</figcaption></figure>
    <figure>{img("docs/03-analysis/screenshots/v2/Batch.png", "v2 Batch")}<figcaption><b>Batch 탭.</b> 폴더 픽커 + 검출 시작/중단 + QThread 워커 + 진행 막대 + 로그.</figcaption></figure>
    <figure>{img("docs/03-analysis/screenshots/v2/Archive.png", "v2 Archive")}<figcaption><b>Archive 탭.</b> 저장된 결과 자동 인덱싱 + 썸네일 그리드 + 미리보기 분할.</figcaption></figure>
    <figure>{img("docs/03-analysis/screenshots/v2/Settings.png", "v2 Settings")}<figcaption><b>Settings 탭.</b> HEF/카메라/임계값/출력 디렉터리 한 화면. 적용 시 다른 탭에 즉시 반영.</figcaption></figure>
  </div>

  <h3>v1 → v2 변경점</h3>
  <ul>
    <li><b>모듈 분해</b>: <code>main_window.py</code> / <code>tabs/{{live,batch,archive,settings}}.py</code> / <code>widgets.py</code> / <code>state.py</code> — 각 파일 250줄 이하.</li>
    <li><b>실 카메라 연결</b>: <code>inference._camera.CV2VideoCapture</code> 사용. 카메라 없으면 <code>ImageFolder</code>로 자동 폴백.</li>
    <li><b>실 추론 파이프라인</b>: <code>HailoInferencePipeline(mock=True)</code>. HEF 컴파일 후 자동으로 NPU 모드로 전환.</li>
    <li><b>전역 상태</b>: <code>AppController.settings_changed</code> 시그널로 Settings 탭 변경이 Live/Batch에 즉시 전파.</li>
    <li><b>비동기 처리</b>: Batch 검출은 <code>QThread BatchWorker</code>로 UI 블로킹 없음.</li>
    <li><b>실 파일 입출력</b>: 촬영 버튼이 <code>detection_results_captured/{{timestamp}}.jpg</code>에 실제 저장.</li>
    <li><b>자체 테스트</b>: <code>--screenshot</code> / <code>--screenshot-all</code> 헤드리스 옵션으로 CI에서도 렌더 검증.</li>
  </ul>
</section>

<!-- =============== STAGE 3 — V3 EDITORIAL COMP =============== -->
<section class="stage">
  <div class="stage-label">Stage 3 · 2026-05-09 (저녁)</div>
  <h2><span class="v">v3</span>에디토리얼 디자인 컴프 — 출판·포스터용</h2>
  <p class="lead">
    v2가 "동작하는 인터페이스"라면, v3는 "그것의 가장 정제된 모습"이다.
    Pretendard + IBM Plex Mono/Sans의 3-폰트 혼합, 따뜻한 종이 베이스,
    SVG 노이즈 그레인, 디샤츄레이트 클래스 컬러, 의도된 비대칭—
    "AI 슬랍" 미학을 정면 회피하는 것이 목적.
  </p>

  <figure>
    {img("docs/03-analysis/screenshots/v3/neomscope_design_mockup_full.png", "v3 design comp")}
    <figcaption>
      <b>그림 7.</b>
      v3 디자인 컴프 (1024×640 자연 높이). 메리스템 셀룰러 패턴 + 12 detection 박스 + 좌상단 LIVE LED와
      4-모서리 메타데이터(scan A · field 03 · 400× / timecode / datetime)가 정밀 계측기 분위기를 연출한다.
      각 클래스 chip은 컬러 바 + 이름 + 미니 분포 막대 + 카운트로 한 줄에 4개 정보를 담는다.
    </figcaption>
  </figure>

  <h3>v2 → v3 변경점</h3>
  <ul>
    <li><b>타이포그래피 의도화</b>: 검출 카운트 36px Mono · "+3" 델타는 11px green — 4× 스케일 대비.</li>
    <li><b>색감</b>: 순백색 → 따뜻한 종이 <code>#F6F4ED</code>. 클래스 컬러는 OpenCV 풀채도 → 출판용 디샤츄레이트.</li>
    <li><b>텍스처</b>: 평면 배경 → SVG <code>feTurbulence</code> 미세 그레인. 평면 AI-look 회피.</li>
    <li><b>이중 라벨</b>: <code>Detection Results · 검출 결과</code> 영문 + 한글 페어링.</li>
    <li><b>현미경 뷰</b>: 실 영상 + bbox → 셀룰러 SVG 패턴 + 4-모서리 메타 + 미세 reticle.</li>
    <li><b>탭 넘버링</b>: <code>01 / 02 / 03 / 04</code> 모노 넘버링 + 활성 보더 청색.</li>
    <li><b>모서리 곡률</b>: 일반적인 8px → 의도적 3px (브루탈리즘과 마시멜로 사이의 의도된 선택).</li>
  </ul>

  <div class="callout">
    <div class="label">사용 영역 분리</div>
    <p>
      v3는 <em>실행 가능한 프로그램이 아니다</em>. 클릭이 없는 정적 HTML이고
      포스터·논문 figure·전람회 스탠드 패널·README 헤더 이미지로 사용한다.
      실제 Pi에서 돌리는 인터페이스는 v2이며, 두 가지가 같은 시각 어휘를 공유한다.
    </p>
  </div>
</section>

<!-- =============== COMPARISON =============== -->
<section>
  <h2>4단계 비교표</h2>
  <table class="compare">
    <thead>
      <tr>
        <th>항목</th>
        <th>v0 — Legacy CLI</th>
        <th>v1 — 첫 mockup</th>
        <th>v2 — 프로덕션 GUI</th>
        <th>v3 — 디자인 컴프</th>
      </tr>
    </thead>
    <tbody>
      <tr><td class="label">출시</td><td>2022</td><td>2026-05-09 오전</td><td>2026-05-09 오후</td><td>2026-05-09 저녁</td></tr>
      <tr><td class="label">기술</td><td>cv2.imshow</td><td>PySide6 정적</td><td class="win">PySide6 + QThread</td><td>HTML + SVG</td></tr>
      <tr><td class="label">실 카메라</td><td>✓</td><td>—</td><td class="win">✓</td><td>—</td></tr>
      <tr><td class="label">실 추론</td><td>Mask R-CNN</td><td>—</td><td class="win">mock pipeline</td><td>—</td></tr>
      <tr><td class="label">탭/창</td><td>창 1개</td><td>4탭 (모형)</td><td>4탭 (동작)</td><td>4탭 (정적)</td></tr>
      <tr><td class="label">설정 인라인</td><td>—</td><td>✓ 시각</td><td class="win">✓ 동작</td><td>✓ 시각</td></tr>
      <tr><td class="label">한국어</td><td>일부</td><td>전체</td><td>전체</td><td class="win">전체 (이중 라벨)</td></tr>
      <tr><td class="label">화면 크기</td><td>가변</td><td>1024×600</td><td class="win">1024×600</td><td>1024×600/640</td></tr>
      <tr><td class="label">디자인 정제</td><td>기능 우선</td><td>레이아웃</td><td>밸런스</td><td class="win">에디토리얼</td></tr>
      <tr><td class="label">사용 영역</td><td>일상 운영</td><td>합의</td><td class="win">실 운영</td><td>발표/문서</td></tr>
      <tr><td class="label">상태</td><td>레거시 보존</td><td>대체됨</td><td class="win">메인</td><td>병행</td></tr>
    </tbody>
  </table>
</section>

<!-- =============== TECH EVOLUTION =============== -->
<section>
  <h2>기술 스택 진화</h2>

  <h3>인터페이스 레이어</h3>
  <table class="compare">
    <thead><tr><th>역할</th><th>v0</th><th>v1·v2</th><th>v3</th></tr></thead>
    <tbody>
      <tr><td class="label">윈도잉</td><td>OpenCV HighGUI</td><td>PySide6 (Qt 6.11)</td><td>HTML5</td></tr>
      <tr><td class="label">레이아웃</td><td>없음</td><td>QHBoxLayout/QGridLayout</td><td>CSS Grid + Flex</td></tr>
      <tr><td class="label">스타일</td><td>OpenCV native</td><td>QSS</td><td>인라인 CSS</td></tr>
      <tr><td class="label">아이콘</td><td>없음</td><td>유니코드/SVG</td><td>인라인 SVG</td></tr>
      <tr><td class="label">한글 폰트</td><td>시스템 기본</td><td>시스템 기본</td><td>Pretendard CDN</td></tr>
    </tbody>
  </table>

  <h3>도메인 레이어 (모든 단계 공통)</h3>
  <ul>
    <li><code>inference.types</code> — Detection / PipelineConfig / CLASS_NAMES / CLASS_COLORS (Source of truth)</li>
    <li><code>inference.pipeline.HailoInferencePipeline</code> — mock=True / NPU 자동 분기</li>
    <li><code>inference.postprocess</code> — YOLO det decode + render_overlay</li>
    <li><code>inference._camera</code> — CV2VideoCapture / ImageFolder 추상화</li>
  </ul>
  <p>
    이 4개 모듈은 v0/v1/v2가 모두 그대로 쓴다. UI 진화의 진짜 의미는
    <em>도메인 코드를 건드리지 않고 표현 계층만 4번 다시 그렸다</em>는 점.
    덕분에 검출 결과 자체는 모든 단계에서 동일하다.
  </p>
</section>

<!-- =============== CONCLUSION =============== -->
<section>
  <h2>결론</h2>
  <p>
    NeoMscope의 인터페이스는 <strong>"동작하는 가장 작은 것에서 시작해 점진적으로 정교화"</strong>의
    교과서적 사례를 따랐다. v0의 OpenCV 한 줄짜리 창은 4년 동안 학교 생물 실험을 지원했고,
    v1의 mockup은 새 디자인 방향에 대한 합의 도구였으며, v2는 그 합의를 작동하는 코드로 옮겼고,
    v3는 그 작동물의 가장 발표 적합한 면을 분리해 출판용 figure로 만들었다.
  </p>
  <p>
    각 단계가 다음 단계를 <em>덮어쓰지 않고 공존</em>한다는 점이 중요하다. Pi 5에는
    v0(<code>neomscope-live</code>) · v2(<code>neomscope-gui</code>)가 같은 venv 안에서 함께 설치되며,
    v3 HTML은 보고서·포스터에 그대로 임베드된다. 사용자는 시나리오에 따라 도구를 고른다 —
    빠른 일회성 스캔이면 v0, 데모와 검수 워크플로면 v2, 인쇄물이면 v3.
  </p>
  <p>
    다음 단계는 학습 데이터셋 라벨링 완료 후 HEF 파일 컴파일이며, 이때 자동으로 v2의
    <code>HailoInferencePipeline(mock=True)</code>가 <code>mock=False</code>로 전환되어 실제 NPU 추론을
    수행한다. 인터페이스는 변경 없이 검출 결과만 진짜로 바뀐다.
  </p>
</section>

<!-- =============== FOOTER =============== -->
<div class="colophon-bottom">
  <span>NeoMscope Interface Evolution · {generated}</span>
  <span>Pretendard · IBM Plex · BGR(0,50,255) #1D49C0</span>
</div>

</div>
</body>
</html>
"""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = build()
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
