"""PCIP Digital Twin — landing page.

All CSS is injected via st.markdown(..., unsafe_allow_html=True) because
st.html() strips <style> blocks. The body markup goes through st.html().
"""
import base64
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="PCIP Digital Twin",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Paper link --------------------------------------------------------------
# When the preprint is on bioRxiv, set PAPER_URL to its full URL.
# When the journal version is accepted, swap to the journal DOI URL.
# While empty, the cover-page button shows "Paper coming soon" and is inert.
PAPER_URL = ""   # e.g. "https://www.biorxiv.org/content/10.1101/2026.0X.YY"

# --- Author ------------------------------------------------------------------
# Shown in the cover-page footer. Set to empty string to hide the line.
AUTHOR = "Mohammad Fili"
# -----------------------------------------------------------------------------

COVER = Path(__file__).parent / "CoverImage.png"
cover_b64 = base64.b64encode(COVER.read_bytes()).decode() if COVER.exists() else ""

CSS = r"""
<style>
  @import url("https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,300;1,9..144,400&family=Instrument+Serif:ital@0;1&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap");

  :root {
    --navy-000: #050D1C; --navy-100: #0A1628; --navy-200: #122038;
    --ink:      #F1F5FB;
    --ink-dim:  rgba(241,245,251,0.72);
    --ink-mute: rgba(241,245,251,0.52);
    --hair:      rgba(241,245,251,0.14);
    --hair-soft: rgba(241,245,251,0.08);
    --cyan:   #6FD9FF;
    --violet: #B794FF;
    --amber:  #FFC66B;
  }

  /* --- Nuke Streamlit chrome / padding for full-bleed landing ---------- */
  header[data-testid="stHeader"] { display: none !important; }
  [data-testid="stToolbar"] { display: none !important; }
  [data-testid="stDecoration"] { display: none !important; }
  #MainMenu { display: none !important; }

  .stApp { background: var(--navy-000); }

  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="stMainBlockContainer"],
  .main, .main .block-container, .block-container,
  section.main > div, section.main > .block-container {
    padding: 0 !important;
    max-width: 100% !important;
    margin: 0 !important;
  }
  [data-testid="stHtml"], .element-container, .stHtml, .stMarkdown {
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
  }

  /* Dark sidebar (when user expands it) */
  [data-testid="stSidebar"] {
    background: rgba(5,13,28,0.92) !important;
    border-right: 1px solid var(--hair-soft);
  }
  [data-testid="stSidebar"] * { color: var(--ink-dim); }

  /* --- Stage ---------------------------------------------------------- */
  .pcip-stage * { box-sizing: border-box; }
  .pcip-stage {
    position: relative;
    width: 100%;
    min-height: 100vh;
    overflow: hidden;
    isolation: isolate;
    color: var(--ink);
    background: var(--navy-000);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .pcip-bg {
    position: absolute; inset: 0; z-index: 0;
    background-image: url("data:image/png;base64,__COVER__");
    background-size: cover; background-position: center; background-repeat: no-repeat;
  }
  .pcip-scrim {
    position: absolute; inset: 0; z-index: 1; pointer-events: none;
    background:
      radial-gradient(ellipse 70% 90% at 18% 45%, rgba(5,13,28,0.88) 0%,
                      rgba(5,13,28,0.55) 45%, rgba(5,13,28,0.10) 75%, rgba(5,13,28,0.0) 100%),
      linear-gradient(180deg, rgba(5,13,28,0.55) 0%, rgba(5,13,28,0.15) 30%,
                      rgba(5,13,28,0.10) 60%, rgba(5,13,28,0.75) 100%),
      linear-gradient(90deg, rgba(5,13,28,0.40) 0%, rgba(5,13,28,0.08) 45%, rgba(5,13,28,0.0) 70%);
  }
  .pcip-grain {
    position: absolute; inset: 0; z-index: 2; pointer-events: none;
    opacity: 0.06; mix-blend-mode: overlay;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)' opacity='0.9'/></svg>");
  }
  .pcip-topbar, .pcip-hero, .pcip-meta, .pcip-cards-wrap, .pcip-footer {
    position: relative; z-index: 10;
  }

  /* --- Topbar --------------------------------------------------------- */
  .pcip-topbar { display: flex; align-items: center; justify-content: space-between; padding: 24px 56px; }
  .pcip-brand { display: flex; align-items: center; gap: 12px; }
  .pcip-brand-mark {
    width: 30px; height: 30px; border-radius: 8px;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--violet) 100%);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.08), 0 10px 30px -10px rgba(111,217,255,0.6);
    position: relative;
  }
  .pcip-brand-mark::after {
    content: ""; position: absolute; inset: 6px; border-radius: 4px; background: var(--navy-000);
  }
  .pcip-brand-mark::before {
    content: ""; position: absolute; left: 50%; top: 50%; width: 8px; height: 8px; border-radius: 50%;
    transform: translate(-50%,-50%); background: var(--amber);
    box-shadow: 0 0 10px var(--amber); z-index: 1;
  }
  .pcip-brand-text { display: flex; flex-direction: column; line-height: 1; }
  .pcip-brand-text b { font-weight: 600; letter-spacing: 0.02em; font-size: 14px; color: var(--ink); }
  .pcip-brand-text span {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--ink-mute);
    margin-top: 4px; letter-spacing: 0.12em; text-transform: uppercase;
  }
  .pcip-topnav { display: flex; gap: 28px; font-size: 13px; color: var(--ink-dim); align-items: center; }
  .pcip-topnav a { color: inherit !important; text-decoration: none !important; }
  .pcip-topnav a:hover { color: var(--ink) !important; }
  .pcip-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 14px; border: 1px solid var(--hair); border-radius: 999px;
    background: rgba(255,255,255,0.02); backdrop-filter: blur(8px);
  }
  .pcip-dot-green {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4ADE80; box-shadow: 0 0 8px #4ADE80;
  }

  /* --- Hero ----------------------------------------------------------- */
  .pcip-hero { padding: 64px 56px 40px; max-width: 900px; }
  .pcip-eyebrow {
    display: inline-flex; align-items: center; gap: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--cyan);
    padding: 6px 12px 6px 10px;
    border: 1px solid rgba(111,217,255,0.28); border-radius: 999px;
    background: rgba(111,217,255,0.06); margin-bottom: 28px;
  }
  .pcip-eyebrow .tick {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--cyan); box-shadow: 0 0 10px var(--cyan);
  }
  h1.pcip-title {
    font-family: 'Fraunces', serif; font-optical-sizing: auto; font-weight: 400;
    font-size: clamp(48px, 7.2vw, 104px); line-height: 0.96;
    letter-spacing: -0.025em; margin: 0; color: var(--ink); text-wrap: balance;
  }
  h1.pcip-title .pcip-chip {
    font-family: 'Inter', sans-serif; font-weight: 500; letter-spacing: 0.02em;
    font-size: 0.58em; display: inline-block;
    padding: 0.12em 0.38em 0.16em;
    border: 1px solid var(--hair); border-radius: 0.22em;
    vertical-align: 0.18em; margin-right: 0.18em;
    background: rgba(255,255,255,0.04);
  }
  h1.pcip-title em {
    font-style: italic; font-weight: 300;
    background: linear-gradient(100deg, var(--cyan) 0%, var(--violet) 60%, var(--amber) 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  p.pcip-subtitle {
    font-weight: 300; font-size: clamp(17px, 1.3vw, 20px); line-height: 1.55;
    color: var(--ink-dim); margin: 28px 0 0; max-width: 640px; text-wrap: pretty;
  }
  p.pcip-subtitle cite {
    font-style: italic; font-family: 'Fraunces', serif; font-weight: 300; color: var(--ink);
  }
  .pcip-ctas { display: flex; gap: 12px; margin-top: 36px; flex-wrap: wrap; }
  a.pcip-btn {
    display: inline-flex; align-items: center; gap: 10px;
    padding: 13px 20px; border-radius: 10px;
    font: 500 14px/1 'Inter', sans-serif; letter-spacing: 0.01em;
    cursor: pointer; border: 1px solid transparent;
    transition: transform .15s ease, background .2s ease, border-color .2s ease;
    text-decoration: none !important;
  }
  a.pcip-btn-primary {
    background: var(--ink) !important; color: var(--navy-000) !important;
    box-shadow: 0 10px 40px -10px rgba(241,245,251,0.35);
  }
  a.pcip-btn-primary:hover { transform: translateY(-1px); }
  a.pcip-btn-ghost {
    background: rgba(255,255,255,0.04) !important; border-color: var(--hair);
    color: var(--ink) !important; backdrop-filter: blur(10px);
  }
  a.pcip-btn-ghost:hover { background: rgba(255,255,255,0.08) !important; border-color: rgba(255,255,255,0.22); }
  .pcip-btn-disabled {
    background: rgba(255,255,255,0.025) !important;
    color: var(--ink-mute) !important;
    border: 1px dashed var(--hair-soft) !important;
    cursor: default !important;
  }
  .pcip-btn-disabled:hover { transform: none !important; }
  .pcip-arr { font-family: 'JetBrains Mono', monospace; font-size: 12px; opacity: 0.8; }

  /* --- Meta strip ----------------------------------------------------- */
  .pcip-meta {
    display: flex; gap: 40px; flex-wrap: wrap;
    padding: 0 56px; margin-top: 56px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-mute);
  }
  .pcip-meta .m { display: flex; flex-direction: column; gap: 6px; }
  .pcip-meta .m b {
    font-family: 'Inter', sans-serif; font-weight: 500; letter-spacing: 0.01em;
    text-transform: none; font-size: 14px; color: var(--ink);
  }

  /* --- Cards ---------------------------------------------------------- */
  .pcip-cards-wrap { padding: 72px 56px 48px; }
  .pcip-cards-head {
    display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 24px;
  }
  .pcip-cards-head h3 {
    font-family: 'Fraunces', serif; font-weight: 400; font-style: italic;
    font-size: 22px; margin: 0; color: var(--ink); letter-spacing: -0.01em;
  }
  .pcip-cards-head .rule { flex: 1; height: 1px; background: var(--hair-soft); margin: 0 24px; }
  .pcip-cards-head .count {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--ink-mute); letter-spacing: 0.18em;
  }
  .pcip-cards {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1px; background: var(--hair-soft);
    border: 1px solid var(--hair-soft); border-radius: 16px;
    overflow: hidden; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  }
  .pcip-card {
    background: rgba(10,22,40,0.55); padding: 28px;
    min-height: 240px; display: flex; flex-direction: column;
    position: relative; transition: background 0.25s ease;
    text-decoration: none !important; color: inherit !important;
  }
  a.pcip-card:hover { background: rgba(18,32,56,0.72); }
  .pcip-card .idx {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.22em; color: var(--ink-mute);
  }
  .pcip-card h4 {
    font-family: 'Fraunces', serif; font-weight: 400;
    font-size: 26px; line-height: 1.1; margin: 14px 0 10px;
    color: var(--ink); letter-spacing: -0.01em;
  }
  .pcip-card p { font-size: 14px; line-height: 1.55; color: var(--ink-dim); margin: 0; }
  .pcip-card .status {
    margin-top: auto; padding-top: 20px;
    display: flex; align-items: center; gap: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
  }
  .pcip-card .status .d { width: 6px; height: 6px; border-radius: 50%; }
  .pcip-card.available .status { color: #86EFAC; }
  .pcip-card.available .d { background: #4ADE80; box-shadow: 0 0 8px #4ADE80; }
  .pcip-card.soon .status { color: var(--ink-mute); }
  .pcip-card.soon .d { background: var(--ink-mute); }
  .pcip-card .accent {
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    opacity: 0; transition: opacity .3s;
  }
  .pcip-card.available .accent { opacity: 0.85; background: linear-gradient(90deg, var(--cyan), transparent); }
  .pcip-card:nth-child(2) .accent { background: linear-gradient(90deg, var(--violet), transparent); }
  .pcip-card:nth-child(3) .accent { background: linear-gradient(90deg, var(--amber), transparent); }

  /* --- Footer --------------------------------------------------------- */
  .pcip-footer {
    padding: 28px 56px 36px;
    display: flex; justify-content: space-between; align-items: center;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.14em;
    color: var(--ink-mute); text-transform: uppercase;
  }
  .pcip-footer a { color: var(--ink-dim) !important; text-decoration: none !important; margin-left: 20px; }
  .pcip-footer a:hover { color: var(--ink) !important; }
  .pcip-author { color: var(--ink-mute); }
  .pcip-author .pcip-name {
    color: var(--ink-dim);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    text-transform: none;
    letter-spacing: 0;
  }

  @media (max-width: 880px) {
    .pcip-cards { grid-template-columns: 1fr; }
    .pcip-topbar, .pcip-hero, .pcip-meta, .pcip-cards-wrap, .pcip-footer {
      padding-left: 24px; padding-right: 24px;
    }
    .pcip-topnav { display: none; }
  }
</style>
"""

BODY = r"""
<div class="pcip-stage">
  <div class="pcip-bg"></div>
  <div class="pcip-scrim"></div>
  <div class="pcip-grain"></div>

  <div class="pcip-topbar">
    <div class="pcip-brand">
      <div class="pcip-brand-mark"></div>
      <div class="pcip-brand-text">
        <b>PCIP &middot; Digital Twin</b>
        <span>v0.1 &middot; research preview</span>
      </div>
    </div>
    <div class="pcip-topnav">
      <a href="/Sensitivity_Analysis" target="_self">Sensitivity</a>
      <a href="#" target="_self">Simulation</a>
      <a href="#" target="_self">Estimation</a>
      <span class="pcip-pill"><span class="pcip-dot-green"></span> Streamlit live</span>
    </div>
  </div>

  <div class="pcip-hero">
    <div class="pcip-eyebrow"><span class="tick"></span> Research preview &middot; companion to the paper</div>
    <h1 class="pcip-title">
      <span class="pcip-chip">PCIP</span>
      <br>
      Digital Twin for<br>
      <em>personalized cognition.</em>
    </h1>
    <p class="pcip-subtitle">
      A browser-based companion to <cite>A Digital-Twin Model for Personalized Cognitive Intervention Programs</cite>. Explore the model interactively &mdash; vary parameters, run full-scale simulations, and recover latent structure from synthetic data.
    </p>
    <div class="pcip-ctas">
      __PAPER_BTN__
    </div>
  </div>

  <div class="pcip-meta">
    <div class="m"><span>Model</span><b>Digital-twin &middot; latent-state</b></div>
    <div class="m"><span>Policies</span><b>Staircase / random</b></div>
    <div class="m"><span>Runtime</span><b>Streamlit Cloud &middot; free tier</b></div>
    <div class="m"><span>Status</span><b>3 of 3 modules live</b></div>
  </div>

  <div class="pcip-cards-wrap">
    <div class="pcip-cards-head">
      <h3>Three ways in</h3>
      <div class="rule"></div>
      <div class="count">01 &mdash; 03</div>
    </div>
    <div class="pcip-cards">
      <a class="pcip-card available" href="/Sensitivity_Analysis" target="_self">
        <div class="accent"></div>
        <div class="idx">01 &middot; INSPECT</div>
        <h4>Sensitivity Analysis</h4>
        <p>Inspect each model component in isolation. Sliders let you vary parameters and watch the response shift in real time.</p>
        <div class="status"><span class="d"></span> Available now</div>
      </a>
      <a class="pcip-card available" href="/Simulation" target="_self">
        <div class="accent"></div>
        <div class="idx">02 &middot; RUN</div>
        <h4>Simulation</h4>
        <p>Run a full-scale intervention program under different policies &mdash; staircase vs. random &mdash; and compare learning trajectories.</p>
        <div class="status"><span class="d"></span> Available now</div>
      </a>
      <a class="pcip-card available" href="/Estimation" target="_self">
        <div class="accent"></div>
        <div class="idx">03 &middot; RECOVER</div>
        <h4>Estimation</h4>
        <p>Recover latent parameters from simulated data. Restricted to a small configuration so it fits within the Streamlit free tier.</p>
        <div class="status"><span class="d"></span> Available now</div>
      </a>
    </div>
  </div>

  <div class="pcip-footer">
    <div>&copy; 2026 &middot; PCIP Digital Twin</div>
    <div class="pcip-author">__AUTHOR_HTML__</div>
  </div>
</div>
"""

if PAPER_URL:
    paper_btn_html = (
        f'<a class="pcip-btn pcip-btn-ghost" href="{PAPER_URL}" '
        f'target="_blank" rel="noopener">Read the paper '
        f'<span class="pcip-arr">&rarr;</span></a>'
    )
else:
    paper_btn_html = (
        '<span class="pcip-btn pcip-btn-ghost pcip-btn-disabled" '
        'title="Preprint will be linked here once posted.">Paper coming soon</span>'
    )

if AUTHOR:
    author_html = f'Built by <span class="pcip-name">{AUTHOR}</span>'
else:
    author_html = ""

st.markdown(CSS.replace("__COVER__", cover_b64), unsafe_allow_html=True)
st.html(
    BODY.replace("__PAPER_BTN__", paper_btn_html)
        .replace("__AUTHOR_HTML__", author_html)
)
