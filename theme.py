"""Shared theme + page header for internal pages.

The landing page (`app.py`) keeps its bespoke hero CSS — it's a one-off.
Internal pages (SA / Simulation / Estimation) all call
`setup_internal_page(title, subtitle)` to get:

- Dark navy palette + Google Fonts (Fraunces / Inter / JetBrains Mono)
- Streamlit chrome hidden, content padding tightened
- A branded header bar with a gradient mark + the page title +
  a "← Home" link back to the landing page
- Matplotlib rcParams adjusted to match (dark figure / axes, brand palette)

The CSS is intentionally minimal — Streamlit's built-in dark theme (set in
.streamlit/config.toml) handles sliders, selects, expanders, dataframes,
tabs, etc. natively. We only restyle what Streamlit doesn't.
"""

import matplotlib.pyplot as plt
import streamlit as st

# Palette mirrors landing-page design tokens
NAVY_000 = "#050D1C"
NAVY_100 = "#0A1628"
NAVY_200 = "#122038"
INK = "#F1F5FB"
INK_DIM = "rgba(241,245,251,0.72)"
INK_MUTE = "rgba(241,245,251,0.52)"
HAIR = "rgba(241,245,251,0.14)"
HAIR_SOFT = "rgba(241,245,251,0.08)"
CYAN = "#6FD9FF"
VIOLET = "#B794FF"
AMBER = "#FFC66B"
GREEN = "#86EFAC"
CORAL = "#F87171"
ROSE = "#FFADC9"

# Plot color cycle — high-contrast on dark, used by every figure
PLOT_PALETTE = [CYAN, VIOLET, AMBER, ROSE, GREEN, "#A78BFA"]

FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;"
    "0,9..144,600;1,9..144,300;1,9..144,400"
    "&family=Inter:wght@300;400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500&display=swap"
)

_GLOBAL_CSS = """
<style>
@import url("__FONTS_URL__");

:root {
  --navy-000: #050D1C; --navy-100: #0A1628; --navy-200: #122038;
  --ink: #F1F5FB; --ink-dim: rgba(241,245,251,0.72); --ink-mute: rgba(241,245,251,0.52);
  --hair: rgba(241,245,251,0.14); --hair-soft: rgba(241,245,251,0.08);
  --cyan: #6FD9FF; --violet: #B794FF; --amber: #FFC66B;
}

/* Hide Streamlit's top-right toolbar but keep header space minimal */
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"] { display: none; }
#MainMenu { display: none; }

/* Reduce default top padding so the header sits close to the top */
.block-container { padding-top: 1.2rem !important; padding-bottom: 3rem; }

/* Default body font */
.stApp, .stMarkdown, .stMarkdown p {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
}

/* Headings — serif display */
h1, h2, h3 {
  font-family: 'Fraunces', 'Inter', serif !important;
  font-weight: 400 !important;
  letter-spacing: -0.01em;
  color: var(--ink) !important;
}
h1 { font-size: 2.0rem !important; line-height: 1.05 !important; }
h2 { font-size: 1.4rem !important; }
h3 { font-size: 1.15rem !important; }

/* Captions / monospace flavoring */
.stCaption, [data-testid="stCaptionContainer"], small {
  color: var(--ink-mute) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.04em !important;
}

/* Sidebar refinements */
[data-testid="stSidebar"] {
  border-right: 1px solid var(--hair-soft);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--ink) !important; }
[data-testid="stSidebar"] .stSubheader { font-family: 'JetBrains Mono', monospace; }

/* Tab styling */
[data-baseweb="tab-list"] {
  gap: 6px;
  border-bottom: 1px solid var(--hair-soft);
}
[data-baseweb="tab"] {
  color: var(--ink-mute) !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em;
}
[data-baseweb="tab"][aria-selected="true"] {
  color: var(--cyan) !important;
}

/* Page-header bar */
.pcip-pagehdr {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 0 18px 0;
  margin-bottom: 14px;
  border-bottom: 1px solid var(--hair-soft);
}
.pcip-pagehdr .mark {
  width: 26px; height: 26px; border-radius: 7px;
  background: linear-gradient(135deg, var(--cyan) 0%, var(--violet) 100%);
  position: relative; flex-shrink: 0;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.06);
}
.pcip-pagehdr .mark::after {
  content: ""; position: absolute; inset: 5px; border-radius: 3px;
  background: var(--navy-000);
}
.pcip-pagehdr .mark::before {
  content: ""; position: absolute; left: 50%; top: 50%;
  width: 7px; height: 7px; border-radius: 50%;
  transform: translate(-50%,-50%);
  background: var(--amber); box-shadow: 0 0 8px var(--amber); z-index: 1;
}
.pcip-pagehdr .ttl {
  font-family: 'Fraunces', serif; font-weight: 400;
  font-size: 22px; color: var(--ink); letter-spacing: -0.01em;
  line-height: 1; padding-right: 10px;
}
.pcip-pagehdr .crumb {
  font-family: 'JetBrains Mono', monospace; font-size: 10px;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--ink-mute);
}
.pcip-pagehdr .spacer { flex: 1; }
.pcip-pagehdr a.home {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--ink-dim); text-decoration: none !important;
  padding: 6px 12px; border: 1px solid var(--hair);
  border-radius: 999px; background: rgba(255,255,255,0.02);
}
.pcip-pagehdr a.home:hover {
  color: var(--ink); border-color: rgba(255,255,255,0.22);
}
.pcip-pagehdr .sub {
  color: var(--ink-mute); font-size: 13px; margin-top: 4px;
}

/* Borderless containers should still feel grouped on dark bg */
[data-testid="stExpander"] details {
  border: 1px solid var(--hair-soft) !important;
  border-radius: 8px !important;
  background: rgba(255,255,255,0.02);
}
</style>
"""


def _setup_matplotlib() -> None:
    """Set matplotlib rcParams for dark figures + brand palette.

    matplotlib doesn't parse CSS-style `rgba(...)` strings, so we use
    `(r, g, b, a)` tuples (0–1) where transparency matters.
    """
    _ink = (241/255, 245/255, 251/255)             # INK as RGB
    _ink_dim = (*_ink, 0.72)                        # ink with alpha
    _hair = (*_ink, 0.14)
    _hair_soft = (*_ink, 0.08)

    plt.rcParams.update({
        "figure.facecolor": NAVY_100,
        "figure.edgecolor": NAVY_100,
        "savefig.facecolor": NAVY_100,
        "savefig.edgecolor": NAVY_100,
        "axes.facecolor": NAVY_000,
        "axes.edgecolor": _hair,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": _ink_dim,
        "ytick.color": _ink_dim,
        "text.color": INK,
        "grid.color": _hair_soft,
        "grid.linewidth": 0.6,
        "axes.prop_cycle": plt.cycler(color=PLOT_PALETTE),
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.labelcolor": INK,
    })


def setup_internal_page(title: str, subtitle: str | None = None,
                        crumb: str | None = None) -> None:
    """Apply theme CSS + render a branded header. Call at top of every page.

    Layout:
        [mark]  TITLE                                    [← Home]
                CRUMB · subtitle text wraps below

    `crumb` defaults to a short uppercase label inferred from the title.
    """
    _setup_matplotlib()
    st.markdown(_GLOBAL_CSS.replace("__FONTS_URL__", FONTS_URL),
                unsafe_allow_html=True)

    crumb_text = (crumb or title).upper()
    sub_html = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="pcip-pagehdr">
          <div class="mark"></div>
          <div>
            <div class="crumb">PCIP · {crumb_text}</div>
            <div class="ttl">{title}</div>
            {sub_html}
          </div>
          <div class="spacer"></div>
          <a class="home" href="/" target="_self">← Home</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
