# PCIP Digital Twin — Web App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://picpdigitaltwin.streamlit.app/)

**Live demo:** <https://picpdigitaltwin.streamlit.app/>

Browser-based companion to *A Digital-Twin Model for Personalized Cognitive
Intervention Programs*. Dark-themed Streamlit app with three live modules:

1. **Sensitivity Analysis** — interactive sliders for each of the 9 model
   components (ES, Z, Ψ, E, F, A, V, Emot, Bias). Single-curve view plus
   paper-style comparison tabs and downloadable PNG/CSV per component.
2. **Simulation** — full forward simulation. Sidebar exposes program
   structure, policy, seed, and all 30+ model parameters in collapsible
   sections. Save / load YAML configs, pin runs to overlay trajectories
   from prior runs, download a `.pt` dataset bundle.
3. **Estimation** — fit the model to either freshly simulated data or an
   uploaded `.pt` bundle. Live progress bar + loss chart during fitting,
   then four diagnostic tabs (convergence, global params recovery, latent
   recovery, OGS fit + residuals). Restricted scale (I ≤ 30, T ≤ 15,
   M ≤ 10) so it fits the Streamlit Cloud free-tier compute budget.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` isn't on your PATH (Windows is common), use
`python -m streamlit run app.py` instead. Opens at <http://localhost:8501>.

## Deploy to Streamlit Cloud

1. Push the `picp_digital_twin_app/` directory to a public GitHub
   repository (the simulator is **vendored inside** — see "Layout" — so
   the deploy is self-contained).
2. Go to <https://share.streamlit.io>, click **New app**.
3. Point it at the repo, set **Main file path** to `app.py`.
4. Default Python version (3.11+) is fine.

`requirements.txt` pulls torch from PyTorch's CPU-only wheel index
(`--extra-index-url https://download.pytorch.org/whl/cpu`) — about ~200 MB
instead of ~1 GB for the default GPU wheel.

## Configurable bits at the top of `app.py`

```python
PAPER_URL = ""               # set to bioRxiv URL when posted; later, journal DOI
AUTHOR    = "Mohammad Fili"  # cover-page footer attribution
```

While `PAPER_URL` is empty, the cover-page button reads "Paper coming
soon" in a dimmed, dashed style. Setting it makes the button live.

## Layout

```
picp_digital_twin_app/
├── app.py                              # landing page (dark theme + cover image)
├── pages/
│   ├── 1_Sensitivity_Analysis.py       # 9 components, sliders + tabs
│   ├── 2_Simulation.py                 # forward sim + 30+ params + pin/compare
│   └── 3_Estimation.py                 # fit + recovery diagnostics
├── theme.py                            # shared CSS, palette, page header
├── plotting.py                         # prettify, add_subplot_label, PNG export
├── components.py                       # numpy implementations of the 9 SA fns
├── Simulation_Estimation/              # vendored PyTorch simulator + estimator
│   ├── core/, policy/, estimator/
│   ├── games.py, participants.py, simulator.py
│   ├── config/default_params.yaml
│   └── SYNC_NOTE.md                    # how/when to refresh from the master
├── CoverImage.png                      # landing-page hero
├── requirements.txt
├── .streamlit/config.toml              # native dark theme + brand colors
├── .gitignore
├── _smoke_test.py                      # 11-case end-to-end test
└── pages_test_helpers.py               # smoke-test helper
```

The simulator at `Simulation_Estimation/` is a vendored copy. Pages
prefer it over the sibling `../Simulation_Estimation/` (used during local
development from the parent project). See `Simulation_Estimation/SYNC_NOTE.md`
for the re-sync recipe.

## Smoke test

```bash
python _smoke_test.py
```

Runs 11 cases: imports, all 9 SA components, plotting + PNG export, full
simulator pass (Staircase + Random + Null), bundle round-trip, estimator
with progress callback, forward pass, recovery dict, YAML config
round-trip, and the upload error path. Useful before committing
non-trivial changes.

## Tech notes

- **Dark theme** is set via `.streamlit/config.toml` (Streamlit's native
  primitives) plus shared CSS in `theme.py` (Fraunces / Inter / JetBrains
  Mono fonts, branded page header, palette). Matplotlib `rcParams` are
  set in the same place so figures match.
- **Caching**: the Simulation page wraps `run_simulation()` with
  `@st.cache_data(max_entries=8)` — identical parameter sets return
  instantly.
- **`fit()` callback**: a small backwards-compatible addition to the
  vendored `estimator/fit.py` adds an optional `progress_callback`
  parameter. The Estimation page uses it to drive the live progress bar
  and loss chart.
- **No equations on screen**: while the paper is unpublished the SA page
  shows sliders + plots + prose explanations only — closed-form
  equations are intentionally omitted.

## Roadmap

- Wire `PAPER_URL` to the bioRxiv preprint once posted; later, swap to
  the journal DOI.
- Optional polish: unified colormaps for the SA heatmaps, mobile-tighter
  layouts, save/load configs on the Estimation page too.
