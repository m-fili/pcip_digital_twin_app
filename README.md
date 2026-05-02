# PCIP Digital Twin

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://digital-twin-pcip.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**Live demo:** <https://digital-twin-pcip.streamlit.app/>

An interactive web companion to the manuscript *A Digital-Twin Model for
Personalized Cognitive Intervention Programs* (PCIP).

---

## About the project

Cognitive intervention programs (CIPs) — structured regimens of
brain-training games targeting executive function, memory, and
visuospatial reasoning — produce highly heterogeneous outcomes across
older adults. Two participants on the same protocol can finish with very
different trajectories, and one-size-fits-all schedules leave gains on
the table.

**PCIP is a digital-twin framework that models a participant's
session-by-session cognitive trajectory** under a given intervention
program. It couples three layers:

- a **psychometric layer** that turns latent ability and game difficulty
  into observed game scores via an IRT-style link, modulated by
  practice effects and an emotional regulator (arousal × valence);
- a **gain layer** that decomposes per-session learning into game
  effectiveness, ability-difficulty mismatch, engagement, and fatigue;
- a **dynamics layer** that accumulates gains into a latent ability
  trajectory across sessions and games, with power-law forgetting and
  adaptive delta-rule updating.

The model was developed against the ADNI cohort (N = 309 cognitively
normal older adults) and covers three cognitive domains — executive
function (EF), memory (MEM), and visuospatial (VS). Once fit, the
digital twin enables what-if analysis on schedules, game pools, and
difficulty policies *before* enrolling real participants.

This repository hosts the **public-facing web app** that accompanies
the paper. It lets readers reproduce every figure, run their own
simulations, and fit the estimator end-to-end in the browser.

---

## What the app does

The app is organized as three modules, each accessible from the
landing page.

### 1. Sensitivity Analysis

Interactive sliders for each of the nine model components — game
score (ES), game effectiveness Z(n), mismatch Ψ(δ), engagement E(u),
fatigue F(m, δ), arousal A, valence V, the emotional multiplier Emot,
and practice bias. Every component shows a single-curve "live" tab
plus paper-style comparison panels, with PNG and CSV downloads of
the underlying curves.

### 2. Simulation

A full forward simulation of a CIP. The sidebar exposes program
structure (participants, sessions, games-per-session), the difficulty
policy (staircase or random), the random seed, and all 30+ model
parameters in collapsible sections. You can save and load YAML
configurations, **pin runs to overlay** their trajectories against a
new run, and download a `.pt` dataset bundle to feed back into the
Estimation page.

### 3. Estimation

Fit the PCIP model to data — either freshly simulated in-app or
uploaded as a `.pt` bundle. A live progress bar and loss chart track
the optimization in real time, then four diagnostic tabs report
**convergence**, **global parameter recovery**, **latent recovery**
(individual abilities, arousal targets, game preferences), and
**OGS fit** with residual diagnostics. Fitting is restricted to a
small problem size (I ≤ 30, T ≤ 15, M ≤ 10) so it completes within
the Streamlit Cloud free-tier compute budget; full-scale runs should
be done locally or against the source repository.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If `streamlit` is not on your `PATH` (common on Windows), use
`python -m streamlit run app.py`. The app opens at
<http://localhost:8501>.

`requirements.txt` pulls **CPU-only PyTorch** via
`--extra-index-url https://download.pytorch.org/whl/cpu` (~200 MB) so
the deploy fits within Streamlit Cloud's free-tier limits. If you have
a CUDA build of PyTorch already installed locally, you can drop the
`torch` line from `requirements.txt`.

### Smoke test

```bash
python _smoke_test.py
```

Runs 11 end-to-end cases — imports, all SA components, simulator
(staircase / random / null policies), `.pt` bundle round-trip,
estimator with progress callback, recovery diagnostics, YAML config
round-trip, and the upload-validation error path. Useful before
committing non-trivial changes.

---

## Repository layout

```
picp_digital_twin_app/
├── app.py                        # Landing page (dark theme + cover image)
├── pages/
│   ├── 1_Sensitivity_Analysis.py
│   ├── 2_Simulation.py
│   └── 3_Estimation.py
├── theme.py                      # Shared CSS, palette, page header bar
├── plotting.py                   # prettify, add_subplot_label, PNG export
├── components.py                 # NumPy implementations of the 9 SA functions
├── Simulation_Estimation/        # Vendored PyTorch simulator + estimator
│   ├── core/, policy/, estimator/
│   ├── games.py, participants.py, simulator.py
│   └── config/default_params.yaml
├── CoverImage.png                # Landing-page hero
├── requirements.txt
├── .streamlit/config.toml        # Native dark theme + brand colors
├── _smoke_test.py                # 11-case test harness
└── LICENSE                       # MIT
```

The PyTorch simulator under `Simulation_Estimation/` is a vendored copy
of the master implementation, kept in-tree so the deploy is
self-contained. The pages prefer this local copy over any sibling
`../Simulation_Estimation/` used during local development.

---

## Deployment

The live app is hosted on **Streamlit Community Cloud** from this
repository's `main` branch, with `app.py` as the entry point. To
redeploy a fork:

1. Push the directory to a public GitHub repository.
2. Visit <https://share.streamlit.io>, click **New app**, and point it
   at the repo with `app.py` as the main file.
3. Default Python version (3.11+) is fine.

---

## Citation

The accompanying paper is currently in preparation. Once the preprint
is available, the landing-page **"Paper"** button will go live and a
citation block will be added here.

---

## License

Released under the [MIT License](LICENSE). © 2026 Mohammad Fili.
