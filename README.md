# PCIP Digital Twin

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://digital-twin-pcip.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**Live demo:** <https://digital-twin-pcip.streamlit.app/>

An interactive web companion to the manuscript *A Digital-Twin Model
for Personalized Cognitive Intervention Programs* (PCIP).

---

## About the project

Cognitive intervention programs — structured regimens of brain-training
games for older adults — produce highly heterogeneous outcomes. Two
participants on the same protocol can finish with very different
trajectories, and one-size-fits-all schedules leave gains on the table.

**PCIP is a digital-twin framework that models a participant's
session-by-session cognitive trajectory** under a given intervention
program. Once fit, the digital twin enables what-if analysis on
schedules, game pools, and difficulty policies *before* enrolling real
participants.

This repository hosts the public-facing web app that accompanies the
paper. It lets readers reproduce key figures, run their own
simulations, and fit the estimator end-to-end in the browser.

---

## What the app does

The app is organized as three modules, each accessible from the
landing page.

### 1. Sensitivity Analysis

Interactive sliders for each model component — game score, game
effectiveness, ability-difficulty mismatch, engagement, fatigue,
arousal, valence, the emotional multiplier, and practice bias. Every
component shows a single-curve "live" tab plus paper-style comparison
panels, with PNG and CSV downloads of the underlying curves.

### 2. Simulation

A full forward simulation of a cognitive intervention program. The
sidebar exposes program structure, the difficulty policy, the random
seed, and the model parameters in collapsible sections. You can save
and load configurations, **pin runs to overlay** their trajectories
against a new run, and download a dataset bundle to feed back into the
Estimation page.

### 3. Estimation

Fit the PCIP model to data — either freshly simulated in-app or
uploaded as a dataset bundle. A live progress bar and loss chart track
the optimization in real time, then four diagnostic tabs report
**convergence**, **global parameter recovery**, **latent recovery**,
and **observed game-score fit** with residual diagnostics. Problem size
is restricted in the hosted app so it completes within the Streamlit
Cloud free-tier compute budget; full-scale runs should be done locally.

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

Runs an end-to-end suite covering imports, all sensitivity-analysis
components, the simulator, the dataset round-trip, and the estimator.
Useful before committing non-trivial changes.

---

## Citation

If you use this app or the underlying model in your research, please
cite the accompanying paper. Full citation details will be added here
shortly.

---

## License

Released under the [MIT License](LICENSE). © 2026 Mohammad Fili.
