# Simulation_Estimation — vendored copy

This is a **copy** of `Code/Simulation_Estimation/` at the repo root, included so
this app deploys self-contained (Streamlit Cloud needs everything in the same
repo).

The pages prefer this local copy via `sys.path` lookup; they fall back to the
sibling `../Simulation_Estimation/` only if this directory is missing.

## Files included
- `core/` (kernels, observation, gain, dynamics)
- `policy/` (base, random, staircase)
- `estimator/` (parameters, loss, fit, diagnostics)
- `games.py`, `participants.py`, `simulator.py`
- `config/default_params.yaml`

## Files NOT included (intentionally)
- `*.ipynb` — notebooks are reference material, not runtime
- `data/` — outputs from local runs
- `*.md`, `*.pdf`, `*.txt` — manuscript / session notes

## Local modifications
- `estimator/fit.py` — adds an optional `progress_callback` parameter for the
  Estimation page's live progress bar. Backwards-compatible (default `None`).

## Re-syncing
When you change the master at `../Simulation_Estimation/`, re-copy the affected
files. Quick refresh of everything (from `Code/`):

```bash
DST=picp_digital_twin_app/Simulation_Estimation
rm -rf $DST/core $DST/policy $DST/estimator $DST/config
cp -r Simulation_Estimation/core Simulation_Estimation/policy \
      Simulation_Estimation/estimator $DST/
cp Simulation_Estimation/games.py Simulation_Estimation/participants.py \
   Simulation_Estimation/simulator.py $DST/
mkdir -p $DST/config
cp Simulation_Estimation/config/default_params.yaml $DST/config/
find $DST -type d -name __pycache__ -exec rm -rf {} +
```

Then re-apply the `progress_callback` patch to `fit.py` (or upstream the change
to the master so the next sync includes it automatically).
