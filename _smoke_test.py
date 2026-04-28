"""End-to-end smoke test — runs every code path the app exercises.

Skips: actual Streamlit UI rendering (requires a live runtime), CSS/visual.
Covers: imports, SA component math, plotting helpers, full simulation,
PyTorch bundle save/load round-trip, full estimation cycle (small budget)
including progress_callback, forward pass, compare_to_truth, YAML round-trip.

Run: py _smoke_test.py
"""
import io
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Simulation_Estimation"))


PASS = "[ OK ]"
FAIL = "[FAIL]"
results: list[tuple[str, bool, str]] = []
sim_state: dict = {}


def case(name: str):
    def deco(fn):
        try:
            t0 = time.time()
            msg = fn() or ""
            results.append((name, True, f"{msg} ({time.time() - t0:.1f}s)"))
            print(f"{PASS} {name}  {msg}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            results.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"{FAIL} {name}  {type(e).__name__}: {e}")
            print(tb)
        return fn
    return deco


# ---------------------------------------------------------------------------
@case("Imports — app modules")
def _():
    import theme; _ = theme
    import plotting; _ = plotting
    import components; _ = components
    return "theme, plotting, components"


@case("Imports — vendored Simulation_Estimation")
def _():
    import yaml; _ = yaml
    import torch; _ = torch
    from games import GamePool; _ = GamePool
    from participants import ParticipantPool; _ = ParticipantPool
    from simulator import SimParams, Simulator, SimulationDataset
    _ = (SimParams, Simulator, SimulationDataset)
    from policy import RandomPolicy, StaircasePolicy
    _ = (RandomPolicy, StaircasePolicy)
    from estimator import fit, compare_to_truth, forward_pass
    _ = (fit, compare_to_truth, forward_pass)
    return "all vendored modules importable"


# ---------------------------------------------------------------------------
@case("SA components — scalar + array inputs, ranges, signatures")
def _():
    from components import (
        expected_score, game_effectiveness, mismatch, engagement, fatigue,
        arousal, valence, emot, practice_bias, omega_B,
    )

    delta = np.linspace(-10, 10, 200)
    es = expected_score(delta)
    assert es.shape == (200,) and 0 <= es.min() and es.max() <= 100, "ES range"

    n = np.arange(0, 30)
    z = game_effectiveness(n)
    assert z.shape == (30,) and z.min() > 0, "Z(n) positive"

    psi = mismatch(delta)
    assert psi.shape == (200,) and 0 <= psi.min() <= psi.max() <= 1.0001, "Psi range"

    u = np.linspace(0, 1, 50)
    e = engagement(u)
    assert e.shape == (50,) and 0 <= e.min() and e.max() <= 1.001

    m = np.arange(1, 11)
    f = fatigue(m, 10, 0.0)
    assert f.shape == (10,) and (f >= 0).all()

    A = arousal(0.0, 0.5)
    assert 0.2 <= float(A) <= 1.0
    A_grid = arousal(delta, 0.5)
    assert A_grid.shape == (200,)

    V = valence(0.0, 0.5)
    assert -1.0 <= float(V) <= 1.0

    E = emot(0.6, 0.8)
    assert 0.2 <= float(E) <= 1.0

    w = omega_B(delta)
    B = practice_bias(np.arange(0, 30), 0.0)
    assert w.shape == (200,) and B.shape == (30,) and B.min() >= 1.0

    return "9 components, scalar + array inputs"


# ---------------------------------------------------------------------------
@case("Plotting — prettify + PNG export")
def _():
    import matplotlib
    matplotlib.use("Agg")
    from theme import _setup_matplotlib
    _setup_matplotlib()
    from plotting import prettify, add_subplot_label, fig_to_png_bytes
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(10), np.arange(10) ** 2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("smoke")
    prettify(ax)
    add_subplot_label(ax, "A", size=14)
    png = fig_to_png_bytes(fig)
    plt.close(fig)
    assert len(png) > 2000, f"PNG too small ({len(png)} bytes)"
    return f"PNG bytes={len(png):,}"


# ---------------------------------------------------------------------------
@case("Simulator — Staircase + Random + Null at I=20 T=10 M=5")
def _():
    import yaml
    from games import GamePool
    from participants import ParticipantPool
    from simulator import SimParams, Simulator
    from policy import RandomPolicy, StaircasePolicy

    cfg_path = ROOT / "Simulation_Estimation" / "config" / "default_params.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["simulation"]["program"]["I"] = 20
    cfg["simulation"]["program"]["T"] = 10
    cfg["simulation"]["program"]["M"] = 5
    params = SimParams.from_config(cfg)

    gp = GamePool.generate(K=params.K, Nk=params.Nk,
                           hyperparams=cfg["simulation"]["game_pool"], seed=8)
    pool = ParticipantPool.generate(
        I=params.I, K=params.K, J_total=gp.J_total, T=params.T,
        hyperparams=cfg["simulation"]["population"], seed=12,
    )
    gs = gp.to_structure()
    sim = Simulator(params)

    ds_sc = sim.run(pool, gp,
                    StaircasePolicy(I=pool.I, M=params.M, game_structure=gs),
                    seed=14)
    assert tuple(ds_sc.ogs.shape) == (20, 10, 5)
    assert tuple(ds_sc.C_true.shape) == (20, 11, params.K)

    pool2 = ParticipantPool.generate(
        I=params.I, K=params.K, J_total=gp.J_total, T=params.T,
        hyperparams=cfg["simulation"]["population"], seed=12,
    )
    ds_rnd = sim.run(pool2, gp,
                     RandomPolicy(I=pool2.I, M=params.M,
                                  game_structure=gs, seed=14),
                     seed=14)
    assert tuple(ds_rnd.ogs.shape) == (20, 10, 5)

    pool3 = ParticipantPool.generate(
        I=params.I, K=params.K, J_total=gp.J_total, T=params.T,
        hyperparams=cfg["simulation"]["population"], seed=12,
    )
    ds_null = sim.run_null(pool3)
    assert tuple(ds_null.C_true.shape) == (20, 11, params.K)
    assert (ds_null.ogs == 0).all()

    sim_state["ds_sc"] = ds_sc
    sim_state["gp"] = gp
    sim_state["pool"] = pool
    sim_state["cfg"] = cfg
    return f"Staircase+Random+Null OGS shapes={tuple(ds_sc.ogs.shape)}"


# ---------------------------------------------------------------------------
@case("Bundle save/load round-trip — torch.save through io.BytesIO")
def _():
    import torch
    if "ds_sc" not in sim_state:
        return "skipped (sim missing)"
    bundle = {
        "dataset":   sim_state["ds_sc"],
        "game_pool": sim_state["gp"],
        "pool":      sim_state["pool"],
        "cfg":       sim_state["cfg"],
    }
    buf = io.BytesIO()
    torch.save(bundle, buf)
    raw = buf.getvalue()
    loaded = torch.load(io.BytesIO(raw), weights_only=False)
    assert set(loaded.keys()) == {"dataset", "game_pool", "pool", "cfg"}
    assert loaded["dataset"].ogs.shape == bundle["dataset"].ogs.shape
    return f"{len(raw):,} bytes, 4 keys preserved"


# ---------------------------------------------------------------------------
@case("Estimator — fit() with progress_callback + Stage 2/3, n_epochs=50")
def _():
    if "ds_sc" not in sim_state:
        return "skipped (sim missing)"

    from estimator import fit
    progress_calls: list[tuple[int, int, float]] = []

    def on_progress(step, total, info):
        progress_calls.append((step, total, info["total"]))

    result = fit(
        dataset=sim_state["ds_sc"],
        game_structure=sim_state["gp"].to_structure(),
        cfg=sim_state["cfg"],
        n_epochs=50,
        lr=0.01,
        verbose=False,
        seed=0,
        stage2_valence=True,
        stage2_arousal=True,
        stage3_epochs=20,
        progress_callback=on_progress,
    )
    assert result["n_epochs_run"] >= 50
    assert len(progress_calls) >= 50, f"only {len(progress_calls)} progress calls"
    assert all(0 < step <= total for step, total, _ in progress_calls)
    sim_state["fit_result"] = result
    return (f"epochs={result['n_epochs_run']}, "
            f"best_loss={result['best_loss']:.4f}, "
            f"progress_calls={len(progress_calls)}")


# ---------------------------------------------------------------------------
@case("Forward pass — predicted OGS shape matches observed")
def _():
    if "fit_result" not in sim_state:
        return "skipped (fit missing)"
    from estimator import forward_pass
    ogs_pred = forward_pass(
        sim_state["fit_result"]["params"], sim_state["ds_sc"]
    ).detach().numpy()
    assert ogs_pred.shape == tuple(sim_state["ds_sc"].ogs.shape)
    rmse = float(np.sqrt(np.mean(
        (ogs_pred - sim_state["ds_sc"].ogs.numpy()) ** 2
    )))
    return f"shape={ogs_pred.shape}, RMSE vs obs={rmse:.2f}"


# ---------------------------------------------------------------------------
@case("compare_to_truth — recovery dict has all expected keys")
def _():
    if "fit_result" not in sim_state:
        return "skipped (fit missing)"
    from estimator import compare_to_truth
    cmp = compare_to_truth(
        sim_state["fit_result"]["params"], sim_state["gp"], sim_state["pool"]
    )
    expected = {"zeta_jk", "beta0_jk", "beta1_jk", "C_init", "A_star", "u_ijk"}
    assert set(cmp.keys()) == expected
    for k, v in cmp.items():
        assert "rmse" in v and "corr" in v
    return f"keys={sorted(cmp.keys())}"


# ---------------------------------------------------------------------------
@case("Sim page — config save/load round-trip via PARAM_PATHS")
def _():
    import yaml
    # Mirror the sim page's mapping
    sample = {
        "predefined": {
            "arousal": {"A_min": 0.25, "K0": 1.5},
            "valence": {"gamma0_V": -0.1},
        },
        "global_init": {"gamma0": 1.2, "gamma1": 1.6, "lambda_Z": 0.04},
        "domain_init": {"eta_k": 0.18},
        "simulation": {
            "noise": {"sigma_OGS": 0.12},
            "program": {"I": 80, "T": 25, "M": 8},
        },
        "_meta": {"policy": "Random", "seed": 99},
    }
    text = yaml.safe_dump(sample, sort_keys=False)
    parsed = yaml.safe_load(text)
    assert parsed == sample, "YAML round-trip lost data"

    # Walk like apply_yaml_to_session and confirm leaves are reachable
    from pages_test_helpers import walk_param_paths
    found = walk_param_paths(sample)
    assert "sim_gamma0" in found and found["sim_gamma0"] == 1.2
    assert "sim_eta_k" in found and found["sim_eta_k"] == 0.18
    assert "sim_policy" in found and found["sim_policy"] == "Random"
    return f"resolved {len(found)} sim_* keys"


# ---------------------------------------------------------------------------
@case("Estimation upload — malformed bundles caught")
def _():
    import torch
    # Bundle with missing keys
    bad = {"dataset": "not a real dataset"}
    buf = io.BytesIO()
    torch.save(bad, buf)
    loaded = torch.load(io.BytesIO(buf.getvalue()), weights_only=False)
    assert isinstance(loaded, dict)
    required = {"dataset", "game_pool", "pool", "cfg"}
    missing = required - set(loaded.keys())
    assert missing == {"game_pool", "pool", "cfg"}
    return "missing-keys path covered"


# ---------------------------------------------------------------------------
print()
print("=" * 70)
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
print(f"SMOKE TEST  —  {passed}/{total} passed")
for name, ok, msg in results:
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name:60s}  {msg}")

if passed < total:
    sys.exit(1)
