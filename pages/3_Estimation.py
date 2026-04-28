"""Estimation page — fits PCIP model to simulated or uploaded data.

Restricted to small configurations (I≤30, T≤15, M≤10) to fit the Streamlit Cloud
free-tier compute budget. Data source is explicit: either re-simulate fresh
data here (with a transparent summary of the config used) or upload a `.pt`
dataset bundle saved from the Simulation page.

Uses fit()'s progress_callback to drive a live progress bar + loss readout.
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

# Prefer the in-app copy (so deploys are self-contained); fall back to the
# sibling Simulation_Estimation/ for local dev if someone hasn't synced.
_LOCAL = APP_ROOT / "Simulation_Estimation"
_SIBLING = APP_ROOT.parent / "Simulation_Estimation"
SIM_ROOT = _LOCAL if _LOCAL.exists() else _SIBLING
sys.path.insert(0, str(SIM_ROOT))

import yaml
import torch

from games import GamePool
from participants import ParticipantPool
from simulator import SimParams, Simulator, SimulationDataset
from policy import RandomPolicy, StaircasePolicy
from estimator import fit, compare_to_truth, forward_pass

from theme import setup_internal_page  # noqa: E402
from plotting import prettify, add_subplot_label, fig_to_png_bytes  # noqa: E402

st.set_page_config(page_title="Estimation", layout="wide")
setup_internal_page(
    "Estimation",
    subtitle="Fit the PCIP model to data and inspect parameter recovery. "
             "Restricted to small populations and short programs so it fits "
             "inside the Streamlit Cloud free-tier compute budget.",
    crumb="Estimation",
)

CONFIG_PATH = SIM_ROOT / "config" / "default_params.yaml"


@st.cache_data
def load_defaults():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


CFG_DEFAULT = load_defaults()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Training data")
    data_source = st.radio(
        "Source",
        ["Simulate here", "Upload .pt dataset"],
        index=0, key="est_source",
        help="`Simulate here` runs a fresh simulation with default model "
             "parameters at restricted size. `Upload` accepts a `.pt` bundle "
             "saved from the Simulation page (full custom config).",
    )

    upload = None
    if data_source == "Simulate here":
        st.caption("Restricted scale. Adjust on Simulation page for full-scale.")
        I = st.slider("I  (participants)", 5, 30, 20, 1, key="est_I")
        T = st.slider("T  (sessions)", 5, 15, 10, 1, key="est_T")
        M = st.slider("M  (games / session)", 3, 10, 5, 1, key="est_M")
        policy_name = st.selectbox("Policy", ["Staircase", "Random"], 0, key="est_policy")
        seed_sim = st.number_input("Sim seed", value=14, step=1, key="est_seed_sim")
    else:
        upload = st.file_uploader(
            "Upload .pt bundle",
            type=["pt", "pth"],
            help="Click 'Download dataset' on the Simulation page to produce one.",
            key="est_upload",
        )
        st.caption("File must contain: `dataset`, `game_pool`, `pool`, `cfg`.")

    st.subheader("Estimation")
    n_epochs = st.slider("n_epochs", 100, 1500, 500, 50, key="est_epochs")
    if n_epochs > 1000:
        st.caption(":warning: >1000 epochs may exceed free-tier compute on cloud deploy.")
    lr = st.slider("Learning rate (Adam)", 0.001, 0.05, 0.01, 0.001,
                   format="%.3f", key="est_lr")
    seed_est = st.number_input("Estimator seed", value=0, step=1, key="est_seed_est")

    with st.expander("Advanced"):
        stage2_valence = st.checkbox("Stage 2a — valence inversion (u_ijk)",
                                     value=True, key="est_s2v")
        stage2_arousal = st.checkbox("Stage 2b — arousal inversion (A*)",
                                     value=True, key="est_s2a")
        stage3_epochs = st.slider("Stage 3 re-opt epochs", 0, 500, 200, 25,
                                  key="est_s3e")
        lambda_gamma = st.slider("λ_γ₀ prior", 0.0, 5.0, 0.5, 0.1, key="est_lg0")
        lambda_gamma1 = st.slider("λ_γ₁ prior", 0.0, 5.0, 0.1, 0.05, key="est_lg1")

    run_clicked = st.button("Run estimation", type="primary", key="est_run")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _build_cfg() -> dict:
    cfg = load_defaults()
    cfg["simulation"]["program"]["I"] = int(I)
    cfg["simulation"]["program"]["T"] = int(T)
    cfg["simulation"]["program"]["M"] = int(M)
    return cfg


def simulate_here() -> tuple:
    cfg = _build_cfg()
    params = SimParams.from_config(cfg)
    SEED_GAME, SEED_POOL = 8, 12
    game_pool = GamePool.generate(
        K=params.K, Nk=params.Nk,
        hyperparams=cfg["simulation"]["game_pool"], seed=SEED_GAME,
    )
    pool = ParticipantPool.generate(
        I=params.I, K=params.K, J_total=game_pool.J_total, T=params.T,
        hyperparams=cfg["simulation"]["population"], seed=SEED_POOL,
    )
    gs = game_pool.to_structure()
    if policy_name == "Staircase":
        policy = StaircasePolicy(I=pool.I, M=params.M, game_structure=gs)
    else:
        policy = RandomPolicy(I=pool.I, M=params.M, game_structure=gs, seed=int(seed_sim))
    sim = Simulator(params)
    dataset = sim.run(pool, game_pool, policy, seed=int(seed_sim))
    src = {
        "kind": "simulate_here",
        "policy": policy_name, "seed_sim": int(seed_sim),
        "filename": None,
    }
    return dataset, gs, game_pool, pool, cfg, src


def load_uploaded(file) -> tuple:
    raw = file.read()
    try:
        bundle = torch.load(io.BytesIO(raw), weights_only=False)
    except Exception as e:
        st.error(
            f"Couldn't read `{file.name}` as a PyTorch checkpoint: {type(e).__name__}: {e}\n\n"
            "The Estimation page expects the `.pt` produced by clicking "
            "**Download dataset** on the Simulation page (or the format saved "
            "by `01_simulate.ipynb`)."
        )
        st.stop()

    if not isinstance(bundle, dict):
        st.error(
            f"`{file.name}` parsed but is not a dict (got `{type(bundle).__name__}`). "
            "Expected a dict with keys: `dataset`, `game_pool`, `pool`, `cfg`."
        )
        st.stop()

    required = {"dataset", "game_pool", "pool", "cfg"}
    missing = required - set(bundle.keys())
    if missing:
        st.error(
            f"`{file.name}` is missing keys: {sorted(missing)}.  "
            f"Found: {sorted(bundle.keys())}.  Expected: {sorted(required)}."
        )
        st.stop()

    dataset = bundle["dataset"]
    if not (hasattr(dataset, "ogs") and hasattr(dataset, "C_true")):
        st.error(
            "The `dataset` field doesn't look like a `SimulationDataset` "
            "(missing `.ogs` / `.C_true`). Was the file produced by a "
            "different version?"
        )
        st.stop()

    game_pool = bundle["game_pool"]
    pool = bundle["pool"]
    cfg = bundle["cfg"]
    try:
        gs = game_pool.to_structure()
    except Exception as e:
        st.error(f"`game_pool.to_structure()` failed: {type(e).__name__}: {e}")
        st.stop()

    src = {
        "kind": "upload",
        "policy": "(from file)", "seed_sim": None,
        "filename": file.name,
    }
    return dataset, gs, game_pool, pool, cfg, src


# ---------------------------------------------------------------------------
# Training-data summary card
# ---------------------------------------------------------------------------
def render_data_card(dataset, cfg, src):
    I_, T_, M_ = dataset.ogs.shape
    K_ = dataset.C_true.shape[-1]
    Nk = cfg["simulation"]["program"]["Nk"]

    glob = cfg["global_init"]
    dom = cfg["domain_init"]
    noise = cfg["simulation"]["noise"]

    if src["kind"] == "simulate_here":
        source_str = "Simulated on this page (default model parameters)"
    else:
        source_str = f"Uploaded — `{src['filename']}`"

    with st.container(border=True):
        st.markdown("**Training data**")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f"""
- **Source**: {source_str}
- **Population**: I = {I_} participants, K = {K_} domains, Nk = {Nk} games/domain
- **Program**: T = {T_} sessions, M = {M_} games/session
- **Policy**: {src['policy']}
- **Sim seed**: {src['seed_sim'] if src['seed_sim'] is not None else '—'}
"""
            )
        with c2:
            st.markdown(
                f"""
**Model parameters used to generate data**

| param | value |  | param | value |
|---|---|---|---|---|
| γ₀ | {glob['gamma0']} || λ_Z | {glob['lambda_Z']} |
| γ₁ | {glob['gamma1']} || κ_L / κ_R | {glob['kappa_L']} / {glob['kappa_R']} |
| α_u / u_min | {glob['alpha_u']} / {glob['u_min']} || α_A / α_V | {glob['alpha_A']} / {glob['alpha_V']} |
| η_k | {dom['eta_k']} || ρ_Q | {glob['rho_Q']} |
| σ_OGS | {noise['sigma_OGS']} || σ_π | {noise['sigma_pi']} |
"""
            )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_estimation(dataset, gs, game_pool, pool, cfg, src):
    with st.status("Fitting estimator...", expanded=True) as status:
        progress_bar = st.progress(0.0, text="Starting fit…")
        chart_placeholder = st.empty()

        loss_log: list[dict] = []

        def on_progress(step: int, total: int, info: dict):
            loss_log.append(info)
            frac = min(1.0, step / max(total, 1))
            progress_bar.progress(
                frac,
                text=f"Epoch {step}/{total}  ·  "
                     f"total={info['total']:.4f}  obs={info['obs']:.4f}  "
                     f"reg={info['reg']:.4f}",
            )
            if step % 20 == 0 or step == total:
                df = pd.DataFrame(loss_log)
                df["epoch"] = np.arange(1, len(df) + 1)
                chart_placeholder.line_chart(
                    df.set_index("epoch")[["total", "obs", "reg"]],
                    height=220,
                )

        t0 = time.time()
        result = fit(
            dataset=dataset,
            game_structure=gs,
            cfg=cfg,
            n_epochs=int(n_epochs),
            lr=float(lr),
            verbose=False,
            seed=int(seed_est),
            stage2_valence=bool(stage2_valence),
            stage2_arousal=bool(stage2_arousal),
            stage3_epochs=int(stage3_epochs),
            lambda_gamma=float(lambda_gamma),
            lambda_gamma1=float(lambda_gamma1),
            progress_callback=on_progress,
        )
        fit_time = time.time() - t0
        st.write(f"Fit done in {fit_time:.1f}s — {result['n_epochs_run']} epochs ran.")

        comparison = compare_to_truth(result["params"], game_pool, pool)
        ogs_pred = forward_pass(result["params"], dataset).detach().numpy()

        status.update(label=f"Done ({fit_time:.1f}s fit)", state="complete",
                      expanded=False)

    fitted = result["params"]
    I_, T_, M_ = dataset.ogs.shape
    return {
        "I": int(I_), "T": int(T_), "M": int(M_), "K": int(dataset.C_true.shape[-1]),
        "src": src,
        "n_epochs_run": result["n_epochs_run"],
        "best_loss": result["best_loss"],
        "loss_history": result["loss_history"],
        "comparison": comparison,
        "fit_time": fit_time,
        "ogs_obs": dataset.ogs.numpy(),
        "ogs_pred": ogs_pred,
        "true": {
            "zeta_jk":  game_pool.parameter_tensors()["zeta"].numpy(),
            "beta0_jk": game_pool.parameter_tensors()["beta0"].numpy(),
            "beta1_jk": game_pool.parameter_tensors()["beta1"].numpy(),
            "C_init":   pool.C_init_tensor().numpy(),
            "A_star":   pool.A_star_tensor().numpy(),
            "u_ijk":    pool.u_tensor().numpy(),
        },
        "est": {
            "zeta_jk":  fitted.zeta_jk.detach().numpy(),
            "beta0_jk": fitted.beta0_jk.detach().numpy(),
            "beta1_jk": fitted.beta1_jk.detach().numpy(),
            "C_init":   fitted.C_init.detach().numpy(),
            "A_star":   fitted.A_star.detach().numpy(),
            "u_ijk":    fitted.u_ijk.detach().numpy(),
        },
        "global_pairs": _global_param_pairs(SimParams.from_config(cfg), fitted),
        "noise_floor": float(cfg["simulation"]["noise"]["sigma_OGS"]) ** 2,
        "cfg_for_card": cfg,
    }


def _global_param_pairs(true_p, fitted) -> list[tuple[str, float, float]]:
    return [
        ("gamma0",   float(true_p.gamma0),   float(fitted.gamma0.detach())),
        ("gamma1",   float(true_p.gamma1),   float(fitted.gamma1.detach())),
        ("E_min",    float(true_p.E_min),    float(fitted.E_min.detach())),
        ("kappa_A",  float(true_p.kappa_A),  float(fitted.kappa_A.detach())),
        ("kappa_V",  float(true_p.kappa_V),  float(fitted.kappa_V.detach())),
        ("alpha_A",  float(true_p.alpha_A),  float(fitted.alpha_A.detach())),
        ("alpha_V",  float(true_p.alpha_V),  float(fitted.alpha_V.detach())),
        ("kappa_B",  float(true_p.kappa_B),  float(fitted.kappa_B.detach())),
        ("zeta_min", float(true_p.zeta_min), float(fitted.zeta_min.detach())),
        ("lambda_Z", float(true_p.lambda_Z), float(fitted.lambda_Z.detach())),
        ("kappa_L",  float(true_p.kappa_L),  float(fitted.kappa_L.detach())),
        ("kappa_R",  float(true_p.kappa_R),  float(fitted.kappa_R.detach())),
        ("u_min",    float(true_p.u_min),    float(fitted.u_min.detach())),
        ("alpha_u",  float(true_p.alpha_u),  float(fitted.alpha_u.detach())),
        ("rho0_F",   float(true_p.rho0_F),   float(fitted.rho0_F.detach())),
        ("rho_h_F",  float(true_p.rho_h_F),  float(fitted.rho_h_F.detach())),
        ("rho_e_F",  float(true_p.rho_e_F),  float(fitted.rho_e_F.detach())),
        ("tau_F",    float(true_p.tau_F),    float(fitted.tau_F.detach())),
        ("rho_Q",    float(true_p.rho_Q),    float(fitted.rho_Q.detach())),
    ]


# ---------------------------------------------------------------------------
# Show data-source preview (before Run)
# ---------------------------------------------------------------------------
preview = None
if data_source == "Simulate here":
    preview = {
        "kind": "simulate_here", "I": I, "T": T, "M": M,
        "policy": policy_name, "seed_sim": int(seed_sim),
        "Nk": int(CFG_DEFAULT["simulation"]["program"]["Nk"]),
        "K": int(CFG_DEFAULT["simulation"]["program"]["K"]),
        "cfg": CFG_DEFAULT,
        "filename": None,
    }
elif data_source == "Upload .pt dataset" and upload is not None:
    try:
        raw = upload.getvalue()
        bundle = torch.load(io.BytesIO(raw), weights_only=False)
        ds_p = bundle["dataset"]
        cfg_p = bundle["cfg"]
        preview = {
            "kind": "upload",
            "I": int(ds_p.ogs.shape[0]), "T": int(ds_p.ogs.shape[1]),
            "M": int(ds_p.ogs.shape[2]), "K": int(ds_p.C_true.shape[-1]),
            "Nk": int(cfg_p["simulation"]["program"]["Nk"]),
            "policy": "(from file)", "seed_sim": None,
            "cfg": cfg_p, "filename": upload.name,
        }
    except Exception as e:
        st.error(f"Couldn't read uploaded file: {e}")


def render_preview_card(p):
    glob = p["cfg"]["global_init"]
    dom = p["cfg"]["domain_init"]
    noise = p["cfg"]["simulation"]["noise"]
    if p["kind"] == "simulate_here":
        source_str = "Will be simulated on Run (default model parameters)"
    else:
        source_str = f"Uploaded — `{p['filename']}`"
    with st.container(border=True):
        st.markdown("**Training data — preview**")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"""
- **Source**: {source_str}
- **Population**: I = {p['I']} participants, K = {p['K']} domains, Nk = {p['Nk']} games/domain
- **Program**: T = {p['T']} sessions, M = {p['M']} games/session
- **Policy**: {p['policy']}
- **Sim seed**: {p['seed_sim'] if p['seed_sim'] is not None else '—'}
""")
        with c2:
            st.markdown(f"""
**Model parameters used to generate data**

| param | value |  | param | value |
|---|---|---|---|---|
| γ₀ | {glob['gamma0']} || λ_Z | {glob['lambda_Z']} |
| γ₁ | {glob['gamma1']} || κ_L / κ_R | {glob['kappa_L']} / {glob['kappa_R']} |
| α_u / u_min | {glob['alpha_u']} / {glob['u_min']} || α_A / α_V | {glob['alpha_A']} / {glob['alpha_V']} |
| η_k | {dom['eta_k']} || ρ_Q | {glob['rho_Q']} |
| σ_OGS | {noise['sigma_OGS']} || σ_π | {noise['sigma_pi']} |
""")


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_clicked:
    if data_source == "Upload .pt dataset" and upload is None:
        st.warning("No file uploaded. Pick a `.pt` file or switch to `Simulate here`.")
        st.stop()

    if data_source == "Simulate here":
        with st.spinner("Simulating data..."):
            dataset, gs, game_pool, pool, cfg, src = simulate_here()
    else:
        # rewind for the second read inside load_uploaded
        upload.seek(0)
        dataset, gs, game_pool, pool, cfg, src = load_uploaded(upload)

    st.session_state["est_result"] = run_estimation(
        dataset, gs, game_pool, pool, cfg, src,
    )

result = st.session_state.get("est_result")

if result is None:
    if preview is not None:
        render_preview_card(preview)
    st.info("Pick / upload data, configure estimation, click **Run estimation**. "
            "The training data summary above shows what will be fit.")
    st.stop()


# ---------------------------------------------------------------------------
# Show what was fit
# ---------------------------------------------------------------------------
class _FakeDataset:
    """Just enough shape info for render_data_card without re-loading tensors."""
    def __init__(self, I, T, M, K):
        self.ogs = type("S", (), {"shape": (I, T, M)})
        self.C_true = type("S", (), {"shape": (I, T + 1, K)})


fake_ds = _FakeDataset(result["I"], result["T"], result["M"], result["K"])
render_data_card(fake_ds, result["cfg_for_card"], result["src"])

st.caption(
    f"**Estimator:** epochs={result['n_epochs_run']}  ·  "
    f"fit time = {result['fit_time']:.1f}s  ·  "
    f"best loss = {result['best_loss']:.4f}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def downloads(fig, df, stem):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download PNG", data=fig_to_png_bytes(fig),
                           file_name=f"{stem}.png", mime="image/png",
                           key=f"png_{stem}")
    with c2:
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"{stem}.csv", mime="text/csv",
                           key=f"csv_{stem}")


def show_fig(fig):
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def status_color(rel_err: float) -> str:
    if rel_err < 0.15:
        return "background-color: #DCFCE7; color: #14532D;"
    if rel_err < 0.30:
        return "background-color: #FEF3C7; color: #78350F;"
    return "background-color: #FEE2E2; color: #7F1D1D;"


COLORS = ["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B3"]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_loss, tab_global, tab_latent, tab_ogs = st.tabs([
    "Convergence", "Global Parameters", "Latent Recovery", "OGS Fit",
])


with tab_loss:
    losses = pd.DataFrame(result["loss_history"])
    losses["epoch"] = np.arange(1, len(losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax = axes[0]
    ax.plot(losses["epoch"], losses["total"], color=COLORS[0], linewidth=1.5,
            label="total")
    ax.plot(losses["epoch"], losses["obs"], color=COLORS[1], linewidth=1.2,
            linestyle="--", label="obs")
    ax.plot(losses["epoch"], losses["reg"], color=COLORS[2], linewidth=1.2,
            linestyle=":", label="reg")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_yscale("log")
    ax.set_title("Loss curves  (log scale)")
    ax.legend(); prettify(ax)
    add_subplot_label(ax, "A", size=18)

    ax = axes[1]
    nf = result["noise_floor"]
    final = losses.iloc[-1]
    summary_lines = [
        f"Best total loss : {result['best_loss']:.5f}",
        f"Final obs loss  : {final['obs']:.5f}",
        f"Noise floor     : {nf:.5f}  (σ_OGS²)",
        f"Obs / floor     : {final['obs']/nf:.2f}×  (1.0 = perfect)",
        f"Epochs run      : {int(result['n_epochs_run'])}",
        f"Fit time        : {result['fit_time']:.1f} s",
    ]
    ax.axis("off")
    ax.text(0.05, 0.95, "\n".join(summary_lines),
            transform=ax.transAxes, va="top", ha="left",
            fontfamily="monospace", fontsize=11)
    add_subplot_label(ax, "B", size=18)

    fig.tight_layout()
    show_fig(fig)
    downloads(fig, losses, "convergence")


with tab_global:
    rows = []
    for name, true_v, est_v in result["global_pairs"]:
        rel = abs(est_v - true_v) / (abs(true_v) + 1e-9)
        flag = "OK" if rel < 0.15 else ("WARN" if rel < 0.30 else "BAD")
        rows.append({
            "parameter": name, "true": true_v, "estimated": est_v,
            "rel_error": rel, "status": flag,
        })
    df = pd.DataFrame(rows)
    n_ok = int((df["status"] == "OK").sum())
    n_warn = int((df["status"] == "WARN").sum())
    n_bad = int((df["status"] == "BAD").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total params", len(df))
    c2.metric("OK  (<15%)", n_ok)
    c3.metric("WARN  (15–30%)", n_warn)
    c4.metric("BAD  (≥30%)", n_bad)

    styled = df.style.format(
        {"true": "{:.4f}", "estimated": "{:.4f}", "rel_error": "{:.1%}"}
    ).map(lambda v: status_color(0 if v == "OK" else 0.2 if v == "WARN" else 0.4),
          subset=["status"])
    st.dataframe(styled, use_container_width=True, height=560)

    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="global_recovery.csv", mime="text/csv",
                       key="csv_global")


with tab_latent:
    panels = [
        ("ζ_jk",     "zeta_jk"),
        ("β₀_jk",    "beta0_jk"),
        ("β₁_jk",    "beta1_jk"),
        ("C_init",   "C_init"),
        ("A*",       "A_star"),
        ("u_ijk",    "u_ijk"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (label, key) in zip(axes.flat, panels):
        true = np.asarray(result["true"][key]).ravel()
        est = np.asarray(result["est"][key]).ravel()
        rmse = float(np.sqrt(np.mean((true - est) ** 2)))
        if true.std() > 1e-8 and est.std() > 1e-8:
            corr = float(np.corrcoef(true, est)[0, 1])
        else:
            corr = float("nan")
        ax.scatter(true, est, alpha=0.4, s=14, color=COLORS[0])
        lo = min(true.min(), est.min())
        hi = max(true.max(), est.max())
        ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{label}  (true)")
        ax.set_ylabel(f"{label}  (est)")
        ax.set_title(f"{label}   RMSE={rmse:.3f}   r={corr:.3f}")
        prettify(ax)
    fig.tight_layout()
    show_fig(fig)

    rec_df = pd.DataFrame([
        {"param": p[0],
         "rmse": float(np.sqrt(np.mean(
             (np.asarray(result["true"][p[1]]).ravel()
              - np.asarray(result["est"][p[1]]).ravel()) ** 2))),
         "corr": (np.corrcoef(
             np.asarray(result["true"][p[1]]).ravel(),
             np.asarray(result["est"][p[1]]).ravel(),
         )[0, 1] if np.asarray(result["true"][p[1]]).std() > 1e-8 else float("nan"))}
        for p in panels
    ])
    downloads(fig, rec_df, "latent_recovery")


with tab_ogs:
    obs = result["ogs_obs"].ravel()
    pred = result["ogs_pred"].ravel()
    resid = pred - obs                                          # bias check: + means over-prediction
    sessions = np.arange(1, result["T"] + 1)
    obs_t = result["ogs_obs"].mean(axis=(0, 2))
    pred_t = result["ogs_pred"].mean(axis=(0, 2))
    resid_t = pred_t - obs_t                                    # mean residual per session

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))

    # Sample for the scatter plot to keep it responsive
    rng = np.random.default_rng(0)
    if obs.size > 5000:
        idx = rng.choice(obs.size, 5000, replace=False)
        obs_s = obs[idx]; pred_s = pred[idx]
    else:
        obs_s = obs; pred_s = pred

    # A — observed vs predicted scatter
    ax = axes[0, 0]
    ax.scatter(obs_s, pred_s, alpha=0.25, s=6, color=COLORS[0])
    ax.plot([0, 100], [0, 100], color="grey", linestyle="--", linewidth=1)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    corr = float(np.corrcoef(obs, pred)[0, 1])
    ax.set_xlabel("OGS observed"); ax.set_ylabel("OGS predicted")
    ax.set_title(f"Observed vs predicted   RMSE={rmse:.2f}   r={corr:.3f}")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    prettify(ax); add_subplot_label(ax, "A", size=18)

    # B — session-mean OGS (true vs predicted)
    ax = axes[0, 1]
    ax.plot(sessions, obs_t, color=COLORS[0], linewidth=2, label="Observed")
    ax.plot(sessions, pred_t, color=COLORS[3], linewidth=2,
            linestyle="--", label="Predicted")
    ax.set_xlabel("Session"); ax.set_ylabel("Mean OGS")
    ax.set_title("Session-mean OGS  (true vs predicted)")
    ax.legend(); prettify(ax); add_subplot_label(ax, "B", size=18)

    # C — residual histogram (pred - obs)
    ax = axes[1, 0]
    ax.hist(resid, bins=60, color=COLORS[2], edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    bias = float(resid.mean())
    sd = float(resid.std())
    ax.axvline(bias, color="red", linestyle="--", linewidth=1.4,
               label=f"mean = {bias:+.2f}")
    ax.set_xlabel("Residual  (predicted − observed)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual distribution   bias={bias:+.2f}   sd={sd:.2f}")
    ax.legend(); prettify(ax); add_subplot_label(ax, "C", size=18)

    # D — residual vs session (band of ±1 SD across (i, m))
    ax = axes[1, 1]
    resid_full = result["ogs_pred"] - result["ogs_obs"]          # [I, T, M]
    resid_session_mean = resid_full.mean(axis=(0, 2))             # [T]
    resid_session_sd = resid_full.std(axis=(0, 2))                # [T]
    ax.plot(sessions, resid_session_mean, color=COLORS[3], linewidth=2)
    ax.fill_between(sessions, resid_session_mean - resid_session_sd,
                    resid_session_mean + resid_session_sd,
                    color=COLORS[3], alpha=0.2)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Session"); ax.set_ylabel("Residual  (mean ± 1 SD)")
    ax.set_title("Residual vs session  (systematic drift?)")
    prettify(ax); add_subplot_label(ax, "D", size=18)

    fig.tight_layout()
    show_fig(fig)

    df = pd.DataFrame({
        "session": sessions,
        "ogs_obs": obs_t, "ogs_pred": pred_t,
        "resid_mean": resid_session_mean, "resid_sd": resid_session_sd,
    })
    downloads(fig, df, "ogs_fit")
