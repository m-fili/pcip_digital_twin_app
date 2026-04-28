"""Simulation page — runs the PCIP forward simulator from Simulation_Estimation/.

Sidebar exposes program structure, policy, seed, and all 30+ model parameters
(grouped into collapsible sections, defaults from default_params.yaml).
Outputs: 3 tabs (OGS Trajectories, Ability Growth, Gain & Affect Dynamics).

Features:
- Save / Load config (YAML) at top + bottom of sidebar
- Cached simulation (identical params return instantly)
- Pin runs to overlay trajectories from prior runs on the current plots
- Download dataset bundle (.pt) for the Estimation page
"""
import io
import sys
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
from simulator import SimParams, Simulator
from policy import RandomPolicy, StaircasePolicy

from theme import setup_internal_page  # noqa: E402
from plotting import prettify, add_subplot_label, fig_to_png_bytes  # noqa: E402

st.set_page_config(page_title="Simulation", layout="wide")
setup_internal_page(
    "Simulation",
    subtitle="Forward-simulate the PCIP intervention program. "
             "Adjust program structure, policy, and any model parameter, then hit Run.",
    crumb="Simulation",
)

CONFIG_PATH = SIM_ROOT / "config" / "default_params.yaml"


@st.cache_data
def load_defaults():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


CFG = load_defaults()
PRE = CFG["predefined"]
GLOB = CFG["global_init"]
DOM = CFG["domain_init"]
NOISE = CFG["simulation"]["noise"]
PROG = CFG["simulation"]["program"]


# ---------------------------------------------------------------------------
# Save / Load config helpers
# ---------------------------------------------------------------------------
# Maps YAML leaf path -> session_state widget key. Order doesn't matter; this
# is the single source of truth for "what's tweakable from the sidebar".
PARAM_PATHS = [
    # (yaml_path_tuple, widget_key)
    (("simulation", "program", "I"), "sim_I"),
    (("simulation", "program", "T"), "sim_T"),
    (("simulation", "program", "M"), "sim_M"),
    (("_meta", "policy"),             "sim_policy"),
    (("_meta", "seed"),               "sim_seed"),
    (("predefined", "arousal", "A_min"),    "sim_A_min"),
    (("predefined", "arousal", "gamma0_A"), "sim_gamma0_A"),
    (("predefined", "arousal", "gamma1_A"), "sim_gamma1_A"),
    (("predefined", "arousal", "K0"),       "sim_K0"),
    (("predefined", "arousal", "rho_u"),    "sim_rho_u"),
    (("predefined", "valence", "gamma0_V"), "sim_gamma0_V"),
    (("predefined", "valence", "gamma_Vu"), "sim_gamma_Vu"),
    (("predefined", "valence", "gamma_Vh"), "sim_gamma_Vh"),
    (("predefined", "valence", "gamma_Ve"), "sim_gamma_Ve"),
    (("predefined", "valence", "tau_V"),    "sim_tau_V"),
    (("predefined", "valence", "beta_Vh"),  "sim_beta_Vh"),
    (("predefined", "valence", "beta_Ve"),  "sim_beta_Ve"),
    (("global_init", "gamma0"),       "sim_gamma0"),
    (("global_init", "gamma1"),       "sim_gamma1"),
    (("global_init", "E_min"),        "sim_E_min"),
    (("global_init", "kappa_A"),      "sim_kappa_A"),
    (("global_init", "kappa_V"),      "sim_kappa_V"),
    (("global_init", "alpha_A"),      "sim_alpha_A"),
    (("global_init", "alpha_V"),      "sim_alpha_V"),
    (("global_init", "kappa_B"),      "sim_kappa_B"),
    (("global_init", "delta_B_star"), "sim_delta_B_star"),
    (("global_init", "zeta_min"),     "sim_zeta_min"),
    (("global_init", "lambda_Z"),     "sim_lambda_Z"),
    (("global_init", "kappa_L"),      "sim_kappa_L"),
    (("global_init", "kappa_R"),      "sim_kappa_R"),
    (("global_init", "delta_star"),   "sim_delta_star"),
    (("global_init", "u_min"),        "sim_u_min"),
    (("global_init", "alpha_u"),      "sim_alpha_u"),
    (("global_init", "rho0_F"),       "sim_rho0_F"),
    (("global_init", "rho_h_F"),      "sim_rho_h_F"),
    (("global_init", "rho_e_F"),      "sim_rho_e_F"),
    (("global_init", "tau_F"),        "sim_tau_F"),
    (("global_init", "rho_Q"),        "sim_rho_Q"),
    (("domain_init", "eta_k"),        "sim_eta_k"),
    (("domain_init", "Bk_max"),       "sim_Bk_max"),
    (("domain_init", "Bk_min"),       "sim_Bk_min"),
    (("simulation", "noise", "sigma_OGS"), "sim_sigma_OGS"),
    (("simulation", "noise", "sigma_pi"),  "sim_sigma_pi"),
]


def session_to_yaml() -> str:
    """Build a structured YAML reflecting current sidebar state."""
    out: dict = {}
    for path, key in PARAM_PATHS:
        if key not in st.session_state:
            continue
        cur = out
        for p in path[:-1]:
            cur = cur.setdefault(p, {})
        cur[path[-1]] = st.session_state[key]
    return yaml.safe_dump(out, sort_keys=False)


def apply_yaml_to_session(text: str) -> tuple[int, list[str]]:
    """Parse YAML and apply leaf values to session_state. Returns (n_applied, warnings)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return 0, [f"YAML parse error: {e}"]
    if not isinstance(data, dict):
        return 0, ["Top-level YAML must be a mapping."]

    n = 0
    warnings = []
    for path, key in PARAM_PATHS:
        cur = data
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False; break
        if ok:
            st.session_state[key] = cur
            n += 1
    return n, warnings


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    with st.expander("⚙ Configuration", expanded=False):
        st.caption("Save the current sidebar as YAML, or load a saved config.")
        loaded = st.file_uploader("Load config (.yaml)",
                                   type=["yaml", "yml"], key="sim_load")
        if loaded is not None and st.session_state.get("sim_loaded_name") != loaded.name:
            n, warns = apply_yaml_to_session(loaded.read().decode("utf-8"))
            for w in warns:
                st.warning(w)
            if n > 0:
                st.session_state["sim_loaded_name"] = loaded.name
                st.toast(f"Applied {n} parameter values from {loaded.name}")
                st.rerun()

    if st.button("Reset all to defaults", key="sim_reset_all"):
        for k in [k for k in list(st.session_state.keys()) if k.startswith("sim_")
                  and k not in {"sim_run", "sim_result", "sim_key",
                                "sim_reset_all", "sim_pins", "sim_loaded_name"}]:
            st.session_state.pop(k, None)
        st.rerun()

    st.subheader("Program")
    I = st.slider("I  (participants)", 10, 200, int(PROG["I"]), 10, key="sim_I")
    T = st.slider("T  (sessions)", 5, 50, int(PROG["T"]), 1, key="sim_T")
    M = st.slider("M  (games / session)", 3, 15, int(PROG["M"]), 1, key="sim_M")

    st.subheader("Policy")
    policy_name = st.selectbox("Policy", ["Staircase", "Random"], index=0, key="sim_policy")

    st.subheader("Seed")
    seed = st.number_input("Random seed", value=14, step=1, key="sim_seed")

    run_clicked = st.button("Run simulation", type="primary", key="sim_run")

    st.divider()
    st.caption("**Model parameters** — click a section to adjust.")

    with st.expander("Expected Score"):
        gamma0 = st.slider("γ₀", -3.0, 7.0, float(GLOB["gamma0"]), 0.1, key="sim_gamma0")
        gamma1 = st.slider("γ₁", 0.1, 5.0, float(GLOB["gamma1"]), 0.1, key="sim_gamma1")

    with st.expander("Game Effectiveness  Z(n)"):
        zeta_min = st.slider("ζ_min", 0.0, 1.0, float(GLOB["zeta_min"]), 0.05, key="sim_zeta_min")
        lambda_Z = st.slider("λ_Z", 0.001, 0.5, float(GLOB["lambda_Z"]), 0.001,
                             format="%.3f", key="sim_lambda_Z")

    with st.expander("Mismatch  Ψ(δ)"):
        delta_star = st.slider("δ*", -5.0, 5.0, float(GLOB["delta_star"]), 0.1, key="sim_delta_star")
        kappa_L = st.slider("κ_L", 0.2, 6.0, float(GLOB["kappa_L"]), 0.1, key="sim_kappa_L")
        kappa_R = st.slider("κ_R", 0.2, 6.0, float(GLOB["kappa_R"]), 0.1, key="sim_kappa_R")

    with st.expander("Engagement  E(u)"):
        u_min = st.slider("u_min", 0.0, 0.8, float(GLOB["u_min"]), 0.05, key="sim_u_min")
        alpha_u = st.slider("α_u", 0.2, 5.0, float(GLOB["alpha_u"]), 0.1, key="sim_alpha_u")

    with st.expander("Fatigue  F(m,δ)"):
        rho0_F = st.slider("ρ₀_F", 0.0, 1.0, float(GLOB["rho0_F"]), 0.01, key="sim_rho0_F")
        rho_h_F = st.slider("ρ_h_F", 0.0, 2.0, float(GLOB["rho_h_F"]), 0.05, key="sim_rho_h_F")
        rho_e_F = st.slider("ρ_e_F", 0.0, 2.0, float(GLOB["rho_e_F"]), 0.05, key="sim_rho_e_F")
        tau_F = st.slider("τ_F", 0.2, 5.0, float(GLOB["tau_F"]), 0.1, key="sim_tau_F")

    with st.expander("Arousal  A(δ,u)"):
        A_min = st.slider("A_min", 0.0, 0.5, float(PRE["arousal"]["A_min"]), 0.02, key="sim_A_min")
        gamma0_A = st.slider("γ₀_A", -3.0, 3.0, float(PRE["arousal"]["gamma0_A"]), 0.1, key="sim_gamma0_A")
        gamma1_A = st.slider("γ₁_A", 0.0, 30.0, float(PRE["arousal"]["gamma1_A"]), 0.5, key="sim_gamma1_A")
        K0 = st.slider("K₀", 0.05, 5.0, float(PRE["arousal"]["K0"]), 0.05, key="sim_K0")
        rho_u = st.slider("ρ_u", 0.0, 60.0, float(PRE["arousal"]["rho_u"]), 1.0, key="sim_rho_u")

    with st.expander("Valence  V(δ,u)"):
        gamma0_V = st.slider("γ₀_V", -3.0, 3.0, float(PRE["valence"]["gamma0_V"]), 0.1, key="sim_gamma0_V")
        gamma_Vu = st.slider("γ_Vu", 0.0, 20.0, float(PRE["valence"]["gamma_Vu"]), 0.5, key="sim_gamma_Vu")
        gamma_Vh = st.slider("γ_Vh", 0.0, 0.5, float(PRE["valence"]["gamma_Vh"]), 0.01, key="sim_gamma_Vh")
        gamma_Ve = st.slider("γ_Ve", 0.0, 0.5, float(PRE["valence"]["gamma_Ve"]), 0.01, key="sim_gamma_Ve")
        tau_V = st.slider("τ_V", 0.5, 10.0, float(PRE["valence"]["tau_V"]), 0.1, key="sim_tau_V")
        beta_Vh = st.slider("β_Vh", 0.5, 10.0, float(PRE["valence"]["beta_Vh"]), 0.5, key="sim_beta_Vh")
        beta_Ve = st.slider("β_Ve", 0.5, 10.0, float(PRE["valence"]["beta_Ve"]), 0.5, key="sim_beta_Ve")

    with st.expander("Emotional Regulator  Emot(A,V)"):
        E_min = st.slider("E_min", 0.0, 0.5, float(GLOB["E_min"]), 0.02, key="sim_E_min")
        kappa_A = st.slider("κ_A", 0.01, 1.0, float(GLOB["kappa_A"]), 0.01, key="sim_kappa_A")
        kappa_V = st.slider("κ_V", 0.05, 2.0, float(GLOB["kappa_V"]), 0.05, key="sim_kappa_V")
        alpha_A = st.slider("α_A", 0.05, 2.0, float(GLOB["alpha_A"]), 0.05, key="sim_alpha_A")
        alpha_V = st.slider("α_V", 0.05, 2.0, float(GLOB["alpha_V"]), 0.05, key="sim_alpha_V")

    with st.expander("Practice Bias"):
        kappa_B = st.slider("κ_B", 0.1, 30.0, float(GLOB["kappa_B"]), 0.1, key="sim_kappa_B")
        delta_B_star = st.slider("δ_B*", -3.0, 3.0, float(GLOB["delta_B_star"]), 0.1, key="sim_delta_B_star")

    with st.expander("Dynamics"):
        rho_Q = st.slider("ρ_Q", 0.01, 0.99, float(GLOB["rho_Q"]), 0.01, key="sim_rho_Q")
        eta_k = st.slider("η_k  (per-domain learning rate)", 0.01, 1.0,
                          float(DOM["eta_k"]), 0.01, key="sim_eta_k")
        Bk_max = st.slider("B_k^max", 0.0, 1.0, float(DOM["Bk_max"]), 0.05, key="sim_Bk_max")
        Bk_min = st.slider("B_k^min", 0.0, 0.5, float(DOM["Bk_min"]), 0.01, key="sim_Bk_min")

    with st.expander("Noise"):
        sigma_OGS = st.slider("σ_OGS  (log-scale)", 0.0, 0.5,
                              float(NOISE["sigma_OGS"]), 0.01, key="sim_sigma_OGS")
        sigma_pi = st.slider("σ_π", 0.0, 0.2, float(NOISE["sigma_pi"]), 0.005,
                             format="%.3f", key="sim_sigma_pi")

    # Save button at bottom — captures everything above
    st.divider()
    st.download_button(
        "💾 Save current config (.yaml)",
        data=session_to_yaml().encode("utf-8"),
        file_name="sim_config.yaml",
        mime="application/x-yaml",
        key="sim_save_config",
    )


# ---------------------------------------------------------------------------
# Build the parameter dict and run
# ---------------------------------------------------------------------------
def make_cfg_overrides() -> dict:
    return {
        "predefined": {
            "arousal": {"A_min": A_min, "gamma0_A": gamma0_A, "gamma1_A": gamma1_A,
                        "K0": K0, "rho_u": rho_u},
            "valence": {"gamma0_V": gamma0_V, "gamma_Vu": gamma_Vu,
                        "gamma_Vh": gamma_Vh, "gamma_Ve": gamma_Ve,
                        "tau_V": tau_V, "beta_Vh": beta_Vh, "beta_Ve": beta_Ve},
        },
        "global_init": {
            "gamma0": gamma0, "gamma1": gamma1,
            "E_min": E_min, "kappa_A": kappa_A, "kappa_V": kappa_V,
            "alpha_A": alpha_A, "alpha_V": alpha_V,
            "kappa_B": kappa_B, "delta_B_star": delta_B_star,
            "zeta_min": zeta_min, "lambda_Z": lambda_Z,
            "kappa_L": kappa_L, "kappa_R": kappa_R, "delta_star": delta_star,
            "u_min": u_min, "alpha_u": alpha_u,
            "rho0_F": rho0_F, "rho_h_F": rho_h_F, "rho_e_F": rho_e_F, "tau_F": tau_F,
            "rho_Q": rho_Q,
        },
        "domain_init": {"eta_k": eta_k, "Bk_max": Bk_max, "Bk_min": Bk_min},
        "simulation": {
            "noise": {"sigma_OGS": sigma_OGS, "sigma_pi": sigma_pi},
            "program": {"I": int(I), "T": int(T), "M": int(M),
                        "K": int(PROG["K"]), "Nk": int(PROG["Nk"])},
        },
        "_meta": {"policy": policy_name, "seed": int(seed)},
    }


def cfg_to_key(o: dict):
    return tuple(sorted([
        (a, b, round(float(v), 6) if isinstance(v, (int, float)) else v)
        for a, sub in o.items() if isinstance(sub, dict)
        for b, v in sub.items()
    ]))


@st.cache_data(show_spinner=False, max_entries=8)
def run_simulation(o: dict):
    cfg = load_defaults()
    cfg["predefined"]["arousal"].update(o["predefined"]["arousal"])
    cfg["predefined"]["valence"].update(o["predefined"]["valence"])
    cfg["global_init"].update(o["global_init"])
    cfg["domain_init"].update(o["domain_init"])
    cfg["simulation"]["noise"].update(o["simulation"]["noise"])
    cfg["simulation"]["program"].update(o["simulation"]["program"])

    params = SimParams.from_config(cfg)
    SEED_GAME, SEED_POOL = 8, 12

    game_pool = GamePool.generate(
        K=params.K, Nk=params.Nk,
        hyperparams=cfg["simulation"]["game_pool"],
        seed=SEED_GAME,
    )
    pool = ParticipantPool.generate(
        I=params.I, K=params.K, J_total=game_pool.J_total, T=params.T,
        hyperparams=cfg["simulation"]["population"],
        seed=SEED_POOL,
    )

    gs = game_pool.to_structure()
    seed_run = int(o["_meta"]["seed"])
    if o["_meta"]["policy"] == "Staircase":
        policy = StaircasePolicy(I=pool.I, M=params.M, game_structure=gs)
    else:
        policy = RandomPolicy(I=pool.I, M=params.M, game_structure=gs, seed=seed_run)

    sim = Simulator(params)
    ds = sim.run(pool, game_pool, policy, seed=seed_run)

    pool_null = ParticipantPool.generate(
        I=params.I, K=params.K, J_total=game_pool.J_total, T=params.T,
        hyperparams=cfg["simulation"]["population"], seed=SEED_POOL,
    )
    ds_null = sim.run_null(pool_null)

    K = params.K
    domain_names = list(cfg["simulation"]["game_pool"].get("domain_names",
                        [f"Domain {k}" for k in range(K)]))[:K]
    return {
        "I": int(params.I), "T": int(params.T), "M": int(params.M), "K": K,
        "policy": o["_meta"]["policy"], "seed": seed_run,
        "domain_names": domain_names,
        "ogs":     ds.ogs.numpy(),
        "C_true":  ds.C_true.numpy(),
        "Q_true":  ds.Q_true.numpy(),
        "pi_true": ds.pi_true.numpy(),
        "Pi_true": ds.Pi_true.numpy(),
        "A_obs":   ds.A_obs.numpy(),
        "V_obs":   ds.V_obs.numpy(),
        "C_init":  ds.C_true.numpy()[:, 0, :],
        "C_null":  ds_null.C_true.numpy(),
        "_bundle": {
            "dataset":   ds,
            "game_pool": game_pool,
            "pool":      pool,
            "cfg":       cfg,
        },
    }


overrides = make_cfg_overrides()
current_key = cfg_to_key(overrides)

if run_clicked:
    with st.spinner(f"Running {policy_name} on I={I}, T={T}, M={M}..."):
        st.session_state["sim_result"] = run_simulation(overrides)
        st.session_state["sim_key"] = current_key

result = st.session_state.get("sim_result")
last_key = st.session_state.get("sim_key")

if result is None:
    st.info("Adjust parameters in the sidebar and click **Run simulation**.")
    st.stop()

stale = (last_key is not None and last_key != current_key)
if stale:
    st.warning("Sidebar parameters have changed since the last run. "
               "Click **Run simulation** to refresh.")

st.caption(
    f"**Policy:** {result['policy']}  ·  "
    f"I={result['I']}  ·  T={result['T']}  ·  M={result['M']}  ·  "
    f"K={result['K']}  ·  seed={result['seed']}"
)


# ---------------------------------------------------------------------------
# Pin / compare runs
# ---------------------------------------------------------------------------
PIN_COLORS = ["#7C3AED", "#EA580C", "#0891B2", "#16A34A"]
MAX_PINS = 4


def pin_summary(r):
    """A compact, comparison-ready snapshot of a run."""
    return {
        "name": f"{r['policy']} · I={r['I']} T={r['T']} M={r['M']} seed={r['seed']}",
        "policy": r["policy"],
        "T": r["T"], "K": r["K"],
        "domain_names": r["domain_names"],
        "ogs_session_mean": r["ogs"].mean(axis=(0, 2)),       # [T]
        "C_mean":           r["C_true"].mean(axis=0),         # [T+1, K]
    }


pins = st.session_state.setdefault("sim_pins", [])

c1, c2, c3 = st.columns([1, 1, 6])
with c1:
    if st.button("📌 Pin this run", key="sim_pin",
                 disabled=len(pins) >= MAX_PINS):
        pins.append(pin_summary(result))
        st.toast(f"Pinned. {len(pins)}/{MAX_PINS}")
with c2:
    if st.button("Clear pins", key="sim_clear_pins", disabled=not pins):
        st.session_state["sim_pins"] = []
        st.rerun()
with c3:
    if pins:
        st.caption(" · ".join(f"#{i+1} {p['name']}" for i, p in enumerate(pins)))

# Offer the full bundle as a download
_buf = io.BytesIO()
torch.save(result["_bundle"], _buf)
st.download_button(
    "Download dataset (.pt) — usable on the Estimation page",
    data=_buf.getvalue(),
    file_name=f"sim_{result['policy']}_I{result['I']}_T{result['T']}_M{result['M']}_seed{result['seed']}.pt",
    mime="application/octet-stream",
    key="dl_sim_bundle",
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


COLORS = ["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B3"]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_ogs, tab_ability, tab_gain = st.tabs([
    "OGS Trajectories", "Ability Growth", "Gain & Affect Dynamics",
])

T = result["T"]
K = result["K"]
sessions = np.arange(1, T + 1)
t_axis = np.arange(T + 1)
domain_names = result["domain_names"]


with tab_ogs:
    ogs = result["ogs"]
    ogs_session = ogs.mean(axis=2)
    mean_t = ogs_session.mean(axis=0)
    std_t = ogs_session.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    ax = axes[0]
    ax.plot(sessions, mean_t, color=COLORS[0], linewidth=2,
            label=f"Mean OGS  ({result['policy']})")
    ax.fill_between(sessions, mean_t - std_t, mean_t + std_t,
                    alpha=0.2, color=COLORS[0])

    # Pin overlays
    for i, p in enumerate(pins):
        if len(p["ogs_session_mean"]) == len(sessions):
            ax.plot(sessions, p["ogs_session_mean"],
                    color=PIN_COLORS[i % len(PIN_COLORS)],
                    linewidth=1.4, linestyle="--",
                    label=f"📌 #{i+1} {p['policy']} seed={p['name'].split('seed=')[-1]}")

    ax.set_xlabel("Session"); ax.set_ylabel("OGS  (0–100)")
    ax.set_title("Mean OGS over sessions  (±1 SD across participants)")
    ax.legend(fontsize=9); prettify(ax); add_subplot_label(ax, "A", size=18)

    ax = axes[1]
    ax.hist(ogs[:, -1, :].ravel(), bins=24,
            color=COLORS[1], edgecolor="white", alpha=0.85)
    ax.set_xlabel("OGS at session T"); ax.set_ylabel("Count")
    ax.set_title("OGS distribution at final session")
    prettify(ax); add_subplot_label(ax, "B", size=18)

    fig.tight_layout()
    show_fig(fig)
    df = pd.DataFrame({"session": sessions, "ogs_mean": mean_t, "ogs_std": std_t})
    downloads(fig, df, "ogs_trajectory")


with tab_ability:
    C = result["C_true"]
    C_null = result["C_null"]
    C_init = result["C_init"]

    fig, axes = plt.subplots(1, K, figsize=(4.4 * K, 4.2), sharey=False)
    if K == 1:
        axes = [axes]

    for k in range(K):
        ax = axes[k]
        Ck = C[:, :, k]
        m = Ck.mean(axis=0); s = Ck.std(axis=0)
        ax.plot(t_axis, m, color=COLORS[k], linewidth=2, label=result["policy"])
        ax.fill_between(t_axis, m - s, m + s, alpha=0.2, color=COLORS[k])
        ax.plot(t_axis, C_null[:, :, k].mean(axis=0),
                color="grey", linewidth=1.2, linestyle=":", label="No intervention")
        for i in range(min(8, Ck.shape[0])):
            ax.plot(t_axis, Ck[i], color=COLORS[k], alpha=0.25, linewidth=0.7)

        # Pin overlays
        for j, p in enumerate(pins):
            if p["C_mean"].shape == (T + 1, K) and k < p["K"]:
                ax.plot(t_axis, p["C_mean"][:, k],
                        color=PIN_COLORS[j % len(PIN_COLORS)],
                        linewidth=1.3, linestyle="--",
                        label=f"📌 #{j+1} {p['policy']}")

        ax.set_xlabel("Session  (t)"); ax.set_ylabel("Latent ability  C")
        ax.set_title(domain_names[k])
        ax.legend(fontsize=8); prettify(ax)

    fig.tight_layout()
    show_fig(fig)

    final_gain_total = C[:, -1, :].sum(axis=1) - C_init.sum(axis=1)
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.hist(final_gain_total, bins=20, color=COLORS[2],
            edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(final_gain_total.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"mean = {final_gain_total.mean():.2f}")
    ax.set_xlabel(r"Total ability gain $\sum_k (C_{T+1} - C_1)$")
    ax.set_ylabel("Count")
    ax.set_title("Final ability gain  (summed over domains)")
    ax.legend(); prettify(ax)
    fig2.tight_layout()
    show_fig(fig2)

    df = pd.DataFrame({"session": t_axis})
    for k in range(K):
        df[f"C_mean_{domain_names[k]}"] = C[:, :, k].mean(axis=0)
        df[f"C_std_{domain_names[k]}"] = C[:, :, k].std(axis=0)
    downloads(fig, df, "ability_trajectory")


with tab_gain:
    pi = result["pi_true"]
    Pi = result["Pi_true"]
    Q = result["Q_true"]
    A = result["A_obs"]
    V = result["V_obs"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.hist(pi.ravel(), bins=40, color=COLORS[2], edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Single-game gain  $\pi$"); ax.set_ylabel("Count")
    ax.set_title("Single-game gain distribution")
    prettify(ax); add_subplot_label(ax, "A", size=18)

    ax = axes[1]
    Pi_mean = Pi.mean(axis=0)
    for k in range(K):
        ax.plot(sessions, Pi_mean[:, k], color=COLORS[k],
                linewidth=2, label=domain_names[k])
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Session"); ax.set_ylabel(r"$\Pi$  (session gain)")
    ax.set_title("Mean session gain per domain")
    ax.legend(fontsize=9); prettify(ax); add_subplot_label(ax, "B", size=18)

    ax = axes[2]
    Q_mean = Q.mean(axis=0)
    for k in range(K):
        ax.plot(sessions, Q_mean[:, k], color=COLORS[k],
                linewidth=2, label=domain_names[k])
    ax.set_xlabel("Session"); ax.set_ylabel("Cumulative impact  Q")
    ax.set_title("Mean cumulative impact per domain")
    ax.legend(fontsize=9); prettify(ax); add_subplot_label(ax, "C", size=18)

    fig.tight_layout()
    show_fig(fig)

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    A_session = A.mean(axis=2)
    V_session = V.mean(axis=2)
    for ax, vals, name, col in [
        (axes[0], A_session, "Arousal  A", COLORS[0]),
        (axes[1], V_session, "Valence  V", COLORS[3]),
    ]:
        m = vals.mean(axis=0); s = vals.std(axis=0)
        ax.plot(sessions, m, color=col, linewidth=2)
        ax.fill_between(sessions, m - s, m + s, alpha=0.2, color=col)
        ax.set_xlabel("Session"); ax.set_ylabel(name)
        ax.set_title(f"Mean {name} over sessions  (±1 SD)")
        prettify(ax)
    add_subplot_label(axes[0], "D", size=18)
    add_subplot_label(axes[1], "E", size=18)
    fig2.tight_layout()
    show_fig(fig2)

    df = pd.DataFrame({"session": sessions})
    for k in range(K):
        df[f"Pi_{domain_names[k]}"] = Pi_mean[:, k]
        df[f"Q_{domain_names[k]}"] = Q_mean[:, k]
    df["A_mean"] = A_session.mean(axis=0)
    df["V_mean"] = V_session.mean(axis=0)
    downloads(fig, df, "gain_affect_dynamics")
