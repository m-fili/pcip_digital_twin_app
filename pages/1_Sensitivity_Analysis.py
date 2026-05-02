"""Sensitivity Analysis page — 9 model components, each with sliders and tabs.

Layout for every component:
  1. LaTeX equation at the top of the main area
  2. Sidebar sliders (shared params) + Reset button
  3. Tabs: 'Single curve' (live), then paper-style comparison views.
     Tab-local sliders for any value used only in that tab.
  4. Interpretation expander
  5. PNG + CSV downloads
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import setup_internal_page  # noqa: E402
from components import (  # noqa: E402
    expected_score, game_effectiveness, mismatch, engagement, fatigue,
    arousal, valence, emot, practice_bias, omega_B,
)
from plotting import (  # noqa: E402
    prettify, add_subplot_label, fig_to_png_bytes,
    NAVY_200, INK,
)


# Theme-friendly callout box for in-plot value annotations.
# Dark facecolor + soft hairline edge so labels read against the dark axes
# background. matplotlib doesn't accept CSS-style rgba strings, so the edge
# is given as an (r, g, b, a) tuple.
_CALLOUT_BBOX = dict(
    facecolor=NAVY_200,
    edgecolor=(241/255, 245/255, 251/255, 0.18),
    boxstyle="round,pad=0.3", linewidth=0.8,
)


st.set_page_config(page_title="Sensitivity Analysis", layout="wide")
setup_internal_page(
    "Sensitivity Analysis",
    subtitle="Interactive exploration of each model component. "
             "Default values match the paper's Section 5.1 figures.",
    crumb="Sensitivity",
)

COMPONENTS = {
    "ES — Expected Score": "es",
    "Gain · Z(n) — Game Effectiveness": "z",
    "Gain · Ψ(δ) — Mismatch Effect": "psi",
    "Gain · E(u) — Engagement": "eng",
    "Gain · F(m, δ) — Fatigue": "fat",
    "Affect · A(δ, u) — Arousal": "arousal",
    "Affect · V(δ, u) — Valence": "valence",
    "Affect · Emot(A, V) — Emotional Regulator": "emot",
    "Practice Bias B(n, δ)": "bias",
}

with st.sidebar:
    label = st.selectbox("Component", list(COMPONENTS.keys()), index=0)
    component_key = COMPONENTS[label]
    st.divider()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def parse_floats(raw, fallback):
    """Parse comma-separated floats; fall back silently on error."""
    raw = (raw or "").strip()
    if not raw:
        return list(fallback)
    try:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        return vals if vals else list(fallback)
    except ValueError:
        return list(fallback)


def reset_button(component_id, keys):
    """Render a Reset button that clears only this component's widget state."""
    if st.sidebar.button("Reset to defaults", key=f"{component_id}_reset"):
        for k in keys:
            st.session_state.pop(k, None)
        st.rerun()


def downloads(fig, df, filename_stem):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download PNG",
            data=fig_to_png_bytes(fig),
            file_name=f"{filename_stem}.png",
            mime="image/png",
            key=f"png_{filename_stem}",
        )
    with c2:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{filename_stem}.csv",
            mime="text/csv",
            key=f"csv_{filename_stem}",
        )


def show_fig(fig):
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Expected Score
# ---------------------------------------------------------------------------
def render_es():
    keys = ["es_g0", "es_g1", "es_drange", "es_cmp_g0", "es_cmp_g1"]
    with st.sidebar:
        st.subheader("Parameters")
        gamma0 = st.slider("γ₀  (intercept)", -3.0, 7.0, 1.0, 0.1, key="es_g0")
        gamma1 = st.slider("γ₁  (slope)", 0.1, 5.0, 1.5, 0.1, key="es_g1")
        delta_range = st.slider("δ range", -15.0, 15.0, (-10.0, 10.0), 0.5, key="es_drange")
    reset_button("es", keys)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Single curve", "Compare γ₀", "Compare γ₁", "Heatmap"]
    )
    delta = np.linspace(delta_range[0], delta_range[1], 400)

    with tab1:
        es = expected_score(delta, gamma0, gamma1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(delta, es, linewidth=2)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        es_match = float(expected_score(0.0, gamma0, gamma1))
        ax.scatter([0], [es_match], zorder=5, color="C3")
        ax.annotate(f"{es_match:.1f}", (0, es_match),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=INK,
                    bbox=_CALLOUT_BBOX)
        ax.set_xlabel(r"Ability–Difficulty Gap ($\delta$)")
        ax.set_ylabel("Expected Score")
        ax.set_ylim(-5, 105)
        prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"delta": delta, "ES": es}), "ES_single")

    with tab2:
        raw = st.text_input("γ₀ values (comma-separated)",
                            value="-3, -1, 0, 1, 3, 7", key="es_cmp_g0")
        g0_list = parse_floats(raw, [-3.0, -1.0, 0.0, 1.0, 3.0, 7.0])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"delta": delta}
        for g0 in g0_list:
            es = expected_score(delta, g0, gamma1)
            ax.plot(delta, es, label=f"γ₀ = {g0}", linewidth=2)
            s = float(expected_score(0.0, g0, gamma1))
            ax.scatter([0], [s], zorder=5)
            ax.annotate(f"{s:.1f}", (0, s), xytext=(6, 0),
                        textcoords="offset points", fontsize=9, color=INK,
                        bbox=_CALLOUT_BBOX)
            data[f"ES_g0={g0}"] = es
        ax.axhline(50, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel(r"$\delta$"); ax.set_ylabel("Expected Score")
        ax.set_ylim(-5, 105); ax.legend(loc="lower right")
        prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "ES_compare_gamma0")

    with tab3:
        raw = st.text_input("γ₁ values (comma-separated)",
                            value="0.5, 1.0, 1.5, 2.0, 3.0", key="es_cmp_g1")
        g1_list = parse_floats(raw, [0.5, 1.0, 1.5, 2.0, 3.0])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"delta": delta}
        for g1 in g1_list:
            es = expected_score(delta, gamma0, g1)
            ax.plot(delta, es, label=f"γ₁ = {g1}", linewidth=2)
            data[f"ES_g1={g1}"] = es
        ax.axhline(50, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel(r"$\delta$"); ax.set_ylabel("Expected Score")
        ax.set_ylim(-5, 105); ax.legend(loc="lower right")
        prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "ES_compare_gamma1")

    with tab4:
        g0_grid = np.linspace(-3.0, 7.0, 80)
        g1_grid = np.linspace(0.1, 5.0, 80)
        G0, _ = np.meshgrid(g0_grid, g1_grid)
        Z = 100.0 / (1.0 + np.exp(-G0))
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(Z, origin="lower", aspect="auto",
                       extent=[g0_grid.min(), g0_grid.max(), g1_grid.min(), g1_grid.max()],
                       cmap="magma_r")
        plt.colorbar(im, ax=ax, label="ES at δ = 0")
        ax.scatter([gamma0], [gamma1], color="white", edgecolor="black", zorder=5, s=60)
        ax.set_xlabel(r"$\gamma_0$"); ax.set_ylabel(r"$\gamma_1$")
        prettify(ax); fig.tight_layout()
        show_fig(fig)

    with st.expander("Interpretation"):
        st.markdown(
            "The Expected Score maps the ability–difficulty gap δ to a percentage "
            "score via an IRT logistic link. **γ₀** sets the score at exact match "
            "(δ = 0): the default γ₀ = 1.0 yields ES ≈ 73. **γ₁** controls the "
            "steepness of the transition between floor and ceiling."
        )


# ---------------------------------------------------------------------------
# 2. Game Effectiveness Z(n)
# ---------------------------------------------------------------------------
def render_z():
    keys = ["z_zeta", "z_zmin", "z_lam", "z_nmax", "z_cmp_lam", "z_cmp_zmin"]
    with st.sidebar:
        st.subheader("Parameters")
        zeta = st.slider("ζ  (initial effectiveness)", 0.0, 1.0, 0.8, 0.05, key="z_zeta")
        zeta_min = st.slider("ζ_min  (floor)", 0.0, 1.0, 0.5, 0.05, key="z_zmin")
        lambda_Z = st.slider("λ_Z  (decay rate)", 0.001, 0.5, 0.03, 0.001,
                             format="%.3f", key="z_lam")
        n_max = st.slider("n range (max)", 10, 100, 30, 5, key="z_nmax")
    reset_button("z", keys)

    tab1, tab2, tab3 = st.tabs(["Single curve", "Compare λ_Z", "Compare ζ_min"])
    n = np.arange(0, n_max + 1)

    with tab1:
        z = game_effectiveness(n, zeta, zeta_min, lambda_Z)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(n, z, linewidth=2, marker="o", markersize=3)
        ax.axhline(zeta_min, color="gray", linestyle="--", alpha=0.7,
                   label=f"ζ_min = {zeta_min}")
        ax.set_xlabel(r"Prior plays $n_{ijk}^{t}$")
        ax.set_ylabel(r"Effectiveness $Z$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"n": n, "Z": z}), "Z_single")

    with tab2:
        raw = st.text_input("λ_Z values (comma-separated)",
                            value="0.01, 0.03, 0.05, 0.1, 0.3", key="z_cmp_lam")
        lambdas = parse_floats(raw, [0.01, 0.03, 0.05, 0.1, 0.3])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"n": n}
        for lam in lambdas:
            z = game_effectiveness(n, zeta, zeta_min, lam)
            ax.plot(n, z, label=f"λ_Z = {lam}", linewidth=2)
            data[f"Z_lam={lam}"] = z
        ax.set_xlabel("Prior plays"); ax.set_ylabel(r"$Z$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "Z_compare_lambda")

    with tab3:
        raw = st.text_input("ζ_min values (comma-separated)",
                            value="0.0, 0.2, 0.5, 0.7", key="z_cmp_zmin")
        zmins = parse_floats(raw, [0.0, 0.2, 0.5, 0.7])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"n": n}
        for zm in zmins:
            z = game_effectiveness(n, zeta, zm, lambda_Z)
            ax.plot(n, z, label=f"ζ_min = {zm}", linewidth=2)
            data[f"Z_zmin={zm}"] = z
        ax.set_xlabel("Prior plays"); ax.set_ylabel(r"$Z$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "Z_compare_zmin")

    with st.expander("Interpretation"):
        st.markdown(
            "Game effectiveness decays exponentially with prior plays. At the "
            "default λ_Z = 0.03, Z retains ~80% of its initial value after 10 "
            "plays; larger λ_Z causes rapid novelty exhaustion."
        )


# ---------------------------------------------------------------------------
# 3. Mismatch Ψ(δ)
# ---------------------------------------------------------------------------
def render_psi():
    keys = ["psi_ds", "psi_kL", "psi_kR", "psi_drange", "psi_cmp_bw", "psi_cmp_ds"]
    with st.sidebar:
        st.subheader("Parameters")
        delta_star = st.slider("δ*", -5.0, 5.0, 0.0, 0.1, key="psi_ds")
        kappa_L = st.slider("κ_L  (too-hard side)", 0.2, 6.0, 3.0, 0.1, key="psi_kL")
        kappa_R = st.slider("κ_R  (too-easy side)", 0.2, 6.0, 1.0, 0.1, key="psi_kR")
        delta_range = st.slider("δ range", -15.0, 15.0, (-8.0, 8.0), 0.5, key="psi_drange")
    reset_button("psi", keys)

    tab1, tab2, tab3 = st.tabs(["Single curve", "Compare bandwidths", "Compare δ*"])
    delta = np.linspace(delta_range[0], delta_range[1], 400)

    with tab1:
        psi = mismatch(delta, delta_star, kappa_L, kappa_R)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(delta, psi, linewidth=2)
        ax.axvline(delta_star, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel(r"$\delta$"); ax.set_ylabel(r"$\Psi(\delta)$")
        prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"delta": delta, "Psi": psi}), "Psi_single")

    with tab2:
        st.caption("Enter (κ_L, κ_R) pairs separated by ';'  —  e.g. 3,1 ; 4,2 ; 2,0.6")
        raw = st.text_input("Bandwidth pairs", value="3,1 ; 4,2 ; 2,0.6", key="psi_cmp_bw")
        presets = []
        for chunk in raw.split(";"):
            parts = [p.strip() for p in chunk.split(",") if p.strip()]
            if len(parts) == 2:
                try:
                    presets.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        if not presets:
            presets = [(3.0, 1.0), (4.0, 2.0), (2.0, 0.6)]
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"delta": delta}
        for kL, kR in presets:
            psi = mismatch(delta, delta_star, kL, kR)
            ax.plot(delta, psi, label=f"κ_L={kL}, κ_R={kR}", linewidth=2)
            data[f"Psi_kL={kL}_kR={kR}"] = psi
        ax.axvline(delta_star, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel(r"$\delta$"); ax.set_ylabel(r"$\Psi(\delta)$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "Psi_compare_bandwidth")

    with tab3:
        raw = st.text_input("δ* values (comma-separated)",
                            value="-2, 0, 2", key="psi_cmp_ds")
        ds_list = parse_floats(raw, [-2.0, 0.0, 2.0])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"delta": delta}
        for ds in ds_list:
            psi = mismatch(delta, ds, kappa_L, kappa_R)
            ax.plot(delta, psi, label=f"δ* = {ds}", linewidth=2)
            ax.axvline(ds, linestyle="--", linewidth=1, alpha=0.5)
            data[f"Psi_ds={ds}"] = psi
        ax.set_xlabel(r"$\delta$"); ax.set_ylabel(r"$\Psi(\delta)$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "Psi_compare_deltaStar")

    with st.expander("Interpretation"):
        st.markdown(
            "Asymmetric Laplace kernel peaked at δ*. Default (κ_L = 3, κ_R = 1) "
            "implements *desirable difficulty*: moderate overchallenge preserves "
            "more learning than equivalent underchallenge."
        )


# ---------------------------------------------------------------------------
# 4. Engagement E(u)
# ---------------------------------------------------------------------------
def render_eng():
    keys = ["eng_umin", "eng_alpha", "eng_cmp_alpha", "eng_cmp_umin"]
    with st.sidebar:
        st.subheader("Parameters")
        u_min = st.slider("u_min  (floor)", 0.0, 0.8, 0.2, 0.05, key="eng_umin")
        alpha_u = st.slider("α_u  (convexity)", 0.2, 5.0, 2.0, 0.1, key="eng_alpha")
    reset_button("eng", keys)

    tab1, tab2, tab3 = st.tabs(["Single curve", "Compare α_u", "Compare u_min"])
    u = np.linspace(0.0, 1.0, 300)

    with tab1:
        e = engagement(u, u_min, alpha_u)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(u, e, linewidth=2)
        ax.axhline(u_min, color="gray", linestyle="--", alpha=0.7, label=f"u_min = {u_min}")
        ax.set_xlabel(r"Utility $u$"); ax.set_ylabel(r"$E(u)$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"u": u, "E": e}), "E_single")

    with tab2:
        raw = st.text_input("α_u values (comma-separated)",
                            value="0.8, 1.0, 2.0, 4.0", key="eng_cmp_alpha")
        alphas = parse_floats(raw, [0.8, 1.0, 2.0, 4.0])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"u": u}
        for a in alphas:
            e = engagement(u, u_min, a)
            ax.plot(u, e, label=f"α_u = {a}", linewidth=2)
            data[f"E_alpha={a}"] = e
        ax.axhline(u_min, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel(r"$u$"); ax.set_ylabel(r"$E(u)$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "E_compare_alpha")

    with tab3:
        raw = st.text_input("u_min values (comma-separated)",
                            value="0.0, 0.2, 0.5", key="eng_cmp_umin")
        umins = parse_floats(raw, [0.0, 0.2, 0.5])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"u": u}
        for um in umins:
            e = engagement(u, um, alpha_u)
            ax.plot(u, e, label=f"u_min = {um}", linewidth=2)
            data[f"E_umin={um}"] = e
        ax.set_xlabel(r"$u$"); ax.set_ylabel(r"$E(u)$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "E_compare_umin")

    with st.expander("Interpretation"):
        st.markdown(
            "Maps utility u ∈ [0, 1] to engagement with a convexity parameter α_u "
            "and a floor u_min. At α_u = 2 (default), low-utility games are "
            "penalized disproportionately while preferred games still yield "
            "near-maximal engagement."
        )


# ---------------------------------------------------------------------------
# 5. Fatigue F(m, δ)
# ---------------------------------------------------------------------------
def render_fat():
    keys = ["fat_rho0", "fat_rhoh", "fat_rhoe", "fat_tauh", "fat_taue",
            "fat_M", "fat_ds", "fat_delta_t1", "fat_delta_t2"]
    with st.sidebar:
        st.subheader("Parameters")
        rho0 = st.slider("ρ₀", 0.0, 1.0, 0.2, 0.05, key="fat_rho0")
        rho_h = st.slider("ρ_h  (hard penalty)", 0.0, 2.0, 0.8, 0.05, key="fat_rhoh")
        rho_e = st.slider("ρ_e  (easy penalty)", 0.0, 2.0, 0.4, 0.05, key="fat_rhoe")
        tau_h = st.slider("τ_h", 0.2, 5.0, 1.5, 0.1, key="fat_tauh")
        tau_e = st.slider("τ_e", 0.2, 5.0, 1.5, 0.1, key="fat_taue")
        M = st.slider("M  (games/session)", 5, 20, 12, 1, key="fat_M")
        delta_star = st.slider("δ*", -3.0, 3.0, 0.0, 0.1, key="fat_ds")
    reset_button("fat", keys)

    tab1, tab2, tab3 = st.tabs(["Per-game", "Cumulative", "Compare mismatch"])
    m = np.arange(1, M + 1)

    with tab1:
        delta_t1 = st.slider("δ (current game)", -6.0, 6.0, 0.0, 0.1, key="fat_delta_t1")
        F = fatigue(m, M, delta_t1, delta_star, rho0, rho_h, rho_e, tau_h, tau_e)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(m, F, marker="o", linewidth=2, label=f"δ = {delta_t1}")
        ax.set_xlabel(r"Game order $m$"); ax.set_ylabel(r"$F_{ijk}^{t,m}$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"m": m, "F": F}), "F_per_game")

    with tab2:
        delta_t2 = st.slider("δ (current game)", -6.0, 6.0, 0.0, 0.1, key="fat_delta_t2")
        F = fatigue(m, M, delta_t2, delta_star, rho0, rho_h, rho_e, tau_h, tau_e)
        cumF = np.cumsum(F)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(m, cumF, marker="o", linewidth=2, label=f"δ = {delta_t2}")
        ax.set_xlabel(r"Game order $m$"); ax.set_ylabel(r"Cumulative fatigue")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"m": m, "cumF": cumF}), "F_cumulative")

    with tab3:
        regimes = {
            "Optimal (δ = δ*)": delta_star,
            "Too hard (δ = δ* − 3)": delta_star - 3.0,
            "Too easy (δ = δ* + 3)": delta_star + 3.0,
        }
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        data = {"m": m}
        for lbl, dlt in regimes.items():
            F = fatigue(m, M, dlt, delta_star, rho0, rho_h, rho_e, tau_h, tau_e)
            axes[0].plot(m, F, marker="o", label=lbl)
            axes[1].plot(m, np.cumsum(F), marker="o", label=lbl)
            data[f"F_{lbl}"] = F
            data[f"cumF_{lbl}"] = np.cumsum(F)
        axes[0].set_xlabel(r"$m$"); axes[0].set_ylabel(r"$F$")
        axes[1].set_xlabel(r"$m$"); axes[1].set_ylabel(r"$\sum F$")
        for ax in axes:
            ax.legend(fontsize=9); prettify(ax)
        add_subplot_label(axes[0], "A", size=20)
        add_subplot_label(axes[1], "B", size=20)
        fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "F_compare_mismatch")

    with st.expander("Interpretation"):
        st.markdown(
            "Fatigue ramps linearly with game order within a session. Asymmetric "
            "penalties (ρ_h > ρ_e) encode that cognitive overload is more "
            "fatiguing than boredom. Cumulative fatigue grows superlinearly."
        )


# ---------------------------------------------------------------------------
# 6. Arousal A(δ, u)
# ---------------------------------------------------------------------------
def render_arousal():
    keys = ["ar_Amin", "ar_g0", "ar_g1", "ar_k0", "ar_rhou",
            "ar_d_t1", "ar_u_t3", "ar_d_t3"]
    with st.sidebar:
        st.subheader("Parameters")
        A_min = st.slider("A_min", 0.0, 0.5, 0.2, 0.02, key="ar_Amin")
        gamma0_A = st.slider("γ₀_A", -3.0, 3.0, -0.5, 0.1, key="ar_g0")
        gamma1_A = st.slider("γ₁_A", 0.0, 30.0, 15.0, 0.5, key="ar_g1")
        kappa0_Au = st.slider("κ₀_Au", 0.05, 5.0, 0.1, 0.05, key="ar_k0")
        rho_u = st.slider("ρ_u", 0.0, 60.0, 20.0, 1.0, key="ar_rhou")
    reset_button("arousal", keys)

    args = (A_min, gamma0_A, gamma1_A, kappa0_Au, rho_u)
    tab1, tab2, tab3 = st.tabs(["A vs u", "A vs δ", "Heatmap"])
    u_grid = np.linspace(0.0, 1.0, 200)
    delta_grid = np.linspace(-10.0, 10.0, 300)

    with tab1:
        d = st.slider("δ (slice)", -10.0, 10.0, 0.0, 0.1, key="ar_d_t1")
        A = arousal(d, u_grid, *args)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(u_grid, A, linewidth=2)
        ax.axhline(A_min, color="k", linestyle="--", alpha=0.5, label=f"A_min = {A_min}")
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$u$"); ax.set_ylabel(r"$A$")
        ax.set_ylim(0, 1.05); ax.legend()
        prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"u": u_grid, "A": A}), "A_vs_u")

    with tab2:
        raw = st.text_input("u values (comma-separated)",
                            value="0.0, 0.25, 0.5, 0.75, 1.0", key="ar_cmp_u")
        u_levels = parse_floats(raw, [0.0, 0.25, 0.5, 0.75, 1.0])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"delta": delta_grid}
        for u in u_levels:
            A = arousal(delta_grid, u, *args)
            ax.plot(delta_grid, A, label=f"u = {u}", linewidth=2)
            data[f"A_u={u}"] = A
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$\delta$"); ax.set_ylabel(r"$A$")
        ax.set_ylim(0, 1.05); ax.legend()
        prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "A_vs_delta")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            u_mark = st.slider("u (marker)", 0.0, 1.0, 0.5, 0.05, key="ar_u_t3")
        with c2:
            d_mark = st.slider("δ (marker)", -10.0, 10.0, 0.0, 0.1, key="ar_d_t3")
        U, D = np.meshgrid(u_grid, delta_grid)
        A = arousal(D, U, *args)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(A, origin="lower", aspect="auto",
                       extent=[u_grid.min(), u_grid.max(),
                               delta_grid.min(), delta_grid.max()], cmap="plasma")
        plt.colorbar(im, ax=ax, label=r"$A$")
        ax.scatter([u_mark], [d_mark], color="white", edgecolor="black", zorder=5, s=80)
        ax.set_xlabel(r"$u$"); ax.set_ylabel(r"$\delta$")
        prettify(ax); fig.tight_layout()
        show_fig(fig)

    with st.expander("Interpretation"):
        st.markdown(
            "Arousal rises sigmoidally with utility and decays as a Gaussian in "
            "δ. The Gaussian bandwidth grows with u, so high-preference games "
            "sustain elevated arousal across a wider range of mismatch."
        )


# ---------------------------------------------------------------------------
# 7. Valence V(δ, u)
# ---------------------------------------------------------------------------
def render_valence():
    keys = ["val_g0", "val_gvu", "val_gvh", "val_gve", "val_tau",
            "val_d_t1", "val_cmp_u", "val_u_t3", "val_d_t3"]
    with st.sidebar:
        st.subheader("Parameters")
        gamma0_v = st.slider("γ₀_v", -3.0, 3.0, 0.0, 0.1, key="val_g0")
        gamma_Vu = st.slider("γ_Vu", 0.0, 20.0, 10.0, 0.5, key="val_gvu")
        gamma_vh = st.slider("γ_vh", 0.0, 0.5, 0.06, 0.01, key="val_gvh")
        gamma_ve = st.slider("γ_ve", 0.0, 0.5, 0.06, 0.01, key="val_gve")
        tau = st.slider("τ", 0.5, 10.0, 5.0, 0.1, key="val_tau")
    reset_button("valence", keys)

    args = dict(gamma0_v=gamma0_v, gamma_Vu=gamma_Vu,
                gamma_vh=gamma_vh, gamma_ve=gamma_ve, tau=tau)
    tab1, tab2, tab3 = st.tabs(["V vs u", "V vs δ", "Heatmap"])
    u_grid = np.linspace(0.0, 1.0, 200)
    delta_grid = np.linspace(-10.0, 10.0, 300)

    with tab1:
        d = st.slider("δ (slice)", -10.0, 10.0, 0.0, 0.1, key="val_d_t1")
        V = valence(d, u_grid, **args)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(u_grid, V, linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$u$"); ax.set_ylabel(r"$V$")
        ax.set_ylim(-1.05, 1.05)
        prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"u": u_grid, "V": V}), "V_vs_u")

    with tab2:
        raw = st.text_input("u values (comma-separated)",
                            value="0.0, 0.25, 0.5, 0.75, 1.0", key="val_cmp_u")
        u_levels = parse_floats(raw, [0.0, 0.25, 0.5, 0.75, 1.0])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"delta": delta_grid}
        for u in u_levels:
            V = valence(delta_grid, u, **args)
            ax.plot(delta_grid, V, label=f"u = {u}", linewidth=2)
            data[f"V_u={u}"] = V
        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$\delta$"); ax.set_ylabel(r"$V$")
        ax.set_ylim(-1.05, 1.05); ax.legend()
        prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "V_vs_delta")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            u_mark = st.slider("u (marker)", 0.0, 1.0, 0.5, 0.05, key="val_u_t3")
        with c2:
            d_mark = st.slider("δ (marker)", -10.0, 10.0, 0.0, 0.1, key="val_d_t3")
        U, D = np.meshgrid(u_grid, delta_grid)
        V = valence(D, U, **args)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(V, origin="lower", aspect="auto",
                       extent=[u_grid.min(), u_grid.max(),
                               delta_grid.min(), delta_grid.max()],
                       cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label=r"$V$")
        cs = ax.contour(U, D, V, levels=np.linspace(-1, 1, 11),
                        colors="k", linewidths=0.6, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        ax.scatter([u_mark], [d_mark], color="white", edgecolor="black", zorder=5, s=80)
        ax.set_xlabel(r"$u$"); ax.set_ylabel(r"$\delta$")
        prettify(ax); fig.tight_layout()
        show_fig(fig)

    with st.expander("Interpretation"):
        st.markdown(
            "Valence is dominated by utility, with secondary quadratic penalties "
            "from hard (δ < 0) and easy (δ > 0) mismatch. The tanh link bounds "
            "V in [−1, 1]."
        )


# ---------------------------------------------------------------------------
# 8. Emotional Regulator Emot(A, V)
# ---------------------------------------------------------------------------
def render_emot():
    keys = ["em_As", "em_Emin", "em_kA", "em_kV", "em_aA", "em_aV",
            "em_A_t1", "em_V_t2"]
    with st.sidebar:
        st.subheader("Parameters")
        A_star = st.slider("A*", 0.1, 1.0, 0.6, 0.05, key="em_As")
        E_min = st.slider("E_min", 0.0, 0.5, 0.2, 0.02, key="em_Emin")
        kappa_A = st.slider("κ_A", 0.01, 1.0, 0.15, 0.01, key="em_kA")
        kappa_V = st.slider("κ_V", 0.05, 2.0, 0.5, 0.05, key="em_kV")
        alpha_A = st.slider("α_A", 0.05, 2.0, 0.1, 0.05, key="em_aA")
        alpha_V = st.slider("α_V", 0.05, 2.0, 0.4, 0.05, key="em_aV")
    reset_button("emot", keys)

    args = dict(A_star=A_star, E_min=E_min, kappa_A=kappa_A, kappa_V=kappa_V,
                alpha_A=alpha_A, alpha_V=alpha_V)
    tab1, tab2, tab3 = st.tabs(["Emot vs V", "Emot vs A", "Heatmap"])
    V_grid = np.linspace(-1.0, 1.0, 200)
    A_grid = np.linspace(0.0, 1.0, 200)

    with tab1:
        A_slice = st.slider("A (slice)", 0.0, 1.0, float(A_star), 0.05, key="em_A_t1")
        E = emot(A_slice, V_grid, **args)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(V_grid, E, linewidth=2)
        ax.axhline(E_min, color="gray", linestyle="--", alpha=0.5, label=f"E_min = {E_min}")
        ax.set_xlabel(r"$V$"); ax.set_ylabel(r"$\mathrm{Emot}$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"V": V_grid, "Emot": E}), "Emot_vs_V")

    with tab2:
        V_slice = st.slider("V (slice)", -1.0, 1.0, 0.5, 0.05, key="em_V_t2")
        E = emot(A_grid, V_slice, **args)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(A_grid, E, linewidth=2)
        ax.axvline(A_star, color="gray", linestyle="--", alpha=0.7, label=f"A* = {A_star}")
        ax.set_xlabel(r"$A$"); ax.set_ylabel(r"$\mathrm{Emot}$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"A": A_grid, "Emot": E}), "Emot_vs_A")

    with tab3:
        Ag, Vg = np.meshgrid(A_grid, V_grid)
        E = emot(Ag, Vg, **args)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(E, origin="lower", aspect="auto",
                       extent=[A_grid.min(), A_grid.max(),
                               V_grid.min(), V_grid.max()], cmap="viridis")
        plt.colorbar(im, ax=ax, label=r"$\mathrm{Emot}$")
        cs = ax.contour(Ag, Vg, E, levels=np.linspace(E_min, 1.0, 10),
                        colors="white", linewidths=0.8, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        ax.axvline(A_star, linestyle="--", color="white", alpha=0.8)
        ax.set_xlabel(r"$A$"); ax.set_ylabel(r"$V$")
        prettify(ax); fig.tight_layout()
        show_fig(fig)

    with st.expander("Interpretation"):
        st.markdown(
            "Product of arousal and valence kernels. With default exponents "
            "(α_A = 0.1, α_V = 0.4) valence dominates: contours run nearly "
            "horizontal. Near-optimal arousal and positive valence push Emot "
            "above 0.9."
        )


# ---------------------------------------------------------------------------
# 9. Practice Bias B(n, δ)
# ---------------------------------------------------------------------------
def render_bias():
    keys = ["bs_b0", "bs_b1", "bs_bmin", "bs_bmax", "bs_dBs", "bs_kB",
            "bs_nmax", "bs_d_t1", "bs_cmp"]
    with st.sidebar:
        st.subheader("Parameters")
        beta0 = st.slider("β₀", 0.01, 0.5, 0.1, 0.01, key="bs_b0")
        beta1 = st.slider("β₁", 0.2, 8.0, 2.0, 0.1, key="bs_b1")
        B_min = st.slider("B_min", 0.0, 0.3, 0.05, 0.01, key="bs_bmin")
        B_max = st.slider("B_max", 0.05, 1.0, 0.4, 0.05, key="bs_bmax")
        delta_B_star = st.slider("δ_B*", -3.0, 3.0, 0.0, 0.1, key="bs_dBs")
        kappa_B = st.slider("κ_B", 0.1, 30.0, 5.0, 0.1, key="bs_kB")
        n_max = st.slider("n max", 5, 60, 30, 1, key="bs_nmax")
    reset_button("bias", keys)

    bias_args = dict(beta0=beta0, beta1=beta1, B_min=B_min, B_max=B_max,
                     delta_B_star=delta_B_star, kappa_B=kappa_B)
    tab1, tab2, tab3 = st.tabs(["Temporal build-up", "Amplitude ω_B", "Combined"])
    n = np.arange(0, n_max + 1)
    delta_grid = np.linspace(-10.0, 10.0, 400)

    with tab1:
        d = st.slider("δ (slice)", -10.0, 10.0, 0.0, 0.1, key="bs_d_t1")
        B = practice_bias(n, d, **bias_args)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(n, B, linewidth=2)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="No bias")
        ax.axhline(1.0 + B_max, color="gray", linestyle="--", alpha=0.5,
                   label=f"1 + B_max = {1 + B_max:.2f}")
        ax.set_xlabel(r"$n$  (plays)"); ax.set_ylabel(r"$B(n,\delta)$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"n": n, "B": B}), "Bias_temporal")

    with tab2:
        w = omega_B(delta_grid, B_min, B_max, delta_B_star, kappa_B)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(delta_grid, w, linewidth=2)
        ax.axhline(B_max, color="gray", linestyle="--", alpha=0.5, label=f"B_max = {B_max}")
        ax.axhline(B_min, color="gray", linestyle="--", alpha=0.5, label=f"B_min = {B_min}")
        ax.axvline(delta_B_star, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$\delta - \delta_B^*$"); ax.set_ylabel(r"$\omega_B$")
        ax.legend(); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame({"delta": delta_grid, "omega_B": w}), "Bias_omega")

    with tab3:
        raw = st.text_input("|δ − δ_B*| values (comma-separated)",
                            value="0, 1, 2, 3, 5, 10", key="bs_cmp")
        mismatch_levels = parse_floats(raw, [0.0, 1.0, 2.0, 3.0, 5.0, 10.0])
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {"n": n}
        for d in mismatch_levels:
            B = practice_bias(n, delta_B_star + d, **bias_args)
            ax.plot(n, B, label=f"|δ − δ_B*| = {d}", linewidth=2)
            data[f"B_mismatch={d}"] = B
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(1.0 + B_max, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$n$"); ax.set_ylabel(r"$B(n,\delta)$")
        ax.legend(fontsize=9); prettify(ax); fig.tight_layout()
        show_fig(fig)
        downloads(fig, pd.DataFrame(data), "Bias_combined")

    with st.expander("Interpretation"):
        st.markdown(
            "Weibull build-up with three qualitative regimes: concave (β₁ < 1), "
            "exponential (β₁ = 1), S-shaped (β₁ > 1). Amplitude ω_B is "
            "strongest near δ_B* and decays with mismatch — practice effects "
            "are a localized phenomenon."
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
RENDERERS = {
    "es": render_es, "z": render_z, "psi": render_psi, "eng": render_eng,
    "fat": render_fat, "arousal": render_arousal, "valence": render_valence,
    "emot": render_emot, "bias": render_bias,
}
RENDERERS[component_key]()
