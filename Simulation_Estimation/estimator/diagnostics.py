"""
estimator/diagnostics.py — Comprehensive estimation quality assessment.

Produces a structured text report with all key metrics, saves it to disk,
and returns a dict of results for programmatic access.

Usage (in notebook, after fit()):
    from estimator.diagnostics import run_diagnostics
    report = run_diagnostics(result, dataset, game_pool, pool, cfg)
"""

from __future__ import annotations

import torch
import numpy as np

from simulator import SimParams, SimulationDataset
from estimator.parameters import ModelParameters
from estimator.loss import forward_pass, _flat_indices
import core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float('nan')
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a.ravel() - b.ravel()) ** 2).mean()))


def _estimated_C_trajectory(params: ModelParameters, dataset: SimulationDataset) -> torch.Tensor:
    """Reconstruct estimated C trajectory, shape [I, T+1, K]."""
    I, T, M = dataset.ogs.shape
    K = params.K

    flat_games = _flat_indices(dataset, params.gs.Nk)
    i_idx = torch.arange(I).view(I, 1, 1).expand(I, T, M)

    C_traj = torch.zeros(I, T + 1, K)
    C_current = params.C_init.detach().clone()
    Q_current = torch.zeros(I, K)
    C_traj[:, 0, :] = C_current

    eta_k = params.eta_k.detach()
    zeta_jk = params.zeta_jk.detach()
    u_all = params.u_ijk.detach()
    m_1idx = torch.arange(1, M + 1, dtype=torch.float32).unsqueeze(0).expand(I, M)

    with torch.no_grad():
        for t in range(T):
            k_t = dataset.game_k[:, t, :]
            d_t = dataset.d[:, t, :]
            n_t = dataset.n_before[:, t, :]
            fg_t = flat_games[:, t, :]

            zeta_t = zeta_jk[fg_t]
            u_t = u_all[i_idx[:, t, :], fg_t]

            C_ik = C_current.gather(1, k_t)
            delta = C_ik - d_t

            Z = core.game_effectiveness(n_t, zeta_t, params.zeta_min, params.lambda_Z)
            Psi = core.mismatch_effect(delta, params.delta_star, params.kappa_L, params.kappa_R)
            E = core.game_engagement(u_t, params.u_min, params.alpha_u)
            F_t = core.fatigue_cost(m_1idx, M, delta, params.delta_star,
                                    params.rho0_F, params.rho_h_F,
                                    params.rho_e_F, params.tau_F)
            pi_t = core.single_game_gain(Z, Psi, E, F_t)

            Pi_t = torch.zeros(I, K)
            Pi_t.scatter_add_(1, k_t, pi_t)

            Q_current = core.update_cumulative_impact(Q_current, Pi_t, params.rho_Q)
            C_current = core.update_ability(C_current, Q_current, eta_k.unsqueeze(0))
            C_traj[:, t + 1, :] = C_current

    return C_traj


def _get_global_pairs(true_p, fitted):
    """Safely extract (name, true_val, est_val) for all global params."""
    names = [
        'gamma0', 'gamma1', 'E_min', 'kappa_A', 'kappa_V',
        'alpha_A', 'alpha_V', 'kappa_B', 'delta_B_star',
        'kappa_L', 'kappa_R', 'delta_star', 'zeta_min',
        'lambda_Z', 'u_min', 'alpha_u', 'rho0_F', 'rho_h_F',
        'rho_e_F', 'tau_F', 'rho_Q',
    ]
    pairs = []
    for name in names:
        true_val = getattr(true_p, name, None)
        est_attr = getattr(fitted, name, None)
        if true_val is None or est_attr is None:
            continue
        est_val = est_attr.item() if hasattr(est_attr, 'item') else float(est_attr)
        pairs.append((name, float(true_val), est_val))
    return pairs


# ---------------------------------------------------------------------------
# Main diagnostic function
# ---------------------------------------------------------------------------

def run_diagnostics(
    result:    dict,
    dataset:   SimulationDataset,
    game_pool,
    pool,
    cfg:       dict,
    save_path: str = 'data/estimation_report.txt',
) -> dict:
    """
    Run comprehensive estimation diagnostics.

    Parameters
    ----------
    result    : dict from fit() — 'params', 'loss_history', 'best_loss'
    dataset   : SimulationDataset
    game_pool : GamePool (has ground-truth parameters)
    pool      : ParticipantPool (has ground-truth values)
    cfg       : full YAML config dict
    save_path : where to save the text report (None to skip)

    Returns
    -------
    report : dict with all metrics (also printed and saved to disk)
    """
    fitted = result['params']
    history = result['loss_history']

    I, T, M = dataset.ogs.shape
    K = dataset.C_true.shape[-1]
    N_obs = I * T * M

    true_p = SimParams.from_config(cfg)
    sigma_OGS = cfg['simulation']['noise']['sigma_OGS']
    noise_floor = sigma_OGS ** 2

    lines: list[str] = []
    report: dict = {}

    def pr(s: str = ''):
        lines.append(s)

    # ===================================================================
    # 1. LOSS SUMMARY
    # ===================================================================
    pr('=' * 70)
    pr('ESTIMATION DIAGNOSTICS REPORT')
    pr('=' * 70)
    pr()
    pr(f'Dataset: I={I}, T={T}, M={M}, K={K}, N_obs={N_obs:,}')
    pr(f'Epochs run: {result["n_epochs_run"]}')
    pr()

    final = history[-1]
    pr('--- 1. Loss Summary ---')
    pr(f'  Final total : {final["total"]:.6f}')
    pr(f'  Final obs   : {final["obs"]:.6f}')
    pr(f'  Final reg   : {final["reg"]:.6f}')
    pr(f'  Noise floor : {noise_floor:.6f}  (sigma_OGS^2 = {sigma_OGS}^2)')
    pr(f'  Obs/floor   : {final["obs"]/noise_floor:.2f}x  (1.0 = perfect fit)')
    pr(f'  Best loss   : {result["best_loss"]:.6f}')
    pr()

    report['loss'] = {
        'final_total': final['total'], 'final_obs': final['obs'],
        'final_reg': final['reg'], 'noise_floor': noise_floor,
        'obs_over_floor': final['obs'] / noise_floor,
        'best_loss': result['best_loss'], 'n_epochs': result['n_epochs_run'],
    }

    # ===================================================================
    # 2. GLOBAL PARAMETER RECOVERY
    # ===================================================================
    pr('--- 2. Global Parameter Recovery ---')
    pr(f'  {"Parameter":<14}  {"True":>8}  {"Estimated":>10}  {"Rel Error":>10}  {"Status"}')
    pr('  ' + '-' * 60)

    global_pairs = _get_global_pairs(true_p, fitted)
    report['global'] = {}

    for name, true_val, est_val in global_pairs:
        rel = abs(est_val - true_val) / (abs(true_val) + 1e-9)
        status = 'OK' if rel < 0.15 else 'WARN' if rel < 0.30 else 'BAD'
        pr(f'  {name:<14}  {true_val:>8.4f}  {est_val:>10.4f}  {rel:>9.1%}  {status}')
        report['global'][name] = {'true': true_val, 'est': est_val, 'rel_err': rel}

    n_ok = sum(1 for v in report['global'].values() if v['rel_err'] < 0.15)
    n_warn = sum(1 for v in report['global'].values() if 0.15 <= v['rel_err'] < 0.30)
    n_bad = sum(1 for v in report['global'].values() if v['rel_err'] >= 0.30)
    pr(f'  Summary: {n_ok} OK, {n_warn} WARN, {n_bad} BAD  '
       f'(thresholds: <15% OK, <30% WARN)')
    pr()

    # ===================================================================
    # 3. DOMAIN PARAMETER RECOVERY
    # ===================================================================
    pr('--- 3. Domain Parameter Recovery ---')

    eta_true = torch.full((K,), float(cfg['domain_init']['eta_k']))
    eta_est = fitted.eta_k.detach()
    pr(f'  eta_k true : {[f"{v:.4f}" for v in eta_true.tolist()]}')
    pr(f'  eta_k est  : {[f"{v:.4f}" for v in eta_est.tolist()]}')

    Bk_true = torch.full((K,), float(cfg['domain_init']['Bk_max']))
    Bk_est = fitted.Bk_max.detach()
    pr(f'  Bk_max true: {[f"{v:.4f}" for v in Bk_true.tolist()]}')
    pr(f'  Bk_max est : {[f"{v:.4f}" for v in Bk_est.tolist()]}')
    pr()

    report['domain'] = {
        'eta_k': {'true': eta_true.tolist(), 'est': eta_est.tolist()},
        'Bk_max': {'true': Bk_true.tolist(), 'est': Bk_est.tolist()},
    }

    # ===================================================================
    # 4. GAME PARAMETER RECOVERY
    # ===================================================================
    pr('--- 4. Game Parameter Recovery ---')

    gp_true = game_pool.parameter_tensors()
    report['game'] = {}

    for name, true_t, est_t in [
        ('zeta_jk',  gp_true['zeta'],  fitted.zeta_jk.detach()),
        ('beta0_jk', gp_true['beta0'], fitted.beta0_jk.detach()),
        ('beta1_jk', gp_true['beta1'], fitted.beta1_jk.detach()),
    ]:
        t = true_t.numpy().ravel()
        e = est_t.numpy().ravel()
        c = _corr(t, e)
        r = _rmse(t, e)
        status = 'OK' if c > 0.7 else 'WARN' if c > 0.4 else 'BAD'
        pr(f'  {name:12s}: corr={c:.3f}  RMSE={r:.4f}  '
           f'true=[{t.min():.3f},{t.max():.3f}]  '
           f'est=[{e.min():.3f},{e.max():.3f}]  {status}')
        report['game'][name] = {'corr': c, 'rmse': r}
    pr()

    # ===================================================================
    # 5. INDIVIDUAL PARAMETER RECOVERY
    # ===================================================================
    pr('--- 5. Individual Parameter Recovery ---')

    C_true_init = pool.C_init_tensor()
    A_true = pool.A_star_tensor()
    u_true = pool.u_tensor()
    report['individual'] = {}

    for name, true_t, est_t in [
        ('C_init', C_true_init.view(-1), fitted.C_init.detach().view(-1)),
        ('A_star', A_true,               fitted.A_star.detach()),
        ('u_ijk',  u_true.view(-1),      fitted.u_ijk.detach().view(-1)),
    ]:
        t = true_t.numpy().ravel()
        e = est_t.numpy().ravel()
        c = _corr(t, e)
        r = _rmse(t, e)
        bias = float(e.mean() - t.mean())
        status = 'OK' if c > 0.7 else 'WARN' if c > 0.4 else 'BAD'
        pr(f'  {name:12s}: corr={c:.3f}  RMSE={r:.4f}  bias={bias:+.3f}  '
           f'true_std={t.std():.3f}  est_std={e.std():.3f}  {status}')
        report['individual'][name] = {
            'corr': c, 'rmse': r, 'bias': bias,
            'true_std': float(t.std()), 'est_std': float(e.std()),
        }
    pr()

    # ===================================================================
    # 6. OGS FIT QUALITY
    # ===================================================================
    pr('--- 6. OGS Fit Quality ---')

    with torch.no_grad():
        ogs_hat = forward_pass(fitted, dataset)

    ogs_obs = dataset.ogs.numpy().ravel()
    ogs_pred = ogs_hat.numpy().ravel()

    corr_ogs = _corr(ogs_obs, ogs_pred)
    rmse_ogs = _rmse(ogs_obs, ogs_pred)
    bias_ogs = float(ogs_pred.mean() - ogs_obs.mean())

    pr(f'  Overall:     corr={corr_ogs:.3f}  RMSE={rmse_ogs:.2f}  bias={bias_ogs:+.2f}')

    obs_sess = dataset.ogs.mean(dim=(0, 2)).numpy()       # [T]
    hat_sess = ogs_hat.mean(dim=(0, 2)).numpy()            # [T]
    sess_err = np.abs(obs_sess - hat_sess)
    pr(f'  Session MAE: {sess_err.mean():.2f}  '
       f'max={sess_err.max():.2f} (session {sess_err.argmax()+1})')

    early_err = np.abs(obs_sess[:5] - hat_sess[:5]).mean()
    late_err = np.abs(obs_sess[-5:] - hat_sess[-5:]).mean()
    pr(f'  Early(1-5):  {early_err:.2f}      Late({T-4}-{T}): {late_err:.2f}')
    pr()

    report['ogs'] = {
        'corr': corr_ogs, 'rmse': rmse_ogs, 'bias': bias_ogs,
        'session_mae': float(sess_err.mean()),
        'session_max_err': float(sess_err.max()),
        'early_err': float(early_err), 'late_err': float(late_err),
    }

    # ===================================================================
    # 7. C TRAJECTORY RECOVERY
    # ===================================================================
    pr('--- 7. C Trajectory Recovery ---')

    C_hat = _estimated_C_trajectory(fitted, dataset)
    C_true_traj = dataset.C_true.numpy()
    C_hat_np = C_hat.numpy()

    rmse_C_all = _rmse(C_hat_np, C_true_traj)
    pr(f'  Overall RMSE: {rmse_C_all:.4f}')

    domain_names = getattr(game_pool, 'domain_names',
                           [f'Dom{k}' for k in range(K)])
    report['C_traj'] = {'overall_rmse': rmse_C_all, 'domains': {}}

    for k in range(K):
        true_k = C_true_traj[:, :, k]
        hat_k = C_hat_np[:, :, k]
        rmse_k = _rmse(true_k, hat_k)
        bias_t0 = float(hat_k[:, 0].mean() - true_k[:, 0].mean())
        bias_tT = float(hat_k[:, -1].mean() - true_k[:, -1].mean())
        corr_t0 = _corr(true_k[:, 0], hat_k[:, 0])
        corr_tT = _corr(true_k[:, -1], hat_k[:, -1])

        dn = domain_names[k] if k < len(domain_names) else f'Dom{k}'
        pr(f'  {dn:6s}: RMSE={rmse_k:.4f}  '
           f'bias(t=0)={bias_t0:+.3f}  bias(t=T)={bias_tT:+.3f}  '
           f'corr(t=0)={corr_t0:.3f}  corr(t=T)={corr_tT:.3f}')
        report['C_traj']['domains'][dn] = {
            'rmse': rmse_k, 'bias_t0': bias_t0, 'bias_tT': bias_tT,
            'corr_t0': corr_t0, 'corr_tT': corr_tT,
        }

    true_gain = C_true_traj[:, -1, :].sum(1) - C_true_traj[:, 0, :].sum(1)
    hat_gain = C_hat_np[:, -1, :].sum(1) - C_hat_np[:, 0, :].sum(1)
    corr_gain = _corr(true_gain, hat_gain)
    pr(f'  Total gain: true={true_gain.mean():.2f}+/-{true_gain.std():.2f}  '
       f'est={hat_gain.mean():.2f}+/-{hat_gain.std():.2f}  corr={corr_gain:.3f}')
    pr()

    report['C_traj']['gain_corr'] = corr_gain
    report['C_traj']['true_gain_mean'] = float(true_gain.mean())
    report['C_traj']['est_gain_mean'] = float(hat_gain.mean())

    # ===================================================================
    # 8. OVERALL ASSESSMENT
    # ===================================================================
    pr('--- 8. Overall Assessment ---')

    checks = [
        ('Obs loss < 3x floor',   report['loss']['obs_over_floor'] < 3.0),
        ('OGS corr > 0.9',        report['ogs']['corr'] > 0.9),
        ('OGS RMSE < 10',         report['ogs']['rmse'] < 10),
        ('C_init corr > 0.7',     report['individual']['C_init']['corr'] > 0.7),
        ('C_init RMSE < 1.0',     report['individual']['C_init']['rmse'] < 1.0),
        ('gamma0 rel err < 15%',  report['global'].get('gamma0', {}).get('rel_err', 1.0) < 0.15),
        ('gamma1 rel err < 15%',  report['global'].get('gamma1', {}).get('rel_err', 1.0) < 0.15),
        ('rho_Q rel err < 15%',   report['global'].get('rho_Q', {}).get('rel_err', 1.0) < 0.15),
        ('zeta corr > 0.5',       report['game']['zeta_jk']['corr'] > 0.5),
        ('C traj RMSE < 1.0',     report['C_traj']['overall_rmse'] < 1.0),
        ('Gain corr > 0.5',       report['C_traj']['gain_corr'] > 0.5),
    ]

    n_pass = sum(1 for _, v in checks if v)
    for label, passed in checks:
        pr(f'  {"PASS" if passed else "FAIL"}  {label}')

    grade = ('A' if n_pass >= 10 else 'B' if n_pass >= 8 else
             'C' if n_pass >= 6  else 'D' if n_pass >= 4 else 'F')
    pr(f'\n  Score: {n_pass}/{len(checks)} checks passed — Grade: {grade}')
    pr()
    pr('=' * 70)

    report['grade'] = grade
    report['checks_passed'] = n_pass
    report['checks_total'] = len(checks)

    # --- Print and save ---
    full_report = '\n'.join(lines)
    print(full_report)

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f'\nReport saved to {save_path}')

    return report
