"""Shared helper for the smoke test — replicates the sim-page YAML walker
without dragging in Streamlit."""

PARAM_PATHS = [
    (("simulation", "program", "I"), "sim_I"),
    (("simulation", "program", "T"), "sim_T"),
    (("simulation", "program", "M"), "sim_M"),
    (("_meta", "policy"), "sim_policy"),
    (("_meta", "seed"), "sim_seed"),
    (("predefined", "arousal", "A_min"), "sim_A_min"),
    (("predefined", "arousal", "K0"), "sim_K0"),
    (("predefined", "valence", "gamma0_V"), "sim_gamma0_V"),
    (("global_init", "gamma0"), "sim_gamma0"),
    (("global_init", "gamma1"), "sim_gamma1"),
    (("global_init", "lambda_Z"), "sim_lambda_Z"),
    (("domain_init", "eta_k"), "sim_eta_k"),
    (("simulation", "noise", "sigma_OGS"), "sim_sigma_OGS"),
]


def walk_param_paths(d: dict) -> dict:
    out: dict = {}
    for path, key in PARAM_PATHS:
        cur = d
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            out[key] = cur
    return out
