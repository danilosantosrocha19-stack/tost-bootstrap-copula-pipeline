"""
Statistical Pipeline for Small-Sample Equivalence RCTs
=======================================================
Bootstrap ANCOVA + TOST + Gaussian Copula Sensitivity Analysis

This pipeline reproduces exactly the analysis reported in:
  Rocha et al. (2025) — Telerehabilitation via Serious Game vs Conventional
  Physiotherapy for Shoulder Rehabilitation in Manual Wheelchair Users.

Authors: Danilo Santos Rocha, Thiago Braga Rodrigues, Eduardo Lázaro Martins Naves
Institution: UFU, Brazil | Technological University of the Shannon (TUS), Ireland
Repository: https://github.com/danilosantosrocha19-stack/tost-bootstrap-copula-pipeline

─────────────────────────────────────────────────────────────────────────────
STATISTICAL APPROACH
─────────────────────────────────────────────────────────────────────────────
Step 1 — Bootstrap ANCOVA
    Model: outcome_pos ~ group + outcome_pre  (OLS, B=1000 resamples)
    Estimates the adjusted group difference (β) with parametric and
    bootstrapped 95% confidence intervals.

Step 2 — TOST (Two One-Sided Tests)
    Applied to post-treatment scores (outcome_pos) using Welch's t-test
    (unequal variances), consistent with the small, unbalanced sample (n=24).
    Equivalence is declared when the 90% CI falls entirely within ±Δ.

Step 3 — Gaussian Copula Sensitivity Analysis
    A rank-based Gaussian copula is fitted to the post-treatment scores of
    each group. 1,000 synthetic replicas are generated and TOST is re-applied
    to each. The robustness index (%) reflects how often equivalence holds
    under perturbations of the joint dependency structure across outcomes.
    Note: the copula operates on post-treatment scores, not deltas,
    because it models the multivariate outcome distribution at follow-up.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm as sp_norm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttost_ind


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: BOOTSTRAP ANCOVA
# ─────────────────────────────────────────────────────────────────────────────

def ancova_bootstrap(df, outcome_pos, outcome_pre, group_col='grupo',
                     B=1000, seed=42):
    """
    Fit ANCOVA (outcome_pos ~ group + outcome_pre) with bootstrap CIs.

    Parameters
    ----------
    df          : pd.DataFrame — full dataset
    outcome_pos : str — column name for post-treatment outcome
    outcome_pre : str — column name for baseline covariate
    group_col   : str — column name for group (0/1)
                  Group 1 = serious game; Group 0 = conventional physiotherapy
    B           : int — number of bootstrap resamples
    seed        : int — random seed for reproducibility

    Returns
    -------
    dict with: beta, p_ancova, ic95_lo_param, ic95_hi_param,
               ic95_lo_boot, ic95_hi_boot, boot_n_failed
    """
    rng = np.random.default_rng(seed)

    tmp = df[[outcome_pos, outcome_pre, group_col]].dropna().copy()
    tmp.columns = ['pos', 'pre', 'grupo']

    modelo      = smf.ols('pos ~ grupo + pre', data=tmp).fit()
    beta_obs    = modelo.params['grupo']
    p_ancova    = modelo.pvalues['grupo']
    ic_lo_param = modelo.conf_int().loc['grupo', 0]
    ic_hi_param = modelo.conf_int().loc['grupo', 1]

    betas_boot = []
    n_failed   = 0
    for _ in range(B):
        sample = tmp.sample(n=len(tmp), replace=True,
                            random_state=int(rng.integers(1_000_000)))
        try:
            m = smf.ols('pos ~ grupo + pre', data=sample).fit()
            betas_boot.append(m.params['grupo'])
        except Exception:
            n_failed += 1
            continue

    ic_lo_boot = np.percentile(betas_boot, 2.5)
    ic_hi_boot = np.percentile(betas_boot, 97.5)

    return {
        'beta'          : round(beta_obs,    3),
        'p_ancova'      : round(p_ancova,    4),
        'ic95_lo_param' : round(ic_lo_param, 3),
        'ic95_hi_param' : round(ic_hi_param, 3),
        'ic95_lo_boot'  : round(ic_lo_boot,  3),
        'ic95_hi_boot'  : round(ic_hi_boot,  3),
        'boot_n_failed' : n_failed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: TOST — TWO ONE-SIDED TESTS
# ─────────────────────────────────────────────────────────────────────────────

def tost_welch(a1, a0, delta):
    """
    Two One-Sided Tests (TOST) using Welch's t-test (unequal variances).

    Applied to post-treatment scores directly, consistent with the
    unbalanced small sample (G1=13, G0=11).

    Parameters
    ----------
    a1    : array-like — Group 1 post-treatment scores
    a0    : array-like — Group 0 post-treatment scores
    delta : float      — equivalence margin (±Δ)

    Returns
    -------
    dict with: diff, ic90_lo, ic90_hi, p_tost, result
    """
    a1, a0 = np.asarray(a1, float), np.asarray(a0, float)

    # TOST via statsmodels (Welch, unequal variances)
    result = ttost_ind(a1, a0, low=-delta, upp=delta, usevar='unequal')
    p_tost = float(result[0])

    # 90% CI (Welch)
    diff = np.mean(a1) - np.mean(a0)
    se   = np.sqrt(np.std(a1, ddof=1)**2 / len(a1) +
                   np.std(a0, ddof=1)**2 / len(a0))
    df_w = se**4 / (
        (np.std(a1, ddof=1)**2 / len(a1))**2 / (len(a1) - 1) +
        (np.std(a0, ddof=1)**2 / len(a0))**2 / (len(a0) - 1)
    )
    t_crit  = stats.t.ppf(0.95, df_w)
    ic90_lo = diff - t_crit * se
    ic90_hi = diff + t_crit * se

    dentro      = (ic90_lo >= -delta) and (ic90_hi <= delta)
    exclui_zero = (ic90_lo > 0) or (ic90_hi < 0)

    if not dentro:
        label = 'Inconclusivo'
    elif exclui_zero:
        label = 'G1 superior (dentro da margem)'
    else:
        label = 'Equivalente estrito'

    return {
        'diff'   : round(diff,    3),
        'ic90_lo': round(ic90_lo, 3),
        'ic90_hi': round(ic90_hi, 3),
        'p_tost' : round(p_tost,  4),
        'result' : label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: GAUSSIAN COPULA SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _fit_copula(mat):
    """
    Fit a rank-based Gaussian copula to a data matrix.
    Returns (correlation_matrix, means, stds).
    No external packages required.
    """
    n, p = mat.shape
    ranks = np.column_stack([
        sp_norm.ppf(stats.rankdata(mat[:, j]) / (n + 1))
        for j in range(p)
    ])
    rho = np.corrcoef(ranks.T)
    return rho, mat.mean(axis=0), mat.std(axis=0, ddof=1)


def _sample_copula(rho, mu, sigma, n, rng):
    """
    Generate n synthetic samples from a fitted Gaussian copula.
    """
    z = rng.multivariate_normal(np.zeros(len(mu)), rho, size=n)
    return sp_norm.cdf(z) * sigma + mu


def copula_sensitivity(df_g1, df_g2, pos_columns, delta_dict,
                       B=1000, seed=123):
    """
    Gaussian Copula sensitivity analysis on post-treatment scores.

    Fits a rank-based Gaussian copula to each group's post-treatment
    outcome matrix, generates B synthetic replicas, and re-applies TOST
    to compute a robustness index per outcome.

    The copula operates on post-treatment scores (outcome_pos) because it
    models the full multivariate outcome distribution at follow-up.
    Robustness reflects stability of the equivalence finding under
    perturbations of the joint dependency structure — not baseline similarity.

    Parameters
    ----------
    df_g1       : pd.DataFrame — Group 1 post-treatment data
    df_g2       : pd.DataFrame — Group 0 post-treatment data
    pos_columns : list of str  — outcome _pos columns to include
    delta_dict  : dict         — {column: equivalence_margin}
    B           : int          — number of synthetic replicas
    seed        : int          — random seed (123 matches original analysis)

    Returns
    -------
    pd.DataFrame — robustness index per outcome (index = column names)
    """
    mat1 = df_g1[pos_columns].dropna().values.astype(float)
    mat0 = df_g2[pos_columns].dropna().values.astype(float)

    rho1, mu1, sig1 = _fit_copula(mat1)
    rho0, mu0, sig0 = _fit_copula(mat0)

    rng    = np.random.default_rng(seed)
    counts = {col: 0 for col in pos_columns}
    n1, n2 = len(df_g1), len(df_g2)

    for _ in range(B):
        syn1 = _sample_copula(rho1, mu1, sig1, n1, rng)
        syn0 = _sample_copula(rho0, mu0, sig0, n2, rng)
        for ji, col in enumerate(pos_columns):
            delta = delta_dict.get(col, 10)
            try:
                res = tost_welch(syn1[:, ji], syn0[:, ji], delta)
                if res['result'] != 'Inconclusivo':
                    counts[col] += 1
            except Exception:
                continue

    robustez = {}
    for col in pos_columns:
        pct   = counts[col] / B
        nivel = 'Alta' if pct >= 0.80 else ('Moderada' if pct >= 0.50 else 'Baixa')
        robustez[col] = {
            'robustez_pct': round(pct * 100, 1),
            'nivel'       : nivel,
        }

    return pd.DataFrame(robustez).T


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df, outcomes, margins, group_col='grupo',
                 B=1000, seed=42, verbose=True):
    """
    Run the full 3-step pipeline for all outcomes.

    Parameters
    ----------
    df        : pd.DataFrame — full dataset with group column
    outcomes  : list of str  — outcome base names (e.g. ['TSK', 'PSEQ', ...])
                Expects columns: {name}_pre and {name}_pos in the dataframe.
    margins   : dict         — {outcome_name: equivalence_margin}
    group_col : str          — group column (1 = serious game, 0 = physio)
    B         : int          — bootstrap and copula resamples
    seed      : int          — random seed
    verbose   : bool         — print progress

    Returns
    -------
    pd.DataFrame — combined results table
    """
    import warnings
    warnings.filterwarnings('ignore')

    g1 = df[df[group_col] == 1].copy()
    g0 = df[df[group_col] == 0].copy()

    results = []

    for name in outcomes:
        pos_col = f'{name}_pos'
        pre_col = f'{name}_pre'
        delta   = margins.get(name, 10)

        if verbose:
            print(f'  Processing: {name}...')

        if pos_col not in df.columns:
            print(f'    WARNING: "{pos_col}" not found. Skipping.')
            continue
        if pre_col not in df.columns:
            print(f'    WARNING: "{pre_col}" not found. Skipping.')
            continue

        # Step 1 — Bootstrap ANCOVA
        anc = ancova_bootstrap(df, pos_col, pre_col,
                               group_col=group_col, B=B, seed=seed)

        # Step 2 — TOST (Welch, on post-treatment scores)
        tost_res = tost_welch(
            g1[pos_col].dropna().values,
            g0[pos_col].dropna().values,
            delta
        )

        results.append({
            'Outcome'       : name,
            'beta_ANCOVA'   : anc['beta'],
            'IC95_lo_param' : anc['ic95_lo_param'],
            'IC95_hi_param' : anc['ic95_hi_param'],
            'IC95_lo_boot'  : anc['ic95_lo_boot'],
            'IC95_hi_boot'  : anc['ic95_hi_boot'],
            'p_ANCOVA'      : anc['p_ancova'],
            'margin_delta'  : delta,
            'diff_G1_G0'    : tost_res['diff'],
            'IC90_lo'       : tost_res['ic90_lo'],
            'IC90_hi'       : tost_res['ic90_hi'],
            'p_TOST'        : tost_res['p_tost'],
            'Result_TOST'   : tost_res['result'],
        })

    df_results = pd.DataFrame(results)

    # Step 3 — Gaussian Copula (on _pos columns)
    if verbose:
        print('\n  Running Gaussian Copula sensitivity analysis...')

    pos_cols  = [f'{n}_pos' for n in outcomes if f'{n}_pos' in df.columns]
    delta_map = {f'{n}_pos': margins.get(n, 10) for n in outcomes
                 if f'{n}_pos' in df.columns}

    if pos_cols:
        cop_res = copula_sensitivity(
            g1[pos_cols].copy(),
            g0[pos_cols].copy(),
            pos_cols, delta_map,
            B=B, seed=123   # seed=123 matches original notebook
        )
        df_results['Robustez_Copula_%'] = df_results['Outcome'].map(
            lambda x: cop_res['robustez_pct'].get(f'{x}_pos', None)
        )
        df_results['Robustez_nivel'] = df_results['Outcome'].map(
            lambda x: cop_res['nivel'].get(f'{x}_pos', None)
        )

    return df_results


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print('Loading example data...')
    df = pd.read_csv('example_data.csv')

    # Outcome base names (expects {name}_pre and {name}_pos columns)
    outcomes = [
        'TSK', 'PCS', 'PSEQ', 'ADAP', 'SPADI_tot',
        'WHOQOL', 'FLEX_D', 'FLEX_E', 'ABD_D', 'ABD_E',
        'ROT_INT', 'ROT_EXT',
    ]

    margins = {
        'TSK'      : 5,
        'PCS'      : 5,
        'PSEQ'     : 5,
        'ADAP'     : 10,
        'SPADI_tot': 10,
        'WHOQOL'   : 8,
        'FLEX_D'   : 15,
        'FLEX_E'   : 15,
        'ABD_D'    : 15,
        'ABD_E'    : 15,
        'ROT_INT'  : 15,
        'ROT_EXT'  : 15,
    }

    # Validate columns before running
    expected = ([f'{n}_pre' for n in outcomes] +
                [f'{n}_pos' for n in outcomes] +
                ['grupo'])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in CSV: {missing}')

    print('\nRunning full pipeline (B=1000)...\n')
    results = run_pipeline(df, outcomes, margins, B=1000, seed=42, verbose=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print('\n=== PIPELINE RESULTS ===')
    print(results.to_string(index=False))

    results.to_csv('pipeline_results.csv', index=False)
    print('\nResults saved to pipeline_results.csv')
