"""
Statistical Pipeline for Small-Sample Equivalence RCTs
=======================================================
Bootstrap ANCOVA + TOST + Gaussian Copula Sensitivity Analysis

Authors: Danilo Santos Rocha, Thiago Braga Rodrigues, Eduardo Lázaro Martins Naves
Institution: UFU, Brazil | Technological University of the Shannon (TUS), Ireland
Repository: https://github.com/danilosantosrocha19-stack/tost-bootstrap-copula-pipeline
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: BOOTSTRAP ANCOVA
# ─────────────────────────────────────────────────────────────────────────────

def ancova_bootstrap(df, outcome_delta, outcome_pre, group_col='Grupo',
                     B=1000, seed=42):
    """
    Fit ANCOVA (outcome_post ~ Group + outcome_pre) with bootstrap CIs.

    Parameters
    ----------
    df           : pd.DataFrame — full dataset
    outcome_delta: str — column name for outcome delta (post - pre)
    outcome_pre  : str — column name for baseline covariate
    group_col    : str — column name for group (0/1)
    B            : int — number of bootstrap resamples
    seed         : int — random seed for reproducibility

    Returns
    -------
    dict with: beta_obs, ic_lo_boot, ic_hi_boot, p_ancova, ic_lo_param, ic_hi_param
    """
    rng = np.random.default_rng(seed)

    formula = f'{outcome_delta} ~ {group_col} + {outcome_pre}'
    modelo  = smf.ols(formula, data=df).fit()

    beta_obs   = modelo.params[group_col]
    p_ancova   = modelo.pvalues[group_col]
    ic_lo_param = modelo.conf_int().loc[group_col, 0]
    ic_hi_param = modelo.conf_int().loc[group_col, 1]

    betas_boot = []
    for _ in range(B):
        sample = df.sample(n=len(df), replace=True,
                           random_state=int(rng.integers(1_000_000)))
        try:
            m = smf.ols(formula, data=sample).fit()
            betas_boot.append(m.params[group_col])
        except Exception:
            continue

    ic_lo_boot = np.percentile(betas_boot, 2.5)
    ic_hi_boot = np.percentile(betas_boot, 97.5)

    return {
        'beta'         : round(beta_obs,   3),
        'p_ancova'     : round(p_ancova,   4),
        'ic95_lo_param': round(ic_lo_param,3),
        'ic95_hi_param': round(ic_hi_param,3),
        'ic95_lo_boot' : round(ic_lo_boot, 3),
        'ic95_hi_boot' : round(ic_hi_boot, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: TOST — TWO ONE-SIDED TESTS
# ─────────────────────────────────────────────────────────────────────────────

def tost(beta, se, df_resid, delta):
    """
    Two One-Sided Tests (TOST) for equivalence.

    Parameters
    ----------
    beta     : float — adjusted difference (from ANCOVA)
    se       : float — standard error of beta
    df_resid : float — residual degrees of freedom
    delta    : float — equivalence margin (±Δ)

    Returns
    -------
    dict with: p_tost, ic90_lo, ic90_hi, result_label
    """
    t_low  = (beta - (-delta)) / se
    t_high = (beta -   delta)  / se

    p_low  = stats.t.sf( t_low,  df_resid)
    p_high = stats.t.cdf(t_high, df_resid)
    p_tost = max(p_low, p_high)

    t_crit = stats.t.ppf(0.95, df_resid)
    ic90_lo = beta - t_crit * se
    ic90_hi = beta + t_crit * se

    if ic90_lo >= -delta and ic90_hi <= delta:
        if ic90_lo <= 0 <= ic90_hi:
            label = "Equivalente estrito"
        else:
            label = "G1 superior (dentro da margem)"
    else:
        label = "Inconclusivo"

    return {
        'p_tost' : round(p_tost,  4),
        'ic90_lo': round(ic90_lo, 3),
        'ic90_hi': round(ic90_hi, 3),
        'result' : label,
    }


def run_tost_from_ancova(df, outcome_delta, outcome_pre, delta,
                         group_col='Grupo'):
    """
    Run TOST using beta and SE extracted directly from ANCOVA.
    """
    formula = f'{outcome_delta} ~ {group_col} + {outcome_pre}'
    modelo  = smf.ols(formula, data=df).fit()
    beta    = modelo.params[group_col]
    se      = modelo.bse[group_col]
    df_r    = modelo.df_resid
    return tost(beta, se, df_r, delta)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: GAUSSIAN COPULA SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def copula_sensitivity(df_g1, df_g2, columns, delta_dict, B=1000, seed=42):
    """
    Gaussian Copula sensitivity analysis.
    Fits copulas to each group, generates B synthetic replicas,
    re-applies TOST and computes robustness index.

    Parameters
    ----------
    df_g1      : pd.DataFrame — Group 1 data
    df_g2      : pd.DataFrame — Group 0 data
    columns    : list of str — outcome columns to include
    delta_dict : dict — {column: equivalence_margin}
    B          : int — number of synthetic replicas
    seed       : int — random seed

    Returns
    -------
    pd.DataFrame with robustness index per outcome
    """
    try:
        from copulas.multivariate import GaussianMultivariate
    except ImportError:
        raise ImportError("Install copulas: pip install copulas")

    np.random.seed(seed)

    cop_g1 = GaussianMultivariate()
    cop_g2 = GaussianMultivariate()
    cop_g1.fit(df_g1[columns].dropna())
    cop_g2.fit(df_g2[columns].dropna())

    counts = {col: 0 for col in columns}
    n1, n2 = len(df_g1), len(df_g2)

    for i in range(B):
        syn_g1 = cop_g1.sample(n1)
        syn_g2 = cop_g2.sample(n2)
        for col in columns:
            delta = delta_dict.get(col, 10)
            diff  = syn_g1[col].mean() - syn_g2[col].mean()
            se_   = np.sqrt(syn_g1[col].var() / n1 + syn_g2[col].var() / n2)
            df_r_ = n1 + n2 - 2
            if se_ == 0:
                continue
            res = tost(diff, se_, df_r_, delta)
            if res['result'] != "Inconclusivo":
                counts[col] += 1

    robustez = {}
    for col in columns:
        pct = counts[col] / B
        if pct >= 0.80:
            nivel = "Alta"
        elif pct >= 0.50:
            nivel = "Moderada"
        else:
            nivel = "Baixa"
        robustez[col] = {'robustez_pct': round(pct * 100, 1), 'nivel': nivel}

    return pd.DataFrame(robustez).T


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(df, margins, outcome_pairs=None, group_col='Grupo',
                 B=1000, seed=42, verbose=True):
    """
    Run the full 3-step pipeline for all outcomes.

    Parameters
    ----------
    df           : pd.DataFrame — full dataset with group column
    margins      : dict — {outcome_delta: equivalence_margin}
    outcome_pairs: dict — {outcome_delta: outcome_pre} for ANCOVA baseline
                   If None, inferred by replacing '_delta' with '_pre'
    group_col    : str — group column name (binary: 0/1)
    B            : int — bootstrap and copula resamples
    seed         : int — random seed
    verbose      : bool — print progress

    Returns
    -------
    pd.DataFrame — combined results table
    """
    g1 = df[df[group_col] == 1]
    g0 = df[df[group_col] == 0]
    results = []

    for outcome_delta, delta in margins.items():
        if verbose:
            print(f"  Processing: {outcome_delta}...")

        if outcome_pairs and outcome_delta in outcome_pairs:
            outcome_pre = outcome_pairs[outcome_delta]
        else:
            outcome_pre = outcome_delta.replace('_delta', '_pre')

        if outcome_pre not in df.columns:
            print(f"    WARNING: baseline '{outcome_pre}' not found. Skipping.")
            continue

        anc = ancova_bootstrap(df, outcome_delta, outcome_pre,
                               group_col=group_col, B=B, seed=seed)

        modelo = smf.ols(
            f'{outcome_delta} ~ {group_col} + {outcome_pre}', data=df
        ).fit()
        tost_res = tost(
            modelo.params[group_col],
            modelo.bse[group_col],
            modelo.df_resid,
            delta
        )

        results.append({
            'Desfecho'      : outcome_delta.replace('_delta', ''),
            'beta'          : anc['beta'],
            'IC95_lo_param' : anc['ic95_lo_param'],
            'IC95_hi_param' : anc['ic95_hi_param'],
            'IC95_lo_boot'  : anc['ic95_lo_boot'],
            'IC95_hi_boot'  : anc['ic95_hi_boot'],
            'p_ANCOVA'      : anc['p_ancova'],
            'margem_delta'  : delta,
            'IC90_lo'       : tost_res['ic90_lo'],
            'IC90_hi'       : tost_res['ic90_hi'],
            'p_TOST'        : tost_res['p_tost'],
            'Resultado_TOST': tost_res['result'],
        })

    df_results = pd.DataFrame(results)

    if verbose:
        print("\n  Running Gaussian Copula sensitivity analysis...")

    delta_map_pre = {}
    for outcome_delta in margins:
        pre_col = outcome_delta.replace('_delta', '_pre')
        if pre_col in df.columns:
            delta_map_pre[pre_col] = margins[outcome_delta]

    pre_cols = [c for c in delta_map_pre if c in df.columns]
    if pre_cols:
        cop_res = copula_sensitivity(
            g1, g0, pre_cols, delta_map_pre, B=B, seed=seed
        )
        df_results['Robustez_Copula_%'] = df_results['Desfecho'].map(
            lambda x: cop_res['robustez_pct'].get(x + '_pre', None)
        )
        df_results['Robustez_nivel'] = df_results['Desfecho'].map(
            lambda x: cop_res['nivel'].get(x + '_pre', None)
        )

    return df_results


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading example data...")
    df = pd.read_csv('example_data.csv')

    margins = {
        'TSK_delta'      : 5,
        'PCS_delta'      : 5,
        'PSEQ_delta'     : 5,
        'ADAP_delta'     : 10,
        'SPADI_tot_delta': 10,
        'WHOQOL_delta'   : 8,
        'FLEX_D_delta'   : 15,
        'FLEX_E_delta'   : 15,
        'ABD_D_delta'    : 15,
        'ABD_E_delta'    : 15,
        'ROT_INT_delta'  : 15,
        'ROT_EXT_delta'  : 15,
    }

    print("\nRunning full pipeline (B=1000)...\n")
    results = run_pipeline(df, margins, B=1000, seed=42, verbose=True)

    print("\n=== PIPELINE RESULTS ===")
    print(results.to_string(index=False))

    results.to_csv('pipeline_results.csv', index=False)
    print("\nResults saved to pipeline_results.csv")
