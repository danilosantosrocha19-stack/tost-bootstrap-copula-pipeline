[README (1).md](https://github.com/user-attachments/files/26213039/README.1.md)
# Statistical Pipeline for Small-Sample Equivalence RCTs

## Bootstrap ANCOVA · TOST · Gaussian Copula

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This repository provides a fully reproducible statistical pipeline for **randomized controlled equivalence trials (RCTs) with small samples**. The pipeline addresses three sequential analytical questions:

| Step | Method | Question |
|------|--------|----------|
| 1 | **Bootstrap ANCOVA** | Is there an adjusted difference between groups? |
| 2 | **TOST (Two One-Sided Tests)** | Is the difference within a clinically acceptable margin? |
| 3 | **Gaussian Copula** | Are the equivalence findings robust to sample size? |

This pipeline was developed and demonstrated in the context of a shoulder rehabilitation RCT comparing telerehabilitation via Serious Game (RehaBEAT) versus conventional physiotherapy in manual wheelchair users (n = 24).

---

## Repository Structure

```
tost-bootstrap-copula-pipeline/
├── pipeline.py          # Full pipeline: Bootstrap ANCOVA + TOST + Gaussian Copula
├── requirements.txt     # Python dependencies
├── example_data.csv     # Anonymized example dataset (n=24)
├── README.md            # This file
└── LICENSE              # MIT License
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=2.0
numpy>=1.24
scipy>=1.11
statsmodels>=0.14
matplotlib>=3.7
```

> No external copula package is required. The Gaussian Copula is implemented
> directly using rank-based correlation (scipy + numpy only).

---

## Quick Start

```python
import pandas as pd
from pipeline import run_pipeline

df = pd.read_csv('example_data.csv')

# Outcome base names — expects {name}_pre and {name}_pos columns
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

results = run_pipeline(df, outcomes, margins, B=1000, seed=42)
print(results.to_string(index=False))
```

---

## Pipeline Description

### Step 1 — Bootstrap ANCOVA

Estimates the adjusted treatment effect (beta) controlling for baseline values.
Bootstrap confidence intervals (B = 1,000 resamples) avoid normality assumptions in small samples.

**Model:** `outcome_pos ~ Group + outcome_pre`

### Step 2 — TOST (Two One-Sided Tests)

Formally tests whether the difference between groups falls within a pre-specified equivalence margin [-delta, +delta].
Uses Welch's t-test (unequal variances), appropriate for the unbalanced small sample (G1 = 13, G0 = 11).
A 90% CI is used (equivalent to two one-sided tests at alpha = 0.05 each).

**Classification:**
- **Strict equivalence:** 90% CI entirely within [-delta, +delta] and includes zero
- **Acceptable superiority:** 90% CI within [-delta, +delta] but excludes zero
- **Inconclusive:** 90% CI exceeds margin (insufficient power, not inferiority)

### Step 3 — Gaussian Copula Sensitivity Analysis

Fits a rank-based Gaussian Copula to the **post-treatment scores** of each group,
preserving the multivariate correlation structure. Generates 1,000 synthetic replicas
and re-applies TOST to each. The robustness index reflects how often equivalence
holds under perturbations of the joint dependency structure at follow-up.

**Classification:**
- **High robustness ≥ 80%:** findings stable across simulated scenarios
- **Moderate robustness 50–79%:** findings preliminary, confirm with larger sample
- **Low robustness < 50%:** inconclusiveness likely due to structural variability

---

## Input Data Format

The pipeline expects a CSV or DataFrame with the following column structure:

| Column | Description |
|--------|-------------|
| `grupo` | Group assignment (1 = serious game, 0 = conventional physiotherapy) |
| `{name}_pre` | Baseline score for each outcome |
| `{name}_pos` | Post-treatment score for each outcome |

The `example_data.csv` file contains anonymized data from a shoulder rehabilitation RCT (n = 24).
Variables include pre/post scores for: TSK-17, PCS, PSEQ-10, ADAP, SPADI, WHOQOL-bref,
and goniometric ROM (6 movements).

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{rocha2025pipeline,
  author  = {Rocha, Danilo Santos and Rodrigues, Thiago Braga and Naves, Eduardo Lazaro Martins},
  title   = {Pipeline Estatistico para Ensaios Clinicos Randomizados com Amostras Pequenas:
             ANCOVA Bootstrapada, TOST e Copula Gaussiana},
  journal = {[Journal name]},
  year    = {2025},
  note    = {Under review}
}
```

---

## Authors

- **Danilo Santos Rocha** — Graduate Program in Biomedical Engineering, UFU, Brazil
- **Thiago Braga Rodrigues** — Technological University of the Shannon: Midlands Midwest (TUS), Ireland
- **Eduardo Lazaro Martins Naves** — Graduate Program in Biomedical Engineering, UFU, Brazil

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
