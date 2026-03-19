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
copulas>=0.9
matplotlib>=3.7
```

---

## Quick Start

```python
import pandas as pd
from pipeline import run_pipeline

df = pd.read_csv('example_data.csv')

margins = {
    'TSK_delta': 5,
    'PCS_delta': 5,
    'PSEQ_delta': 5,
    'ADAP_delta': 10,
    'SPADI_tot_delta': 10,
    'WHOQOL_delta': 8,
    'FLEX_D_delta': 15,
    'FLEX_E_delta': 15,
    'ABD_D_delta': 15,
    'ABD_E_delta': 15,
    'ROT_INT_delta': 15,
    'ROT_EXT_delta': 15,
}

results = run_pipeline(df, margins, B=1000, seed=42)
print(results)
```

---

## Pipeline Description

### Step 1 — Bootstrap ANCOVA

Estimates the adjusted treatment effect (beta) controlling for baseline values.
Bootstrap confidence intervals (B = 1,000 resamples) avoid normality assumptions in small samples.

**Model:** `outcome_post ~ Group + outcome_pre`

### Step 2 — TOST (Two One-Sided Tests)

Formally tests whether the difference between groups falls within a pre-specified equivalence margin [-delta, +delta].
A 90% CI is used (equivalent to two one-sided tests at alpha = 0.05 each).

**Classification:**
- Strict equivalence: 90% CI entirely within [-delta, +delta] and includes zero
- Acceptable superiority: 90% CI within [-delta, +delta] but excludes zero
- Inconclusive: 90% CI exceeds margin (insufficient power, not inferiority)

### Step 3 — Gaussian Copula Sensitivity Analysis

Fits a Gaussian Copula to each group, generates 1,000 synthetic replicas preserving the multivariate correlation structure, and re-applies TOST to each replica.
Robustness index = proportion of replicas with p_TOST < 0.05.

**Classification:**
- High robustness >= 80%: findings stable across simulated scenarios
- Moderate robustness 50-79%: findings preliminary, confirm with larger sample
- Low robustness < 50%: inconclusiveness likely due to structural variability

---

## Demonstration Dataset

The `example_data.csv` file contains anonymized data from a shoulder rehabilitation RCT (n = 24).
Variables include pre/post scores and deltas for: TSK-17, PCS, PSEQ-10, ADAP, SPADI, WHOQOL-bref, and goniometric ROM (6 movements).

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

- **Danilo Santos Rocha** — Postgraduate Program in Biomedical Engineering (PPGEB), UFU, Brazil
- **Thiago Braga Rodrigues** — Technological University of the Shannon: Midlands Midwest (TUS), Ireland
- **Eduardo Lazaro Martins Naves** — Graduate Program in Biomedical Engineering, UFU, Brazil

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
