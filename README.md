# VIDRAFT Polaris Toxicity Benchmarks

Structure-only heterogeneous ensemble for [Polaris](https://polarishub.io) toxicity benchmarks,
by **VIDRAFT (Aether PharmaOS)**.

## Method
For each benchmark, four model families are trained on the official Polaris train split and
their positive-class probabilities are averaged:
- **XGBoost**, **RandomForest**, **ExtraTrees**, **HistGradientBoosting**

**Features (structure-only, no external/phenomics data):**
Morgan ECFP4 (2048 bits) + MACCS keys (167) + RDKit 2D descriptors (~208), with train-median
imputation of non-finite descriptors. Class imbalance handled via `scale_pos_weight` / balanced weights.

## Results (held-out test, ROC-AUC)
| Benchmark | VIDRAFT | Prior best | Note |
|---|---|---|---|
| **tdcommons/ames** | **0.877** | 0.868 (MPNN+Phenomics) | **#1** on the Polaris leaderboard |
| tdcommons/dili | 0.9335 | 0.933 | tie with top (test n=96, within noise) |
| tdcommons/herg | 0.844 | 0.892 | mid-pack |

Notably, the Ames #1 result **beats methods that use external Cell-Painting (phenomics) data**, using
molecular structure alone.

## Reproduce
```bash
pip install polaris-lib datamol xgboost scikit-learn
python tox_ensemble.py     # trains + evaluates ames/herg/dili
```

## Honesty
Computational predictions only; intended for triage/prioritization, not clinical decisions.
Each model is trained solely on the official train split; the hidden test is scored by Polaris.

— VIDRAFT · Aether PharmaOS · https://www.vidraft.net
