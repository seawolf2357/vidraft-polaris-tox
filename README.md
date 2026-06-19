# VIDRAFT — Polaris Toxicity Benchmarks

Submissions by **VIDRAFT (Aether PharmaOS)** to [Polaris](https://polarishub.io) drug-discovery
toxicity benchmarks (blind, third-party-scored leaderboards).

## Headline result
🥇 **#1 on `tdcommons/ames` (Ames mutagenicity)** — ROC-AUC **0.877** on the official held-out
test set, **surpassing methods that use external Cell-Painting (phenomics) imaging data — using
molecular structure alone.**

| Benchmark | VIDRAFT (ROC-AUC) | Prior best | Note |
|---|---|---|---|
| tdcommons/ames | **0.877** | 0.868 | **#1** (structure-only) |

## Approach (high level)
A **heterogeneous ensemble of established model families** over standard molecular
representations (fingerprints + physicochemical descriptors), **structure-only** — no external
assay/imaging data. Trained solely on the official Polaris training split; the hidden test set is
scored centrally by Polaris (no leakage).

> Detailed feature engineering, hyperparameters, and ensembling configuration are part of
> VIDRAFT's internal **Aether PharmaOS** toolkit and are **not disclosed**.

## Integrity
Computational predictions intended for triage / prioritization, not clinical decisions.
Evaluation is blind: only the official train split is used for fitting; the test set is held out
and scored by the Polaris Hub.

— VIDRAFT · Aether PharmaOS · https://www.vidraft.net
