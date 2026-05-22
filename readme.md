# Data-Driven Approach for Assessing Child Development

**Dual Degree Thesis (B.Tech + M.Tech) — IIT Bombay**
**Author:** Abhishek Choudhary (Roll No. 210010003)
**Supervisor:** Prof. Sharat Chandran
**Programme:** Centre for Machine Intelligence and Data Science (CMInDS)

---

## Overview

This repository contains the complete codebase accompanying the thesis *"Data Driven Approach for Assessing Child Development"* (`DDP_Report_210010003.pdf`). The work develops a data-driven psychometric framework for measuring early child development (ages 0–6) directly from item-level responses, using approximately 4,000 subjects from the STREAM initiative (India and Malawi).

The core contribution is an **IRT–VAE** model: a Variational Autoencoder whose decoder is constrained to follow psychometric Item Response Theory (IRT) response functions (2-Parameter Logistic for binary items; Graded Response Model for ordinal items). This yields interpretable latent ability scores that are validated along three axes:

1. **Age monotonicity** — latent scores track chronological age.
2. **Convergent validity** — scores correlate with adverse-childhood and protective covariates in theoretically expected directions.
3. **Criterion validity** — a thin neural network maps latent scores to the Griffiths Mental Development Scales (GMDS), an established clinical instrument.

A deployment strategy is proposed: **MDAT for ages 0–3** and **DEEP for ages 3–6**, motivated by instrument design and alignment with the WHO Global Scales for Early Development (GSED).

---

## Repository Structure

```
.
├── DDP_Report_210010003.pdf          # Full thesis report
├── readme.md                         # This file
│
├── VAE/                              # Stage 1: IRT-VAE scoring
│   ├── mdat_scoring.ipynb            # Score MDAT domains (binary 2PL decoder)
│   ├── DEEP_scoring.ipynb            # Score DEEP games (ordinal GRM decoder)
│   └── start_scoring.ipynb           # Score START tasks (linear Gaussian decoder)
│
├── convergent_validity/              # Stage 2: Convergent validity
│   ├── convergent_analysis_of_MDAT_latent.ipynb
│   ├── convergent_analysis_of_DEEP_latent.ipynb
│   ├── convergent_analysis_of_START_latent.ipynb
│   ├── convergent_analysis_of_GMDS_predicted_using_MDAT.ipynb
│   └── convergent_analysis_of_GMDS_predicted_using_DEEP.ipynb
│
├── criterion_validity/               # Stage 3: Criterion validity vs GMDS
│   ├── base_model_single_NN_for_criterion.ipynb
│   └── section_6.1_age_split_criterion.ipynb
│
├── DEEP_optimisation/                # Chapter 7: DEEP game-subset selection
│   ├── DEEP_scoring.ipynb            # Train singleton IRT-VAE per DEEP game
│   ├── explore.ipynb                 # Exhaustive combinatorial subset search
│   └── readme.md
│
└── classification_using_START/       # Chapter 7: START as a screening tool
    ├── classification.ipynb          # Enriched-vs-community classification
    ├── classification_extended.ipynb
    ├── classification_extended_wt_youden.ipynb
    └── readme.md
```

---

## Datasets

The codebase uses four instruments collected under the **STREAM Initiative** (India and Malawi, ~4,000 children aged 0–6):

| Instrument | Description | Age Range | Item Type |
|---|---|---|---|
| **MDAT** | Malawi Developmental Assessment Tool — clinician-administered, 4 domains (social, fine motor, gross motor, language) | 0–6 yrs | Binary (pass/fail) |
| **DEEP** | Developmental Assessment on an E-Platform — 14 tablet games measuring cognition, motor, and language | 3–6 yrs | Ordinal (levels) |
| **START** | Screening Tool for Autism Risk using Technology — 8 automated tasks on a tablet | 0–6 yrs | Continuous |
| **GMDS** | Griffiths Mental Development Scales — criterion variable, clinician-administered subscale scores | 0–6 yrs | Continuous |

Convergent validity uses an additional covariate file (`bhisma_data.csv`) containing psychosocial, anthropometric, and family-level variables.

> **Note:** Raw data files are not included in this repository due to data governance restrictions. Contact the STREAM team or the thesis supervisor for access. Each notebook lists the expected input file(s) at the top under `DATA_PATH`.

---

## Methodology

### IRT–VAE Framework

The model is an unsupervised probabilistic generative model combining deep learning with classical psychometrics:

- **Encoder:** A multi-layer perceptron (MLP) that maps item responses and an observation mask to a low-dimensional latent ability space `θ` (mean `μ` and log-variance `logvar`). Age is passed as an auxiliary input to help orient the latent space.
- **Decoder:** An IRT-constrained response function — **not** a neural network. For binary items it uses the 2-Parameter Logistic (2PL) model; for ordinal items it uses the Graded Response Model (GRM).
- **Training:** The Evidence Lower BOund (ELBO) is maximised: reconstruction log-likelihood (masked for missing responses) minus a KL regularisation term.
- **Missing data:** A binary observation mask `M` is passed to the encoder; masked entries are excluded from the reconstruction loss, implementing masked reconstruction as in [Appendix A.9 of the report].

### Validation Pipeline

```
Raw responses
    │
    ▼
IRT-VAE (VAE/)
    │   → latent ability scores per child per instrument
    │
    ├── Convergent validity (convergent_validity/)
    │       Pearson r with external covariates, BH-FDR corrected
    │
    └── Criterion validity (criterion_validity/)
            MLP: latent scores → GMDS subscales, 5-fold CV
```

---

## Running the Notebooks

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn scipy imbalanced-learn
```

### Execution Order

Run notebooks in this order to reproduce the full pipeline:

1. **VAE scoring** (generates latent ability CSVs):
   - `VAE/mdat_scoring.ipynb` → produces `mdat_Deep_abilities_0.csv`
   - `VAE/DEEP_scoring.ipynb` → produces `DEEP_Deep_abilities_3.csv`
   - `VAE/start_scoring.ipynb` → produces `start_Deep_ability.csv`

2. **Criterion validity** (requires GMDS data + latent scores from step 1):
   - `criterion_validity/base_model_single_NN_for_criterion.ipynb`
   - `criterion_validity/section_6.1_age_split_criterion.ipynb`

3. **Convergent validity** (requires GAMLSS-normalised scores + covariate data):
   - `convergent_validity/convergent_analysis_of_MDAT_latent.ipynb`
   - `convergent_validity/convergent_analysis_of_DEEP_latent.ipynb`
   - `convergent_validity/convergent_analysis_of_START_latent.ipynb`
   - `convergent_validity/convergent_analysis_of_GMDS_predicted_using_MDAT.ipynb`
   - `convergent_validity/convergent_analysis_of_GMDS_predicted_using_DEEP.ipynb`

4. **DEEP optimisation** (game subset selection, Chapter 7):
   - `DEEP_optimisation/DEEP_scoring.ipynb` (run once per game)
   - `DEEP_optimisation/explore.ipynb` (exhaustive search over subsets)

5. **START classification** (screening, Chapter 7):
   - `classification_using_START/classification.ipynb`

### Key Configuration Constants

Each notebook exposes its configuration at the top of the first code cell:

| Constant | Meaning |
|---|---|
| `DATA_PATH` | Path to input CSV |
| `AGE_MIN_DAYS` / `AGE_MIN_YEARS` | Age filter applied before training |
| `LATENT_DIM` | Dimensionality of latent ability space (always 1 in this work) |
| `EPOCHS` | Training epochs |
| `BATCH_SIZE` | Mini-batch size |
| `LEARNING_RATE` | Adam optimiser learning rate |
| `SHOW_PLOTS` | Set `False` to suppress all matplotlib output |

---

## Key Results (Summary)

| Experiment | Finding |
|---|---|
| Age monotonicity | MDAT: r = 0.75, DEEP: r = 0.68 with chronological age |
| Convergent validity | FCI, SES, maternal education positively correlated; PHQ-9, RBS risk negatively correlated |
| Criterion validity (5-fold CV) | MDAT→GMDS: r ≈ 0.85–0.90; DEEP→GMDS: r ≈ 0.85–0.90 across subscales |
| DEEP game reduction | 5 games match the full 14-game battery on GMDS prediction |
| START screening | LR with tuned threshold: CV ROC-AUC ≈ 0.77 for enriched-vs-community |

Full tables, figures, and discussion are in `DDP_Report_210010003.pdf`.

---

## Citation

If you use this code or the methodology, please cite the thesis:

> Choudhary, A. (2026). *Data Driven Approach for Assessing Child Development*. Dual Degree Thesis, Indian Institute of Technology Bombay. Supervisor: Prof. Sharat Chandran.
