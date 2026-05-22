# START as a Screening Tool — Classification Experiments

This folder implements **Chapter 7.3** of the thesis: using the START tablet-based task battery to distinguish *enriched* (developmental concern) children from *community* (neurotypical) children.

## Background

START administers 8 automated games on a tablet to children aged 0–6, producing 16 continuous task-performance features (reaction time, accuracy, social preference ratios, etc.). This folder explores whether those features can screen for developmental atypicality — specifically, separating the clinically enriched group from the general community sample.

The classification task is **highly imbalanced**: approximately 6.5% of the full dataset (and ~3.2% of complete-data rows) are enriched children.

## Notebooks

### `classification.ipynb` — Main Classification Analysis

The primary notebook. Covers the full pipeline:

1. **EDA** — class distribution, missingness per feature, complete-data breakdown by class.
2. **Train/test split** — rows with any missing task feature are dropped before splitting (43% of participants have no task data at all; median-imputing zeros for them would add noise rather than signal).
3. **SMOTE** — synthetic minority oversampling is applied inside each pipeline's `fit()` step, isolated to training folds only.
4. **Models** — Logistic Regression, Random Forest, Gradient Boosting, all wrapped in `imblearn.Pipeline`.
5. **Threshold optimisation** — the default 0.5 threshold is unsuitable for 30:1 imbalance; the optimal threshold is found by maximising enriched-class F1 on the precision-recall curve.
6. **Summary table** — ROC-AUC, CV ROC-AUC, enriched precision/recall/F1 for all models at default and tuned thresholds.

**Key result:** Logistic Regression with a tuned threshold achieves CV ROC-AUC ≈ 0.77 and enriched-class precision of 0.60 at recall 0.23.

### `classification_extended.ipynb` — Extended Analysis

Extends the base analysis with additional feature engineering or model configurations. See notebook for details.

### `classification_extended_wt_youden.ipynb` — Youden Index Threshold

Uses the **Youden Index** (sensitivity + specificity − 1) rather than F1 as the threshold-selection criterion. This favours balanced sensitivity/specificity over precision-dominated F1, which may be preferred in clinical screening contexts where missing a true case (low sensitivity) is more costly than a false alarm.

## File Dependencies

```
classification.ipynb
    reads:   START.csv          (raw START task features, one row per child)
    reads:   bhisma_data.csv    (sample label: 'enriched' or 'community')
```

## Key Findings

| Model | CV ROC-AUC | Enriched Recall | Enriched Precision |
|---|---|---|---|
| Logistic Regression (default t=0.5) | 0.774 ± 0.049 | 0.62 | 0.08 |
| Random Forest | 0.754 ± 0.049 | 0.15 | 0.10 |
| Gradient Boosting | 0.696 ± 0.075 | 0.38 | 0.14 |
| **LR (tuned threshold)** | 0.774 | 0.23 | **0.60** |

The most important features for the enriched class (positive LR coefficients) tend to be timing and jerk metrics from motor tasks, consistent with the literature on motor delays as early developmental markers.

## Interpretation Note

The enriched group in STREAM is not a pure clinical NDD sample — it is oversampled for developmental concern based on initial screening, not a confirmed diagnosis. Classification performance should therefore be interpreted as sensitivity to *risk indicators* rather than clinical diagnosis accuracy. See Section 7.3 of the thesis for full discussion.
