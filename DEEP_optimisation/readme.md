# DEEP Optimisation — Game Subset Selection

This folder implements **Chapter 7.1** of the thesis: reducing the DEEP administration time by identifying a minimal subset of games that retains the predictive power of the full 14-game battery for GMDS criterion validity.

## Background

DEEP administers 14 tablet games to children aged 3–6. Each game produces ordinal responses (highest level reached, accuracy, timing). Running all 14 games takes significant time in a clinical or community setting. The question is:

> **Can a small subset of games predict GMDS subscale scores as well as the full battery?**

A brute-force combinatorial search is tractable here because the number of games (14) limits the total search space to 2^14 − 1 = 16,383 non-empty subsets.

## Notebooks

### 1. `DEEP_scoring.ipynb` — Singleton Scoring

Trains an independent IRT–VAE per DEEP game (one model per game prefix). For each game, the model produces:
- A latent ability score (`total_DEEP`) for each child
- Subscale scores for each response type present in that game (highest level, accuracy, timing)

**Outputs:** One CSV per game under `singles/`, named `DEEP_Deep_abilities_{game_prefix}.csv`.

**Key configuration:**
```python
AGE_MIN_YEARS = 2.5       # include children aged ≥ 2.5 years
TEST_SIZE     = 0.2        # 20% held-out test split
```

**Game prefixes** (14 games):
`ST_`, `AT_`, `PB_`, `GYG_`, `HO_`, `OOO_`, `MS_`, `JIG_`, `LR_`, `PM_`, `SR_`, `SO_`, `SC_`, `SD_`

### 2. `explore.ipynb` — Combinatorial Subset Search

Exhaustively evaluates all 2^14 − 1 non-empty subsets of the 14 DEEP games. For each subset:
1. Loads the pre-computed singleton latent scores from `singles/`.
2. Merges them into a feature matrix.
3. Trains an MLP regressor and evaluates via 5-fold cross-validation.
4. Records mean correlation with GMDS subscales.

The search is **parallelised** using `joblib` (Loky backend). Results are saved incrementally to avoid losing progress on long runs.

**Key result (from thesis):** A **5-game subset** (`GYG`, `LR`, `OOO`, `SC`, `SR`) matches the full 14-game battery on GMDS criterion validity.

**Outputs:**
- `mlp_subset_search_summary.csv` — one row per game combination, ranked by mean GMDS correlation
- `mlp_subset_search_detailed.csv` — per-target breakdown for every combination

**Key configuration:**
```python
MAX_GAMES_PER_SUBSET = None   # None = full exhaustive search; set to e.g. 4 for a quick dry-run
PARALLEL_JOBS        = 7      # number of parallel workers (auto-detected)
SUBSET_BATCH_SIZE    = 32     # subsets per batch dispatched to each worker
```

## File Dependencies

```
DEEP_scoring.ipynb
    reads:   DEEP_processed_data.csv   (raw DEEP responses, one row per child)
    reads:   Gareth_scores_DEEP.csv    (baseline scores for correlation reference)
    writes:  singles/DEEP_Deep_abilities_{game_prefix}.csv  (one per game)

explore.ipynb
    reads:   singles/DEEP_Deep_abilities_*.csv
    reads:   gmds_scores.csv
    writes:  mlp_subset_search_summary.csv
    writes:  mlp_subset_search_detailed.csv
```

## How to Run

```bash
# Step 1: generate singleton scores (runs IRT-VAE for each of the 14 games)
jupyter nbconvert --to notebook --execute DEEP_scoring.ipynb

# Step 2: run the exhaustive combinatorial search (may take 30–60 min)
jupyter nbconvert --to notebook --execute explore.ipynb
```

Alternatively, open both notebooks in JupyterLab and run cells interactively. To do a quick sanity-check run, set `MAX_GAMES_PER_SUBSET = 3` in `explore.ipynb` before running.
