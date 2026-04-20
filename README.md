# FILS threshold estimation pipeline (acute stroke)

This repository contains the analysis code used in our manuscript to estimate contemporaneous swallowing status in acute stroke using routinely available clinical variables.

Primary outcomes:
- **FILS ≥ 3**
- **FILS ≥ 7**

The pipeline implements:
- Nested cross-validation with probability calibration within the outer fold
- Out-of-fold (OOF) ROC, calibration plots, and decision curve analysis
- Bootstrap confidence intervals for AUC and Brier score
- Sensitivity analyses (complete-case; no class weighting)
- Stability selection (L1 logistic regression on repeated subsamples)
- Supplementary ordinal surrogate analysis (regressor → rounding/clipping)

## Quick start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Prepare your dataset (not included)
Place your dataset locally (CSV/XLS/XLSX). **Do not commit patient data to GitHub.**  
See `data/README.md` for expected columns.

### 3) Run the pipeline
From the repository root:
```bash
python scripts/run_pipeline.py --data path/to/your_data.csv
```

If you omit `--data`, the script searches the common paths listed in `src/fils_pipeline.py` (`DATA_PATHS`).

## Outputs

- Figures are saved to `figures/` by default.
- CSV outputs are written to the working directory:
  - `missingness_overall.csv`, `missingness_by_ge3.csv`, `missingness_by_ge7.csv`
  - `stability_ge3.csv`, `stability_ge7.csv`

## What you need to edit (before running on your data)

Open `src/fils_pipeline.py` and edit the **CONFIG** block at the top:

1) `DATA_PATHS`  
   - Add the location(s) where your local data file may exist, or just use `--data`.

2) `FILS_COL`  
   - Column name for FILS (default: `fils`).

3) `RAW_NUMERIC_COLS` / `RAW_CATEGORICAL_COLS`  
   - Column names in your dataset. Remove columns you do not have; add columns you do.

4) Thresholds  
   - `THRESH_A` and `THRESH_B` (default: 3 and 7).

5) Interaction term (optional)  
   - `USE_INTERACTION`, `INTERACTION_A`, `INTERACTION_B`  
   - If your time variable uses a different column name, update `INTERACTION_B`.

6) Cross-validation and calibration  
   - `OUTER_SPLITS`, `OUTER_REPEATS`, `INNER_SPLITS`  
   - `CALIBRATION_METHOD` (`"sigmoid"` is recommended for modest samples).

7) Class weighting  
   - `USE_CLASS_WEIGHT_BALANCED`  
   - If you focus on probability interpretation, consider reporting sensitivity analyses without class weighting.

## Notes
This code is provided for research transparency and reproducibility. External validation is required before clinical deployment.

## License
MIT (see `LICENSE`).

## Citation
See `CITATION.cff`.
