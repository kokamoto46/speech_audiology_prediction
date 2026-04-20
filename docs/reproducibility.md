# Reproducibility notes

- The pipeline uses nested cross-validation with fixed random seeds (see `RANDOM_SEED` and seeds used for bootstrap).
- Figures are saved with the non-GUI backend (`matplotlib.use("Agg")`) for Windows compatibility.
- Outputs:
  - `figures/`: ROC, calibration plots, DCA plots, histograms, confusion heatmap
  - CSV tables in the working directory:
    - `missingness_overall.csv`, `missingness_by_ge3.csv`, `missingness_by_ge7.csv`
    - `stability_ge3.csv`, `stability_ge7.csv`
