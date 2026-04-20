# Configuration guide (what to edit)

The main settings are at the top of `src/fils_pipeline.py` under:

```
# ===========================================================
# CONFIG
# ===========================================================
```

Edit these items to match your dataset and desired analysis:

1) **FILS column**
- `FILS_COL = "fils"`  (1..10)

2) **Candidate predictors**
- `RAW_NUMERIC_COLS = [...]`
- `RAW_CATEGORICAL_COLS = [...]`

3) **Thresholds**
- `THRESH_A = 3`
- `THRESH_B = 7`

4) **Interaction term (optional)**
- `USE_INTERACTION`, `INTERACTION_A`, `INTERACTION_B`

5) **Cross-validation and calibration**
- `OUTER_SPLITS`, `OUTER_REPEATS`, `INNER_SPLITS`
- `CALIBRATION_METHOD` ("sigmoid" recommended)
- `USE_CLASS_WEIGHT_BALANCED` (use with care for probability interpretation)

6) **Output**
- `FIG_DIR = "figures"`
- CSV files are written to the working directory by default.
