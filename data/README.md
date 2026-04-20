# Data

This repository does **not** include patient-level data due to ethical and privacy restrictions.

## Expected input
Provide a CSV/XLS/XLSX file with at least the following columns (names can be changed in `src/fils_pipeline.py`):

- FILS score: `Gr` (integer 1–10)
- Numeric predictors (examples): `Age`, `mRS`, `Alb`, `BMI`, `JCS`, `FIM_motor`, `FIM_cognition`, `Tim_from_onset`
- Categorical predictors (examples): `Sex`, `Stroke_type`

## How to run
From the repository root:

```bash
python scripts/run_pipeline.py --data path/to/your_data.csv
```
