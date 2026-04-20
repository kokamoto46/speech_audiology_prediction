# fils_pipeline.py
# ------------------------------------------------------------
# Reviewer-ready FILS pipeline (Windows-friendly, no-GUI plotting)
#
# What you get:
#  - Method A: matplotlib Agg (no GUI), save all figures to disk
#  - Binary endpoints (FILS>=3, FILS>=7):
#       * Nested CV + probability calibration (Platt/isotonic) inside outer fold
#       * OOF ROC + calibration + decision curve plots
#       * Bootstrap CI for AUC/Brier
#       * Sensitivity: complete-case, no class_weight
#  - Ordinal (FILS 1..10) surrogate (sklearn regressor -> round/clip): MAE, QWK + confusion heatmap
#  - Missingness tables: overall + by boundary (CSV)
#  - ★ Stability selection (requested):
#       * L1 logistic (LASSO) on repeated subsamples
#       * Outputs: stability_ge3.csv, stability_ge7.csv (+ top20 printed)
#
# Notes:
#  - This avoids statsmodels OrderedModel which often fails due to constant columns in folds.
#  - Custom transformer is clone-safe for GridSearchCV.
# ------------------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import List, Optional, Dict, Any, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingRegressor


# ===========================================================
# CONFIG
# ===========================================================

DATA_PATHS = [
    "./enge_data.csv",
]

FIG_DIR = "figures"
SAVE_FIGS = True
DPI = 300

FILS_COL = "fils"     # 1..10 ordinal
THRESH_A = 3
THRESH_B = 7

ID_COL = "Num"
RAW_NUMERIC_COLS = [
    "Age",
    "mRS",
    "Alb",
    "BMI",
    "JCS",
    "JCS_categ",
    "FIM_motor",
    "FIM_cognition",
    "Tim_from_onset",
    # Example: if your dataset uses a different name for time, add it here and update INTERACTION_B
]
RAW_CATEGORICAL_COLS = [
    "Sex",
    "Stroke_type",
]

# Interaction (prior hypothesis)
USE_INTERACTION = True
INTERACTION_A = "Alb"
INTERACTION_B = "Tim_from_onset"
INTERACTION_NAME = "Alb_x_Tim_from_onset"
USE_LOG1P_TIME = False

# Nested CV
OUTER_SPLITS = 5
OUTER_REPEATS = 10
INNER_SPLITS = 5
RANDOM_SEED = 42

# Bootstrap CI
BOOTSTRAP_ROUNDS = 2000

# Calibration inside outer fold
CALIBRATION_METHOD = "sigmoid"  # "sigmoid", "isotonic", or None
MIN_POS_FOR_ISOTONIC = 30

# class_weight
USE_CLASS_WEIGHT_BALANCED = True

# Stability selection
STAB_ROUNDS = 500
STAB_SUBSAMPLE_FRAC = 0.75
STAB_C = 0.2
STAB_USE_CLASS_WEIGHT_BALANCED = False  # usually OFF if you care about probability scale


# ===========================================================
# Utilities
# ===========================================================

def ensure_fig_dir():
    if SAVE_FIGS:
        os.makedirs(FIG_DIR, exist_ok=True)

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)[:200]

def savefig(path: str):
    if SAVE_FIGS:
        plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()

def bootstrap_ci(values, alpha=0.05):
    v = np.asarray(values, dtype=float)
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))

def bootstrap_performance_ci(y_true, y_prob, n_rounds=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    aucs, briers = [], []
    for _ in range(n_rounds):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        briers.append(brier_score_loss(yt, yp))

    auc_lo, auc_hi = bootstrap_ci(aucs)
    br_lo, br_hi = bootstrap_ci(briers)
    return {
        "AUC_mean": float(np.mean(aucs)),
        "AUC_CI": (auc_lo, auc_hi),
        "Brier_mean": float(np.mean(briers)),
        "Brier_CI": (br_lo, br_hi),
    }

def quadratic_weighted_kappa(y_true, y_pred, min_rating=1, max_rating=10):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_ratings = int(max_rating - min_rating + 1)

    O = np.zeros((n_ratings, n_ratings), dtype=float)
    for a, b in zip(y_true, y_pred):
        if min_rating <= a <= max_rating and min_rating <= b <= max_rating:
            O[a - min_rating, b - min_rating] += 1

    if O.sum() == 0:
        return np.nan

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / O.sum()

    W = np.zeros((n_ratings, n_ratings), dtype=float)
    for i in range(n_ratings):
        for j in range(n_ratings):
            W[i, j] = ((i - j) ** 2) / ((n_ratings - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else np.nan

def net_benefit(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    w = threshold / (1 - threshold)
    return (tp / n) - (fp / n) * w


# ===========================================================
# Plot helpers (SAVE ONLY)
# ===========================================================

def plot_bar_distribution(series, xlabel, title, fname):
    ensure_fig_dir()
    counts = series.value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(int), counts.values)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    savefig(os.path.join(FIG_DIR, fname))

def plot_histogram(series, title, fname):
    ensure_fig_dir()
    plt.figure()
    pd.to_numeric(series, errors="coerce").dropna().hist()
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Count")
    savefig(os.path.join(FIG_DIR, fname))

def plot_roc(y_true, y_prob, title, fname):
    ensure_fig_dir()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={auc:.3f})")
    savefig(os.path.join(FIG_DIR, fname))

def plot_calibration_curve(y_true, y_prob, title, fname, n_bins=10):
    ensure_fig_dir()
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = pd.qcut(y_prob, q=n_bins, duplicates="drop")
    dfc = pd.DataFrame({"y": y_true, "p": y_prob, "bin": bins})
    grp = dfc.groupby("bin", observed=True).agg(
        obs=("y", "mean"),
        pred=("p", "mean"),
        n=("y", "size")
    ).reset_index()

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(grp["pred"], grp["obs"], marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(title)
    savefig(os.path.join(FIG_DIR, fname))

def plot_decision_curve(y_true, y_prob, title, fname, thresholds=None):
    ensure_fig_dir()
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    nb_model = [net_benefit(y_true, y_prob, t) for t in thresholds]

    prevalence = float(np.mean(y_true))
    nb_all = []
    for t in thresholds:
        w = t / (1 - t)
        nb_all.append(prevalence - (1 - prevalence) * w)
    nb_none = [0.0 for _ in thresholds]

    plt.figure()
    plt.plot(thresholds, nb_model, label="Model")
    plt.plot(thresholds, nb_all, linestyle="--", label="Treat all")
    plt.plot(thresholds, nb_none, linestyle="--", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.legend()
    savefig(os.path.join(FIG_DIR, fname))

def plot_ordinal_confusion_1to10(y_true, y_pred, title, fname):
    ensure_fig_dir()
    classes = np.arange(1, 11)
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            cm[idx[a], idx[b]] += 1

    plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks(np.arange(K), classes, rotation=90)
    plt.yticks(np.arange(K), classes)
    plt.xlabel("Predicted FILS")
    plt.ylabel("Observed FILS")
    plt.title(title)
    savefig(os.path.join(FIG_DIR, fname))


# ===========================================================
# Missingness helpers
# ===========================================================

def missingness_table(df: pd.DataFrame, cols: List[str], group_col: Optional[str] = None) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if group_col is None:
        out = pd.DataFrame({
            "n": len(df),
            "n_missing": df[cols].isna().sum(),
            "missing_rate": df[cols].isna().mean(),
        }).reset_index().rename(columns={"index": "variable"})
        return out.sort_values("missing_rate", ascending=False)

    parts = []
    for g, sub in df.groupby(group_col):
        tmp = pd.DataFrame({
            "group": g,
            "n": len(sub),
            "variable": cols,
            "n_missing": sub[cols].isna().sum().values,
            "missing_rate": sub[cols].isna().mean().values
        })
        parts.append(tmp)
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out.sort_values(["variable", "group"])

def make_complete_case(df: pd.DataFrame, x_cols: List[str], y_col: str) -> pd.DataFrame:
    use = [c for c in x_cols if c in df.columns] + [y_col]
    return df.dropna(subset=use).reset_index(drop=True)


# ===========================================================
# Clone-safe interaction transformer
# ===========================================================

class NumericInteractionEngineer(BaseEstimator, TransformerMixin):
    """
    Clone-safe:
      - do not modify parameters inside __init__
    """
    def __init__(
        self,
        numeric_cols,
        use_interaction: bool,
        interaction_a: str,
        interaction_b: str,
        interaction_name: str,
        use_log1p_time: bool = False,
    ):
        self.numeric_cols = tuple(numeric_cols) if numeric_cols is not None else tuple()
        self.use_interaction = use_interaction
        self.interaction_a = interaction_a
        self.interaction_b = interaction_b
        self.interaction_name = interaction_name
        self.use_log1p_time = use_log1p_time

    def fit(self, X, y=None):
        Xdf = self._to_dataframe(X)
        self.base_numeric_cols_ = list(self.numeric_cols)
        self.medians_ = {c: pd.to_numeric(Xdf[c], errors="coerce").median() for c in self.base_numeric_cols_}

        if self.use_interaction:
            Xa = self._prepare(Xdf)
            self.mean_a_ = float(Xa[self.interaction_a].mean())
            self.mean_b_ = float(Xa[self.interaction_b].mean())

            names = []
            for c in self.base_numeric_cols_:
                if c == self.interaction_a or c == self.interaction_b:
                    names.append(f"{c}_c")
                else:
                    names.append(c)
            names.append(self.interaction_name)
            self.feature_names_ = names
        else:
            self.feature_names_ = list(self.base_numeric_cols_)

        return self

    def transform(self, X):
        Xdf = self._to_dataframe(X)

        if not self.use_interaction:
            out = pd.DataFrame(index=Xdf.index)
            for c in self.base_numeric_cols_:
                out[c] = pd.to_numeric(Xdf[c], errors="coerce").fillna(self.medians_[c]).astype(float)
            return out[self.feature_names_].to_numpy(dtype=float)

        Xa = self._prepare(Xdf)
        Xa[self.interaction_a] = Xa[self.interaction_a] - self.mean_a_
        Xa[self.interaction_b] = Xa[self.interaction_b] - self.mean_b_
        Xa[self.interaction_name] = Xa[self.interaction_a] * Xa[self.interaction_b]

        out = Xa.rename(columns={
            self.interaction_a: f"{self.interaction_a}_c",
            self.interaction_b: f"{self.interaction_b}_c",
        })
        return out[self.feature_names_].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=list(self.numeric_cols))

    def _prepare(self, Xdf):
        out = pd.DataFrame(index=Xdf.index)
        for c in self.base_numeric_cols_:
            s = pd.to_numeric(Xdf[c], errors="coerce").fillna(self.medians_[c]).astype(float)
            if self.use_log1p_time and c == self.interaction_b:
                s = np.log1p(np.clip(s, a_min=0, a_max=None))
            out[c] = s
        return out


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str], use_interaction: bool) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("feat", NumericInteractionEngineer(
            numeric_cols=numeric_cols,
            use_interaction=use_interaction,
            interaction_a=INTERACTION_A,
            interaction_b=INTERACTION_B,
            interaction_name=INTERACTION_NAME,
            use_log1p_time=USE_LOG1P_TIME,
        )),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


# ===========================================================
# Data loading
# ===========================================================

def resolve_data_path() -> str:
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Input file not found. Put your data in the working directory, update DATA_PATHS, "
        "or run: python fils_pipeline_final_reviewer_ready.py path/to/data.csv"
    )

def load_data() -> pd.DataFrame:
    data_path = resolve_data_path()
    ext = os.path.splitext(data_path)[1].lower()
    raw = pd.read_csv(data_path) if ext == ".csv" else pd.read_excel(data_path)
    df = raw.copy()

    if ID_COL in df.columns:
        df = df[df[ID_COL].notna()].copy()
        tmp = pd.to_numeric(df[ID_COL], errors="coerce")
        df = df[tmp.notna()].copy()
        df[ID_COL] = tmp[tmp.notna()].astype(int).values

    for col in ["Unnamed: 19", "Unnamed: 20", "Unnamed: 21"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    required = [FILS_COL] + RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS
    keep = [c for c in required if c in df.columns]
    dat = df[keep].copy()

    dat[FILS_COL] = pd.to_numeric(dat[FILS_COL], errors="coerce")
    dat = dat[dat[FILS_COL].notna()].copy()
    dat[FILS_COL] = dat[FILS_COL].astype(int)
    dat = dat[(dat[FILS_COL] >= 1) & (dat[FILS_COL] <= 10)].copy()

    dat["FILS_ge_3"] = (dat[FILS_COL] >= THRESH_A).astype(int)
    dat["FILS_ge_7"] = (dat[FILS_COL] >= THRESH_B).astype(int)

    return dat.reset_index(drop=True)


# ===========================================================
# Binary nested CV with calibration
# ===========================================================

def nested_cv_binary_calibrated(
    dat: pd.DataFrame,
    y_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    use_interaction: bool,
    calibrate: bool = True,
    calibration_method: Optional[str] = "sigmoid",
    use_class_weight_balanced: bool = True,
) -> Dict[str, Any]:

    X_all = dat[numeric_cols + categorical_cols].copy()
    y_all = dat[y_col].astype(int).values

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_SEED,
    )

    oof_prob = np.zeros(len(dat), dtype=float)
    fold_auc, fold_brier = [], []
    chosen_params = []

    cw = "balanced" if use_class_weight_balanced else None
    base_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=30000,
        class_weight=cw,
    )

    param_grid = {
        "model__C": np.logspace(-2, 1, 7),
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    for train_idx, test_idx in outer.split(X_all, y_all):
        train_df = dat.iloc[train_idx].copy()
        test_df = dat.iloc[test_idx].copy()

        pre = make_preprocessor(numeric_cols, categorical_cols, use_interaction)
        pipe = Pipeline(steps=[("pre", pre), ("model", base_model)])

        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )
        gs.fit(train_df[numeric_cols + categorical_cols], train_df[y_col].astype(int).values)
        best = gs.best_estimator_
        chosen_params.append(gs.best_params_)

        if calibrate and calibration_method is not None:
            method = calibration_method
            if method == "isotonic":
                pos = int(train_df[y_col].sum())
                if pos < MIN_POS_FOR_ISOTONIC:
                    method = "sigmoid"

            cal = CalibratedClassifierCV(estimator=best, method=method, cv=inner)
            cal.fit(train_df[numeric_cols + categorical_cols], train_df[y_col].astype(int).values)
            prob = cal.predict_proba(test_df[numeric_cols + categorical_cols])[:, 1]
        else:
            prob = best.predict_proba(test_df[numeric_cols + categorical_cols])[:, 1]

        oof_prob[test_idx] = prob
        y_test = test_df[y_col].astype(int).values
        fold_auc.append(roc_auc_score(y_test, prob))
        fold_brier.append(brier_score_loss(y_test, prob))

    return {
        "y_true": y_all,
        "oof_prob": oof_prob,
        "fold_auc": np.array(fold_auc, dtype=float),
        "fold_brier": np.array(fold_brier, dtype=float),
        "chosen_params": chosen_params,
    }


# ===========================================================
# Ordinal surrogate (sklearn): robust, no statsmodels
# ===========================================================

def ordinal_surrogate_cv(
    dat: pd.DataFrame,
    fils_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    use_interaction: bool,
    n_splits=5,
    n_repeats=10,
    seed=321
) -> Dict[str, Any]:
    """
    Predict FILS as a continuous score using HistGradientBoostingRegressor,
    then round/clip to 1..10 to evaluate ordinal metrics (MAE, QWK).
    """
    y = dat[fils_col].astype(int).to_numpy()
    y_bin = np.asarray(pd.cut(y, bins=[0, 2, 6, 10], labels=[0, 1, 2]).astype(int))

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    oof_pred_cont = np.zeros(len(dat), dtype=float)
    oof_pred_ord = np.zeros(len(dat), dtype=int)

    X_cols = numeric_cols + categorical_cols

    for train_idx, test_idx in rskf.split(dat, y_bin):
        train_df = dat.iloc[train_idx].copy()
        test_df = dat.iloc[test_idx].copy()

        pre = make_preprocessor(numeric_cols, categorical_cols, use_interaction)

        reg = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            random_state=seed,
        )

        pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])
        pipe.fit(train_df[X_cols], train_df[fils_col].astype(int).to_numpy())

        pred = pipe.predict(test_df[X_cols])
        oof_pred_cont[test_idx] = pred

        pred_round = np.rint(pred).astype(int)
        pred_round = np.clip(pred_round, 1, 10)
        oof_pred_ord[test_idx] = pred_round

    mae = float(np.mean(np.abs(y - oof_pred_ord)))
    qwk = float(quadratic_weighted_kappa(y, oof_pred_ord, min_rating=1, max_rating=10))

    return {
        "y_true": y,
        "oof_pred_cont": oof_pred_cont,
        "oof_pred_ord": oof_pred_ord,
        "MAE": mae,
        "QWK": qwk,
    }


# ===========================================================
# ★ Stability selection (requested)
# ===========================================================

def _get_feature_names_from_fitted_preprocessor(pre: ColumnTransformer) -> List[str]:
    """
    Extract feature names after ColumnTransformer:
      - numeric: num/feat feature_names_out
      - categorical: ohe get_feature_names_out
    """
    # numeric
    num_names = list(
        pre.named_transformers_["num"]
           .named_steps["feat"]
           .get_feature_names_out()
    )

    # categorical
    if "cat" in pre.named_transformers_:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        # NOTE: get_feature_names_out requires original categorical col names passed to OneHotEncoder
        # ColumnTransformer stores them, but easiest is:
        cat_feature_names = list(ohe.get_feature_names_out())
    else:
        cat_feature_names = []

    return num_names + cat_feature_names


def stability_selection_binary(
    dat: pd.DataFrame,
    y_col: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    use_interaction: bool,
    n_rounds: int = 500,
    subsample_frac: float = 0.75,
    seed: int = 0,
    C: float = 0.2,
    use_class_weight_balanced: bool = False,
) -> pd.DataFrame:
    """
    Stability selection:
      - Repeated subsampling (without replacement)
      - Fit L1 logistic regression (LASSO; saga)
      - Count how often each post-processed feature has non-zero coefficient

    Returns:
      DataFrame sorted by selection_freq desc.
    """
    rng = np.random.default_rng(seed)
    n = len(dat)
    m = max(10, int(np.floor(n * subsample_frac)))

    X_cols = numeric_cols + categorical_cols
    y_all = dat[y_col].astype(int).to_numpy()

    cw = "balanced" if use_class_weight_balanced else None

    # One fit to lock feature space + names
    pre0 = make_preprocessor(numeric_cols, categorical_cols, use_interaction)
    X0 = dat[X_cols]
    y0 = y_all

    # For stability, require both classes
    if len(np.unique(y0)) < 2:
        raise ValueError(f"{y_col}: only one class present; stability selection not possible.")

    model0 = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=C,
        max_iter=30000,
        class_weight=cw,
    )
    pipe0 = Pipeline([("pre", pre0), ("model", model0)])
    pipe0.fit(X0, y0)

    # Build feature names robustly
    # Preprocessor is inside pipeline; now fitted:
    fitted_pre = pipe0.named_steps["pre"]

    # ColumnTransformer's OneHotEncoder: get_feature_names_out can be called with input_features
    # but we call without args; it returns names based on fitted categories.
    num_names = list(
        fitted_pre.named_transformers_["num"]
                 .named_steps["feat"]
                 .get_feature_names_out()
    )
    if len(categorical_cols) > 0:
        ohe = fitted_pre.named_transformers_["cat"].named_steps["ohe"]
        cat_names = list(ohe.get_feature_names_out(categorical_cols))
    else:
        cat_names = []
    feature_names = num_names + cat_names

    selected_counts = np.zeros(len(feature_names), dtype=int)
    used_rounds = 0
    skipped_oneclass = 0
    failed = 0

    for r in range(n_rounds):
        idx = rng.choice(n, size=m, replace=False)
        sub = dat.iloc[idx].copy()
        y_sub = sub[y_col].astype(int).to_numpy()
        if len(np.unique(y_sub)) < 2:
            skipped_oneclass += 1
            continue

        try:
            pre = make_preprocessor(numeric_cols, categorical_cols, use_interaction)
            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=C,
                max_iter=30000,
                class_weight=cw,
            )
            pipe = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(sub[X_cols], y_sub)
            coef = pipe.named_steps["model"].coef_.ravel()
            selected_counts += (np.abs(coef) > 1e-8).astype(int)
            used_rounds += 1
        except Exception:
            failed += 1
            continue

    denom = max(1, used_rounds)
    freq = selected_counts / denom

    out = pd.DataFrame({
        "feature": feature_names,
        "selection_freq": freq,
        "selected_count": selected_counts,
    }).sort_values("selection_freq", ascending=False).reset_index(drop=True)

    out["n_rounds_requested"] = n_rounds
    out["n_rounds_used"] = used_rounds
    out["skipped_oneclass"] = skipped_oneclass
    out["failed_fits"] = failed
    out["subsample_frac"] = subsample_frac
    out["C"] = C
    out["class_weight_balanced"] = bool(use_class_weight_balanced)

    return out


def run_stability_selection_and_save(
    dat: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    use_interaction: bool,
):
    stab_ge3 = stability_selection_binary(
        dat=dat,
        y_col="FILS_ge_3",
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        use_interaction=use_interaction,
        n_rounds=STAB_ROUNDS,
        subsample_frac=STAB_SUBSAMPLE_FRAC,
        seed=100,
        C=STAB_C,
        use_class_weight_balanced=STAB_USE_CLASS_WEIGHT_BALANCED,
    )
    print("\nStability selection (top 20): FILS >= 3")
    print(stab_ge3.head(20))
    stab_ge3.to_csv("stability_ge3.csv", index=False)

    stab_ge7 = stability_selection_binary(
        dat=dat,
        y_col="FILS_ge_7",
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        use_interaction=use_interaction,
        n_rounds=STAB_ROUNDS,
        subsample_frac=STAB_SUBSAMPLE_FRAC,
        seed=200,
        C=STAB_C,
        use_class_weight_balanced=STAB_USE_CLASS_WEIGHT_BALANCED,
    )
    print("\nStability selection (top 20): FILS >= 7")
    print(stab_ge7.head(20))
    stab_ge7.to_csv("stability_ge7.csv", index=False)


# ===========================================================
# Main
# ===========================================================

def main():
    dat = load_data()

    numeric_cols = [c for c in RAW_NUMERIC_COLS if c in dat.columns]
    categorical_cols = [c for c in RAW_CATEGORICAL_COLS if c in dat.columns]

    use_interaction = USE_INTERACTION
    if use_interaction:
        missing = {INTERACTION_A, INTERACTION_B} - set(numeric_cols)
        if missing:
            print(f"[WARN] Interaction disabled because missing columns: {missing}")
            use_interaction = False

    print(f"Analysis rows: n={len(dat)}")
    print("Class balance:")
    print(f"  FILS >= {THRESH_A}: {dat['FILS_ge_3'].mean():.3f} positive")
    print(f"  FILS >= {THRESH_B}: {dat['FILS_ge_7'].mean():.3f} positive")
    print("\nAvailable numeric columns:", numeric_cols)
    print("Available categorical columns:", categorical_cols)

    # Missingness tables
    x_cols = numeric_cols + categorical_cols
    miss_all = missingness_table(dat, [FILS_COL] + x_cols)
    miss_ge3 = missingness_table(dat, [FILS_COL] + x_cols, group_col="FILS_ge_3")
    miss_ge7 = missingness_table(dat, [FILS_COL] + x_cols, group_col="FILS_ge_7")

    print("\nMissingness (overall, top 15):")
    print(miss_all.head(15))

    miss_all.to_csv("missingness_overall.csv", index=False)
    miss_ge3.to_csv("missingness_by_ge3.csv", index=False)
    miss_ge7.to_csv("missingness_by_ge7.csv", index=False)

    # Figures: distributions
    plot_bar_distribution(
        dat[FILS_COL],
        "FILS / swallowing grade",
        f"FILS distribution (n={len(dat)})",
        "fig_fils_distribution.png",
    )

    for col in ["Alb", "Tim_from_onset", "BMI", "FIM_motor", "FIM_cognition", "mRS", "JCS", "JCS_categ"]:
        if col in dat.columns:
            plot_histogram(dat[col], f"Distribution: {col}", f"hist_{safe_filename(col)}.png")

    # Primary binary models (calibrated nested CV)
    for y_col, label in [("FILS_ge_3", f"FILS >= {THRESH_A}"), ("FILS_ge_7", f"FILS >= {THRESH_B}")]:
        print(f"\n=== Primary | {label} | calibration={CALIBRATION_METHOD} | class_weight_balanced={USE_CLASS_WEIGHT_BALANCED} ===")
        res = nested_cv_binary_calibrated(
            dat, y_col, numeric_cols, categorical_cols,
            use_interaction=use_interaction,
            calibrate=(CALIBRATION_METHOD is not None),
            calibration_method=CALIBRATION_METHOD,
            use_class_weight_balanced=USE_CLASS_WEIGHT_BALANCED,
        )

        print(
            f"Nested-CV (outer folds): AUC mean={res['fold_auc'].mean():.3f}, sd={res['fold_auc'].std(ddof=1):.3f} | "
            f"Brier mean={res['fold_brier'].mean():.3f}"
        )

        prefix = safe_filename(label.replace(" ", ""))
        plot_roc(res["y_true"], res["oof_prob"], f"ROC (OOF): {label}", f"roc_{prefix}.png")
        plot_calibration_curve(res["y_true"], res["oof_prob"], f"Calibration (OOF): {label}", f"cal_{prefix}.png")
        plot_decision_curve(res["y_true"], res["oof_prob"], f"Decision curve (OOF): {label}", f"dca_{prefix}.png")

        ci = bootstrap_performance_ci(
            res["y_true"], res["oof_prob"],
            n_rounds=BOOTSTRAP_ROUNDS,
            seed=1 if y_col == "FILS_ge_3" else 2
        )
        print(
            f"Bootstrap CI on OOF: AUC={ci['AUC_mean']:.3f} (95%CI {ci['AUC_CI'][0]:.3f}-{ci['AUC_CI'][1]:.3f}), "
            f"Brier={ci['Brier_mean']:.3f} (95%CI {ci['Brier_CI'][0]:.3f}-{ci['Brier_CI'][1]:.3f})"
        )

    # Sensitivity: complete-case
    for y_col, label in [("FILS_ge_3", f"FILS >= {THRESH_A}"), ("FILS_ge_7", f"FILS >= {THRESH_B}")]:
        dat_cc = make_complete_case(dat, x_cols, y_col)
        print(f"\n=== Sensitivity (complete-case) | {label} ===")
        print(f"Complete-case n={len(dat_cc)} (dropped {len(dat) - len(dat_cc)})")

        res_cc = nested_cv_binary_calibrated(
            dat_cc, y_col, numeric_cols, categorical_cols,
            use_interaction=use_interaction,
            calibrate=(CALIBRATION_METHOD is not None),
            calibration_method=CALIBRATION_METHOD,
            use_class_weight_balanced=USE_CLASS_WEIGHT_BALANCED,
        )
        print(
            f"Complete-case nested-CV: AUC mean={res_cc['fold_auc'].mean():.3f}, sd={res_cc['fold_auc'].std(ddof=1):.3f} | "
            f"Brier mean={res_cc['fold_brier'].mean():.3f}"
        )

    # Sensitivity: no class_weight
    if USE_CLASS_WEIGHT_BALANCED:
        for y_col, label in [("FILS_ge_3", f"FILS >= {THRESH_A}"), ("FILS_ge_7", f"FILS >= {THRESH_B}")]:
            print(f"\n=== Sensitivity (no class_weight) | {label} ===")
            res_nobal = nested_cv_binary_calibrated(
                dat, y_col, numeric_cols, categorical_cols,
                use_interaction=use_interaction,
                calibrate=(CALIBRATION_METHOD is not None),
                calibration_method=CALIBRATION_METHOD,
                use_class_weight_balanced=False,
            )
            print(
                f"No class_weight nested-CV: AUC mean={res_nobal['fold_auc'].mean():.3f}, sd={res_nobal['fold_auc'].std(ddof=1):.3f} | "
                f"Brier mean={res_nobal['fold_brier'].mean():.3f}"
            )

    # Ordinal surrogate
    ord2 = ordinal_surrogate_cv(
        dat, FILS_COL, numeric_cols, categorical_cols,
        use_interaction=use_interaction,
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, seed=321
    )
    print("\nOrdinal surrogate (sklearn) - OOF performance:")
    print(f"  MAE={ord2['MAE']:.3f} grades")
    print(f"  QWK={ord2['QWK']:.3f}")

    plot_ordinal_confusion_1to10(
        ord2["y_true"], ord2["oof_pred_ord"],
        "Ordinal confusion (OOF, surrogate)",
        "ordinal_confusion.png"
    )

    # ★ Stability selection
    run_stability_selection_and_save(dat, numeric_cols, categorical_cols, use_interaction)

    print("\nDONE.")
    print("Figures saved to:", os.path.abspath(FIG_DIR))
    print("Missingness tables saved: missingness_overall.csv, missingness_by_ge3.csv, missingness_by_ge7.csv")
    print("Stability selection saved: stability_ge3.csv, stability_ge7.csv")


if __name__ == "__main__":
    main()