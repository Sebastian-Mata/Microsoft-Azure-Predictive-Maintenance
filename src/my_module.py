# Module containing diverse functions employed in the project.
# # Don't forget to run data aquisition first for data dependent functions
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import OrderedDict
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    fbeta_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.base import BaseEstimator, TransformerMixin


# Data Loading
def load_data():
    """Load datasets from CSV files into pandas DataFrames.
    Returns:
        tuple: A tuple containing DataFrames for telemetry, errors, maintenance,
               failures, and machines. In the order:
               (telemetry_df, errors_df, maint_df, failures_df, machines_df)
    """
    telemetry_df = pd.read_csv("data/telemetry.csv", parse_dates=["datetime"])
    errors_df = pd.read_csv("data/errors.csv", parse_dates=["datetime"])
    maint_df = pd.read_csv("data/maint.csv", parse_dates=["datetime"])
    failures_df = pd.read_csv("data/failures.csv", parse_dates=["datetime"])
    machines_df = pd.read_csv("data/machines.csv")

    return telemetry_df, errors_df, maint_df, failures_df, machines_df


# Load DataFrames for data dependent functions
telemetry, errors, maintenance, failures, machines = load_data()


# Plotting Functions
def plot_hist(df, feature_name, log=False, bins=100):
    """Plot histogram of a feature in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the feature.
        feature_name (str): Name of the feature to plot.
        log (bool, optional): Whether to use logarithmic scale. Defaults to False.
        bins (int, optional): Number of bins for the histogram. Defaults to 100.
    Returns:
        Figure: Matplotlib figure object containing the histogram.
    """
    plt.figure(figsize=(10, 6))
    if log:
        plt.hist(df[feature_name].dropna(), bins=bins, log=True)
    else:
        plt.hist(df[feature_name].dropna(), bins=bins)
    plt.title(f"Distribution of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.grid()
    plt.show(block=False)


def plot_barh(df, feature_name, log=False, title=None, xlabel=None):
    """Plot horizontal bar chart of a feature in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the feature.
        feature_name (str): Name of the feature to plot.
        log (bool, optional): Whether to use logarithmic scale. Defaults to False.
        title (str, optional): Title of the plot. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to None.
    Returns:
        Figure: Matplotlib figure object containing the bar chart.
    """
    plt.figure(figsize=(10, 6))
    if log:
        df[feature_name].value_counts().plot(kind="barh", logx=True)
    else:
        df[feature_name].value_counts().plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(feature_name)
    plt.show(block=False)


def plot_grouped_bar(df, index, columns, values, title=None, xlabel=None, ylabel=None):
    """Plot grouped bar chart from a DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        index (str): Column to use as index.
        columns (str): Column to use as columns.
        values (str): Column to use as values.
        title (str, optional): Title of the plot. Defaults to None.
        xlabel (str, optional): Label for the x-axis. Defaults to None.
        ylabel (str, optional): Label for the y-axis. Defaults to None.
    Returns:
        Figure: Matplotlib figure object containing the grouped bar chart.
    """
    df_temp = df.groupby([index, columns]).size().reset_index()
    df_temp.columns = [index, columns, values]
    df_pivot = pd.pivot(
        df_temp, index=index, columns=columns, values=values
    ).rename_axis(None, axis=1)
    df_pivot.plot.bar(stacked=True, figsize=(20, 6), title=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show(block=False)


# Check if the machine will fail in the near future (time window)
def check_future_failure(current_time, failure_times, window_hours):
    if not failure_times:
        return 0
    window = pd.Timedelta(hours=window_hours)
    for ft in failure_times:
        dt = ft - current_time
        if pd.Timedelta(0) <= dt <= window:
            return 1
    return 0


# Create new columns determining if an error type happened within the time window [Data Dependent Function]
# # List existing machines by ID
machine_ids = telemetry["machineID"].unique().tolist()


# Function
def add_error_flags_per_machine(telemetry_df, errors_df, error_label):
    timecol = f"{error_label}_time"
    telemetry_df[f"{error_label}_last_24h"] = 0
    telemetry_df[f"{error_label}_last_48h"] = 0
    e = errors_df[errors_df["errorID"] == error_label].rename(
        columns={"datetime": timecol}
    )

    for m in machine_ids:
        left_mask = telemetry_df["machineID"] == m
        left = telemetry_df.loc[left_mask, ["datetime"]].sort_values("datetime")
        right = e.loc[e["machineID"] == m, [timecol]].sort_values(timecol)
        if right.empty:
            continue

        merged = pd.merge_asof(
            left,
            right,
            left_on="datetime",
            right_on=timecol,
            direction="backward",
            allow_exact_matches=True,
        )
        delta_h = (merged["datetime"] - merged[timecol]).dt.total_seconds() / 3600.0
        has_prev = merged[timecol].notna()
        telemetry_df.loc[left.index, f"{error_label}_last_24h"] = (
            ((delta_h <= 24) & has_prev).astype(int).values
        )
        telemetry_df.loc[left.index, f"{error_label}_last_48h"] = (
            ((delta_h <= 48) & has_prev).astype(int).values
        )

    return telemetry_df


# Create new columns determining the time since a component had maintenance [Data Dependent Function]
def add_time_since_maint(telemetry_df, maint_df, comp_label):
    timecol = f"{comp_label}_maint_time"
    telemetry_df[f"time_since_maint_{comp_label}_h"] = pd.NA
    telemetry_df[f"time_since_maint_{comp_label}_d"] = pd.NA

    m = maint_df[maint_df["comp"] == comp_label].rename(columns={"datetime": timecol})

    for m_id in machine_ids:
        left_mask = telemetry_df["machineID"] == m_id
        left = telemetry_df.loc[left_mask, ["datetime"]].sort_values("datetime")
        right = m.loc[m["machineID"] == m_id, [timecol]].sort_values(timecol)

        if right.empty:
            continue

        merged = pd.merge_asof(
            left,
            right,
            left_on="datetime",
            right_on=timecol,
            direction="backward",
            allow_exact_matches=True,
        )
        delta_h = (merged["datetime"] - merged[timecol]).dt.total_seconds() / 3600.0
        telemetry_df.loc[left.index, f"time_since_maint_{comp_label}_h"] = (
            delta_h.values
        )
        telemetry_df.loc[left.index, f"time_since_maint_{comp_label}_d"] = (
            delta_h / 24
        ).values

    return telemetry_df


# Add columns for for lags, rolling means, rolling stds, and slopes.
def add_sensor_features(df, sensors, lags, rmeans, rstds, slopes):
    g = df.groupby("machineID", group_keys=False)
    # Lags
    for c in sensors:
        for k in lags:
            df[f"{c}_lag_{k}h"] = g[c].shift(k)
    # Rolling Means
    for c in sensors:
        for k in rmeans:
            df[f"{c}_mean_{k}h"] = (
                g[c]
                .rolling(window=k, min_periods=k)
                .mean()
                .reset_index(level=0, drop=True)
            )
    # Rolling Stds
    for c in sensors:
        for k in rstds:
            df[f"{c}_std_{k}h"] = (
                g[c]
                .rolling(window=k, min_periods=k)
                .std(ddof=0)
                .reset_index(level=0, drop=True)
            )
    # Slopes
    for c in sensors:
        for k in slopes:
            lag_col = f"{c}_lag_{k}h"

            # If the column is non-existent, create it.
            if lag_col not in df.columns:
                df[lag_col] = g[c].shift(k)
            df[f"{c}_slope_{k}h"] = (df[c] - df[lag_col]) / k
    return df


# Make training, validation and test splits
def build_split(X, y, train_mask, val_mask, test_mask, label):
    # Return an ordered dictionary of splits
    ds = OrderedDict()
    ds[f"X_train_{label}"] = X[train_mask]
    ds[f"y_train_{label}"] = y[train_mask]
    ds[f"X_val_{label}"] = X[val_mask]
    ds[f"y_val_{label}"] = y[val_mask]
    ds[f"X_test_{label}"] = X[test_mask]
    ds[f"y_test_{label}"] = y[test_mask]

    # Sizes and balance summmary
    print(f"\n== {label.upper()} ===")
    for split in ["train", "val", "test"]:
        y_split = ds[f"y_{split}_{label}"]
        pos = int(y_split.sum())
        n = int(y_split.shape[0])
        prop = pos / max(n, 1)
        print(f"{split:>5}: n={n:,} | positives={pos:,} ({prop: .4%})")

    # Check for temporal leak (date times)
    for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        print(
            f"{split:>5} range: {telemetry.loc[mask, 'datetime'].min()} → {telemetry.loc[mask, 'datetime'].max()}"
        )

    return ds


## To ensure a determinist behavior and global reproducibility we are gona set a fixed seed value to the model.
SEED = 42
random.seed(SEED)  # For python's own random number generator (RNG)
np.random.seed(SEED)  # For numpy's RNG
os.environ["PYTHONHASHSEED"] = str(SEED)
RANDOM_STATE = 42


# Find the optimal decision threshold for Fbeta score
def best_threshold_by_fbeta(y_true, y_prob, beta=2.0):
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.r_[t, 1.0]  # Lineup
    fbeta = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    idx = np.nanargmax(fbeta)
    return float(t[idx]), float(fbeta[idx])


# Measure metrics for model evaluation
def metrics_block(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp + 1e-12)  # Recall class 0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),  # Sensitivity (class 1)
        "specificity": specificity,  # Recall class 0
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2.0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# Plot reciving operation characteristcs
def plot_roc(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall+)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Display confusion matrix
def print_confmat(y_true, y_prob, threshold, title):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title} | threshold={threshold:.3f}")
    print(cm)
    print(classification_report(y_true, y_pred, digits=4))


# Preprocess data for Logistc Regression model (Applies One Hot Encoder, Imputes median values over Nan's and scales)
def build_preprocessor_lr(X_sample: pd.DataFrame) -> ColumnTransformer:
    cat_features = [c for c in ["model"] if c in X_sample.columns]
    num_features = [c for c in X_sample.columns if c not in cat_features]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
            if cat_features
            else ("cat", "drop", []),
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                num_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# Preprocess data for Random Forest model (Applies ordinal encoder and imputes median)
def build_preprocessor_rf(X_sample: pd.DataFrame) -> ColumnTransformer:
    cat_features = [c for c in ["model"] if c in X_sample.columns]
    num_features = [c for c in X_sample.columns if c not in cat_features]
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                cat_features,
            )
            if cat_features
            else ("cat", "drop", []),
            ("num", SimpleImputer(strategy="median"), num_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# Transform into type float32
class ToFloat32(BaseEstimator, TransformerMixin):
    """Transformador simple para castear a float32"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(np.float32)


# Logistic regression model builder
def make_lr(X_sample):
    return Pipeline(
        [
            ("prep", build_preprocessor_lr(X_sample)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# Random Forest Model builder
def make_rf_fast(X_sample):
    return Pipeline(
        steps=[
            ("prep", build_preprocessor_rf(X_sample)),
            ("to32", ToFloat32()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,  # ↑ for more preformance
                    max_depth=16,  # Limit for more speed
                    min_samples_leaf=5,
                    max_features="sqrt",
                    bootstrap=True,
                    max_samples=0.7,  # Subsample accelerates and regularizes
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# Runner that saves and returns models by horizon (trains, selects F2 threshold and evaluates VAL/TEST)
def evaluate_and_save_models_for_horizon(
    splits, label, save_dir="models_step7", cache_dir=None
):
    X_train, y_train = splits[f"X_train_{label}"], splits[f"y_train_{label}"]
    X_val, y_val = splits[f"X_val_{label}"], splits[f"y_val_{label}"]
    X_test, y_test = splits[f"X_test_{label}"], splits[f"y_test_{label}"]

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    specs = [("LR", make_lr), ("RF_fast", make_rf_fast)]
    rows, models_out = [], {}

    for name, maker in specs:
        print(f"\n===== {label} :: {name} =====")
        pipe = maker(
            X_train
        )  # cache_dir opcional if markers where defined with 'memory'
        pipe.fit(X_train, y_train)

        # Save complete pipeline
        model_path = Path(save_dir) / f"model_{label}_{name}.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved: {model_path}")

        # Predictions
        val_prob = pipe.predict_proba(X_val)[:, 1]
        test_prob = pipe.predict_proba(X_test)[:, 1]

        # Optimal threshold for F2 in validation
        thr, f2_val = best_threshold_by_fbeta(y_val.values, val_prob, beta=2.0)
        print(f"Optimal threshold (F2, val) = {thr:.3f} | F2(val)={f2_val:.4f}")

        print_confmat(y_val, val_prob, thr, title=f"[VAL]  {label}-{name}")
        plot_roc(y_val, val_prob, title=f"ROC (VAL)  — {label} {name}")
        print_confmat(y_test, test_prob, thr, title=f"[TEST] {label}-{name}")
        plot_roc(y_test, test_prob, title=f"ROC (TEST) — {label} {name}")

        # Metrics
        m_val = metrics_block(y_val.values, val_prob, thr)
        m_test = metrics_block(y_test.values, test_prob, thr)
        rows.append(
            {"horizon": label, "model": name, "split": "VAL", "threshold": thr, **m_val}
        )
        rows.append(
            {
                "horizon": label,
                "model": name,
                "split": "TEST",
                "threshold": thr,
                **m_test,
            }
        )

        # Return in memory
        models_out[(label, name)] = pipe

    df = pd.DataFrame(rows)
    return df, models_out


# Load model
def get_model_safe(horizon: str, name: str):
    # 1) Try from dictionaries in memory
    try:
        dct = (
            globals().get("models24") if horizon == "24h" else globals().get("models48")
        )
        if isinstance(dct, dict) and (horizon, name) in dct:
            return dct[(horizon, name)]
    except Exception:
        pass
    # 2) Try from the disk
    p = Path(f"models_step7/model_{horizon}_{name}.joblib")
    if p.exists():
        return joblib.load(p)
    # 3) Diagnostic
    raise FileNotFoundError(
        f"No encontré el modelo {horizon}-{name} ni en memoria ni en disco.\n"
        f"Busca el archivo: {p.resolve()}.\n"
        f"Si no existe, vuelve a correr: df24, models24 = evaluate_and_save_models_for_horizon(splits_24, '24h') y "
        f"df48, models48 = evaluate_and_save_models_for_horizon(splits_48, '48h')."
    )


# Load models from memory or disk without retraining
def group_importance_rf_from_pipeline(pipeline, title: str):
    """Toma un Pipeline guardado (RF_fast), extrae feature_importances_ y agrupa por familias."""
    # Feature names after preprocessor
    feat_names = pipeline.named_steps["prep"].get_feature_names_out()
    rf = pipeline.named_steps["clf"]

    df_imp = pd.DataFrame(
        {"feature": feat_names, "importance": rf.feature_importances_}
    )

    def family(f):
        f = str(f)
        if "mean_" in f:
            return "Rolling mean"
        if "slope_" in f:
            return "Slope"
        if "_lag_" in f:
            return "Lag"
        if "_std_" in f:
            return "Rolling std"
        if "error" in f:
            return "Recent errors"
        if "maint" in f:
            return "Time since maintenance"
        if "model" in f or "age" in f:
            return "Model/Year"
        return "Others"

    df_imp["family"] = df_imp["feature"].apply(family)
    grp = (
        df_imp.groupby("family", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(grp["family"], grp["importance"])
    plt.xlabel("Added importance")
    plt.title(f"Importance by family — {title}")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.show()
    return grp, df_imp.sort_values("importance", ascending=False)
