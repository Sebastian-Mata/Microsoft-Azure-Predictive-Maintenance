import os
import random
import pandas as pd
import numpy as np
import joblib
from IPython.display import display
from functions import (
    build_split,
    group_importance_rf_from_pipeline,
    evaluate_and_save_models_for_horizon,
)


# Load the processed DataFrame
telemetry = pd.read_csv("data/processed/telemetry.csv", parse_dates=["datetime"])


## Data Modeling
def data_modeling(telemetry: pd.DataFrame):
    ## To ensure a determinist behavior and global reproducibility we are gona set a fixed seed value to the model.
    SEED = 42
    random.seed(SEED)  # For python's own random number generator (RNG)
    np.random.seed(SEED)  # For numpy's RNG
    os.environ["PYTHONHASHSEED"] = str(SEED)

    # Change the model column type from object to category
    telemetry["model"] = telemetry["model"].astype("category")

    # Create a DataFrame of all the features
    exclude_cols = {"datetime", "machineID", "will_fail_in_24h", "will_fail_in_48h"}
    features_cols = [c for c in telemetry.columns if c not in exclude_cols]

    # Separate by horizon into variables and features (x,y)
    X = telemetry[features_cols]
    y_24 = telemetry["will_fail_in_24h"].astype(int)
    y_48 = telemetry["will_fail_in_48h"].astype(int)

    # Splits for DataFrame into training, validation and test
    cut_train = pd.Timestamp("2015-09-30 23:59:59")
    cut_val = pd.Timestamp("2015-11-15 23:59:59")
    train_mask = telemetry["datetime"] <= cut_train
    val_mask = (telemetry["datetime"] > cut_train) & (telemetry["datetime"] <= cut_val)
    test_mask = telemetry["datetime"] > cut_val

    # Make splits for each time window
    splits_24 = build_split(
        X,
        y_24,
        train_mask,
        val_mask,
        test_mask,
        label="24h",
        timestamps=telemetry["datetime"],
    )
    splits_48 = build_split(
        X,
        y_48,
        train_mask,
        val_mask,
        test_mask,
        label="48h",
        timestamps=telemetry["datetime"],
    )

    # === Complete evaluation: matrices, metrics y ROC curves (VAL/TEST) for 24h & 48h ===
    df24, models24 = evaluate_and_save_models_for_horizon(
        splits_24, "24h", save_dir="data/models"
    )
    df48, models48 = evaluate_and_save_models_for_horizon(
        splits_48, "48h", save_dir="data/models"
    )

    eval_all = pd.concat([df24, df48], ignore_index=True)
    cols = [
        "horizon",
        "model",
        "split",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    print("\n===== Metrics Summary =====")
    display(
        eval_all[cols].sort_values(["horizon", "model", "split"]).reset_index(drop=True)
    )

    # Save metrics summary
    eval_all.to_csv("eval_summary_step7.csv", index=False)

    return telemetry, X, y_24, y_48


## Actual Model
def main():
    # --- Load Model ---
    rf24 = joblib.load("models_step7/model_24h_RF_fast.joblib")
    rf48 = joblib.load("models_step7/model_48h_RF_fast.joblib")

    grp24, topfeat24 = group_importance_rf_from_pipeline(rf24, "RF_fast — 24h")
    grp48, topfeat48 = group_importance_rf_from_pipeline(rf48, "RF_fast — 48h")

    display(grp24)  # Importance by family (24h)
    display(topfeat24.head(20))  # top 20 features (24h)

    display(grp48)  # Importance by family (48h)
    display(topfeat48.head(20))  # top 20 features (48h)


if __name__ == "__main__":
    main()
