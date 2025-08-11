import kagglehub
import pandas as pd


def save_data(destination_folder: str = "data"):
    # Download latest version of dataset from Kaggle
    path = kagglehub.dataset_download(
        "arnabbiswas1/microsoft-azure-predictive-maintenance"
    )
    print("\nPath to dataset files:", path)

    # Load datasets using pandas into DataFrames
    # Since date is in the ISO 8601 date format, we parse it as datetime
    telemetry_df = pd.read_csv(f"{path}/PdM_telemetry.csv", parse_dates=["datetime"])
    errors_df = pd.read_csv(f"{path}/PdM_errors.csv", parse_dates=["datetime"])
    maint_df = pd.read_csv(f"{path}/PdM_maint.csv", parse_dates=["datetime"])
    failures_df = pd.read_csv(f"{path}/PdM_failures.csv", parse_dates=["datetime"])
    machines_df = pd.read_csv(f"{path}/PdM_machines.csv")

    # Sort DataFrames by machineID and datetime
    tables = [telemetry_df, maint_df, failures_df, errors_df]
    for df in tables:
        df.sort_values(["machineID", "datetime"], inplace=True)  # , ignore_index=True)

    # Save DataFrames to CSV files in the data directory
    telemetry_df.to_csv(f"{destination_folder}/telemetry.csv", index=False)
    errors_df.to_csv(f"{destination_folder}/errors.csv", index=False)
    maint_df.to_csv(f"{destination_folder}/maint.csv", index=False)
    failures_df.to_csv(f"{destination_folder}/failures.csv", index=False)
    machines_df.to_csv(f"{destination_folder}/machines.csv", index=False)


# Data Loading
def load_data() -> tuple[pd.DataFrame, ...]:
    """Load datasets from CSV files into pandas DataFrames.
    Returns:
        tuple: A tuple containing DataFrames for telemetry, errors, maintenance,
               failures, and machines. In the order:
               (telemetry_df, errors_df, maint_df, failures_df, machines_df)
    """
    telemetry_df = pd.read_csv("data/raw/telemetry.csv", parse_dates=["datetime"])
    errors_df = pd.read_csv("data/raw/errors.csv", parse_dates=["datetime"])
    maint_df = pd.read_csv("data/raw/maint.csv", parse_dates=["datetime"])
    failures_df = pd.read_csv("data/raw/failures.csv", parse_dates=["datetime"])
    machines_df = pd.read_csv("data/raw/machines.csv")

    return telemetry_df, errors_df, maint_df, failures_df, machines_df


if __name__ == "__main__":
    save_data("data/raw")
