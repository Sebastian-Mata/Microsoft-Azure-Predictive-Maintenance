import kagglehub
import pandas as pd

# Download latest version of dataset from Kaggle
path = kagglehub.dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")


print("\nPath to dataset files:", path)

# Load datasets using pandas into DataFrames
telemetry_df = pd.read_csv(f"{path}/PdM_telemetry.csv")
errors_df = pd.read_csv(f"{path}/PdM_errors.csv")
maint_df = pd.read_csv(f"{path}/PdM_maint.csv")
failures_df = pd.read_csv(f"{path}/PdM_failures.csv")
machines_df = pd.read_csv(f"{path}/PdM_machines.csv")

# Convert datetime columns to pandas datetime objects and sort DataFrames by datetime and machineID
tables = [telemetry_df, maint_df, failures_df, errors_df]
for df in tables:
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    df.sort_values(["datetime", "machineID"], inplace=True, ignore_index=True)


# Save DataFrames to CSV files in the data directory
telemetry_df.to_csv("data/telemetry.csv", index=False)
errors_df.to_csv("data/errors.csv", index=False)
maint_df.to_csv("data/maint.csv", index=False)
failures_df.to_csv("data/failures.csv", index=False)
machines_df.to_csv("data/machines.csv", index=False)
