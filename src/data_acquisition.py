import kagglehub
import pandas as pd

# Download latest version of dataset from Kaggle
path = kagglehub.dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")
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
telemetry_df.to_csv("data/telemetry.csv", index=False)
errors_df.to_csv("data/errors.csv", index=False)
maint_df.to_csv("data/maint.csv", index=False)
failures_df.to_csv("data/failures.csv", index=False)
machines_df.to_csv("data/machines.csv", index=False)
