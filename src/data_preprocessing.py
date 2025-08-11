import matplotlib.pyplot as plt
from tqdm import tqdm
from my_module import (
    load_data,
    check_future_failure,
    add_error_flags_per_machine,
    add_time_since_maint,
    add_sensor_features,
)

# Load DataFrames
telemetry, errors, maintenance, failures, machines = load_data()

# Generate and initialize target variable
telemetry["will_fail_in_24h"] = 0
telemetry["will_fail_in_48h"] = 0

# Group failures for each machine in a DataFrame
failures_by_machine = failures.groupby("machineID")["datetime"].apply(list).to_dict()

# Check for future failures within 24h and display proccess progress with tqdm
# 24h window
tqdm.pandas(desc="Assign 24h target variable")
telemetry["will_fail_in_24h"] = telemetry.progress_apply(
    lambda row: check_future_failure(
        row["datetime"], failures_by_machine.get(row["machineID"], []), 24
    ),
    axis=1,
).astype("int8")

# 48h window
tqdm.pandas(desc="Assign 48h target variable")
telemetry["will_fail_in_48h"] = telemetry.progress_apply(
    lambda row: check_future_failure(
        row["datetime"], failures_by_machine.get(row["machineID"], []), 48
    ),
    axis=1,
).astype("int8")


# Sanity Check
print(telemetry.head())
print("\nGenerated targets:")
print(
    telemetry[["will_fail_in_24h", "will_fail_in_48h"]].sum()
)  # No. of target generated
print("\nGenerated Nan's:")
print(
    telemetry[["will_fail_in_24h", "will_fail_in_48h"]].isnull().sum()
)  # Check for Nan's

# Merge machine model and age to main DataFrame (telemetry) and  reorder
telemetry = telemetry.merge(machines, on="machineID", how="left")
telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)

# List existing error types
error_types = errors["errorID"].unique().tolist()

# Add columns to the main DataFrame denoting if an error ocurred within the time window
for et in error_types:
    telemetry = add_error_flags_per_machine(telemetry, errors, et)

# Sanity check
print("\nErrors within time window:")
print(telemetry[[c for c in telemetry.columns if "error" in c]].sum().sort_index())

# Reorder maintenance and list existing components
maintenance = maintenance.sort_values(["machineID", "datetime"]).reset_index(drop=True)
components = maintenance["comp"].unique().tolist()

# Add columns to the main DataFrame denoting the last time since a component had maintenance
for comp in components:
    telemetry = add_time_since_maint(telemetry, maintenance, comp)

# Sanity check
maint_cols = [c for c in telemetry.columns if c.startswith("time_since_maint_")]
print("\nNew columns:", maint_cols)
print("\nGenerated Nan's:")
print(
    (telemetry[maint_cols].isna().mean() * 100).round(2).sort_values(ascending=False)
)  # Check for Nan's

# # Telemetry feature engineering (volt, rotate, preassure, vibration)
SENSORS = ["volt", "rotate", "pressure", "vibration"]

# Windows
LAGS = [1, 3, 6, 12, 24]
ROLL_MEANS = [3, 6, 12, 24, 48]
ROLL_STDS = [6, 24, 48]
SLOPES_K = [3, 6, 12]

# Add lags, roll means, rolling stds and slopes for each feature in the main DataFrame
telemetry = add_sensor_features(
    telemetry, SENSORS, LAGS, ROLL_MEANS, ROLL_STDS, SLOPES_K
)

# Sanity check
feature_cols = [
    c
    for c in telemetry.columns
    if any(s in c for s in ["_lag_", "_mean_", "_std_", "_slope_"])
]
print("\nTotal new features:", len(feature_cols))
print("\nNan's within new features:")
print(
    telemetry[feature_cols].isna().mean().sort_values(ascending=False).head(10)
)  # Check for Nan's to ensure there is no data leakage

## Post-Processing EDA
# Save main DataFrame in processed data folder
telemetry.to_csv("processed_data/telemetry.csv", index=False)

# Class Balance
print(
    "\nClass balance 24h:\n", telemetry["will_fail_in_24h"].value_counts(normalize=True)
)
print(
    "\nClass balance 48h:\n", telemetry["will_fail_in_48h"].value_counts(normalize=True)
)

# Preprocessing Summary
print("\nPreprocessing summary:")
print(telemetry.info())

# NaN's
na_ratio = telemetry.isna().mean().sort_values(ascending=False)
print("\nNan percentage (top 15):")
print((na_ratio * 100).round(2).head(15))

# Sensor Distrubution
telemetry[SENSORS].hist(bins=50, figsize=(10, 6))
plt.suptitle("Sensor Distrubution (raw)")
plt.savefig("processed_data/graphs/sensor_distribution.png")


# Temp sanity check (positive events)
for target in ["will_fail_in_24h", "will_fail_in_48h"]:
    plt.figure(figsize=(12, 4))
    telemetry.groupby("datetime")[target].mean().plot()
    plt.title(f"Proportion of positive failures through time ({target})")
    plt.ylabel("Proporción")
    plt.savefig(f"processed_data/graphs/{target}_positive_failures.png")

# Fail rate of each machine model
fail_rate_by_model = (
    telemetry.groupby("model")["will_fail_in_24h"].mean().sort_values(ascending=False)
)
print("\nFail rate of each machine model")
print(fail_rate_by_model)
