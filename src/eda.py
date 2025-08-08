# Preeleminary Exploratory Data Analysis, run after data acquisition.
from my_module import load_data

telemetry_df, errors_df, maint_df, failures_df, machines_df = load_data()

# Display basic information about shape of the datasets
print("\nBasic Information about the datasets:")
print(f"\nShape of the Telemetry Records: {telemetry_df.shape}")
print(telemetry_df.head())
print(f"\nShape of the Errors Records: {errors_df.shape}")
print(errors_df.head())
print(f"\nShape of the Maintenance Records: {maint_df.shape}")
print(maint_df.head())
print(f"\nShape of the Failures Records: {failures_df.shape}")
print(failures_df.head())
print(f"\nShape of the Machines Records: {machines_df.shape}")
print(machines_df.head())

# Check for missing values in each DataFrame
print("\nMissing Values in each DataFrame:")
print(f"\nTelemetry DataFrame:\n{telemetry_df.isnull().sum()}")
print(f"\nErrors DataFrame:\n{errors_df.isnull().sum()}")
print(f"\nMaintenance DataFrame:\n{maint_df.isnull().sum()}")
print(f"\nFailures DataFrame:\n{failures_df.isnull().sum()}")
print(f"\nMachines DataFrame:\n{machines_df.isnull().sum()}")

# Check for duplicate rows in each DataFrame
print("\nDuplicate Rows in each DataFrame:")
print(f"Telemetry {telemetry_df.duplicated().sum()} duplicates")
print(f"Errors {errors_df.duplicated().sum()} duplicates")
print(f"Maintenance {maint_df.duplicated().sum()} duplicates")
print(f"Failures {failures_df.duplicated().sum()} duplicates")
print(f"Machines {machines_df.duplicated().sum()} duplicates")
