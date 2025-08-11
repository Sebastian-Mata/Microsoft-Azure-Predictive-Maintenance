# Preeliminary Exploratory Data Analysis, run after data acquisition.
import matplotlib.pyplot as plt
import seaborn as sns
from my_module import load_data, plot_barh, plot_hist, plot_grouped_bar

telemetry_df, errors_df, maint_df, failures_df, machines_df = load_data()

# Display data types of each DataFrame
print("\nData Types of each DataFrame:")
print(f"Telemetry DataFrame:\n{telemetry_df.dtypes}")
print(f"\nErrors DataFrame:\n{errors_df.dtypes}")
print(f"\nMaintenance DataFrame:\n{maint_df.dtypes}")
print(f"\nFailures DataFrame:\n{failures_df.dtypes}")
print(f"\nMachines DataFrame:\n{machines_df.dtypes}")

# Display basic information about shape of the datasets
print("\nBasic Information about the datasets:")
print(f"Shape of the Telemetry Records: {telemetry_df.shape}")
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
print(f"Telemetry DataFrame Missing:\n{telemetry_df.isnull().sum()}")
print(f"\nErrors DataFrame Missing:\n{errors_df.isnull().sum()}")
print(f"\nMaintenance DataFrame Missing:\n{maint_df.isnull().sum()}")
print(f"\nFailures DataFrame Missing:\n{failures_df.isnull().sum()}")
print(f"\nMachines DataFrame Missing:\n{machines_df.isnull().sum()}")

# Check for duplicate rows in each DataFrame
print("\nDuplicate Rows in each DataFrame:")
print(f"Telemetry {telemetry_df.duplicated().sum()} duplicates")
print(f"Errors {errors_df.duplicated().sum()} duplicates")
print(f"Maintenance {maint_df.duplicated().sum()} duplicates")
print(f"Failures {failures_df.duplicated().sum()} duplicates")
print(f"Machines {machines_df.duplicated().sum()} duplicates")

# Display basic statistics of each DataFrame
print("\nBasic Statistics of each DataFrame:")
print(f"Telemetry DataFrame Statistics:\n{telemetry_df.describe()}")
print(f"\nErrors DataFrame Statistics:\n{errors_df.describe()}")
print(f"\nMaintenance DataFrame Statistics:\n{maint_df.describe()}")
print(f"\nFailures DataFrame Statistics:\n{failures_df.describe()}")
print(f"\nMachines DataFrame Statistics:\n{machines_df.describe()}")

# Plotting histograms for the distribution of the features in the telemetry DataFrame
for name in ["volt", "rotate", "pressure", "vibration"]:
    plot_hist(telemetry_df, feature_name=name, log=False, bins=1000)
    plt.savefig(
        f"data/graphs/{name}_distribution.png"
    )  # Save each plot on data directory
    plt.close()  # Close the figure to free memory

# Plotting error types frecuency in the errors DataFrame
plot_barh(
    errors_df,
    feature_name="errorID",
    log=False,
    title="Error Types Frequency",
    xlabel="Number of Errors",
)
plt.savefig("data/graphs/error_types_frequency.png")  # Save the plot on data directory
plt.close()

# Plotting component failure types in the failures DataFrame
plot_barh(
    failures_df,
    feature_name="failure",
    log=False,
    title="Failure type frequency",
    xlabel="Number of component failure",
)
plt.savefig(
    "data/graphs/failure_types_frequency.png"
)  # Save the plot on data directory
plt.close()

# Plotting type of errors per machines
plot_grouped_bar(
    df=errors_df,
    index="machineID",
    columns="errorID",
    values="errorValues",
    title="Type of Errors per Machine",
    xlabel="Machine ID",
    ylabel="Number of Errors",
)
plt.savefig("data/graphs/errors_per_machine.png")  # Save the plot on data directory
plt.close()

# Plotting component replacements per machine
plot_grouped_bar(
    df=maint_df,
    index="machineID",
    columns="comp",
    values="num_comp",
    title="Components Replaced per Machine",
    xlabel="Machine ID",
    ylabel="Components Replaced",
)
plt.savefig(
    "data/graphs/components_replaced_per_machine.png"
)  # Save the plot on data directory
plt.close()

# Plotting errors per day
errors_df["date"] = errors_df.datetime.dt.date
errors_df.groupby("date").size().hist(bins=20, figsize=(12, 6))
plt.title("Disrubution of Number of Errors per Day")
plt.xlabel("Errors on a day")
plt.ylabel("Number of errors")
plt.savefig("data/graphs/errors_per_day.png")
plt.close()

# Telemetry features correlation
features = ["volt", "rotate", "pressure", "vibration"]
corr = telemetry_df[features].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation between machine features")
plt.savefig("data/graphs/features_correlation.png")
