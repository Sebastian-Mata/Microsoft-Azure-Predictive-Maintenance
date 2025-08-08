# Module containing diverse functions employed in the project.
import pandas as pd


# Data Loading and Preprocessing
def load_data():
    """Load datasets from CSV files into pandas DataFrames.
    Returns:
        tuple: A tuple containing DataFrames for telemetry, errors, maintenance,
               failures, and machines. In the order:
               (telemetry_df, errors_df, maint_df, failures_df, machines_df)
    """
    telemetry_df = pd.read_csv("data/telemetry.csv")
    errors_df = pd.read_csv("data/errors.csv")
    maint_df = pd.read_csv("data/maint.csv")
    failures_df = pd.read_csv("data/failures.csv")
    machines_df = pd.read_csv("data/machines.csv")

    return telemetry_df, errors_df, maint_df, failures_df, machines_df
