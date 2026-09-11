"""
IDS 706 - Week 2 Data Analysis Project
Dataset: Red and White Wine Quality (merged)

Project question:
What characteristics are associated with higher wine quality, and do red and
white wines show different patterns?

This script covers:
1. Dataset import
2. Data inspection
3. Filtering and grouping with Pandas
4. Visualization
5. Equivalent analysis with Polars + simple timing comparison
6. Beginner-friendly machine learning exploration with Linear Regression
7. Saving results for the README
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/wine_quality_merged.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

REQUIRED_COLUMNS = {
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
    "type",
}


# ---------------------------------------------------------------------------
# 1. Import the dataset
# ---------------------------------------------------------------------------

def load_with_pandas() -> pd.DataFrame:
    """Load the comma-separated merged wine-quality dataset with Pandas."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place wine_quality_merged.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "The dataset is missing expected columns: "
            + ", ".join(sorted(missing_columns))
        )

    return df


# ---------------------------------------------------------------------------
# 2. Inspect the data
# ---------------------------------------------------------------------------

def inspect_data(df: pd.DataFrame) -> None:
    """Print the main inspection results required by the assignment."""
    print("\n" + "=" * 72)
    print("1. DATA INSPECTION")
    print("=" * 72)

    print("\nFirst five rows:")
    print(df.head())

    print("\nShape (rows, columns):")
    print(df.shape)

    print("\nColumn names:")
    print(list(df.columns))

    print("\nData types and non-null counts:")
    df.info()

    print("\nSummary statistics for numeric columns:")
    print(df.describe().round(3))

    print("\nMissing values by column:")
    print(df.isna().sum())

    duplicate_count = int(df.duplicated().sum())
    print(f"\nDuplicate rows: {duplicate_count}")

    print("\nWine type counts:")
    print(df["type"].value_counts())

    print("\nQuality score counts:")
    print(df["quality"].value_counts().sort_index())


# ---------------------------------------------------------------------------
# 3. Filtering and grouping with Pandas
# ---------------------------------------------------------------------------

def filter_and_group_pandas(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a meaningful high-quality subset and grouped summaries.

    High-quality wine is defined here as quality >= 7.
    """
    print("\n" + "=" * 72)
    print("2. FILTERING AND GROUPING WITH PANDAS")
    print("=" * 72)

    # Filter: keep wines with quality score 7 or higher.
    high_quality = df[df["quality"] >= 7].copy()

    print(f"\nWines with quality >= 7: {len(high_quality)}")
    print(high_quality.head())

    print("\nHigh-quality wines by type:")
    print(high_quality["type"].value_counts())

    # Group 1: compare red and white wines overall.
    grouped_by_type = (
        df.groupby("type")
        .agg(
            count=("quality", "size"),
            mean_quality=("quality", "mean"),
            mean_alcohol=("alcohol", "mean"),
            mean_volatile_acidity=("volatile acidity", "mean"),
            mean_sulphates=("sulphates", "mean"),
        )
        .reset_index()
        .sort_values("type")
    )

    print("\nGrouped summary by wine type:")
    print(grouped_by_type.round(3))

    # Group 2: examine how selected measurements change with quality score.
    grouped_by_quality = (
        df.groupby("quality")
        .agg(
            count=("quality", "size"),
            mean_alcohol=("alcohol", "mean"),
            mean_volatile_acidity=("volatile acidity", "mean"),
            mean_sulphates=("sulphates", "mean"),
        )
        .reset_index()
        .sort_values("quality")
    )

    print("\nGrouped summary by quality:")
    print(grouped_by_quality.round(3))

    # Group 3: compare red and white wines within each quality level.
    grouped_by_type_quality = (
        df.groupby(["type", "quality"])
        .agg(
            count=("quality", "size"),
            mean_alcohol=("alcohol", "mean"),
        )
        .reset_index()
        .sort_values(["type", "quality"])
    )

    print("\nGrouped summary by wine type and quality:")
    print(grouped_by_type_quality.round(3))

    # Save grouped tables so the analysis is easy to inspect later.
    grouped_by_type.to_csv(OUTPUT_DIR / "grouped_by_type.csv", index=False)
    grouped_by_quality.to_csv(OUTPUT_DIR / "grouped_by_quality.csv", index=False)
    grouped_by_type_quality.to_csv(
        OUTPUT_DIR / "grouped_by_type_quality.csv",
        index=False,
    )

    return (
        high_quality,
        grouped_by_type,
        grouped_by_quality,
        grouped_by_type_quality,
    )


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def create_visualizations(df: pd.DataFrame) -> None:
    """Create and save two clear plots related to the project question."""
    print("\n" + "=" * 72)
    print("3. VISUALIZATION")
    print("=" * 72)

    # Plot 1:
    # Compare the distribution of quality scores for red and white wines.
    quality_type_counts = (
        df.groupby(["quality", "type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    qualities = quality_type_counts.index.to_numpy()
    types = list(quality_type_counts.columns)
    x = np.arange(len(qualities))
    width = 0.8 / len(types)

    plt.figure(figsize=(9, 5))

    for i, wine_type in enumerate(types):
        offset = (i - (len(types) - 1) / 2) * width
        plt.bar(
            x + offset,
            quality_type_counts[wine_type].to_numpy(),
            width=width,
            label=wine_type,
        )

    plt.title("Wine Quality Distribution by Type")
    plt.xlabel("Quality Score")
    plt.ylabel("Number of Wines")
    plt.xticks(x, qualities)
    plt.legend(title="Wine Type")
    plt.tight_layout()

    path1 = OUTPUT_DIR / "quality_distribution_by_type.png"
    plt.savefig(path1, dpi=150)
    plt.close()

    # Plot 2:
    # Compare alcohol distributions across quality scores.
    qualities = sorted(df["quality"].unique())
    alcohol_groups = [
        df.loc[df["quality"] == quality, "alcohol"].to_numpy()
        for quality in qualities
    ]

    plt.figure(figsize=(9, 5))
    plt.boxplot(
        alcohol_groups,
        tick_labels=[str(quality) for quality in qualities],
    )
    plt.title("Alcohol Content by Wine Quality Score")
    plt.xlabel("Quality Score")
    plt.ylabel("Alcohol (%)")
    plt.tight_layout()

    path2 = OUTPUT_DIR / "alcohol_by_quality.png"
    plt.savefig(path2, dpi=150)
    plt.close()

    print(f"Saved: {path1}")
    print(f"Saved: {path2}")
    print(
        "\nWhy these plots?\n"
        "- Plot 1 shows whether red and white wines have different quality-score "
        "distributions.\n"
        "- Plot 2 shows whether alcohol content changes across quality levels."
    )


# ---------------------------------------------------------------------------
# 5. Polars analysis and performance comparison
# ---------------------------------------------------------------------------

def polars_analysis_and_timing() -> tuple[pl.DataFrame, float, float]:
    """
    Repeat key filter/group operations in Polars and compare a small workflow
    with Pandas.
    """
    print("\n" + "=" * 72)
    print("4. POLARS ANALYSIS AND PANDAS VS POLARS PERFORMANCE")
    print("=" * 72)

    # Correctly assign the loaded Polars DataFrame.
    pl_df = pl.read_csv(DATA_PATH)

    pl_high_quality = pl_df.filter(pl.col("quality") >= 7)

    pl_grouped = (
        pl_df.group_by(["type", "quality"])
        .agg(
            pl.len().alias("count"),
            pl.col("alcohol").mean().alias("mean_alcohol"),
            pl.col("volatile acidity")
            .mean()
            .alias("mean_volatile_acidity"),
            pl.col("sulphates").mean().alias("mean_sulphates"),
        )
        .sort(["type", "quality"])
    )

    print("\nPolars high-quality subset:")
    print(pl_high_quality.head())

    print("\nPolars grouped summary by type and quality:")
    print(pl_grouped)

    # Time the same read -> filter -> group workflow in both libraries.
    # The dataset is small, so results may vary from run to run.
    repeats = 50

    pandas_start = perf_counter()
    for _ in range(repeats):
        pd_temp = pd.read_csv(DATA_PATH)
        (
            pd_temp[pd_temp["quality"] >= 7]
            .groupby(["type", "quality"])["alcohol"]
            .mean()
        )
    pandas_seconds = perf_counter() - pandas_start

    polars_start = perf_counter()
    for _ in range(repeats):
        pl_temp = pl.read_csv(DATA_PATH)
        (
            pl_temp.filter(pl.col("quality") >= 7)
            .group_by(["type", "quality"])
            .agg(pl.col("alcohol").mean().alias("mean_alcohol"))
        )
    polars_seconds = perf_counter() - polars_start

    print(f"\nTiming over {repeats} read/filter/group runs:")
    print(f"Pandas: {pandas_seconds:.6f} seconds")
    print(f"Polars: {polars_seconds:.6f} seconds")

    if pandas_seconds > 0 and polars_seconds > 0:
        ratio = pandas_seconds / polars_seconds

        if ratio > 1:
            print(f"Polars was about {ratio:.2f}x faster in this run.")
        else:
            print(f"Pandas was about {1 / ratio:.2f}x faster in this run.")

    print(
        "Note: This dataset is relatively small, so timing results can vary "
        "between computers and runs. The comparison demonstrates equivalent "
        "Pandas and Polars workflows."
    )

    return pl_grouped, pandas_seconds, polars_seconds


# ---------------------------------------------------------------------------
# 6. Machine learning exploration
# ---------------------------------------------------------------------------

def machine_learning_exploration(df: pd.DataFrame) -> dict:
    """
    Train a beginner-friendly Linear Regression model to predict wine quality.

    The merged dataset includes the categorical column 'type'. Machine-learning
    models such as LinearRegression require numeric inputs, so Pandas
    get_dummies() converts wine type into a numeric indicator variable.
    """
    print("\n" + "=" * 72)
    print("5. MACHINE LEARNING EXPLORATION")
    print("=" * 72)

    # Target/output.
    y = df["quality"]

    # Inputs/features.
    # Convert the categorical 'type' variable into a numeric dummy variable.
    X = pd.get_dummies(
        df.drop(columns=["quality"]),
        columns=["type"],
        drop_first=True,
        dtype=int,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Baseline:
    # predict the training-set mean quality for every test observation.
    baseline_predictions = np.full(len(y_test), y_train.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Model input features: {list(X.columns)}")
    print("Model output/target: quality")
    print(f"\nLinear Regression MAE: {mae:.4f}")
    print(f"Linear Regression R^2: {r2:.4f}")
    print(f"Mean-prediction baseline MAE: {baseline_mae:.4f}")

    if mae < baseline_mae:
        print("The Linear Regression model beats the simple mean baseline on MAE.")
    else:
        print("The Linear Regression model does not beat the simple mean baseline on MAE.")

    coefficient_table = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "coefficient": model.coef_,
                "abs_coefficient": np.abs(model.coef_),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
    )

    print("\nLinear regression coefficients:")
    print(coefficient_table.round(4).to_string(index=False))

    coefficient_table.to_csv(
        OUTPUT_DIR / "linear_regression_coefficients.csv",
        index=False,
    )

    return {
        "mae": float(mae),
        "r2": float(r2),
        "baseline_mae": float(baseline_mae),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }


# ---------------------------------------------------------------------------
# 7. Save a compact summary for the README
# ---------------------------------------------------------------------------

def save_summary(
    df: pd.DataFrame,
    high_quality: pd.DataFrame,
    grouped_by_type: pd.DataFrame,
    grouped_by_quality: pd.DataFrame,
    ml_results: dict,
    pandas_seconds: float,
    polars_seconds: float,
) -> None:
    """Save the main results so they can be copied into README findings."""
    quality_min = int(df["quality"].min())
    quality_max = int(df["quality"].max())
    most_common_quality = int(df["quality"].mode().iloc[0])
    mean_alcohol = float(df["alcohol"].mean())

    type_counts = df["type"].value_counts().sort_index()
    high_quality_type_counts = high_quality["type"].value_counts().sort_index()

    high_quality_rate = len(high_quality) / len(df) * 100

    lines = [
        "IDS 706 Data Analysis Summary",
        "=" * 48,
        "",
        "Dataset Inspection",
        f"Rows: {len(df)}",
        f"Columns: {df.shape[1]}",
        f"Missing values total: {int(df.isna().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        f"Observed quality range: {quality_min} to {quality_max}",
        f"Most common quality score: {most_common_quality}",
        f"Mean alcohol content: {mean_alcohol:.3f}",
        "",
        "Wine Type Counts",
        type_counts.to_string(),
        "",
        "Filtering",
        f"High-quality wines (quality >= 7): {len(high_quality)}",
        f"High-quality share of dataset: {high_quality_rate:.2f}%",
        "High-quality wines by type:",
        high_quality_type_counts.to_string(),
        "",
        "Grouped Summary by Wine Type",
        grouped_by_type.round(3).to_string(index=False),
        "",
        "Grouped Summary by Quality",
        grouped_by_quality.round(3).to_string(index=False),
        "",
        "Machine Learning",
        f"Linear Regression MAE: {ml_results['mae']:.4f}",
        f"Linear Regression R^2: {ml_results['r2']:.4f}",
        f"Baseline MAE: {ml_results['baseline_mae']:.4f}",
        "",
        "Performance Comparison",
        f"Pandas elapsed time: {pandas_seconds:.6f} seconds",
        f"Polars elapsed time: {polars_seconds:.6f} seconds",
        "",
        "Visualization Files",
        "- quality_distribution_by_type.png",
        "- alcohol_by_quality.png",
    ]

    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved summary for README: {summary_path}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_with_pandas()

    inspect_data(df)

    (
        high_quality,
        grouped_by_type,
        grouped_by_quality,
        _grouped_by_type_quality,
    ) = filter_and_group_pandas(df)

    create_visualizations(df)

    _, pandas_seconds, polars_seconds = polars_analysis_and_timing()

    ml_results = machine_learning_exploration(df)

    save_summary(
        df=df,
        high_quality=high_quality,
        grouped_by_type=grouped_by_type,
        grouped_by_quality=grouped_by_quality,
        ml_results=ml_results,
        pandas_seconds=pandas_seconds,
        polars_seconds=polars_seconds,
    )

    print("\n" + "=" * 72)
    print("PROJECT RUN COMPLETED SUCCESSFULLY")
    print("=" * 72)
    print("Check the outputs folder for:")
    print("- summary.txt")
    print("- quality_distribution_by_type.png")
    print("- alcohol_by_quality.png")
    print("- grouped_by_type.csv")
    print("- grouped_by_quality.csv")
    print("- grouped_by_type_quality.csv")
    print("- linear_regression_coefficients.csv")
    print("\nUse summary.txt and the plots to finalize the README findings.")


if __name__ == "__main__":
    main()
