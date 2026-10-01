"""
IDS 706 - Data Analysis Project

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
# Project configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/wine_quality_merged.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

HIGH_QUALITY_THRESHOLD = 7
BENCHMARK_REPEATS = 50
OUTLIER_COLUMNS = [
    "alcohol",
    "volatile acidity",
    "residual sugar",
    "chlorides",
    "sulphates",
]

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
# Shared helpers
# ---------------------------------------------------------------------------


def print_section(title: str) -> None:
    """Print a consistent section heading."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if expected columns are missing."""
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "The dataset is missing expected columns: "
            + ", ".join(sorted(missing_columns))
        )


def save_dataframe(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame in the outputs directory and return its path."""
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. Import the dataset
# ---------------------------------------------------------------------------


def load_with_pandas() -> pd.DataFrame:
    """Load the merged wine-quality dataset with Pandas."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place wine_quality_merged.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    validate_required_columns(df)
    return df


# ---------------------------------------------------------------------------
# 2. Inspect the data
# ---------------------------------------------------------------------------


def inspect_data(df: pd.DataFrame) -> None:
    """Print the main inspection results required by the assignment."""
    print_section("1. DATA INSPECTION")

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


def summarize_iqr_outliers(
    df: pd.DataFrame,
    columns: list[str] = OUTLIER_COLUMNS,
) -> pd.DataFrame:
    """Summarize potential outliers using the 1.5 * IQR rule."""
    rows = []

    for column in columns:
        q1 = float(df[column].quantile(0.25))
        q3 = float(df[column].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(
            ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        )

        rows.append(
            {
                "feature": column,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_share_pct": outlier_count / len(df) * 100,
            }
        )

    return pd.DataFrame(rows)


def inspect_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Document missing values, duplicates, and potential numeric outliers."""
    outlier_summary = summarize_iqr_outliers(df)

    print("\nData-quality decisions:")
    print("- Missing values: none were found in the dataset.")
    print(
        "- Potential outliers are flagged with the 1.5*IQR rule for selected "
        "numeric features."
    )
    print(
        "- Outliers are retained because unusual chemical measurements may be "
        "valid wines rather than data-entry errors."
    )
    print(
        "- Duplicate rows are reported but retained to avoid changing the source data."
    )

    print("\nIQR outlier summary:")
    print(outlier_summary.round(3).to_string(index=False))

    save_dataframe(outlier_summary, "outlier_summary.csv")
    return outlier_summary


# ---------------------------------------------------------------------------
# 3. Filtering and grouping with Pandas
# ---------------------------------------------------------------------------


def get_high_quality_wines(df: pd.DataFrame) -> pd.DataFrame:
    """Return wines whose quality score meets the project threshold."""
    return df[df["quality"] >= HIGH_QUALITY_THRESHOLD].copy()


def group_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize overall differences between red and white wines."""
    return (
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


def group_by_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected measurements across quality levels."""
    return (
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


def group_by_type_and_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Compare red and white wines within each quality level."""
    return (
        df.groupby(["type", "quality"])
        .agg(
            count=("quality", "size"),
            mean_alcohol=("alcohol", "mean"),
        )
        .reset_index()
        .sort_values(["type", "quality"])
    )


def save_grouped_tables(
    grouped_by_type: pd.DataFrame,
    grouped_by_quality: pd.DataFrame,
    grouped_by_type_quality: pd.DataFrame,
) -> None:
    """Save grouped analysis tables for later inspection."""
    save_dataframe(grouped_by_type, "grouped_by_type.csv")
    save_dataframe(grouped_by_quality, "grouped_by_quality.csv")
    save_dataframe(grouped_by_type_quality, "grouped_by_type_quality.csv")


def filter_and_group_pandas(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the filtered subset and grouped Pandas summaries."""
    print_section("2. FILTERING AND GROUPING WITH PANDAS")

    high_quality = get_high_quality_wines(df)
    grouped_type = group_by_type(df)
    grouped_quality = group_by_quality(df)
    grouped_type_quality = group_by_type_and_quality(df)

    print(f"\nWines with quality >= {HIGH_QUALITY_THRESHOLD}: {len(high_quality)}")
    print(high_quality.head())

    print("\nHigh-quality wines by type:")
    print(high_quality["type"].value_counts())

    print("\nGrouped summary by wine type:")
    print(grouped_type.round(3))

    print("\nGrouped summary by quality:")
    print(grouped_quality.round(3))

    print("\nGrouped summary by wine type and quality:")
    print(grouped_type_quality.round(3))

    save_grouped_tables(grouped_type, grouped_quality, grouped_type_quality)

    return high_quality, grouped_type, grouped_quality, grouped_type_quality


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------


def plot_quality_distribution_by_type(df: pd.DataFrame) -> Path:
    """Create a bar chart of quality-score counts by wine type."""
    quality_type_counts = (
        df.groupby(["quality", "type"]).size().unstack(fill_value=0).sort_index()
    )

    qualities = quality_type_counts.index.to_numpy()
    wine_types = list(quality_type_counts.columns)
    x = np.arange(len(qualities))
    width = 0.8 / len(wine_types)

    plt.figure(figsize=(9, 5))

    for i, wine_type in enumerate(wine_types):
        offset = (i - (len(wine_types) - 1) / 2) * width
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

    path = OUTPUT_DIR / "quality_distribution_by_type.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_alcohol_by_quality(df: pd.DataFrame) -> Path:
    """Create a box plot of alcohol content across quality levels."""
    qualities = sorted(df["quality"].unique())
    alcohol_groups = [
        df.loc[df["quality"] == quality, "alcohol"].to_numpy() for quality in qualities
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

    path = OUTPUT_DIR / "alcohol_by_quality.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def create_visualizations(df: pd.DataFrame) -> None:
    """Create and save the project visualizations."""
    print_section("3. VISUALIZATION")

    path1 = plot_quality_distribution_by_type(df)
    path2 = plot_alcohol_by_quality(df)

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


def run_pandas_benchmark(repeats: int = BENCHMARK_REPEATS) -> float:
    """Time a small Pandas read/filter/group workflow."""
    start = perf_counter()

    for _ in range(repeats):
        pd_temp = pd.read_csv(DATA_PATH)
        (
            pd_temp[pd_temp["quality"] >= HIGH_QUALITY_THRESHOLD]
            .groupby(["type", "quality"])["alcohol"]
            .mean()
        )

    return perf_counter() - start


def run_polars_benchmark(repeats: int = BENCHMARK_REPEATS) -> float:
    """Time the equivalent Polars read/filter/group workflow."""
    start = perf_counter()

    for _ in range(repeats):
        pl_temp = pl.read_csv(DATA_PATH)
        (
            pl_temp.filter(pl.col("quality") >= HIGH_QUALITY_THRESHOLD)
            .group_by(["type", "quality"])
            .agg(pl.col("alcohol").mean().alias("mean_alcohol"))
        )

    return perf_counter() - start


def print_benchmark_result(
    pandas_seconds: float,
    polars_seconds: float,
    repeats: int,
) -> None:
    """Print the benchmark comparison in a consistent format."""
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


def polars_analysis_and_timing() -> tuple[pl.DataFrame, float, float]:
    """Repeat key analysis in Polars and compare simple workflow timing."""
    print_section("4. POLARS ANALYSIS AND PANDAS VS POLARS PERFORMANCE")

    pl_df = pl.read_csv(DATA_PATH)
    pl_high_quality = pl_df.filter(pl.col("quality") >= HIGH_QUALITY_THRESHOLD)

    pl_grouped = (
        pl_df.group_by(["type", "quality"])
        .agg(
            pl.len().alias("count"),
            pl.col("alcohol").mean().alias("mean_alcohol"),
            pl.col("volatile acidity").mean().alias("mean_volatile_acidity"),
            pl.col("sulphates").mean().alias("mean_sulphates"),
        )
        .sort(["type", "quality"])
    )

    print("\nPolars high-quality subset:")
    print(pl_high_quality.head())

    print("\nPolars grouped summary by type and quality:")
    print(pl_grouped)

    pandas_seconds = run_pandas_benchmark()
    polars_seconds = run_polars_benchmark()

    print_benchmark_result(
        pandas_seconds=pandas_seconds,
        polars_seconds=polars_seconds,
        repeats=BENCHMARK_REPEATS,
    )

    return pl_grouped, pandas_seconds, polars_seconds


# ---------------------------------------------------------------------------
# 6. Machine learning exploration
# ---------------------------------------------------------------------------


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create numeric model features and the target variable."""
    y = df["quality"]

    X = pd.get_dummies(
        df.drop(columns=["quality"]),
        columns=["type"],
        drop_first=True,
        dtype=int,
    )

    return X, y


def build_coefficient_table(
    feature_names: pd.Index,
    coefficients: np.ndarray,
) -> pd.DataFrame:
    """Create a coefficient table ordered by absolute coefficient size."""
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        )
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
    )


def machine_learning_exploration(df: pd.DataFrame) -> dict:
    """Train a simple Linear Regression model to predict wine quality."""
    print_section("5. MACHINE LEARNING EXPLORATION")

    X, y = prepare_model_data(df)

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
        print(
            "The Linear Regression model does not beat the simple mean baseline on MAE."
        )

    coefficient_table = build_coefficient_table(X.columns, model.coef_)

    print("\nLinear regression coefficients:")
    print(coefficient_table.round(4).to_string(index=False))

    save_dataframe(
        coefficient_table,
        "linear_regression_coefficients.csv",
    )

    return {
        "mae": float(mae),
        "r2": float(r2),
        "baseline_mae": float(baseline_mae),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------------------
# 7. Save a compact summary for the README
# ---------------------------------------------------------------------------


def save_summary(
    df: pd.DataFrame,
    high_quality: pd.DataFrame,
    grouped_by_type: pd.DataFrame,
    grouped_by_quality: pd.DataFrame,
    outlier_summary: pd.DataFrame,
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
        "Data Quality Treatment",
        "Missing values: none found",
        "Potential outliers were identified with the 1.5*IQR rule and retained.",
        "Outlier summary:",
        outlier_summary.round(3).to_string(index=False),
        "",
        "Wine Type Counts",
        type_counts.to_string(),
        "",
        "Filtering",
        f"High-quality wines (quality >= {HIGH_QUALITY_THRESHOLD}): {len(high_quality)}",
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


def print_completion_message() -> None:
    """Print the expected project completion message and output files."""
    print_section("PROJECT RUN COMPLETED SUCCESSFULLY")
    print("Check the outputs folder for:")
    print("- summary.txt")
    print("- quality_distribution_by_type.png")
    print("- alcohol_by_quality.png")
    print("- grouped_by_type.csv")
    print("- grouped_by_quality.csv")
    print("- grouped_by_type_quality.csv")
    print("- outlier_summary.csv")
    print("- linear_regression_coefficients.csv")
    print("\nUse summary.txt and the plots to finalize the README findings.")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete data-analysis workflow."""
    df = load_with_pandas()

    inspect_data(df)
    outlier_summary = inspect_data_quality(df)

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
        outlier_summary=outlier_summary,
        ml_results=ml_results,
        pandas_seconds=pandas_seconds,
        polars_seconds=polars_seconds,
    )

    print_completion_message()


if __name__ == "__main__":
    main()
