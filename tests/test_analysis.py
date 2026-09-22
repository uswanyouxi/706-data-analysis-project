import pandas as pd
import pytest

import analysis


def make_sample_dataframe():
    """Create a small valid wine dataset for unit testing."""
    return pd.DataFrame(
        {
            "fixed acidity": [7.0, 8.0, 6.5, 7.5],
            "volatile acidity": [0.5, 0.3, 0.4, 0.2],
            "citric acid": [0.2, 0.4, 0.3, 0.5],
            "residual sugar": [2.0, 2.5, 1.8, 3.0],
            "chlorides": [0.05, 0.04, 0.06, 0.03],
            "free sulfur dioxide": [10.0, 20.0, 15.0, 25.0],
            "total sulfur dioxide": [30.0, 50.0, 40.0, 60.0],
            "density": [0.995, 0.993, 0.996, 0.992],
            "pH": [3.2, 3.1, 3.3, 3.0],
            "sulphates": [0.5, 0.6, 0.55, 0.7],
            "alcohol": [9.5, 11.0, 10.0, 12.0],
            "quality": [5, 7, 6, 8],
            "type": ["red", "white", "red", "white"],
        }
    )


def test_load_with_pandas_success(tmp_path, monkeypatch):
    """A valid CSV should load successfully."""
    sample_df = make_sample_dataframe()

    test_file = tmp_path / "wine_quality_merged.csv"
    sample_df.to_csv(test_file, index=False)

    monkeypatch.setattr(analysis, "DATA_PATH", test_file)

    loaded_df = analysis.load_with_pandas()

    assert len(loaded_df) == 4
    assert set(analysis.REQUIRED_COLUMNS).issubset(loaded_df.columns)


def test_load_with_pandas_missing_file(tmp_path, monkeypatch):
    """Loading a nonexistent file should raise FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.csv"

    monkeypatch.setattr(analysis, "DATA_PATH", missing_file)

    with pytest.raises(FileNotFoundError):
        analysis.load_with_pandas()


def test_load_with_pandas_missing_column(tmp_path, monkeypatch):
    """A dataset missing a required column should raise ValueError."""
    sample_df = make_sample_dataframe().drop(columns=["quality"])

    test_file = tmp_path / "bad_wine_data.csv"
    sample_df.to_csv(test_file, index=False)

    monkeypatch.setattr(analysis, "DATA_PATH", test_file)

    with pytest.raises(ValueError):
        analysis.load_with_pandas()


def test_filter_and_group_pandas(tmp_path, monkeypatch):
    """Filtering and grouping should return the expected results."""
    sample_df = make_sample_dataframe()

    monkeypatch.setattr(analysis, "OUTPUT_DIR", tmp_path)

    (
        high_quality,
        grouped_by_type,
        grouped_by_quality,
        grouped_by_type_quality,
    ) = analysis.filter_and_group_pandas(sample_df)

    assert len(high_quality) == 2
    assert (high_quality["quality"] >= 7).all()

    assert grouped_by_type["count"].sum() == 4
    assert grouped_by_quality["count"].sum() == 4
    assert grouped_by_type_quality["count"].sum() == 4




def test_machine_learning_exploration(tmp_path, monkeypatch):
    """Machine-learning workflow should return valid metrics and save coefficients."""
    sample_df = pd.concat(
        [make_sample_dataframe()] * 10,
        ignore_index=True,
    )

    monkeypatch.setattr(analysis, "OUTPUT_DIR", tmp_path)

    results = analysis.machine_learning_exploration(sample_df)

    expected_keys = {
        "mae",
        "r2",
        "baseline_mae",
        "n_train",
        "n_test",
    }

    assert expected_keys.issubset(results.keys())
    assert results["n_train"] + results["n_test"] == len(sample_df)

    assert results["mae"] >= 0
    assert results["baseline_mae"] >= 0
    assert isinstance(results["r2"], float)

    coefficients_file = tmp_path / "linear_regression_coefficients.csv"
    assert coefficients_file.exists()