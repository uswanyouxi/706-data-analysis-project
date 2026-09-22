import analysis


def test_main_end_to_end(tmp_path, monkeypatch, capsys):
    """Run the complete analysis pipeline and verify its main outputs."""

    # Keep test outputs separate from the real project outputs folder.
    monkeypatch.setattr(analysis, "OUTPUT_DIR", tmp_path)

    # Run the complete workflow.
    analysis.main()

    # Capture terminal output.
    captured = capsys.readouterr()

    expected_files = [
        "summary.txt",
        "quality_distribution_by_type.png",
        "alcohol_by_quality.png",
        "grouped_by_type.csv",
        "grouped_by_quality.csv",
        "grouped_by_type_quality.csv",
        "linear_regression_coefficients.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()

    assert "PROJECT RUN COMPLETED SUCCESSFULLY" in captured.out

    summary_text = (tmp_path / "summary.txt").read_text(encoding="utf-8")

    assert "Dataset Inspection" in summary_text
    assert "Machine Learning" in summary_text