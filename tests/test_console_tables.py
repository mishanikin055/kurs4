"""Saved comparisons can be inspected without ML dependencies or inference."""

import csv

import pytest

from scripts.export_tables import print_table


def fixture_table(tmp_path, rows):
    path = tmp_path / "comparison.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_id", "status", "mAP", "rss_peak_GiB"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_console_preserves_four_participants_and_missing_values(tmp_path, capsys):
    path = fixture_table(
        tmp_path,
        [
            {"model_id": "a", "status": "complete", "mAP": ".42", "rss_peak_GiB": "1.2"},
            {"model_id": "b", "status": "failed"},
            {"model_id": "c", "status": "not_run"},
            {"model_id": "d", "status": "oom"},
        ],
    )
    print_table(path)
    output = capsys.readouterr().out
    for status in ["complete", "failed", "not_run", "oom"]:
        assert status in output
    assert "0.42" in output and "—" in output
    assert "rss_peak_GiB" not in output
    print_table(path, full=True)
    assert "rss_peak_GiB" in capsys.readouterr().out


def test_console_rejects_missing_or_duplicate_participants(tmp_path):
    path = fixture_table(tmp_path, [{"model_id": "a", "status": "complete"}] * 4)
    with pytest.raises(ValueError, match="Duplicate"):
        print_table(path)
    path = fixture_table(tmp_path, [{"model_id": "a", "status": "complete"}])
    with pytest.raises(ValueError, match="four"):
        print_table(path)
