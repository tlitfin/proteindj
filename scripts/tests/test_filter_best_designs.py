"""Tests for filter_best_designs.py."""
import os

import pandas as pd
import pytest

from filter_best_designs import extract_ids_from_pdb, main


class TestExtractIdsFromPdb:
    def test_fold_and_seq_match(self):
        assert extract_ids_from_pdb("design_fold_3_seq_7.pdb") == (3, 7)

    def test_fold_only_match(self):
        assert extract_ids_from_pdb("design_fold_12.pdb") == (12, None)

    def test_no_match_returns_none_none(self):
        assert extract_ids_from_pdb("random_file.pdb") == (None, None)

    def test_uses_basename_not_full_path(self):
        assert extract_ids_from_pdb("/some/dir/fold_1_seq_2.pdb") == (1, 2)

    def test_fold_and_seq_takes_precedence_over_fold_only_pattern(self):
        # fold_(\d+)_seq_(\d+) is tried first
        fold_id, seq_id = extract_ids_from_pdb("fold_5_seq_9_extra.pdb")
        assert (fold_id, seq_id) == (5, 9)


def _write_pdb(directory, filename):
    (directory / filename).write_text("ATOM\n")


class TestMain:
    def test_matches_with_seq_id(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "input.csv"
        pd.DataFrame({
            "fold_id": [1, 1, 2],
            "seq_id": [1, 2, 1],
            "score": [0.1, 0.2, 0.3],
        }).to_csv(csv_path, index=False)

        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        _write_pdb(pdb_dir, "design_fold_1_seq_2.pdb")

        out_csv = tmp_path / "out.csv"
        out_dir = tmp_path / "out_pdbs"

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "filter_best_designs.py",
            "--csv", str(csv_path),
            "--pdb-dir", str(pdb_dir),
            "--output-csv", str(out_csv),
            "--output-dir", str(out_dir),
        ])

        rc = main()
        assert rc == 0

        result = pd.read_csv(out_csv)
        assert len(result) == 1
        assert result.iloc[0]["fold_id"] == 1
        assert result.iloc[0]["seq_id"] == 2
        assert (out_dir / "design_fold_1_seq_2.pdb").exists()

    def test_matches_fold_only_when_csv_lacks_seq_id(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "input.csv"
        pd.DataFrame({
            "fold_id": [1, 2, 3],
            "score": [0.1, 0.2, 0.3],
        }).to_csv(csv_path, index=False)

        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        _write_pdb(pdb_dir, "design_fold_2.pdb")

        out_csv = tmp_path / "out.csv"
        out_dir = tmp_path / "out_pdbs"

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "filter_best_designs.py",
            "--csv", str(csv_path),
            "--pdb-dir", str(pdb_dir),
            "--output-csv", str(out_csv),
            "--output-dir", str(out_dir),
        ])

        rc = main()
        assert rc == 0

        result = pd.read_csv(out_csv)
        assert len(result) == 1
        assert result.iloc[0]["fold_id"] == 2

    def test_mismatch_csv_has_seq_id_pdb_does_not_returns_error(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "input.csv"
        pd.DataFrame({
            "fold_id": [1],
            "seq_id": [1],
            "score": [0.1],
        }).to_csv(csv_path, index=False)

        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        _write_pdb(pdb_dir, "design_fold_1.pdb")  # no seq_id in filename

        out_dir = tmp_path / "out_pdbs"

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "filter_best_designs.py",
            "--csv", str(csv_path),
            "--pdb-dir", str(pdb_dir),
            "--output-csv", str(tmp_path / "out.csv"),
            "--output-dir", str(out_dir),
        ])

        rc = main()
        assert rc == 1

    def test_mismatch_csv_lacks_seq_id_pdb_has_it_returns_error(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "input.csv"
        pd.DataFrame({
            "fold_id": [1],
            "score": [0.1],
        }).to_csv(csv_path, index=False)

        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        _write_pdb(pdb_dir, "design_fold_1_seq_2.pdb")

        out_dir = tmp_path / "out_pdbs"

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "filter_best_designs.py",
            "--csv", str(csv_path),
            "--pdb-dir", str(pdb_dir),
            "--output-csv", str(tmp_path / "out.csv"),
            "--output-dir", str(out_dir),
        ])

        rc = main()
        assert rc == 1

    def test_bad_csv_path_returns_error(self, tmp_path, monkeypatch):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "filter_best_designs.py",
            "--csv", str(tmp_path / "does_not_exist.csv"),
            "--pdb-dir", str(pdb_dir),
            "--output-csv", str(tmp_path / "out.csv"),
            "--output-dir", str(tmp_path / "out_pdbs"),
        ])

        rc = main()
        assert rc == 1
