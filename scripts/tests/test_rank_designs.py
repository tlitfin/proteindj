"""Tests for rank_designs.py."""
import os

import pandas as pd
import pytest

from rank_designs import rank_designs, generate_pdb_filename_from_row, copy_and_rename_pdbs


class TestRankDesigns:
    def test_missing_metric_exits(self):
        df = pd.DataFrame({"fold_id": [1, 2], "score": [0.1, 0.2]})
        with pytest.raises(SystemExit):
            rank_designs(df, "nonexistent_metric")

    def test_drops_nan_metric_rows(self):
        df = pd.DataFrame({
            "fold_id": [1, 2, 3],
            "seq_id": [1, 1, 1],
            "af2_pae_interaction": [5.0, None, 2.0],
        })
        result = rank_designs(df, "af2_pae_interaction")
        assert len(result) == 2
        assert set(result["fold_id"]) == {1, 3}

    def test_all_nan_metric_exits(self):
        df = pd.DataFrame({
            "fold_id": [1, 2],
            "af2_pae_interaction": [None, None],
        })
        with pytest.raises(SystemExit):
            rank_designs(df, "af2_pae_interaction")

    def test_lower_is_better_metric_sorts_ascending(self):
        df = pd.DataFrame({
            "fold_id": [1, 2, 3],
            "af2_pae_interaction": [5.0, 1.0, 3.0],
        })
        result = rank_designs(df, "af2_pae_interaction")
        assert list(result["fold_id"]) == [2, 3, 1]

    def test_higher_is_better_metric_sorts_descending(self):
        df = pd.DataFrame({
            "fold_id": [1, 2, 3],
            "boltz_iptm": [0.5, 0.9, 0.2],
        })
        result = rank_designs(df, "boltz_iptm")
        assert list(result["fold_id"]) == [2, 1, 3]

    def test_ambiguous_metric_defaults_to_higher_is_better(self):
        df = pd.DataFrame({
            "fold_id": [1, 2, 3],
            "custom_score": [0.5, 0.9, 0.2],
        })
        result = rank_designs(df, "custom_score")
        assert list(result["fold_id"]) == [2, 1, 3]

    def test_rank_column_is_1_indexed_and_first_column(self):
        df = pd.DataFrame({
            "fold_id": [1, 2],
            "af2_pae_interaction": [5.0, 1.0],
        })
        result = rank_designs(df, "af2_pae_interaction")
        assert list(result.columns)[0] == "rank"
        assert list(result["rank"]) == [1, 2]

    def test_max_seqs_per_fold_limits_per_group(self):
        df = pd.DataFrame({
            "fold_id": [1, 1, 1, 2, 2],
            "seq_id": [1, 2, 3, 1, 2],
            "af2_pae_interaction": [1.0, 2.0, 3.0, 1.5, 2.5],
        })
        result = rank_designs(df, "af2_pae_interaction", max_seqs_per_fold=2)
        assert len(result) == 4
        fold1_rows = result[result["fold_id"] == 1]
        assert set(fold1_rows["seq_id"]) == {1, 2}

    def test_max_seqs_per_fold_missing_fold_id_column_warns_and_skips(self):
        df = pd.DataFrame({
            "af2_pae_interaction": [1.0, 2.0, 3.0],
        })
        result = rank_designs(df, "af2_pae_interaction", max_seqs_per_fold=1)
        # No fold_id column -> filter skipped, all rows retained
        assert len(result) == 3

    def test_re_sorted_globally_after_per_fold_filter(self):
        df = pd.DataFrame({
            "fold_id": [1, 1, 2, 2],
            "seq_id": [1, 2, 1, 2],
            "af2_pae_interaction": [1.0, 9.0, 2.0, 3.0],
        })
        result = rank_designs(df, "af2_pae_interaction", max_seqs_per_fold=1)
        # top-1 per fold: fold1 -> seq1 (1.0), fold2 -> seq1 (2.0); global sort ascending
        assert list(result["fold_id"]) == [1, 2]
        assert list(result["rank"]) == [1, 2]


class TestGeneratePdbFilenameFromRow:
    def test_boltz_prediction(self):
        row = pd.Series({"fold_id": 3, "seq_id": 5, "boltz_ptm": 0.8})
        assert generate_pdb_filename_from_row(row) == "fold_3_seq_5_boltzpred.pdb"

    def test_af2_prediction_when_boltz_ptm_missing(self):
        row = pd.Series({"fold_id": 3, "seq_id": 5, "boltz_ptm": None})
        assert generate_pdb_filename_from_row(row) == "fold_3_seq_5_af2pred.pdb"

    def test_af2_prediction_when_boltz_ptm_nan(self):
        row = pd.Series({"fold_id": 1, "seq_id": 2, "boltz_ptm": float("nan")})
        assert generate_pdb_filename_from_row(row) == "fold_1_seq_2_af2pred.pdb"

    def test_af2_prediction_when_no_boltz_column(self):
        row = pd.Series({"fold_id": 1, "seq_id": 2})
        assert generate_pdb_filename_from_row(row) == "fold_1_seq_2_af2pred.pdb"


class TestCopyAndRenamePdbs:
    def test_copies_and_pads_rank_prefix(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        out_dir = tmp_path / "out"

        df = pd.DataFrame({
            "rank": [1, 2],
            "fold_id": [1, 2],
            "seq_id": [1, 1],
            "boltz_ptm": [None, None],
        })
        for _, row in df.iterrows():
            (pdb_dir / generate_pdb_filename_from_row(row)).write_text("ATOM\n")

        copied = copy_and_rename_pdbs(df, str(pdb_dir), str(out_dir))
        assert copied == 2
        assert (out_dir / "1_fold_1_seq_1_af2pred.pdb").exists()
        assert (out_dir / "2_fold_2_seq_1_af2pred.pdb").exists()

    def test_zero_padding_width_scales_with_total_designs(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        out_dir = tmp_path / "out"

        # 10 rows -> padding width 2
        rows = []
        for i in range(1, 11):
            rows.append({"rank": i, "fold_id": i, "seq_id": 1, "boltz_ptm": None})
        df = pd.DataFrame(rows)
        for _, row in df.iterrows():
            (pdb_dir / generate_pdb_filename_from_row(row)).write_text("ATOM\n")

        copy_and_rename_pdbs(df, str(pdb_dir), str(out_dir))
        assert (out_dir / "01_fold_1_seq_1_af2pred.pdb").exists()
        assert (out_dir / "10_fold_10_seq_1_af2pred.pdb").exists()

    def test_missing_pdb_files_are_skipped_not_errored(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        out_dir = tmp_path / "out"

        df = pd.DataFrame({
            "rank": [1],
            "fold_id": [1],
            "seq_id": [1],
            "boltz_ptm": [None],
        })
        # Don't create the expected PDB file
        copied = copy_and_rename_pdbs(df, str(pdb_dir), str(out_dir))
        assert copied == 0
