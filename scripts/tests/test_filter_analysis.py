"""Tests for filter_analysis.py."""
import argparse
import json

import pytest

from filter_analysis import read_jsonl, passes_filter, filter_data, copy_pdb_files


def _args(**overrides):
    defaults = dict(
        pr_min_helices=None, pr_max_helices=None,
        pr_min_strands=None, pr_max_strands=None,
        pr_min_total_ss=None, pr_max_total_ss=None,
        pr_min_rog=None, pr_max_rog=None,
        pr_min_intface_bsa=None,
        pr_min_intface_shpcomp=None,
        pr_min_intface_hbonds=None,
        pr_max_intface_unsat_hbonds=None,
        pr_max_intface_deltag=None,
        pr_max_intface_deltagtobsa=None,
        pr_max_surfhphobics=None,
        pr_max_sap=None,
        pr_max_sap_complex=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _entry(description="a", **overrides):
    base = dict(
        description=description,
        pr_helices=3,
        pr_strands=0,
        pr_total_ss=3,
        pr_RoG=15.0,
        pr_intface_BSA=1000.0,
        pr_intface_shpcomp=0.7,
        pr_intface_hbonds=5,
        pr_intface_unsat_hbonds=1,
        pr_intface_deltaG=-20.0,
        pr_intface_deltaGtoBSA=-0.02,
        pr_surfhphobics=10.0,
        pr_SAP=30.0,
        pr_SAP_complex=25.0,
    )
    base.update(overrides)
    return base


class TestReadJsonl:
    def test_reads_valid_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_entry("a")) + "\n")
            f.write(json.dumps(_entry("b")) + "\n")
        data = read_jsonl(str(path))
        assert len(data) == 2

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(_entry("a")) + "\n")
            f.write("\n")
            f.write(json.dumps(_entry("b")) + "\n")
        data = read_jsonl(str(path))
        assert len(data) == 2

    def test_skips_malformed_line_continues(self, tmp_path):
        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            f.write("{not valid json\n")
            f.write(json.dumps(_entry("a")) + "\n")
        data = read_jsonl(str(path))
        assert len(data) == 1

    def test_nonexistent_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            read_jsonl(str(tmp_path / "does_not_exist.jsonl"))


class TestPassesFilter:
    def test_min_only(self):
        assert passes_filter(5, 3, None, "metric") == (True, None)
        passed, reason = passes_filter(2, 3, None, "metric")
        assert passed is False
        assert "metric 2 < 3" == reason

    def test_max_only(self):
        assert passes_filter(5, None, 10, "metric") == (True, None)
        passed, reason = passes_filter(15, None, 10, "metric")
        assert passed is False
        assert "metric 15 > 10" == reason

    def test_min_and_max_in_range(self):
        assert passes_filter(5, 1, 10, "metric") == (True, None)

    def test_min_and_max_out_of_range(self):
        passed, reason = passes_filter(15, 1, 10, "metric")
        assert passed is False
        assert "not in range [1, 10]" in reason

    def test_no_bounds_always_passes(self):
        assert passes_filter(5, None, None, "metric") == (True, None)


class TestFilterData:
    def test_no_filters_all_pass(self):
        data = [_entry("a"), _entry("b")]
        passed, _ = filter_data(data, _args())
        assert passed == ["a", "b"]

    def test_missing_description_rejected(self):
        entry = _entry()
        del entry["description"]
        passed, _ = filter_data([entry], _args())
        assert passed == []

    def test_min_helices_filter(self):
        data = [_entry("a", pr_helices=1)]
        passed, _ = filter_data(data, _args(pr_min_helices=2))
        assert passed == []

    def test_rog_range_filter(self):
        data = [_entry("a", pr_RoG=25.0)]
        passed, _ = filter_data(data, _args(pr_min_rog=5.0, pr_max_rog=20.0))
        assert passed == []

    def test_missing_metric_when_filter_active_rejects(self):
        entry = _entry("a")
        del entry["pr_helices"]
        passed, _ = filter_data([entry], _args(pr_min_helices=1))
        assert passed == []

    def test_missing_metric_when_filter_inactive_still_passes(self):
        entry = _entry("a")
        del entry["pr_helices"]
        passed, _ = filter_data([entry], _args())
        assert passed == ["a"]

    def test_invalid_type_value_rejected(self):
        data = [_entry("a", pr_helices="not_a_number")]
        passed, _ = filter_data(data, _args(pr_min_helices=1))
        assert passed == []

    def test_unsat_hbonds_max_only_filter(self):
        data = [_entry("a", pr_intface_unsat_hbonds=5)]
        passed, _ = filter_data(data, _args(pr_max_intface_unsat_hbonds=2))
        assert passed == []

    def test_multiple_filters_all_must_pass(self):
        data = [_entry("a", pr_helices=1, pr_SAP=100.0)]
        passed, _ = filter_data(data, _args(pr_min_helices=2, pr_max_sap=50.0))
        assert passed == []


class TestCopyPdbFiles:
    def test_missing_pdb_skipped(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        copied = copy_pdb_files(["nonexistent"], str(pdb_dir), str(out_dir))
        assert copied == []

    def test_existing_pdb_copied(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        (pdb_dir / "design_1.pdb").write_text("ATOM\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        copied = copy_pdb_files(["design_1"], str(pdb_dir), str(out_dir))
        assert copied == ["design_1"]
        assert (out_dir / "design_1.pdb").exists()
