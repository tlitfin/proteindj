"""Tests for filter_af2.py."""
import argparse
import json

import pytest

from filter_af2 import read_data_from_directory, filter_data, copy_pdb_files


def _args(**overrides):
    """Namespace with all af2_* threshold args defaulted to None."""
    defaults = dict(
        af2_min_iptm=None,
        af2_max_pae_interaction=None,
        af2_max_pae_overall=None,
        af2_max_pae_binder=None,
        af2_max_pae_target=None,
        af2_min_plddt_overall=None,
        af2_min_plddt_binder=None,
        af2_min_plddt_target=None,
        af2_max_rmsd_overall=None,
        af2_max_rmsd_binder_bndaln=None,
        af2_max_rmsd_binder_tgtaln=None,
        af2_max_rmsd_target=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _entry(description="design_1", **overrides):
    base = dict(
        description=description,
        af2_iptm=0.9,
        af2_pae_interaction=5.0,
        af2_pae_overall=5.0,
        af2_pae_binder=5.0,
        af2_pae_target=5.0,
        af2_plddt_overall=90.0,
        af2_plddt_binder=90.0,
        af2_plddt_target=90.0,
        af2_rmsd_overall=1.0,
        af2_rmsd_binder_bndaln=1.0,
        af2_rmsd_binder_tgtaln=1.0,
        af2_rmsd_target=1.0,
    )
    base.update(overrides)
    return base


class TestReadDataFromDirectory:
    def test_no_matching_files_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            read_data_from_directory(str(tmp_path), "*.json")

    def test_reads_json_array_file(self, tmp_path, write_json_files):
        write_json_files({"scores.json": [_entry("a"), _entry("b")]})
        data = read_data_from_directory(str(tmp_path))
        assert len(data) == 2

    def test_reads_single_json_object_file(self, tmp_path, write_json_files):
        write_json_files({"scores.json": _entry("a")})
        data = read_data_from_directory(str(tmp_path))
        assert len(data) == 1

    def test_reads_jsonl_file(self, tmp_path):
        path = tmp_path / "scores.json"
        with open(path, "w") as f:
            f.write(json.dumps(_entry("a")) + "\n")
            f.write(json.dumps(_entry("b")) + "\n")
        data = read_data_from_directory(str(tmp_path))
        assert len(data) == 2

    def test_skips_empty_file(self, tmp_path):
        (tmp_path / "empty.json").write_text("")
        (tmp_path / "valid.json").write_text(json.dumps(_entry("a")))
        data = read_data_from_directory(str(tmp_path))
        assert len(data) == 1

    def test_malformed_json_is_skipped_not_raised(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not valid json")
        (tmp_path / "good.json").write_text(json.dumps(_entry("a")))
        data = read_data_from_directory(str(tmp_path))
        assert len(data) == 1


class TestFilterData:
    def test_missing_description_rejected(self):
        entry = _entry()
        del entry["description"]
        passed, entries = filter_data([entry], _args())
        assert passed == []

    def test_no_filters_all_pass(self):
        data = [_entry("a"), _entry("b")]
        passed, entries = filter_data(data, _args())
        assert passed == ["a", "b"]

    def test_iptm_below_threshold_rejected(self):
        data = [_entry("a", af2_iptm=0.5)]
        passed, _ = filter_data(data, _args(af2_min_iptm=0.7))
        assert passed == []

    def test_iptm_meets_threshold_accepted(self):
        data = [_entry("a", af2_iptm=0.8)]
        passed, _ = filter_data(data, _args(af2_min_iptm=0.7))
        assert passed == ["a"]

    def test_pae_interaction_above_threshold_rejected(self):
        data = [_entry("a", af2_pae_interaction=15.0)]
        passed, _ = filter_data(data, _args(af2_max_pae_interaction=10.0))
        assert passed == []

    def test_plddt_below_threshold_rejected(self):
        data = [_entry("a", af2_plddt_overall=50.0)]
        passed, _ = filter_data(data, _args(af2_min_plddt_overall=70.0))
        assert passed == []

    def test_rmsd_above_threshold_rejected(self):
        data = [_entry("a", af2_rmsd_overall=5.0)]
        passed, _ = filter_data(data, _args(af2_max_rmsd_overall=2.0))
        assert passed == []

    def test_missing_key_when_filter_active_rejects_design(self):
        entry = _entry("a")
        del entry["af2_iptm"]
        passed, _ = filter_data([entry], _args(af2_min_iptm=0.5))
        assert passed == []

    def test_missing_key_when_filter_inactive_still_passes(self):
        entry = _entry("a")
        del entry["af2_iptm"]
        passed, _ = filter_data([entry], _args())
        assert passed == ["a"]

    def test_multiple_failures_all_reported_and_rejected(self):
        data = [_entry("a", af2_iptm=0.1, af2_plddt_overall=10.0)]
        passed, _ = filter_data(data, _args(af2_min_iptm=0.5, af2_min_plddt_overall=50.0))
        assert passed == []

    def test_passed_entries_match_passed_designs(self):
        data = [_entry("a"), _entry("b", af2_iptm=0.1)]
        passed, entries = filter_data(data, _args(af2_min_iptm=0.5))
        assert passed == ["a"]
        assert len(entries) == 1
        assert entries[0]["description"] == "a"


class TestCopyPdbFiles:
    def test_missing_pdb_skipped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "out").mkdir()
        copied = copy_pdb_files(["nonexistent_design"], str(tmp_path / "out"))
        assert copied == []

    def test_existing_pdb_copied(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "design_1.pdb").write_text("ATOM\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        copied = copy_pdb_files(["design_1"], str(out_dir))
        assert copied == ["design_1"]
        assert (out_dir / "design_1.pdb").exists()
