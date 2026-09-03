"""Tests for filter_boltz.py."""
import argparse
import json

import pytest

from filter_boltz import (
    read_data_from_directory,
    load_unbound_metadata,
    filter_data,
    UNBOUND_METRIC_KEYS,
)


def _args(**overrides):
    """Namespace with all boltz_* threshold args defaulted to None."""
    defaults = dict(
        boltz_max_rmsd_overall=None,
        boltz_max_rmsd_binder=None,
        boltz_max_rmsd_target=None,
        boltz_min_conf_score=None,
        boltz_min_ptm=None,
        boltz_min_ptm_binder=None,
        boltz_min_ptm_target=None,
        boltz_min_iptm=None,
        boltz_min_plddt=None,
        boltz_min_iplddt=None,
        boltz_max_pde=None,
        boltz_max_ipde=None,
        boltz_min_ipSAE_min=None,
        boltz_min_LIS=None,
        boltz_min_pDockQ2_min=None,
        boltz_max_pae_interaction=None,
        boltz_max_unbound_rmsd=None,
        boltz_min_unbound_conf_score=None,
        boltz_min_unbound_ptm=None,
        boltz_min_unbound_plddt=None,
        boltz_max_unbound_pde=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _entry(description="fold_1_seq_1_boltzpred", **overrides):
    base = dict(
        description=description,
        boltz_rmsd_overall=1.0,
        boltz_rmsd_binder=1.0,
        boltz_rmsd_target=1.0,
        boltz_conf_score=0.9,
        boltz_ptm=0.9,
        boltz_ptm_binder=0.9,
        boltz_ptm_target=0.9,
        boltz_iptm=0.9,
        boltz_plddt=90.0,
        boltz_iplddt=90.0,
        boltz_pde=1.0,
        boltz_ipde=1.0,
        ipSAE_min=0.8,
        LIS=0.8,
        pDockQ2_min=0.8,
        boltz_pae_interaction=5.0,
    )
    base.update(overrides)
    return base


class TestReadDataFromDirectory:
    def test_no_matching_files_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            read_data_from_directory(str(tmp_path), "*.json")

    def test_reads_single_json_object_and_injects_metadata(self, tmp_path):
        (tmp_path / "fold_3_seq_7_boltzpred.json").write_text(json.dumps({"boltz_ptm": 0.9}))
        data = read_data_from_directory(str(tmp_path))
        assert len(data) == 1
        assert data[0]["fold_id"] == 3
        assert data[0]["seq_id"] == 7
        assert data[0]["description"] == "fold_3_seq_7_boltzpred"

    def test_skips_unbound_prediction_files(self, tmp_path):
        (tmp_path / "fold_1_seq_1_boltzpred.json").write_text(json.dumps({"boltz_ptm": 0.9}))
        (tmp_path / "fold_1_seq_1_unbound_boltzpred.json").write_text(json.dumps({"boltz_unbound_ptm": 0.5}))
        data = read_data_from_directory(str(tmp_path))
        assert len(data) == 1
        assert data[0]["description"] == "fold_1_seq_1_boltzpred"

    def test_skips_invalid_filename_format(self, tmp_path):
        (tmp_path / "not_a_valid_name.json").write_text(json.dumps({"boltz_ptm": 0.9}))
        data = read_data_from_directory(str(tmp_path))
        assert data == []

    def test_json_array_file(self, tmp_path):
        (tmp_path / "fold_1_seq_1_boltzpred.json").write_text(
            json.dumps([{"boltz_ptm": 0.9}, {"boltz_ptm": 0.5}])
        )
        data = read_data_from_directory(str(tmp_path))
        assert len(data) == 2
        assert all(e["fold_id"] == 1 and e["seq_id"] == 1 for e in data)

    def test_skips_empty_file(self, tmp_path):
        (tmp_path / "fold_1_seq_1_boltzpred.json").write_text("")
        data = read_data_from_directory(str(tmp_path))
        assert data == []


class TestLoadUnboundMetadata:
    def test_loads_and_filters_to_unbound_keys_only(self, tmp_path):
        (tmp_path / "fold_2_seq_4_unbound_boltzpred.json").write_text(json.dumps({
            "boltz_unbound_rmsd": 1.5,
            "boltz_unbound_conf_score": 0.7,
            "description": "should_not_leak_through",
        }))
        lookup = load_unbound_metadata(str(tmp_path))
        assert (2, 4) in lookup
        entry = lookup[(2, 4)]
        assert entry == {"boltz_unbound_rmsd": 1.5, "boltz_unbound_conf_score": 0.7}
        assert "description" not in entry

    def test_no_matching_files_returns_empty_dict(self, tmp_path):
        lookup = load_unbound_metadata(str(tmp_path))
        assert lookup == {}

    def test_only_keeps_keys_in_unbound_metric_keys_constant(self):
        assert UNBOUND_METRIC_KEYS == (
            'boltz_unbound_rmsd', 'boltz_unbound_conf_score', 'boltz_unbound_ptm',
            'boltz_unbound_plddt', 'boltz_unbound_pde'
        )


class TestFilterData:
    def test_no_filters_all_pass(self):
        data = [_entry("a"), _entry("b")]
        passed, _ = filter_data(data, _args())
        assert passed == ["a", "b"]

    def test_rmsd_above_threshold_rejected(self):
        data = [_entry("a", boltz_rmsd_overall=5.0)]
        passed, _ = filter_data(data, _args(boltz_max_rmsd_overall=2.0))
        assert passed == []

    def test_ptm_below_threshold_rejected(self):
        data = [_entry("a", boltz_ptm=0.3)]
        passed, _ = filter_data(data, _args(boltz_min_ptm=0.5))
        assert passed == []

    def test_missing_key_defaults_used_not_keyerror(self):
        # entry.get(..., default) is used throughout filter_data, so a missing metric
        # falls back to a sentinel default (1000 for max-checks, 0 for min-checks)
        # rather than raising/rejecting.
        entry = _entry("a")
        del entry["boltz_ptm"]
        passed, _ = filter_data([entry], _args(boltz_min_ptm=0.5))
        assert passed == []  # default 0 < 0.5 -> rejected, but not via KeyError path

    def test_zero_threshold_is_not_silently_skipped(self):
        # boltz_max_rmsd_overall=0.0 must still be applied (uses `is not None`, not truthy check)
        data = [_entry("a", boltz_rmsd_overall=5.0)]
        passed, _ = filter_data(data, _args(boltz_max_rmsd_overall=0.0))
        assert passed == []

    def test_unbound_metric_filter_applies_after_merge(self):
        data = [_entry("a", boltz_unbound_rmsd=3.0)]
        passed, _ = filter_data(data, _args(boltz_max_unbound_rmsd=1.0))
        assert passed == []

    def test_multiple_thresholds_all_must_pass(self):
        data = [_entry("a", boltz_ptm=0.9, boltz_iptm=0.2)]
        passed, _ = filter_data(data, _args(boltz_min_ptm=0.5, boltz_min_iptm=0.5))
        assert passed == []

    def test_entries_without_description_key_raise_and_are_skipped(self):
        entry = _entry("a")
        del entry["description"]
        passed, entries = filter_data([entry], _args())
        # KeyError on entry['description'] in the "passed" branch is caught -> design dropped
        assert passed == []
        assert entries == []
