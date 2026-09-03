"""Tests for filter_seq.py."""
import argparse
import json

import pytest

from filter_seq import (
    extract_designed_sequence,
    calculate_seq_metrics,
    load_json_data,
    filter_by_score,
    filter_by_seq_metrics,
    copy_filtered_designs,
    write_seq_metrics_jsonl,
)


def _args(**overrides):
    defaults = dict(
        seq_min_ext_coef=None,
        seq_max_ext_coef=None,
        seq_min_pi=None,
        seq_max_pi=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestExtractDesignedSequence:
    def test_plain_sequence(self):
        assert extract_designed_sequence("MKTAYIAKQR") == "MKTAYIAKQR"

    def test_fampnn_multi_chain_takes_first_chain(self):
        assert extract_designed_sequence("A:MKTAYI|B:QRSTUV") == "MKTAYI"

    def test_single_chain_with_prefix(self):
        assert extract_designed_sequence("A:MKTAYI") == "MKTAYI"

    def test_empty_string(self):
        assert extract_designed_sequence("") == ""


class TestCalculateSeqMetrics:
    def test_valid_sequence_returns_metrics(self):
        result = calculate_seq_metrics("fold_1_seq_2", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWELVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL")
        assert result is not None
        assert result["description"] == "fold_1_seq_2"
        assert result["fold_id"] == 1
        assert result["seq_id"] == 2
        assert result["seq_length"] > 0
        assert isinstance(result["seq_MW"], int)
        assert isinstance(result["seq_pI"], float)

    def test_no_regex_match_gives_none_ids(self):
        result = calculate_seq_metrics("some_other_name", "MKTAYIAKQRQISFVKSHFSRQ")
        assert result is not None
        assert result["fold_id"] is None
        assert result["seq_id"] is None

    def test_empty_sequence_returns_none(self):
        assert calculate_seq_metrics("fold_1_seq_1", "") is None

    def test_invalid_sequence_returns_none(self):
        # ProteinAnalysis raises on characters outside the standard amino acid alphabet
        # in molecular_weight/molar_extinction_coefficient depending on biopython version;
        # this uses digits which are never valid residues.
        result = calculate_seq_metrics("fold_1_seq_1", "1234567890")
        assert result is None


class TestLoadJsonData:
    def test_loads_mpnn_score_field(self, tmp_path, write_jsonl_file):
        write_jsonl_file("scores.json", [
            {"design": "fold_1_seq_1", "score": 1.23, "sequence": "MKTAYI"},
        ])
        data_map = load_json_data(str(tmp_path), "mpnn")
        assert data_map["fold_1_seq_1"]["score"] == pytest.approx(1.23)
        assert data_map["fold_1_seq_1"]["sequence"] == "MKTAYI"

    def test_loads_fampnn_score_field(self, tmp_path, write_jsonl_file):
        write_jsonl_file("scores.json", [
            {"design": "fold_1_seq_1", "fampnn_avg_psce": 0.5, "sequence": "MKTAYI"},
        ])
        data_map = load_json_data(str(tmp_path), "fampnn")
        assert data_map["fold_1_seq_1"]["score"] == pytest.approx(0.5)

    def test_missing_sequence_defaults_to_empty_string(self, tmp_path, write_jsonl_file):
        write_jsonl_file("scores.json", [
            {"design": "fold_1_seq_1", "score": 1.0},
        ])
        data_map = load_json_data(str(tmp_path), "mpnn")
        assert data_map["fold_1_seq_1"]["sequence"] == ""

    def test_missing_required_key_skips_line(self, tmp_path):
        path = tmp_path / "scores.json"
        with open(path, "w") as f:
            f.write(json.dumps({"design": "fold_1_seq_1"}) + "\n")  # missing 'score'
            f.write(json.dumps({"design": "fold_1_seq_2", "score": 1.0, "sequence": "MK"}) + "\n")
        data_map = load_json_data(str(tmp_path), "mpnn")
        assert "fold_1_seq_1" not in data_map
        assert "fold_1_seq_2" in data_map

    def test_malformed_json_line_skipped(self, tmp_path):
        path = tmp_path / "scores.json"
        with open(path, "w") as f:
            f.write("{not valid json\n")
            f.write(json.dumps({"design": "fold_1_seq_2", "score": 1.0, "sequence": "MK"}) + "\n")
        data_map = load_json_data(str(tmp_path), "mpnn")
        assert len(data_map) == 1

    def test_ignores_non_json_files(self, tmp_path, write_jsonl_file):
        write_jsonl_file("scores.json", [
            {"design": "fold_1_seq_1", "score": 1.0, "sequence": "MK"},
        ])
        (tmp_path / "readme.txt").write_text("not json")
        data_map = load_json_data(str(tmp_path), "mpnn")
        assert len(data_map) == 1


class TestFilterByScore:
    def test_none_threshold_returns_all(self):
        data_map = {"a": {"score": 1.0}, "b": {"score": 2.0}}
        assert filter_by_score(data_map, None) == data_map

    def test_filters_above_threshold(self):
        data_map = {"a": {"score": 1.0}, "b": {"score": 5.0}}
        result = filter_by_score(data_map, 2.0)
        assert list(result.keys()) == ["a"]

    def test_boundary_value_inclusive(self):
        data_map = {"a": {"score": 2.0}}
        result = filter_by_score(data_map, 2.0)
        assert "a" in result


class TestFilterBySeqMetrics:
    def test_no_metrics_passes_through_with_warning(self):
        data_map = {"a": {"score": 1.0}}
        result = filter_by_seq_metrics(data_map, {}, _args(seq_min_pi=5.0))
        assert "a" in result

    def test_ext_coef_min_filter_rejects(self):
        data_map = {"a": {"score": 1.0}}
        metrics_map = {"a": {"seq_ext_coef": 100, "seq_pI": 7.0}}
        result = filter_by_seq_metrics(data_map, metrics_map, _args(seq_min_ext_coef=500))
        assert "a" not in result

    def test_ext_coef_max_filter_rejects(self):
        data_map = {"a": {"score": 1.0}}
        metrics_map = {"a": {"seq_ext_coef": 1000, "seq_pI": 7.0}}
        result = filter_by_seq_metrics(data_map, metrics_map, _args(seq_max_ext_coef=500))
        assert "a" not in result

    def test_pi_range_filter(self):
        data_map = {"a": {"score": 1.0}, "b": {"score": 1.0}}
        metrics_map = {
            "a": {"seq_ext_coef": 100, "seq_pI": 4.0},
            "b": {"seq_ext_coef": 100, "seq_pI": 8.0},
        }
        result = filter_by_seq_metrics(data_map, metrics_map, _args(seq_min_pi=5.0, seq_max_pi=9.0))
        assert "a" not in result
        assert "b" in result

    def test_no_filters_all_pass(self):
        data_map = {"a": {"score": 1.0}}
        metrics_map = {"a": {"seq_ext_coef": 100, "seq_pI": 7.0}}
        result = filter_by_seq_metrics(data_map, metrics_map, _args())
        assert "a" in result


class TestCopyFilteredDesigns:
    def test_copies_pdb_and_json_when_present(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        json_dir = tmp_path / "jsons"
        out_dir = tmp_path / "out"
        pdb_dir.mkdir()
        json_dir.mkdir()
        (pdb_dir / "fold_1_seq_1.pdb").write_text("ATOM\n")
        (json_dir / "fold_1_seq_1.json").write_text("{}")

        copied = copy_filtered_designs(["fold_1_seq_1"], str(pdb_dir), str(json_dir), str(out_dir))
        assert copied == 1
        assert (out_dir / "fold_1_seq_1.pdb").exists()
        assert (out_dir / "fold_1_seq_1.json").exists()

    def test_missing_pdb_not_counted(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        json_dir = tmp_path / "jsons"
        out_dir = tmp_path / "out"
        pdb_dir.mkdir()
        json_dir.mkdir()

        copied = copy_filtered_designs(["missing_design"], str(pdb_dir), str(json_dir), str(out_dir))
        assert copied == 0


class TestWriteSeqMetricsJsonl:
    def test_writes_only_non_none_metrics(self, tmp_path):
        metrics_map = {
            "a": {"description": "a", "seq_length": 10},
            "b": None,
        }
        out_path = tmp_path / "metrics.jsonl"
        write_seq_metrics_jsonl(metrics_map, str(out_path))
        lines = out_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["description"] == "a"
