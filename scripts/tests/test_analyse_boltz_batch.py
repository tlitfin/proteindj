"""Tier A unit tests for analyse_boltz_batch.py."""
import json

import numpy as np
import pytest

from analyse_boltz_batch import (
    build_file_index,
    locate_files,
    expected_txt_path,
    get_ipsae_min_max,
    get_pDockQ_min_max,
    min_max_pae_for_chain_contacts,
    find_ipsae_txts,
)


# ---------------------------------------------------------------------------
# build_file_index
# ---------------------------------------------------------------------------

def test_build_file_index_groups_by_binder_id(tmp_path):
    (tmp_path / "fold_1_seq_1_boltzpred.pdb").write_text("pdb")
    (tmp_path / "pae_fold_1_seq_1_boltzpred.npz").write_bytes(b"npz")
    (tmp_path / "fold_1_seq_1_boltzpred.json").write_text("{}")
    (tmp_path / "unrelated.txt").write_text("x")

    index = build_file_index(str(tmp_path))
    assert set(index.keys()) == {"fold_1_seq_1_boltzpred"}
    files = index["fold_1_seq_1_boltzpred"]
    assert files["structure"].endswith("fold_1_seq_1_boltzpred.pdb")
    assert files["confidence"].endswith("pae_fold_1_seq_1_boltzpred.npz")
    assert files["json"].endswith("fold_1_seq_1_boltzpred.json")


def test_build_file_index_missing_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        build_file_index(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# locate_files
# ---------------------------------------------------------------------------

def test_locate_files_found():
    index = {"bid1": {"structure": "s", "confidence": "c", "json": "j"}}
    result, error = locate_files("bid1", index)
    assert result == ("s", "c", "j")
    assert error is None


def test_locate_files_case_variant():
    index = {"BID1": {"structure": "s", "confidence": "c", "json": "j"}}
    result, error = locate_files("bid1", index)
    assert result == ("s", "c", "j")


def test_locate_files_missing_returns_error():
    index = {}
    result, error = locate_files("bid1", index)
    assert result is None
    assert "missing structure" in error


# ---------------------------------------------------------------------------
# expected_txt_path
# ---------------------------------------------------------------------------

def test_expected_txt_path_format():
    path = expected_txt_path("/some/dir/fold_1_seq_1_boltzpred.pdb", 10.0, 10.0)
    assert path.name == "fold_1_seq_1_boltzpred_pae10_dist10.txt"


# ---------------------------------------------------------------------------
# get_ipsae_min_max
# ---------------------------------------------------------------------------

_IPSAE_HEADER = (
    "Chn1 Chn2 Type ipSAE ipSAE_avg ipSAE_min_in_calculation "
    "LIS ipSAE_d0chn ipSAE_d0dom dist1 dist2 ipae"
)


def _write_ipsae_txt(tmp_path, lines):
    path = tmp_path / "result.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_get_ipsae_min_max_aggregates_single_partner(tmp_path):
    lines = [
        _IPSAE_HEADER,
        "A B max 0.5 0.4 0.3 0.6 0.55 0.45 1.0 1.0 2.0",
        "A B asym 0.2 0.1 0.1 0.1 0.1 0.1 1.0 1.0 9.9",
    ]
    path = _write_ipsae_txt(tmp_path, lines)
    result = get_ipsae_min_max(str(path))
    avg_min, avg_max, avg_ipsae_avg, avg_lis, avg_ipsae_min, avg_d0chn, avg_d0dom, avg_ipae = result
    assert avg_min == pytest.approx(0.2)
    assert avg_max == pytest.approx(0.5)
    assert avg_ipsae_avg == pytest.approx(0.4)
    assert avg_lis == pytest.approx(0.6)
    assert avg_ipsae_min == pytest.approx(0.3)
    assert avg_d0chn == pytest.approx(0.55)
    assert avg_d0dom == pytest.approx(0.45)
    assert avg_ipae == pytest.approx(2.0)


def test_get_ipsae_min_max_excludes_zero_distance_pairs(tmp_path):
    lines = [
        _IPSAE_HEADER,
        "A B max 0.5 0.4 0.3 0.6 0.55 0.45 0.0 1.0 2.0",
    ]
    path = _write_ipsae_txt(tmp_path, lines)
    avg_min, avg_max, *_ = get_ipsae_min_max(str(path))
    assert avg_min == 0.0
    assert avg_max == 0.0


def test_get_ipsae_min_max_only_target_chain_a_supported(tmp_path):
    path = _write_ipsae_txt(tmp_path, [_IPSAE_HEADER])
    with pytest.raises(ValueError, match="only supports target_chain='A'"):
        get_ipsae_min_max(str(path), target_chain='B')


def test_get_ipsae_min_max_empty_file_returns_none_tuple(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    result = get_ipsae_min_max(str(path))
    assert result == (None,) * 8


def test_get_ipsae_min_max_missing_column_raises(tmp_path):
    path = _write_ipsae_txt(tmp_path, ["Chn1 Chn2 Type ipSAE"])
    with pytest.raises(ValueError, match="Missing columns"):
        get_ipsae_min_max(str(path))


# ---------------------------------------------------------------------------
# get_pDockQ_min_max
# ---------------------------------------------------------------------------

def test_get_pdockq_min_max_aggregates(tmp_path):
    lines = [
        "Chn1 Chn2 pDockQ pDockQ2 dist1 dist2",
        "A B 0.3 0.4 1.0 1.0",
        "A B 0.5 0.6 1.0 1.0",
    ]
    path = _write_ipsae_txt(tmp_path, lines)
    result = get_pDockQ_min_max(str(path))
    assert result["pDockQ"] == [pytest.approx(0.3), pytest.approx(0.5)]
    assert result["pDockQ2"] == [pytest.approx(0.4), pytest.approx(0.6)]


def test_get_pdockq_min_max_no_partner_data(tmp_path):
    lines = [
        "Chn1 Chn2 pDockQ pDockQ2 dist1 dist2",
        "A B 0.3 0.4 0.0 1.0",  # zero distance excluded
    ]
    path = _write_ipsae_txt(tmp_path, lines)
    result = get_pDockQ_min_max(str(path))
    assert result == {"pDockQ": [0, 0], "pDockQ2": [0, 0]}


def test_get_pdockq_min_max_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    result = get_pDockQ_min_max(str(path))
    assert result == {"pDockQ": [None, None], "pDockQ2": [None, None]}


# ---------------------------------------------------------------------------
# min_max_pae_for_chain_contacts
# ---------------------------------------------------------------------------

def test_min_max_pae_for_chain_contacts_with_contacts(tmp_path):
    chain_ids = ['A', 'A', 'B', 'B']
    contact_probs = [
        [0, 0, 0.9, 0.1],
        [0, 0, 0.2, 0.9],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    pae = [
        [0, 0, 5.0, 1.0],
        [0, 0, 1.0, 10.0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps({
        'token_chain_ids': chain_ids, 'contact_probs': contact_probs, 'pae': pae,
    }))
    mn, mx, count = min_max_pae_for_chain_contacts(str(json_path), threshold=0.5)
    assert mn == pytest.approx(5.0)
    assert mx == pytest.approx(10.0)
    assert count == 2


def test_min_max_pae_for_chain_contacts_no_contacts_above_threshold(tmp_path):
    chain_ids = ['A', 'A', 'B', 'B']
    contact_probs = [
        [0, 0, 0.9, 0.1],
        [0, 0, 0.2, 0.9],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    pae = [[0] * 4 for _ in range(4)]
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps({
        'token_chain_ids': chain_ids, 'contact_probs': contact_probs, 'pae': pae,
    }))
    mn, mx, count = min_max_pae_for_chain_contacts(str(json_path), threshold=2.0)
    assert (mn, mx, count) == (25, 25, 0)


# ---------------------------------------------------------------------------
# find_ipsae_txts
# ---------------------------------------------------------------------------

def test_find_ipsae_txts_excludes_byres_and_done(tmp_path):
    struct_path = tmp_path / "fold_1_seq_1_boltzpred.pdb"
    struct_path.write_text("pdb")
    (tmp_path / "fold_1_seq_1_boltzpred_pae10_dist10.txt").write_text("data")
    (tmp_path / "fold_1_seq_1_boltzpred_byres.txt").write_text("data")
    (tmp_path / "fold_1_seq_1_boltzpred_done.txt").write_text("data")

    result = find_ipsae_txts(str(struct_path), "fold_1_seq_1_boltzpred")
    assert len(result) == 1
    assert result[0].endswith("_pae10_dist10.txt")


def test_find_ipsae_txts_falls_back_to_lowercase(tmp_path):
    struct_path = tmp_path / "FOLD_1_seq_1_boltzpred.pdb"
    struct_path.write_text("pdb")
    (tmp_path / "fold_1_seq_1_boltzpred_pae10_dist10.txt").write_text("data")

    result = find_ipsae_txts(str(struct_path), "FOLD_1_seq_1_boltzpred")
    assert len(result) == 1
