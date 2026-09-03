"""Tier A unit tests for prep_fampnn_csv.py."""
import json

from prep_fampnn_csv import (
    parse_pdb_chains,
    indices_to_chain_ranges,
    process_file_pair,
)


def _pdb_line(chain, resnum):
    return f"ATOM      1  CA  ALA {chain}{resnum:>4}      0.000   0.000   0.000  1.00  0.00           C"


def _write_pdb(path, chain_resnums):
    """chain_resnums: list of (chain, resnum) tuples in file order."""
    with open(path, 'w') as f:
        for chain, resnum in chain_resnums:
            f.write(_pdb_line(chain, resnum) + "\n")


class TestParsePdbChains:
    def test_single_chain(self, tmp_path):
        pdb = tmp_path / "a.pdb"
        _write_pdb(pdb, [('A', 1), ('A', 2), ('A', 3)])
        chains = parse_pdb_chains(str(pdb))
        assert chains == {'A': [1, 2, 3]}

    def test_multi_chain_preserves_order(self, tmp_path):
        pdb = tmp_path / "a.pdb"
        _write_pdb(pdb, [('A', 1), ('A', 2), ('B', 10), ('B', 11)])
        chains = parse_pdb_chains(str(pdb))
        assert chains == {'A': [1, 2], 'B': [10, 11]}

    def test_dedupes_multiple_atoms_per_residue(self, tmp_path):
        pdb = tmp_path / "a.pdb"
        # Two atom lines for the same (chain, resnum) should count as one residue
        _write_pdb(pdb, [('A', 1), ('A', 1), ('A', 2)])
        chains = parse_pdb_chains(str(pdb))
        assert chains == {'A': [1, 2]}


class TestIndicesToChainRanges:
    def test_empty_indices_returns_empty_string(self):
        assert indices_to_chain_ranges([], {'A': [1, 2, 3]}) == ""

    def test_contiguous_range_single_chain(self):
        chain_residues = {'A': [1, 2, 3, 4, 5]}
        result = indices_to_chain_ranges([2, 3, 4], chain_residues)
        assert result == "A3-5"

    def test_non_contiguous_positions(self):
        chain_residues = {'A': [1, 2, 3, 4, 5]}
        result = indices_to_chain_ranges([0, 2, 4], chain_residues)
        assert result == "A1-1,A3-3,A5-5"

    def test_multi_chain_ranges(self):
        chain_residues = {'A': [1, 2, 3], 'B': [10, 11, 12]}
        # indices 1,2 -> A2,A3 ; indices 3,4 -> B10,B11
        result = indices_to_chain_ranges([1, 2, 3, 4], chain_residues)
        assert result == "A2-3,B10-11"

    def test_out_of_range_indices_are_dropped(self):
        chain_residues = {'A': [1, 2, 3]}
        result = indices_to_chain_ranges([0, 99], chain_residues)
        assert result == "A1-1"

    def test_non_contiguous_pdb_numbering_not_merged(self):
        # PDB residue numbers 1, 2, 4 (gap at 3) - indices 0,1,2 should NOT merge
        # index 2 (resnum 4) into the same range as indices 0-1 (resnum 1-2)
        chain_residues = {'A': [1, 2, 4]}
        result = indices_to_chain_ranges([0, 1, 2], chain_residues)
        assert result == "A1-2,A4-4"


class TestProcessFilePair:
    def test_rfd_inpaint_seq_key(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1), ('A', 2), ('A', 3)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({'rfd_inpaint_seq': [True, False, True]}))
        result = process_file_pair(json_path, pdb)
        assert result == "A1-1,A3-3"

    def test_bc_inpaint_seq_key(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1), ('A', 2), ('A', 3)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({'bc_inpaint_seq': [False, True, True]}))
        result = process_file_pair(json_path, pdb)
        assert result == "A2-3"

    def test_bg_inpaint_seq_key(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1), ('A', 2), ('A', 3)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({'bg_inpaint_seq': [True, True, False]}))
        result = process_file_pair(json_path, pdb)
        assert result == "A1-2"

    def test_no_inpaint_key_returns_empty(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1), ('A', 2)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({'some_other_key': 1}))
        result = process_file_pair(json_path, pdb)
        assert result == ""

    def test_all_false_mask_returns_empty(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1), ('A', 2)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({'rfd_inpaint_seq': [False, False]}))
        result = process_file_pair(json_path, pdb)
        assert result == ""

    def test_rfd_key_takes_priority_over_bc_and_bg(self, tmp_path):
        # Only one key should ever be present in real usage, but confirm the `or` chain
        # prefers rfd_inpaint_seq first if multiple happened to be present.
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1), ('A', 2)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({
            'rfd_inpaint_seq': [True, False],
            'bc_inpaint_seq': [False, True],
        }))
        result = process_file_pair(json_path, pdb)
        assert result == "A1-1"
