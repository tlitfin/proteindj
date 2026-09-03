"""Tier A unit tests for prep_mpnn_designs.py."""
import json

from prep_mpnn_designs import (
    map_inpaint_to_residues,
    get_fixed_from_bfactor,
    get_fixed_residues,
    modify_pdb_file,
)


def _pdb_line(chain, resnum, bfactor=0.0):
    return (
        f"ATOM      1  CA  ALA {chain}{resnum:>4}      0.000   0.000   0.000  1.00{bfactor:>6.2f}"
        "           C"
    )


def _write_pdb(path, residues):
    """residues: list of (chain, resnum, bfactor) tuples in file order."""
    with open(path, 'w') as f:
        for chain, resnum, bfactor in residues:
            f.write(_pdb_line(chain, resnum, bfactor) + "\n")


class TestMapInpaintToResidues:
    def test_true_positions_are_fixed(self, tmp_path):
        pdb = tmp_path / "a.pdb"
        _write_pdb(pdb, [('A', 1, 0.0), ('A', 2, 0.0), ('A', 3, 0.0)])
        result = map_inpaint_to_residues(str(pdb), [True, False, True])
        assert result == [('A', 1), ('A', 3)]

    def test_all_false_returns_empty(self, tmp_path):
        pdb = tmp_path / "a.pdb"
        _write_pdb(pdb, [('A', 1, 0.0), ('A', 2, 0.0)])
        result = map_inpaint_to_residues(str(pdb), [False, False])
        assert result == []

    def test_inpaint_array_shorter_than_residues_is_bounds_checked(self, tmp_path):
        # inpaint_array has fewer entries than PDB residues; residues beyond the
        # array length must not be treated as fixed (no IndexError either).
        pdb = tmp_path / "a.pdb"
        _write_pdb(pdb, [('A', 1, 0.0), ('A', 2, 0.0), ('A', 3, 0.0)])
        result = map_inpaint_to_residues(str(pdb), [True])
        assert result == [('A', 1)]

    def test_inpaint_array_longer_than_residues_is_ignored(self, tmp_path):
        pdb = tmp_path / "a.pdb"
        _write_pdb(pdb, [('A', 1, 0.0)])
        result = map_inpaint_to_residues(str(pdb), [True, True, True])
        assert result == [('A', 1)]


class TestGetFixedFromBfactor:
    def test_all_atoms_bfactor_1_is_fixed(self, tmp_path):
        pdb = tmp_path / "a.pdb"
        _write_pdb(pdb, [('A', 1, 1.0), ('A', 2, 0.0)])
        result = get_fixed_from_bfactor(str(pdb))
        assert result == [('A', 1)]

    def test_mixed_bfactor_within_residue_is_not_fixed(self, tmp_path):
        # Residue with two atoms, one 1.00 and one not -> not fixed (all() must be True)
        with open(tmp_path / "a.pdb", 'w') as f:
            f.write(_pdb_line('A', 1, 1.0) + "\n")
            f.write(_pdb_line('A', 1, 0.5) + "\n")
        result = get_fixed_from_bfactor(str(tmp_path / "a.pdb"))
        assert result == []

    def test_no_fixed_residues(self, tmp_path):
        pdb = tmp_path / "a.pdb"
        _write_pdb(pdb, [('A', 1, 0.0), ('A', 2, 0.5)])
        result = get_fixed_from_bfactor(str(pdb))
        assert result == []


class TestGetFixedResidues:
    def test_uses_json_rfd_inpaint_seq_when_present(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1, 0.0), ('A', 2, 0.0)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({'rfd_inpaint_seq': [True, False]}))
        result = get_fixed_residues(str(pdb), str(json_path))
        assert result == [('A', 1)]

    def test_uses_json_bc_inpaint_seq_when_present(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1, 0.0), ('A', 2, 0.0)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({'bc_inpaint_seq': [False, True]}))
        result = get_fixed_residues(str(pdb), str(json_path))
        assert result == [('A', 2)]

    def test_uses_json_bg_inpaint_seq_when_present(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1, 0.0), ('A', 2, 0.0)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({'bg_inpaint_seq': [True, True]}))
        result = get_fixed_residues(str(pdb), str(json_path))
        assert result == [('A', 1), ('A', 2)]

    def test_no_json_path_falls_back_to_bfactor(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1, 1.0), ('A', 2, 0.0)])
        result = get_fixed_residues(str(pdb), None)
        assert result == [('A', 1)]

    def test_missing_json_file_falls_back_to_bfactor(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1, 1.0), ('A', 2, 0.0)])
        missing_json = str(tmp_path / "does_not_exist.json")
        result = get_fixed_residues(str(pdb), missing_json)
        assert result == [('A', 1)]

    def test_json_without_any_inpaint_key_falls_back_to_bfactor(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1, 1.0), ('A', 2, 0.0)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text(json.dumps({'some_other_key': 1}))
        result = get_fixed_residues(str(pdb), str(json_path))
        assert result == [('A', 1)]

    def test_malformed_json_falls_back_to_bfactor(self, tmp_path):
        pdb = tmp_path / "fold_0.pdb"
        _write_pdb(pdb, [('A', 1, 1.0), ('A', 2, 0.0)])
        json_path = tmp_path / "fold_0.json"
        json_path.write_text("{not valid json")
        result = get_fixed_residues(str(pdb), str(json_path))
        assert result == [('A', 1)]


class TestModifyPdbFile:
    def test_adds_fixed_remarks_sorted(self, tmp_path):
        pdb = tmp_path / "in.pdb"
        _write_pdb(pdb, [('A', 1, 0.0), ('A', 2, 0.0)])
        out_pdb = tmp_path / "out.pdb"
        modify_pdb_file(str(pdb), str(out_pdb), [('A', 2), ('A', 1)])
        content = out_pdb.read_text()
        lines = content.splitlines()
        remark_lines = [l for l in lines if l.startswith('REMARK')]
        assert remark_lines == [
            "REMARK PDBinfo-LABEL: 1 FIXED",
            "REMARK PDBinfo-LABEL: 2 FIXED",
        ]

    def test_preserves_original_pdb_content(self, tmp_path):
        pdb = tmp_path / "in.pdb"
        _write_pdb(pdb, [('A', 1, 0.0)])
        original_content = pdb.read_text()
        out_pdb = tmp_path / "out.pdb"
        modify_pdb_file(str(pdb), str(out_pdb), [])
        out_content = out_pdb.read_text()
        assert out_content == original_content

    def test_no_fixed_residues_writes_no_remarks(self, tmp_path):
        pdb = tmp_path / "in.pdb"
        _write_pdb(pdb, [('A', 1, 0.0)])
        out_pdb = tmp_path / "out.pdb"
        modify_pdb_file(str(pdb), str(out_pdb), [])
        content = out_pdb.read_text()
        assert 'REMARK' not in content
