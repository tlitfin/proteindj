"""Tier A unit tests for prep_fampnn_designs.py (pure-function parts only;
restore_backbone_atoms/restore_sidechains depend on PDBFixer/FASPR external
binaries and are out of scope for Tier A)."""
import pytest

from prep_fampnn_designs import get_residue_metadata


class TestGetResidueMetadata:
    def test_extracts_bfactor_and_occupancy_from_ca(self, pdb_atom_line):
        pdb_text = pdb_atom_line(serial=1, name='CA', chain='A', resnum=1, occ=0.80, temp=45.5) + "\n"
        metadata = get_residue_metadata(pdb_text)
        assert metadata == {('A', 1): (45.5, 0.80)}

    def test_only_ca_atom_used_when_residue_has_multiple_atoms(self, pdb_atom_line):
        pdb_text = (
            pdb_atom_line(serial=1, name='N', chain='A', resnum=1, occ=1.0, temp=10.0) + "\n"
            + pdb_atom_line(serial=2, name='CA', chain='A', resnum=1, occ=0.5, temp=20.0) + "\n"
            + pdb_atom_line(serial=3, name='C', chain='A', resnum=1, occ=1.0, temp=30.0) + "\n"
        )
        metadata = get_residue_metadata(pdb_text)
        assert metadata == {('A', 1): (20.0, 0.5)}

    def test_residue_missing_ca_is_not_in_metadata(self, pdb_atom_line):
        pdb_text = pdb_atom_line(serial=1, name='N', chain='A', resnum=1, occ=1.0, temp=10.0) + "\n"
        metadata = get_residue_metadata(pdb_text)
        assert metadata == {}

    def test_empty_pdb_raises_value_error(self):
        # Bio.PDB's PDBParser raises ValueError("Empty file.") for empty input
        # rather than returning an empty structure.
        with pytest.raises(ValueError):
            get_residue_metadata("")

    def test_multiple_chains_and_residues(self, pdb_atom_line):
        pdb_text = (
            pdb_atom_line(serial=1, name='CA', chain='A', resnum=1, occ=1.0, temp=1.0) + "\n"
            + pdb_atom_line(serial=2, name='CA', chain='A', resnum=2, occ=1.0, temp=2.0) + "\n"
            + pdb_atom_line(serial=3, name='CA', chain='B', resnum=1, occ=1.0, temp=3.0) + "\n"
        )
        metadata = get_residue_metadata(pdb_text)
        assert metadata == {
            ('A', 1): (1.0, 1.0),
            ('A', 2): (2.0, 1.0),
            ('B', 1): (3.0, 1.0),
        }
