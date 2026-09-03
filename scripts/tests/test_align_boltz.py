"""Tier A unit tests for align_boltz.py.

align_boltz.py runs `setup_logging()` at import time, which creates an
"alignment.log" FileHandler in the current working directory as a side
effect. We import it with cwd temporarily pointed at a throwaway directory
so this doesn't litter the repo (this only matters on the first import in
the whole test session, since module imports are cached).
"""
import os
import tempfile

import pytest
from Bio.PDB import PDBParser

_prev_cwd = os.getcwd()
os.chdir(tempfile.mkdtemp())
try:
    from align_boltz import get_all_ca_atoms, get_chain_ca_atoms, get_target_ca_atoms
finally:
    os.chdir(_prev_cwd)


def _write_pdb(tmp_path, lines, name="input.pdb"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def _parse(path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure('s', path)


def _multi_chain_structure(tmp_path, pdb_atom_line):
    lines = (
        [pdb_atom_line(serial=1, name='CA', chain='A', resnum=1)]
        + [pdb_atom_line(serial=2, name='CA', chain='B', resnum=1, x=20.0)]
        + [pdb_atom_line(serial=3, name='CA', chain='C', resnum=1, x=40.0)]
    )
    return _parse(_write_pdb(tmp_path, lines))


# ---------------------------------------------------------------------------
# get_all_ca_atoms
# ---------------------------------------------------------------------------

def test_get_all_ca_atoms_collects_across_chains(tmp_path, pdb_atom_line):
    structure = _multi_chain_structure(tmp_path, pdb_atom_line)
    atoms = get_all_ca_atoms(structure)
    assert len(atoms) == 3


def test_get_all_ca_atoms_no_ca_raises(tmp_path, pdb_atom_line):
    lines = [pdb_atom_line(serial=1, name='N', chain='A', resnum=1)]
    structure = _parse(_write_pdb(tmp_path, lines))
    with pytest.raises(ValueError, match="No CA atoms found in structure"):
        get_all_ca_atoms(structure)


# ---------------------------------------------------------------------------
# get_chain_ca_atoms
# ---------------------------------------------------------------------------

def test_get_chain_ca_atoms_filters_by_chain(tmp_path, pdb_atom_line):
    structure = _multi_chain_structure(tmp_path, pdb_atom_line)
    atoms = get_chain_ca_atoms(structure, 'B')
    assert len(atoms) == 1


def test_get_chain_ca_atoms_missing_chain_raises(tmp_path, pdb_atom_line):
    structure = _multi_chain_structure(tmp_path, pdb_atom_line)
    with pytest.raises(ValueError, match="No CA atoms found in chain Z"):
        get_chain_ca_atoms(structure, 'Z')


# ---------------------------------------------------------------------------
# get_target_ca_atoms
# ---------------------------------------------------------------------------

def test_get_target_ca_atoms_excludes_binder_chain_handles_split_targets(tmp_path, pdb_atom_line):
    structure = _multi_chain_structure(tmp_path, pdb_atom_line)
    atoms = get_target_ca_atoms(structure, binder_chain='A')
    assert len(atoms) == 2


def test_get_target_ca_atoms_no_target_chains_raises(tmp_path, pdb_atom_line):
    lines = [pdb_atom_line(serial=1, name='CA', chain='A', resnum=1)]
    structure = _parse(_write_pdb(tmp_path, lines))
    with pytest.raises(ValueError, match="No CA atoms found in target chains"):
        get_target_ca_atoms(structure, binder_chain='A')
