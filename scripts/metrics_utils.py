"""Shared secondary-structure (DSSP) and radius-of-gyration metrics.

Used by both analyse_best_designs.py and filter_fold.py.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
from Bio.PDB import DSSP

# 8-state DSSP to 3-state mapping.
# Note: isolated beta-bridge residues ('B') are classified as loop, not
# strand, matching Rosetta's dssp.Dssp().get_dssp_secstruct() (used by the
# original PyRosetta implementation) and structure viewers such as ChimeraX,
# which only render a strand where there is a continuous, multi-residue
# H-bonded ladder ('E').
DSSP_HELIX = frozenset(('H', 'G', 'I'))
DSSP_STRAND = frozenset(('E',))


def _prepare_dssp_input(pdb_path):
    """Ensure a PDB file is recognized as PDB format by mkdssp (>=4.0).

    mkdssp>=4.0 decides whether to parse a file as PDB or mmCIF based on
    whether it starts with a HEADER record; without one it assumes mmCIF and
    fails. AF2/Boltz/RFdiffusion output PDBs have no HEADER line, so we
    prepend one to a temporary copy when needed.

    Returns a (path, tmp_path) tuple where path is what should be passed to
    DSSP and tmp_path is the temporary file to clean up afterwards (None if
    no temporary file was created).
    """
    pdb_path = Path(pdb_path)
    with open(pdb_path) as f:
        content = f.read()

    if content.startswith('HEADER'):
        return str(pdb_path), None

    fd, tmp_path = tempfile.mkstemp(suffix='.pdb', dir=pdb_path.parent)
    with os.fdopen(fd, 'w') as dst:
        dst.write('HEADER\n')
        dst.write(content)

    return tmp_path, tmp_path


def compute_dssp_chars_by_chain(model, pdb_path):
    """Run mkdssp once for the whole structure and map codes to 3 states.

    Returns {chain_id: [ss chars in residue order]} so callers needing
    per-chain counts (e.g. multi-chain oligomers) don't have to re-run
    mkdssp once per chain.
    """
    dssp_path, tmp_path = _prepare_dssp_input(pdb_path)
    try:
        dssp_obj = DSSP(model, dssp_path, dssp="mkdssp")
        chars_by_chain = {}
        for key in dssp_obj.keys():
            ss = dssp_obj[key][2]
            if ss in DSSP_HELIX:
                mapped = 'H'
            elif ss in DSSP_STRAND:
                mapped = 'E'
            else:
                mapped = 'L'
            chars_by_chain.setdefault(key[0], []).append(mapped)
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

    return chars_by_chain


def count_ss_elements(dssp_chars):
    """Count discrete helix/strand elements in an ordered list of H/E/L codes."""
    helix_count = 0
    strand_count = 0
    current_helix = False
    current_strand = False

    for ss in dssp_chars:
        if ss == 'H':
            if not current_helix:
                helix_count += 1
                current_helix = True
            current_strand = False
        elif ss == 'E':
            if not current_strand:
                strand_count += 1
                current_strand = True
            current_helix = False
        else:
            current_helix = False
            current_strand = False

    return helix_count, strand_count


def count_secondary_structures(model, pdb_path, chain_id=None):
    """Count secondary structure elements for one chain (or the whole model).

    Runs mkdssp once via compute_dssp_chars_by_chain(). When counting
    multiple chains from the same structure, call
    compute_dssp_chars_by_chain() once yourself and pass each chain's list
    to count_ss_elements() instead, to avoid re-running mkdssp per chain.
    """
    chars_by_chain = compute_dssp_chars_by_chain(model, pdb_path)
    if chain_id is not None:
        dssp_chars = chars_by_chain.get(chain_id, [])
    else:
        dssp_chars = [c for chain_chars in chars_by_chain.values() for c in chain_chars]
    return count_ss_elements(dssp_chars)


def calculate_rog(chain_or_model):
    """Calculate mass-weighted radius of gyration for a chain or model."""
    coords = np.array([atom.get_coord() for atom in chain_or_model.get_atoms()])
    masses = np.array([atom.mass for atom in chain_or_model.get_atoms()])

    if len(coords) == 0:
        return 0.0

    com = np.average(coords, axis=0, weights=masses)
    diff = coords - com
    return round(float(np.sqrt(np.sum(masses * np.sum(diff**2, axis=1)) / masses.sum())), 2)
