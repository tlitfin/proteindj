"""Tier A unit tests for metrics_utils.py.

compute_dssp_chars_by_chain is tested via a real (but geometrically
irregular) mkdssp invocation: a planar zigzag backbone has no consistent
H-bond pattern, so DSSP reliably assigns loop ('-') to every residue. This
gives real integration coverage of the mkdssp subprocess/parsing path
without depending on a brittle, hand-built helix/strand geometry.
"""
import math

import numpy as np
import pytest
from Bio.PDB import PDBParser

from metrics_utils import (
    _prepare_dssp_input,
    compute_dssp_chars_by_chain,
    count_ss_elements,
    count_secondary_structures,
    calculate_rog,
)


def _zigzag_backbone_lines(chain='A', n_res=6, bond_len=1.45, angle_deg=111.0,
                            resname='ALA', x_offset=0.0, serial_start=1):
    """Planar, fully-extended (non-regular) N/CA/C/O backbone - real bond-like
    distances but no consistent secondary-structure H-bond pattern."""
    angle = math.radians(180 - angle_deg)
    pos = [x_offset, 0.0, 0.0]
    direction = 0.0
    coords = [tuple(pos)]
    for i in range(1, n_res * 3):
        direction += angle if i % 2 == 0 else -angle
        pos = [pos[0] + bond_len * math.cos(direction), pos[1] + bond_len * math.sin(direction), 0.0]
        coords.append(tuple(pos))

    per_res_atoms = {}
    for i, c in enumerate(coords):
        resnum = i // 3 + 1
        name = ['N', 'CA', 'C'][i % 3]
        per_res_atoms.setdefault(resnum, {})[name] = c

    lines = []
    serial = serial_start
    for resnum in range(1, n_res + 1):
        atoms = per_res_atoms[resnum]
        for name in ('N', 'CA', 'C'):
            x, y, z = atoms[name]
            lines.append(
                f"ATOM  {serial:>5} {name:<4} {resname:>3} {chain}{resnum:>4}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {name[0]:>2}"
            )
            serial += 1
        cx, cy, cz = atoms['C']
        cax, cay, caz = atoms['CA']
        dx, dy = cx - cax, cy - cay
        norm = math.hypot(dx, dy)
        dx, dy = dx / norm, dy / norm
        px, py = -dy, dx
        ox, oy = cx + 1.23 * px, cy + 1.23 * py
        lines.append(
            f"ATOM  {serial:>5} O    {resname:>3} {chain}{resnum:>4}    "
            f"{ox:>8.3f}{oy:>8.3f}{cz:>8.3f}{1.0:>6.2f}{0.0:>6.2f}          {'O':>2}"
        )
        serial += 1
    return lines, serial


def _write_zigzag_pdb(tmp_path, chains=('A',), n_res=6, with_header=False):
    lines = []
    serial = 1
    for i, chain in enumerate(chains):
        chain_lines, serial = _zigzag_backbone_lines(
            chain=chain, n_res=n_res, x_offset=i * 100.0, serial_start=serial
        )
        lines += chain_lines
    lines.append("END")
    text = ("HEADER\n" if with_header else "") + "\n".join(lines) + "\n"
    path = tmp_path / "input.pdb"
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# _prepare_dssp_input
# ---------------------------------------------------------------------------

def test_prepare_dssp_input_passthrough_when_header_present(tmp_path):
    path = _write_zigzag_pdb(tmp_path, with_header=True)
    dssp_path, tmp_created = _prepare_dssp_input(path)
    assert dssp_path == str(path)
    assert tmp_created is None


def test_prepare_dssp_input_prepends_header_when_missing(tmp_path):
    path = _write_zigzag_pdb(tmp_path, with_header=False)
    original_text = path.read_text()
    dssp_path, tmp_created = _prepare_dssp_input(path)
    try:
        assert tmp_created is not None
        assert dssp_path == tmp_created
        new_text = open(dssp_path).read()
        assert new_text == "HEADER\n" + original_text
    finally:
        import os
        if tmp_created:
            os.unlink(tmp_created)


# ---------------------------------------------------------------------------
# compute_dssp_chars_by_chain (real mkdssp invocation)
# ---------------------------------------------------------------------------

def test_compute_dssp_chars_by_chain_single_chain(tmp_path):
    path = _write_zigzag_pdb(tmp_path, chains=('A',), n_res=6)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', path)
    model = structure[0]

    chars_by_chain = compute_dssp_chars_by_chain(model, path)

    assert set(chars_by_chain.keys()) == {'A'}
    assert len(chars_by_chain['A']) == 6
    assert all(c in ('H', 'E', 'L') for c in chars_by_chain['A'])
    # Irregular zigzag backbone has no consistent H-bond pattern -> all loop.
    assert chars_by_chain['A'] == ['L'] * 6


def test_compute_dssp_chars_by_chain_multi_chain(tmp_path):
    path = _write_zigzag_pdb(tmp_path, chains=('A', 'B'), n_res=4)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', path)
    model = structure[0]

    chars_by_chain = compute_dssp_chars_by_chain(model, path)

    assert set(chars_by_chain.keys()) == {'A', 'B'}
    assert len(chars_by_chain['A']) == 4
    assert len(chars_by_chain['B']) == 4


def test_compute_dssp_chars_by_chain_cleans_up_tmp_file(tmp_path):
    path = _write_zigzag_pdb(tmp_path, chains=('A',), n_res=4, with_header=False)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', path)
    model = structure[0]

    compute_dssp_chars_by_chain(model, path)

    leftover_tmp_files = [p for p in tmp_path.iterdir() if p.name != path.name]
    assert leftover_tmp_files == []


# ---------------------------------------------------------------------------
# count_secondary_structures (thin wrapper around compute_dssp_chars_by_chain)
# ---------------------------------------------------------------------------

def test_count_secondary_structures_on_irregular_backbone(tmp_path):
    path = _write_zigzag_pdb(tmp_path, chains=('A',), n_res=6)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', path)
    model = structure[0]

    helix_count, strand_count = count_secondary_structures(model, path, chain_id='A')
    assert (helix_count, strand_count) == (0, 0)


# ---------------------------------------------------------------------------
# count_ss_elements (pure)
# ---------------------------------------------------------------------------

def test_count_ss_elements_counts_discrete_runs():
    dssp_chars = list("HHHLLLEEELHH")
    helix_count, strand_count = count_ss_elements(dssp_chars)
    assert (helix_count, strand_count) == (2, 1)


def test_count_ss_elements_empty():
    assert count_ss_elements([]) == (0, 0)


def test_count_ss_elements_no_helix_or_strand():
    assert count_ss_elements(list("LLLLL")) == (0, 0)


def test_count_ss_elements_adjacent_runs_not_merged():
    # H immediately followed by E: two separate elements, not merged.
    assert count_ss_elements(list("HHHEEE")) == (1, 1)


# ---------------------------------------------------------------------------
# calculate_rog
# ---------------------------------------------------------------------------

def test_calculate_rog_known_value(tmp_path, pdb_atom_line):
    # Two carbon atoms 4A apart, equal mass -> RoG = half the separation = 2.0
    lines = [
        pdb_atom_line(serial=1, name='CA', chain='A', resnum=1, x=0.0, element='C'),
        pdb_atom_line(serial=2, name='CA', chain='A', resnum=2, x=4.0, element='C'),
    ]
    path = tmp_path / "input.pdb"
    path.write_text("\n".join(lines) + "\n")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', path)
    rog = calculate_rog(structure[0]['A'])
    assert rog == pytest.approx(2.0, abs=1e-6)


def test_calculate_rog_empty_chain_is_zero(tmp_path):
    from Bio.PDB.Chain import Chain
    empty_chain = Chain('A')
    assert calculate_rog(empty_chain) == 0.0
