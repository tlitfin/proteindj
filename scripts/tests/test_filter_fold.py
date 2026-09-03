"""Tier A unit tests for filter_fold.py.

analyze_structure is exercised via real mkdssp runs (same irregular
zigzag-backbone strategy used in test_metrics_utils.py) since it is not
practical to isolate its internal `passes_filter` closure.
"""
import math
from pathlib import Path

import pytest
from Bio.PDB import PDBParser

from filter_fold import (
    extract_fold_id,
    parse_structure,
    get_chain_ids,
    analyze_structure,
)


def _zigzag_backbone_lines(chain='A', n_res=6, bond_len=1.45, angle_deg=111.0,
                            resname='ALA', x_offset=0.0, serial_start=1):
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


def _write_zigzag_pdb(tmp_path, filename, chains=('A',), n_res=6):
    lines = []
    serial = 1
    for i, chain in enumerate(chains):
        chain_lines, serial = _zigzag_backbone_lines(
            chain=chain, n_res=n_res, x_offset=i * 100.0, serial_start=serial
        )
        lines += chain_lines
    lines.append("END")
    path = tmp_path / filename
    path.write_text("HEADER\n" + "\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# extract_fold_id
# ---------------------------------------------------------------------------

def test_extract_fold_id_matches_pattern():
    assert extract_fold_id(Path("fold_5_seq_2.pdb")) == 5


def test_extract_fold_id_no_match_returns_none():
    assert extract_fold_id(Path("design_output.pdb")) is None


def test_extract_fold_id_first_number_group():
    assert extract_fold_id(Path("prefix_fold_12_suffix_99.pdb")) == 12


# ---------------------------------------------------------------------------
# parse_structure / get_chain_ids
# ---------------------------------------------------------------------------

def test_parse_structure_and_get_chain_ids(tmp_path):
    path = _write_zigzag_pdb(tmp_path, "fold_1_seq_1.pdb", chains=('A', 'B'), n_res=4)
    structure, model = parse_structure(path)
    assert get_chain_ids(model) == ['A', 'B']


# ---------------------------------------------------------------------------
# analyze_structure (real mkdssp integration)
# ---------------------------------------------------------------------------

def test_analyze_structure_monomer_no_filters_passes_and_copies(tmp_path):
    pdb_file = _write_zigzag_pdb(tmp_path, "fold_3_seq_1.pdb", chains=('A',), n_res=6)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    args = (pdb_file, None, None, None, None, None, None, None, None, output_dir, None)
    result = analyze_structure(args)

    assert result == {
        "fold_id": 3,
        "fold_helices": 0,
        "fold_strands": 0,
        "fold_total_ss": 0,
        "fold_RoG": pytest.approx(result["fold_RoG"]),
    }
    assert (output_dir / "fold_3_seq_1.pdb").exists()


def test_analyze_structure_binder_two_chains_analyzes_first_only(tmp_path):
    pdb_file = _write_zigzag_pdb(tmp_path, "fold_7_seq_1.pdb", chains=('A', 'B'), n_res=4)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    args = (pdb_file, None, None, None, None, None, None, None, None, output_dir, None)
    result = analyze_structure(args)

    assert result["fold_id"] == 7
    assert result["fold_helices"] == 0
    assert result["fold_strands"] == 0


def test_analyze_structure_oligomer_three_chains_aggregates(tmp_path):
    pdb_file = _write_zigzag_pdb(tmp_path, "fold_2_seq_1.pdb", chains=('A', 'B', 'C'), n_res=3)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    args = (pdb_file, None, None, None, None, None, None, None, None, output_dir, None)
    result = analyze_structure(args)

    assert result["fold_id"] == 2
    assert result["fold_helices"] == 0
    assert result["fold_strands"] == 0


def test_analyze_structure_fails_filter_does_not_copy(tmp_path):
    pdb_file = _write_zigzag_pdb(tmp_path, "fold_9_seq_1.pdb", chains=('A',), n_res=6)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Require at least 1 helix; the irregular backbone has none, so it fails.
    args = (pdb_file, None, None, 1, None, None, None, None, None, output_dir, None)
    result = analyze_structure(args)

    assert result["fold_helices"] == 0
    assert not (output_dir / "fold_9_seq_1.pdb").exists()


def test_analyze_structure_copies_matching_json(tmp_path):
    pdb_file = _write_zigzag_pdb(tmp_path, "fold_4_seq_1.pdb", chains=('A',), n_res=4)
    json_dir = tmp_path
    (json_dir / "fold_4_seq_1.json").write_text('{"score": 1.0}')
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    args = (pdb_file, None, None, None, None, None, None, None, None, output_dir, json_dir)
    analyze_structure(args)

    assert (output_dir / "fold_4_seq_1.pdb").exists()
    assert (output_dir / "fold_4_seq_1.json").exists()


def test_analyze_structure_missing_json_does_not_error(tmp_path):
    pdb_file = _write_zigzag_pdb(tmp_path, "fold_6_seq_1.pdb", chains=('A',), n_res=4)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    args = (pdb_file, None, None, None, None, None, None, None, None, output_dir, tmp_path)
    result = analyze_structure(args)

    assert result is not None
    assert (output_dir / "fold_6_seq_1.pdb").exists()
    assert not (output_dir / "fold_6_seq_1.json").exists()
