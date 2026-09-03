"""Tier A unit tests for create_scaffolds.py.

Only pure/deterministic functions are tested. mask_ss (uses unseeded
random.uniform/randint) and DSSP-dependent extract_secstruc/dssp_biopython
are intentionally excluded, per the Phase 4 test plan.
"""
import numpy as np
import pytest
import torch

from create_scaffolds import (
    ss_to_tensor,
    generate_Cbeta,
    get_pair_dist,
    construct_block_adj_matrix,
    parse_pdb_lines_torch,
    parse_pdb_lines,
    aa2num,
)


# ---------------------------------------------------------------------------
# ss_to_tensor
# ---------------------------------------------------------------------------

def test_ss_to_tensor_converts_codes_and_idx():
    ss = {'idx': [1, 2, 3], 'ss': ['H', 'E', 'L']}
    ss_int, idx = ss_to_tensor(ss)
    assert list(ss_int) == [0, 1, 2]
    assert list(idx) == [1, 2, 3]


def test_ss_to_tensor_all_loop():
    ss = {'idx': [10, 11], 'ss': ['L', 'L']}
    ss_int, idx = ss_to_tensor(ss)
    assert list(ss_int) == [2, 2]


# ---------------------------------------------------------------------------
# generate_Cbeta
# ---------------------------------------------------------------------------

def test_generate_cbeta_known_geometry():
    N = torch.tensor([0.0, 0.0, 0.0])
    Ca = torch.tensor([1.0, 0.0, 0.0])
    C = torch.tensor([1.0, 1.0, 0.0])

    Cb = generate_Cbeta(N, Ca, C)

    expected = torch.tensor([1.5689693, -0.5441217, -0.57910144])
    assert torch.allclose(Cb, expected, atol=1e-5)


def test_generate_cbeta_translation_invariant_offset():
    # Translating N/Ca/C by a fixed offset should translate Cb by the same offset.
    N = torch.tensor([0.0, 0.0, 0.0])
    Ca = torch.tensor([1.0, 0.0, 0.0])
    C = torch.tensor([1.0, 1.0, 0.0])
    offset = torch.tensor([5.0, -2.0, 3.0])

    Cb_base = generate_Cbeta(N, Ca, C)
    Cb_translated = generate_Cbeta(N + offset, Ca + offset, C + offset)

    assert torch.allclose(Cb_translated, Cb_base + offset, atol=1e-5)


# ---------------------------------------------------------------------------
# get_pair_dist
# ---------------------------------------------------------------------------

def test_get_pair_dist_known_distance():
    a = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    b = a.clone()
    dist = get_pair_dist(a, b)
    assert dist.shape == (2, 2)
    assert dist[0, 1].item() == pytest.approx(5.0)
    assert dist[0, 0].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# construct_block_adj_matrix
# ---------------------------------------------------------------------------

def _residue_backbone(offset):
    """N/Ca/C triple in a fixed relative shape, translated by offset."""
    N = torch.tensor([0.0, 0.0, 0.0]) + offset
    Ca = torch.tensor([1.0, 0.0, 0.0]) + offset
    C = torch.tensor([1.0, 1.0, 0.0]) + offset
    return N, Ca, C


def _build_xyz(offsets):
    Ns, Cas, Cs = [], [], []
    for off in offsets:
        N, Ca, C = _residue_backbone(torch.tensor(off))
        Ns.append(N)
        Cas.append(Ca)
        Cs.append(C)
    return torch.stack([torch.stack(Ns), torch.stack(Cas), torch.stack(Cs)], dim=1)


def test_construct_block_adj_matrix_connects_close_segments():
    # Two 2-residue segments (helix=0, strand=1); Cb atoms end up close (<6A apart).
    sstruct = torch.tensor([0, 0, 1, 1])
    xyz = _build_xyz([[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]])

    block_adj = construct_block_adj_matrix(sstruct, xyz, cutoff=6)

    assert block_adj.shape == (4, 4)
    # Cross-segment block (helix vs strand) should be connected.
    assert torch.all(block_adj[0:2, 2:4] == 1)
    assert torch.all(block_adj[2:4, 0:2] == 1)
    # Within-segment blocks are never populated by the algorithm.
    assert torch.all(block_adj[0:2, 0:2] == 0)


def test_construct_block_adj_matrix_no_connection_when_far_apart():
    sstruct = torch.tensor([0, 0, 1, 1])
    xyz = _build_xyz([[0, 0, 0], [2, 0, 0], [20, 0, 0], [22, 0, 0]])

    block_adj = construct_block_adj_matrix(sstruct, xyz, cutoff=6)

    assert torch.all(block_adj[0:2, 2:4] == 0)


def test_construct_block_adj_matrix_excludes_loops_by_default():
    # sstruct label 2 = loop; segments are close together but loop is excluded.
    sstruct = torch.tensor([0, 0, 2, 2])
    xyz = _build_xyz([[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]])

    block_adj = construct_block_adj_matrix(sstruct, xyz, cutoff=6, include_loops=False)

    assert torch.all(block_adj == 0)


def test_construct_block_adj_matrix_includes_loops_when_flagged():
    sstruct = torch.tensor([0, 0, 2, 2])
    xyz = _build_xyz([[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]])

    block_adj = construct_block_adj_matrix(sstruct, xyz, cutoff=6, include_loops=True)

    assert torch.all(block_adj[0:2, 2:4] == 1)


# ---------------------------------------------------------------------------
# parse_pdb_lines_torch / parse_pdb_lines
# ---------------------------------------------------------------------------

def _two_residue_ala_lines(pdb_atom_line):
    lines = []
    serial = 1
    for resnum, base_x in ((1, 0.0), (2, 10.0)):
        for name, x in (('N', base_x), ('CA', base_x + 1.0), ('C', base_x + 2.0),
                        ('O', base_x + 3.0), ('CB', base_x + 4.0)):
            lines.append(pdb_atom_line(serial=serial, name=name, resname='ALA',
                                        chain='A', resnum=resnum, x=x, y=0.0, z=0.0))
            serial += 1
    return lines


def test_parse_pdb_lines_torch_shapes_and_values(pdb_atom_line):
    lines = _two_residue_ala_lines(pdb_atom_line)
    xyz, mask, pdb_idx = parse_pdb_lines_torch(lines)

    assert xyz.shape == (2, 27, 3)
    assert mask.shape == (2, 27)
    assert pdb_idx[0][0] == 'A' and int(pdb_idx[0][1]) == 1
    assert pdb_idx[1][0] == 'A' and int(pdb_idx[1][1]) == 2

    # N, CA, C, O, CB are atoms 0-4 in the aa2long ALA ordering.
    assert mask[0, 0:5].all()
    assert not mask[0, 5:].any()
    assert xyz[0, 0] == pytest.approx([0.0, 0.0, 0.0])
    assert xyz[0, 1] == pytest.approx([1.0, 0.0, 0.0])
    assert xyz[1, 0] == pytest.approx([10.0, 0.0, 0.0])


def test_parse_pdb_lines_returns_seq_and_idx(pdb_atom_line):
    lines = _two_residue_ala_lines(pdb_atom_line)
    out = parse_pdb_lines(lines)

    assert list(out['seq']) == [aa2num['ALA'], aa2num['ALA']]
    assert list(out['idx']) == [1, 2]
    assert out['pdb_idx'] == [('A', 1), ('A', 2)]
    assert out['xyz'].shape == (2, 27, 3)
    assert out['mask'][0, 0:5].all()


def test_parse_pdb_lines_unknown_residue_raises_keyerror(pdb_atom_line):
    # seq-building tolerates unknown resnames (maps to UNK=20), but the
    # per-atom aa2long lookup does not guard against unknown resnames and
    # raises KeyError - this documents the actual (unguarded) behavior.
    lines = [
        pdb_atom_line(serial=1, name='CA', resname='XYZ', chain='A', resnum=1),
    ]
    with pytest.raises(KeyError):
        parse_pdb_lines(lines)
