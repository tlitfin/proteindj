"""Tier A unit tests for generate_contigs.py."""
import pytest
from Bio.PDB import PDBParser

from generate_contigs import (
    get_protein_chains,
    find_continuous_ranges,
    format_chain_contig,
    validate_design_length,
    generate_contig_binder_denovo,
    generate_contig_monomer_denovo,
    generate_contig_monomer_partialdiff,
    generate_contig_binder_partialdiff,
    generate_contig_string,
)


def _write_pdb(tmp_path, pdb_atom_line, lines):
    path = tmp_path / "input.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def _parse(tmp_path, pdb_atom_line, lines):
    path = _write_pdb(tmp_path, pdb_atom_line, lines)
    parser = PDBParser(QUIET=True)
    return parser.get_structure('protein', path)


# ---------------------------------------------------------------------------
# find_continuous_ranges
# ---------------------------------------------------------------------------

def test_find_continuous_ranges_empty():
    assert find_continuous_ranges([]) == []


def test_find_continuous_ranges_single_residue():
    assert find_continuous_ranges([5]) == [(5, 5)]


def test_find_continuous_ranges_no_gaps():
    assert find_continuous_ranges([1, 2, 3, 4]) == [(1, 4)]


def test_find_continuous_ranges_with_gaps():
    assert find_continuous_ranges([1, 2, 3, 10, 11, 20]) == [(1, 3), (10, 11), (20, 20)]


# ---------------------------------------------------------------------------
# format_chain_contig
# ---------------------------------------------------------------------------

def test_format_chain_contig_single_range_with_chain_id():
    assert format_chain_contig('A', [(1, 77)], include_chain_id=True) == "A1-77"


def test_format_chain_contig_multi_range_with_chain_id():
    assert format_chain_contig('B', [(23, 77), (80, 105)], include_chain_id=True) == "B23-77/B80-105"


def test_format_chain_contig_single_residue_range():
    assert format_chain_contig('A', [(5, 5)], include_chain_id=True) == "A5"


def test_format_chain_contig_without_chain_id():
    assert format_chain_contig('A', [(23, 77), (80, 105)], include_chain_id=False) == "23-77/80-105"


# ---------------------------------------------------------------------------
# validate_design_length
# ---------------------------------------------------------------------------

def test_validate_design_length_empty_is_valid():
    assert validate_design_length('') is True
    assert validate_design_length(None) is True


def test_validate_design_length_single_number():
    assert validate_design_length('60') is True


def test_validate_design_length_range():
    assert validate_design_length('70-80') is True


def test_validate_design_length_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid design_length format"):
        validate_design_length('abc')


def test_validate_design_length_range_start_after_end_raises():
    with pytest.raises(ValueError, match="Start must be less than or equal to end"):
        validate_design_length('80-70')


# ---------------------------------------------------------------------------
# get_protein_chains
# ---------------------------------------------------------------------------

def test_get_protein_chains_filters_hetatm_and_sorts(tmp_path, pdb_atom_line):
    lines = [
        pdb_atom_line(serial=1, chain='A', resnum=3),
        pdb_atom_line(serial=2, chain='A', resnum=1),
        pdb_atom_line(serial=3, chain='A', resnum=2),
        "HETATM    4  O   HOH A  50      10.000  10.000  10.000  1.00  0.00           O",
    ]
    structure = _parse(tmp_path, pdb_atom_line, lines)
    chains = get_protein_chains(structure)
    assert dict(chains) == {'A': [1, 2, 3]}


# ---------------------------------------------------------------------------
# generate_contig_binder_denovo / monomer_denovo / partialdiff variants
# ---------------------------------------------------------------------------

def test_generate_contig_binder_denovo_with_design_length():
    chain_residues = {'A': list(range(1, 78)), 'B': list(range(23, 78)) + list(range(80, 106))}
    contig = generate_contig_binder_denovo(chain_residues, design_length='60')
    assert contig == "[A1-77/0 B23-77/B80-105/0 60]"


def test_generate_contig_binder_denovo_without_design_length():
    chain_residues = {'A': [1, 2, 3]}
    assert generate_contig_binder_denovo(chain_residues) == "[A1-3/0]"


def test_generate_contig_monomer_denovo_requires_design_length():
    with pytest.raises(ValueError, match="design_length is required"):
        generate_contig_monomer_denovo(None)


def test_generate_contig_monomer_denovo_with_length():
    # Bare single-value design_length is expanded to 'N-N' so RFdiffusion's Hydra CLI parses the
    # contig as a string rather than an int (see generate_contig_monomer_denovo docstring).
    assert generate_contig_monomer_denovo('60') == "[60-60]"
    assert generate_contig_monomer_denovo('70-80') == "[70-80]"


def test_generate_contig_monomer_partialdiff():
    chain_residues = {'A': list(range(1, 156))}
    assert generate_contig_monomer_partialdiff(chain_residues) == "[155-155]"


def test_generate_contig_binder_partialdiff():
    chain_residues = {'A': list(range(1, 89)), 'B': list(range(89, 204))}
    assert generate_contig_binder_partialdiff(chain_residues) == "[88-88/0 B89-203]"


def test_generate_contig_binder_partialdiff_requires_two_chains():
    with pytest.raises(ValueError, match="requires at least 2 chains"):
        generate_contig_binder_partialdiff({'A': [1, 2, 3]})


# ---------------------------------------------------------------------------
# generate_contig_string (main dispatcher)
# ---------------------------------------------------------------------------

def test_generate_contig_string_denovo_monomer_no_pdb_parse_needed(tmp_path, pdb_atom_line):
    # design_length-only monomer denovo path shouldn't need a valid pdb_file
    result = generate_contig_string('/nonexistent/path.pdb', design_mode='rfd_denovo',
                                     design_length='60', is_binder=False)
    assert result == "[60-60]"


def test_generate_contig_string_denovo_binder(tmp_path, pdb_atom_line):
    lines = [pdb_atom_line(serial=i, chain='A', resnum=i) for i in range(1, 4)]
    path = _write_pdb(tmp_path, pdb_atom_line, lines)
    result = generate_contig_string(str(path), design_mode='rfd_denovo', design_length='60', is_binder=True)
    assert result == "[A1-3/0 60]"


def test_generate_contig_string_partialdiff_monomer(tmp_path, pdb_atom_line):
    lines = [pdb_atom_line(serial=i, chain='A', resnum=i) for i in range(1, 4)]
    path = _write_pdb(tmp_path, pdb_atom_line, lines)
    result = generate_contig_string(str(path), design_mode='rfd_partialdiff', is_binder=False)
    assert result == "[3-3]"


def test_generate_contig_string_partialdiff_binder(tmp_path, pdb_atom_line):
    lines = (
        [pdb_atom_line(serial=i, chain='A', resnum=i) for i in range(1, 4)]
        + [pdb_atom_line(serial=i + 3, chain='B', resnum=i, x=20.0) for i in range(1, 4)]
    )
    path = _write_pdb(tmp_path, pdb_atom_line, lines)
    result = generate_contig_string(str(path), design_mode='rfd_partialdiff', is_binder=True)
    assert result == "[3-3/0 B1-3]"


def test_generate_contig_string_invalid_design_mode_raises(tmp_path, pdb_atom_line):
    with pytest.raises(ValueError, match="Invalid design_mode"):
        generate_contig_string('/nonexistent/path.pdb', design_mode='bogus_mode')


def test_generate_contig_string_invalid_design_length_raises(tmp_path, pdb_atom_line):
    with pytest.raises(ValueError, match="Invalid design_length format"):
        generate_contig_string('/nonexistent/path.pdb', design_mode='rfd_denovo',
                                design_length='abc', is_binder=False)


def test_generate_contig_string_no_protein_chains_raises(tmp_path, pdb_atom_line):
    lines = ["HETATM    1  O   HOH A  50      10.000  10.000  10.000  1.00  0.00           O"]
    path = _write_pdb(tmp_path, pdb_atom_line, lines)
    with pytest.raises(ValueError, match="No protein chains found"):
        generate_contig_string(str(path), design_mode='rfd_partialdiff', is_binder=False)
