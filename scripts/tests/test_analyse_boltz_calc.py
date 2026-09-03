"""Tests for analyse_boltz_calc.py (a.k.a. ipsae.py).

This script is NOT structured as an importable module: it parses sys.argv and opens
output files for writing at module scope, then runs its entire pipeline (reading a PDB
file, loading Boltz2 PAE/pLDDT npz files, etc.) as top-level code immediately on import.
Rather than fabricating a full set of valid Boltz2 npz/json/pdb fixtures just to reach
import time safely, we extract only the pure, side-effect-free top-level function
definitions via the `ast` module and exec them in isolation. This avoids executing any
of the file-opening / sys.argv-parsing code while still testing the real function bodies
verbatim from the script.
"""
import ast
import math
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "analyse_boltz_calc.py"

_PURE_FUNCTION_NAMES = {
    "ptm_func",
    "calc_d0",
    "calc_d0_array",
    "parse_pdb_atom_line",
    "contiguous_ranges",
    "init_chainpairdict_zeros",
    "init_chainpairdict_npzeros",
    "init_chainpairdict_set",
    "classify_chains",
}


def _load_pure_functions():
    source = SCRIPT_PATH.read_text()
    tree = ast.parse(source, filename=str(SCRIPT_PATH))
    nodes = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in _PURE_FUNCTION_NAMES
    ]
    found_names = {n.name for n in nodes}
    missing = _PURE_FUNCTION_NAMES - found_names
    assert not missing, f"Expected functions not found in {SCRIPT_PATH}: {missing}"

    module_ast = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)
    namespace = {"np": np, "math": math}
    exec(compile(module_ast, filename=str(SCRIPT_PATH), mode="exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in _PURE_FUNCTION_NAMES})


@pytest.fixture(scope="module")
def ipsae():
    return _load_pure_functions()


class TestPtmFunc:
    def test_zero_distance_gives_ptm_of_one(self, ipsae):
        assert ipsae.ptm_func(0.0, 10.0) == pytest.approx(1.0)

    def test_distance_equal_to_d0_gives_half(self, ipsae):
        assert ipsae.ptm_func(10.0, 10.0) == pytest.approx(0.5)

    def test_larger_distance_gives_smaller_value(self, ipsae):
        low = ipsae.ptm_func(5.0, 10.0)
        high = ipsae.ptm_func(20.0, 10.0)
        assert high < low


class TestCalcD0:
    def test_short_length_clamped_to_27(self, ipsae):
        assert ipsae.calc_d0(10, "protein") == ipsae.calc_d0(27, "protein")

    def test_protein_formula_value_at_min_length(self, ipsae):
        expected = 1.24 * (27 - 15) ** (1.0 / 3.0) - 1.8
        assert ipsae.calc_d0(27, "protein") == pytest.approx(expected)

    def test_protein_min_value_clamp_applies_when_formula_below_one(self, ipsae):
        # Formula value is only < 1.0 for L below ~26.3 (pre-clamp), which calc_d0
        # never reaches since L is clamped to >= 27 first -- so min_value=1.0 is
        # effectively a defensive floor rather than one hit by real inputs. Confirm
        # calc_d0 never returns below 1.0 regardless.
        assert ipsae.calc_d0(1, "protein") >= 1.0

    def test_nucleic_acid_min_value_is_two(self, ipsae):
        assert ipsae.calc_d0(27, "nucleic_acid") == pytest.approx(2.0)

    def test_large_length_uses_formula_value(self, ipsae):
        expected = 1.24 * (200 - 15) ** (1.0 / 3.0) - 1.8
        assert ipsae.calc_d0(200, "protein") == pytest.approx(expected)


class TestCalcD0Array:
    def test_matches_scalar_calc_d0_elementwise(self, ipsae):
        lengths = [10, 27, 100, 200]
        array_result = ipsae.calc_d0_array(lengths, "protein")
        for length, value in zip(lengths, array_result):
            assert value == pytest.approx(ipsae.calc_d0(length, "protein"))

    def test_nucleic_acid_min_value_applied_elementwise(self, ipsae):
        result = ipsae.calc_d0_array([10, 27], "nucleic_acid")
        assert all(v == pytest.approx(2.0) for v in result)


class TestParsePdbAtomLine:
    def test_parses_fixed_column_atom_line(self, ipsae):
        line = "ATOM    123  CA  ALA A  15      11.111  22.222  33.333  1.00 20.00           C"
        atom = ipsae.parse_pdb_atom_line(line)
        assert atom["atom_num"] == 123
        assert atom["atom_name"] == "CA"
        assert atom["residue_name"] == "ALA"
        assert atom["chain_id"] == "A"
        assert atom["residue_seq_num"] == 15
        assert atom["x"] == pytest.approx(11.111)
        assert atom["y"] == pytest.approx(22.222)
        assert atom["z"] == pytest.approx(33.333)


class TestContiguousRanges:
    def test_empty_returns_none(self, ipsae):
        assert ipsae.contiguous_ranges(set()) is None

    def test_single_number(self, ipsae):
        assert ipsae.contiguous_ranges({5}) == "5"

    def test_contiguous_range_formatted_with_dash(self, ipsae):
        assert ipsae.contiguous_ranges({1, 2, 3, 4}) == "1-4"

    def test_gaps_produce_multiple_ranges_joined_by_plus(self, ipsae):
        assert ipsae.contiguous_ranges({1, 2, 5, 7, 8, 9}) == "1-2+5+7-9"


class TestClassifyChains:
    def test_all_protein_residues(self, ipsae):
        chains = np.array(["A", "A", "B", "B"])
        residue_types = np.array(["ALA", "GLY", "SER", "LEU"])
        result = ipsae.classify_chains(chains, residue_types)
        assert result == {"A": "protein", "B": "protein"}

    def test_nucleic_acid_chain_detected(self, ipsae):
        chains = np.array(["A", "A", "B", "B"])
        residue_types = np.array(["ALA", "GLY", "DA", "DC"])
        result = ipsae.classify_chains(chains, residue_types)
        assert result == {"A": "protein", "B": "nucleic_acid"}


class TestInitChainpairdicts:
    def test_zeros_dict_excludes_self_pairs(self, ipsae):
        result = ipsae.init_chainpairdict_zeros(["A", "B"])
        assert result == {"A": {"B": 0}, "B": {"A": 0}}

    def test_npzeros_dict_creates_arrays_of_given_size(self, ipsae):
        result = ipsae.init_chainpairdict_npzeros(["A", "B"], 5)
        assert result["A"]["B"].shape == (5,)
        assert np.all(result["A"]["B"] == 0)

    def test_set_dict_creates_empty_sets(self, ipsae):
        result = ipsae.init_chainpairdict_set(["A", "B"])
        assert result["A"]["B"] == set()
        result["A"]["B"].add(1)
        assert result["B"]["A"] == set()  # independent sets, not shared reference
