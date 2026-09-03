"""Tier A unit tests for analyse_bindcraft.py."""
import pytest

from analyse_bindcraft import (
    parse_trajectory_time,
    get_chain_length,
    transform_interface_residues,
    parse_interface_residues,
    generate_inpaint_seq,
    parse_design_name,
    construct_pdb_filename,
    create_metadata_from_row,
    swap_and_renumber_chains,
)
from Bio.PDB import PDBParser


def _write_pdb(path, chain_resnums, pdb_atom_line):
    """chain_resnums: list of (chain, resnum) tuples in file order."""
    with open(path, 'w') as f:
        for chain, resnum in chain_resnums:
            f.write(pdb_atom_line(chain=chain, resnum=resnum) + "\n")


class TestParseTrajectoryTime:
    def test_hours_minutes_seconds(self):
        assert parse_trajectory_time("1 hours, 2 minutes, 3 seconds") == 3723

    def test_singular_forms(self):
        assert parse_trajectory_time("1 hour, 1 minute, 1 second") == 3661

    def test_only_minutes(self):
        assert parse_trajectory_time("5 minutes") == 300

    def test_only_seconds(self):
        assert parse_trajectory_time("45 seconds") == 45

    def test_empty_string_returns_zero(self):
        assert parse_trajectory_time("") == 0

    def test_no_matches_returns_zero(self):
        assert parse_trajectory_time("unknown format") == 0


class TestGetChainLength:
    def test_counts_only_standard_residues(self, pdb_atom_line, tmp_path):
        pdb = tmp_path / "a.pdb"
        _write_pdb(pdb, [('A', 1), ('A', 2), ('A', 3)], pdb_atom_line)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('p', str(pdb))
        chain = structure[0]['A']
        assert get_chain_length(chain) == 3


class TestTransformInterfaceResidues:
    def test_b_prefix_becomes_a(self):
        assert transform_interface_residues("B1,B2,B3") == "A1,A2,A3"

    def test_non_b_chains_unchanged(self):
        assert transform_interface_residues("C1,B2") == "C1,A2"

    def test_strips_whitespace(self):
        assert transform_interface_residues("B1, B2 , B3") == "A1,A2,A3"


class TestParseInterfaceResidues:
    def test_extracts_residue_numbers(self):
        assert parse_interface_residues("A1,A2,A3") == {1, 2, 3}

    def test_handles_whitespace(self):
        assert parse_interface_residues("A1, A2 , A3") == {1, 2, 3}

    def test_multi_digit_numbers(self):
        assert parse_interface_residues("A10,A123") == {10, 123}


class TestGenerateInpaintSeq:
    def test_default_no_interface(self):
        result = generate_inpaint_seq(3, 2)
        assert result == [False, False, False, True, True]

    def test_fix_interface_sets_chain_a_positions_true(self):
        result = generate_inpaint_seq(5, 2, interface_res_str="A2,A4", fix_interface=True)
        assert result == [False, True, False, True, False, True, True]

    def test_fix_interface_false_ignores_interface_str(self):
        result = generate_inpaint_seq(3, 2, interface_res_str="A1,A2", fix_interface=False)
        assert result == [False, False, False, True, True]

    def test_interface_residue_out_of_chain_a_range_is_ignored(self):
        # res_num 99 is beyond chain_a_length=3, must not raise or corrupt the array
        result = generate_inpaint_seq(3, 2, interface_res_str="A99", fix_interface=True)
        assert result == [False, False, False, True, True]

    def test_no_interface_res_str_with_fix_interface_true(self):
        result = generate_inpaint_seq(2, 2, interface_res_str=None, fix_interface=True)
        assert result == [False, False, True, True]


class TestParseDesignName:
    def test_valid_design_name(self):
        assert parse_design_name("Batch_0_l146_s894229") == (0, 146, 894229)

    def test_case_insensitive(self):
        assert parse_design_name("batch_1_L50_S123") == (1, 50, 123)

    def test_non_matching_returns_none(self):
        assert parse_design_name("not_a_valid_name") is None


class TestConstructPdbFilename:
    def test_construct_pdb_filename(self):
        assert construct_pdb_filename(0, 146, 894229) == "batch_0_l146_s894229.pdb"

    def test_roundtrip_with_parse_design_name(self):
        parsed = parse_design_name("Batch_2_l80_s42")
        assert construct_pdb_filename(*parsed) == "batch_2_l80_s42.pdb"


class TestCreateMetadataFromRow:
    def _row(self, **overrides):
        row = {
            'InterfaceResidues': 'B1,B2',
            'Length': '10',
            'pLDDT': '85.5',
            'Target_RMSD': '1.2',
            'TrajectoryTime': '1 hours, 0 minutes, 0 seconds',
        }
        row.update(overrides)
        return row

    def test_builds_expected_metadata_dict(self):
        row = self._row()
        metadata = create_metadata_from_row(row, fold_id=3, chain_a_length=5, chain_b_length=4)
        assert metadata == {
            'fold_id': 3,
            'bc_length': 10,
            'bc_plddt': 85.5,
            'bc_rmsd_target': 1.2,
            'bc_intface_res': 'A1,A2',
            'bc_time': 3600,
            'bc_inpaint_seq': [False, False, False, False, False, True, True, True, True],
        }

    def test_fix_interface_true_sets_positions(self):
        row = self._row(InterfaceResidues='B1,B3')
        metadata = create_metadata_from_row(
            row, fold_id=0, chain_a_length=5, chain_b_length=2, fix_interface=True
        )
        assert metadata['bc_inpaint_seq'] == [True, False, True, False, False, True, True]


class TestSwapAndRenumberChains:
    def test_swaps_and_renumbers(self, pdb_atom_line, tmp_path):
        input_pdb = tmp_path / "in.pdb"
        # Original chain A (target, 2 residues), chain B (binder, 3 residues)
        with open(input_pdb, 'w') as f:
            f.write(pdb_atom_line(chain='A', resnum=1) + "\n")
            f.write(pdb_atom_line(chain='A', resnum=2) + "\n")
            f.write(pdb_atom_line(chain='B', resnum=1) + "\n")
            f.write(pdb_atom_line(chain='B', resnum=2) + "\n")
            f.write(pdb_atom_line(chain='B', resnum=3) + "\n")

        output_pdb = tmp_path / "out.pdb"
        chain_a_length, chain_b_length = swap_and_renumber_chains(str(input_pdb), str(output_pdb))

        assert (chain_a_length, chain_b_length) == (3, 2)

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('p', str(output_pdb))
        model = structure[0]
        new_chain_a_resnums = sorted(res.id[1] for res in model['A'])
        new_chain_b_resnums = sorted(res.id[1] for res in model['B'])
        assert new_chain_a_resnums == [1, 2, 3]
        assert new_chain_b_resnums == [4, 5]

    def test_missing_chain_raises_value_error(self, pdb_atom_line, tmp_path):
        input_pdb = tmp_path / "in.pdb"
        with open(input_pdb, 'w') as f:
            f.write(pdb_atom_line(chain='A', resnum=1) + "\n")

        output_pdb = tmp_path / "out.pdb"
        with pytest.raises(ValueError):
            swap_and_renumber_chains(str(input_pdb), str(output_pdb))
