"""Tests for binder validator."""

import pytest

from bindsweeper.binder_validator import BinderValidator


class TestBinderValidator:
    """Test BinderValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return BinderValidator()

    def test_supported_modes(self, validator):
        """Test that all expected modes are supported."""
        modes = validator.get_supported_modes()
        assert "rfd_denovo" in modes
        assert "rfd_foldcond" in modes
        assert "rfd_motifscaff" in modes
        assert "rfd_partialdiff" in modes
        assert "bindcraft_denovo" in modes
        assert "boltzgen_denovo" in modes
        assert "boltzgen_motifscaff" in modes

    def test_rfd_denovo_hotspot_null(self, validator):
        """Test rfd_denovo with null hotspot_residues."""
        # Should accept None
        assert validator.validate_parameter("rfd_denovo", "hotspot_residues", None)
        
        # Should accept valid string
        assert validator.validate_parameter("rfd_denovo", "hotspot_residues", "A56,A115")

    def test_rfd_denovo_hotspot_empty_string_fails(self, validator):
        """Test rfd_denovo rejects empty string for hotspot_residues."""
        with pytest.raises(ValueError, match="does not match pattern"):
            validator.validate_parameter("rfd_denovo", "hotspot_residues", "")

    def test_rfd_denovo_hotspot_invalid_format(self, validator):
        """Test rfd_denovo rejects invalid hotspot format."""
        with pytest.raises(ValueError, match="does not match pattern"):
            validator.validate_parameter("rfd_denovo", "hotspot_residues", "A56,115")  # Missing chain
        
        with pytest.raises(ValueError, match="does not match pattern"):
            validator.validate_parameter("rfd_denovo", "hotspot_residues", "56,115")  # Missing chains

    def test_bindcraft_denovo_hotspot_null(self, validator):
        """Test bindcraft_denovo with null hotspot_residues."""
        # Should accept None
        assert validator.validate_parameter("bindcraft_denovo", "hotspot_residues", None)
        
        # Should accept valid string
        assert validator.validate_parameter("bindcraft_denovo", "hotspot_residues", "B135,B146")

    def test_bindcraft_denovo_hotspot_empty_string_fails(self, validator):
        """Test bindcraft_denovo rejects empty string for hotspot_residues."""
        with pytest.raises(ValueError, match="does not match pattern"):
            validator.validate_parameter("bindcraft_denovo", "hotspot_residues", "")

    def test_bindcraft_denovo_required_params(self, validator):
        """Test bindcraft_denovo has expected parameters."""
        params = validator.get_supported_parameters("bindcraft_denovo")
        assert "input_pdb" in params
        assert "design_length" in params
        assert "hotspot_residues" in params

    def test_bindcraft_denovo_input_pdb_validation(self, validator):
        """Test bindcraft_denovo input_pdb validation."""
        # Valid PDB path
        assert validator.validate_parameter("bindcraft_denovo", "input_pdb", "/path/to/file.pdb")
        
        # Invalid - not ending with .pdb
        with pytest.raises(ValueError, match="does not match pattern"):
            validator.validate_parameter("bindcraft_denovo", "input_pdb", "/path/to/file.txt")

    def test_bindcraft_denovo_design_length_validation(self, validator):
        """Test bindcraft_denovo design_length validation."""
        # Valid single value
        assert validator.validate_parameter("bindcraft_denovo", "design_length", "60")
        
        # Valid range
        assert validator.validate_parameter("bindcraft_denovo", "design_length", "60-100")
        
        # Invalid format
        with pytest.raises(ValueError, match="does not match pattern"):
            validator.validate_parameter("bindcraft_denovo", "design_length", "60-100-150")

    def test_rfd_foldcond_hotspot_null(self, validator):
        """Test rfd_foldcond with null hotspot_residues."""
        # Should accept None
        assert validator.validate_parameter("rfd_foldcond", "hotspot_residues", None)
        
        # Should accept valid string
        assert validator.validate_parameter("rfd_foldcond", "hotspot_residues", "A56")

    def test_validate_config_with_null_hotspots(self, validator):
        """Test validating full config with null hotspot sweep."""
        mode = "bindcraft_denovo"
        sweep_params = {
            "hotspot_residues": {
                "values": [None, "B135,B146"]
            }
        }
        fixed_params = {
            "input_pdb": "/path/to/target.pdb",
            "design_length": "60-100"
        }
        
        # Should not raise
        assert validator.validate_config(mode, sweep_params, fixed_params)

    def test_unknown_mode_raises(self, validator):
        """Test that unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            validator.validate_parameter("unknown_mode", "some_param", "value")

    def test_unknown_parameter_raises(self, validator):
        """Test that unknown parameter raises ValueError."""
        with pytest.raises(ValueError, match="not supported for mode"):
            validator.validate_parameter("rfd_denovo", "unknown_param", "value")

    def test_boltzgen_denovo_required_params(self, validator):
        """Test boltzgen_denovo has expected parameters."""
        params = validator.get_supported_parameters("boltzgen_denovo")
        assert "input_pdb" in params
        assert "design_length" in params
        assert "hotspot_residues" in params
        assert "bg_not_binding_residues" in params
        assert "flexible_residues" in params

    def test_boltzgen_denovo_hotspot_null(self, validator):
        """Test boltzgen_denovo with null hotspot_residues."""
        assert validator.validate_parameter("boltzgen_denovo", "hotspot_residues", None)

    def test_boltzgen_denovo_hotspot_requires_chain(self, validator):
        """Test boltzgen_denovo hotspot_residues supports multi-letter chains, ranges, and bare chains."""
        assert validator.validate_parameter("boltzgen_denovo", "hotspot_residues", "A56,A115-120,B")

        with pytest.raises(ValueError, match="does not match pattern"):
            # Missing chain identifier is invalid for BoltzGen modes
            validator.validate_parameter("boltzgen_denovo", "hotspot_residues", "56,115")

    def test_boltzgen_motifscaff_required_params(self, validator):
        """Test boltzgen_motifscaff has expected parameters."""
        params = validator.get_supported_parameters("boltzgen_motifscaff")
        assert "input_pdb" in params
        assert "motifscaff_spec" in params
        assert "motifscaff_inpaint_seq" in params
        assert "flexible_residues" in params
        assert "design_length" not in params  # architecture length is unchanged unless motifscaff_spec is set

    def test_boltzgen_motifscaff_spec_validation(self, validator):
        """Test boltzgen_motifscaff motifscaff_spec validation."""
        assert validator.validate_parameter(
            "boltzgen_motifscaff", "motifscaff_spec", "7-10,A1-60,5,A70-100,10"
        )
        assert validator.validate_parameter("boltzgen_motifscaff", "motifscaff_spec", None)

        with pytest.raises(ValueError, match="does not match pattern"):
            validator.validate_parameter("boltzgen_motifscaff", "motifscaff_spec", "B1-60")

    def test_boltzgen_motifscaff_inpaint_seq_validation(self, validator):
        """Test boltzgen_motifscaff motifscaff_inpaint_seq only accepts chain A tokens."""
        assert validator.validate_parameter(
            "boltzgen_motifscaff", "motifscaff_inpaint_seq", "A10-50,A60"
        )

        with pytest.raises(ValueError, match="does not match pattern"):
            validator.validate_parameter("boltzgen_motifscaff", "motifscaff_inpaint_seq", "B10-50")

    def test_rfd_partialdiff_required_params(self, validator):
        """Test rfd_partialdiff has expected parameters."""
        params = validator.get_supported_parameters("rfd_partialdiff")
        assert "input_pdb" in params
        assert "rfd_partialdiff_timesteps" in params
        assert "rfd_partialdiff_spec" in params

    def test_rfd_partialdiff_spec_validation(self, validator):
        """Test rfd_partialdiff_spec validation: keep tokens plus fixed-count diffuse tokens only."""
        assert validator.validate_parameter("rfd_partialdiff", "rfd_partialdiff_spec", "A1-19,59")
        assert validator.validate_parameter("rfd_partialdiff", "rfd_partialdiff_spec", None)

        with pytest.raises(ValueError, match="does not match pattern"):
            # Ranges are not allowed for diffuse tokens (length cannot change)
            validator.validate_parameter("rfd_partialdiff", "rfd_partialdiff_spec", "A1-19,50-59")
