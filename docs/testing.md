[🏠 ProteinDJ](../README.md) > Testing

# ProteinDJ Testing

ProteinDJ has three complementary test suites, from fastest/most granular to slowest/most end-to-end:

| Layer | Location | Runtime | What it checks |
|-------|----------|---------|-----------------|
| pytest | `scripts/`, `bindsweeper/` | seconds | Pure-Python logic in the scripts invoked by Nextflow modules |
| nf-test unit | `tests/unit` | seconds | Groovy helper classes/functions (`lib/*.groovy`) - no containers, no GPU |
| nf-test module (CPU) | `tests/modules` (tag `cpu`) | ~5-30s each | Real `python_tools.sif` container process wiring/logic, no GPU |
| nf-test module (GPU) | `tests/modules` (tag `gpu`) | ~1-10 min each | Real GPU processes (RFdiffusion, ProteinMPNN, FAMPNN, AF2, Boltz, BindCraft, BoltzGen) |
| `end2end_test.sh` | `scripts/end2end_test.sh` | ~60 min | Full pipeline, all 13 supported modes, on 4 A30 GPUs |

## Using the testing suite

A [Makefile](../Makefile) at the repo root wraps the common commands:

- `make test-pytest` - run both pytest suites (fast, run on every script change).
- `make test-nf-unit` - run the Groovy unit tests (fast, no containers).
- `make test-nf-cpu` - run all CPU/`python_tools` nf-test module tests serially.
- `make test-nf-gpu` - run all GPU-tagged nf-test module tests serially (slow - RunBC alone takes ~8 min).
- `make test-all` - pytest -> nf-test unit -> nf-test cpu -> nf-test gpu, serially.

When iterating on a single module, target just that file instead of a whole tag, e.g.:

```bash
nf-test test tests/modules/bindcraft.runbc.nf.test --tag gpu --profile milton
```

## End-to-end pipeline testing (`end2end_test.sh`)

### Overview

The `end2end_test.sh` script is a comprehensive testing framework for the ProteinDJ pipeline that automatically tests all supported pipeline modes and generates detailed reports. It helps ensure pipeline reliability and provides insights into the performance of different protein design modes. A full test takes an hour on 4 A30 GPUs.

### Features

- ✅ **Automated Testing**: Tests all 13 pipeline modes automatically
- 📊 **Comprehensive Reporting**: 

      - Generates detailed text reports
      - Tracks execution time and success rates
      - Analyzes generated files and directory structures
- 🔍 **Error Capture**: Captures and reports detailed error information


### Basic Usage

```bash
# Load nextflow module/environment if needed
module load nextflow/24.10.5

# Run tests with compute profile and output directory
./scripts/end2end_test.sh <compute_profile> <output_directory>

# Examples with different compute profiles
./scripts/end2end_test.sh milton /vast/scratch/users/$USER     # WEHI HPC cluster environment
./scripts/end2end_test.sh apptainer /home/$USER/test_outputs   # Apptainer containers
./scripts/end2end_test.sh singularity /tmp/proteindj_tests     # Singularity containers
```

#### Prerequisites

1. **Nextflow**: Version 24.10.5 or later
2. **Environment**: Access to the compute environment specified
3. **Storage**: Sufficient space in the specified output directory for test outputs 

### Pipeline Modes Tested

The script tests the following ProteinDJ pipeline profiles:

#### Monomer Profiles
- `rfd_denovo_monomer` - De novo protein design (RFdiffusion)
- `rfd_foldcond_monomer` - Conditional folding design (RFdiffusion)
- `rfd_motifscaff_monomer` - Motif-based scaffolding (RFdiffusion)
- `rfd_partialdiff_monomer` - Partial diffusion design (RFdiffusion)
- `boltzgen_denovo_monomer` - De novo protein design (BoltzGen)
- `boltzgen_motifscaff_monomer` - Motif-based scaffolding (BoltzGen)

#### Binder Profiles
- `bindcraft_denovo` - De novo binder design (BindCraft)
- `rfd_denovo_binder` - De novo binder design (RFdiffusion)
- `rfd_foldcond_binder` - Conditional binder folding (RFdiffusion)
- `rfd_motifscaff_binder` - Binder motif scaffolding (RFdiffusion)
- `rfd_partialdiff_binder` - Partial diffusion binder design (RFdiffusion)
- `boltzgen_denovo_binder` - De novo binder design (BoltzGen)
- `boltzgen_motifscaff_binder` - Binder motif scaffolding (BoltzGen)

### Command Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `compute_profile` | Yes | The compute profile to use for testing | e.g. `milton`, `apptainer`, `singularity` |
| `output_directory` | Yes | Base directory where test outputs will be stored (< 50 MB) | e.g. `/vast/scratch/user/$USER`, `/home/$USER/tests` |

**Available Compute Profiles:**
- `milton` - WEHI HPC cluster environment
- `apptainer` - Uses Apptainer containers for execution
- `singularity` - Uses Singularity containers for execution

### Output Structure

The script creates a timestamped directory with the following structure:

```
test_results_YYYYMMDD_HHMMSS/
├── logs/
│   ├── test_execution.log          # Main execution log
│   ├── rfd_denovo_monomer.log       # Individual mode logs
│   ├── rfd_denovo_binder.log
│   └── ...
└── analysis.txt                    # Comprehensive analysis and summary
```

### Report Types

#### Text Analysis (`analysis.txt`)
- **Comprehensive report** combining all mode results
- **Summary statistics** with pass/fail counts and success rates
- **Output analysis** for each successful run
- **Performance metrics** and timing data

#### Performance Metrics
- **Duration**: Time taken for each mode (in seconds)
- **Success Rate**: Percentage of modes that passed
- **Output Analysis**: Count of generated files (PDB, CSV) and number of designs in CSV files

#### Output Directory

Test outputs are saved to:
```
<output_directory>/pdj_test_${mode}_${TIMESTAMP}
```

#### Error Recovery

If tests fail:
1. Check the text analysis report (`analysis.txt`) for detailed information
2. Review individual log files in the `logs/` directory
3. Verify input data and configuration files
4. Ensure compute resources are available

---

**Note**: This testing framework is designed to validate the ProteinDJ pipeline functionality across all supported modes. Regular testing helps ensure pipeline reliability and catches regressions early in the development process.

[⬅️ Back to Main README](../README.md)