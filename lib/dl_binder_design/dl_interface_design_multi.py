#!/usr/bin/env python

import argparse
import gc
import glob
import io
import json
import os
import re
import sys
import tempfile
import time
from itertools import zip_longest

import numpy as np
import torch

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent, 'include'))
sys.path.append(os.path.join(parent, 'af2_initial_guess'))

import util_protein_mpnn as mpnn_util
import common_util


def range1(iterable):
    return range(1, iterable + 1)


# ── OpenMM module-level imports ───────────────────────────────────────────────
try:
    import openmm as _openmm
    from openmm import app as _openmm_app, unit as _openmm_unit
    from openmm import Platform as _openmm_Platform, OpenMMException as _OpenMMException
    from pdbfixer import PDBFixer as _PDBFixer
    _HAS_OPENMM = True
except ImportError:
    _HAS_OPENMM = False

# ForceField singleton (Amber14 + OBC2 implicit solvent) — avoids repeated XML parsing
_OPENMM_FF_SINGLETON = None


def _get_openmm_forcefield():
    """Return a cached ForceField instance (Amber14 + OBC2 implicit solvent)."""
    global _OPENMM_FF_SINGLETON
    if not _HAS_OPENMM:
        raise ImportError("OpenMM is not available")
    if _OPENMM_FF_SINGLETON is None:
        _OPENMM_FF_SINGLETON = _openmm_app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    return _OPENMM_FF_SINGLETON


def _k_kj_per_nm2(k_kcal_a2):
    """Convert kcal/mol/Å² to kJ/mol/nm²."""
    return k_kcal_a2 * 4.184 * 100.0


def _extract_residue_bfactors(pdb_text):
    """Extract per-residue B-factors from PDB text (CA preferred, else first ATOM per residue).
    Returns {(chain, resseq, icode): float} for B-factor restoration after relaxation.
    """
    bfactors = {}
    seen_ca = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        chain = line[21]
        resseq = line[22:26]
        icode = line[26]
        key = (chain, resseq, icode)
        try:
            b = float(line[60:66])
        except (ValueError, IndexError):
            continue
        if atom_name == "CA":
            bfactors[key] = b
            seen_ca.add(key)
        elif key not in seen_ca and key not in bfactors:
            bfactors[key] = b
    return bfactors


def _apply_residue_bfactors(pdb_text, bfactors):
    """Write per-residue B-factors back to all ATOM/HETATM records in PDB text."""
    if not bfactors:
        return pdb_text
    lines = []
    for line in pdb_text.splitlines(keepends=True):
        if (line.startswith("ATOM") or line.startswith("HETATM")) and len(line) >= 66:
            key = (line[21], line[22:26], line[26])
            if key in bfactors:
                line = line[:60] + f"{bfactors[key]:6.2f}" + line[66:]
        lines.append(line)
    return "".join(lines)


def _openmm_relax_pdb_text(pdb_text, max_iterations, restraint_k, platform_name):
    """
    3-stage ramped OpenMM minimisation (FreeBindCraft protocol).

    Stages:
      Pre-ramp: full backbone restraint, no LJ-repulsion, L-BFGS minimisation
      Stage 1:  full restraint (x1.0), no LJ-rep (x0.0), MD shake + minimise
      Stage 2:  reduced restraint (x0.4), LJ-rep (x1.5), MD shake + minimise
      Stage 3:  no restraint (x0.0), max LJ-rep (x3.0), tight tolerance, no MD

    Accept-to-best uses physical energy only (force groups 0+1; LJ-rep group 2
    excluded) so the varying LJ-rep strength does not bias cross-stage comparisons.
    Context is reset to best positions at the start of each stage after the first.
    """
    if not _HAS_OPENMM:
        raise ImportError(
            "OpenMM/PDBFixer is required for relaxation. "
            "Install with: conda install -c conda-forge openmm pdbfixer"
        )

    # Ramp parameters (same as FreeBindCraft defaults)
    RESTRAINT_RAMP = (1.0, 0.4, 0.0)
    LJ_REP_RAMP    = (0.0, 1.5, 3.0)
    LJ_REP_BASE    = 10.0   # kJ/mol base strength
    MD_STEPS       = 1000   # Langevin steps per shake (stages 1 and 2 only)
    RAMP_TOL       = 2.0    # kJ/mol/nm force tolerance (ramp stages)
    FINAL_TOL      = 0.1    # kJ/mol/nm force tolerance (final stage)

    # Preserve input B-factors (AF2 pLDDT) before PDBFixer overwrites them
    _input_bfactors = _extract_residue_bfactors(pdb_text)

    # Write input to a temp file for PDBFixer
    tmp_in = tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False)
    tmp_in.write(pdb_text)
    tmp_in.close()
    in_path = tmp_in.name

    try:
        # ── PDBFixer preparation ───────────────────────────────────────────
        fixer = _PDBFixer(filename=in_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        # ── System setup (Amber14 + OBC2 implicit solvent) ────────────────
        forcefield = _get_openmm_forcefield()
        system = forcefield.createSystem(
            fixer.topology,
            nonbondedMethod=_openmm_app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * _openmm_unit.nanometer,
            constraints=_openmm_app.HBonds,
        )

        # Extract per-atom sigma values from NonbondedForce for LJ-rep
        original_sigmas = []
        nb_force_idx = -1
        for i in range(system.getNumForces()):
            f = system.getForce(i)
            if isinstance(f, _openmm.NonbondedForce):
                nb_force_idx = i
                for p in range(f.getNumParticles()):
                    _, sigma, _ = f.getParticleParameters(p)
                    original_sigmas.append(sigma.value_in_unit(_openmm_unit.nanometer))
                break

        # ── Custom LJ-repulsive force (force group 2) ─────────────────────
        lj_rep_force = None
        k_lj_idx = -1
        if original_sigmas:
            lj_rep_force = _openmm.CustomNonbondedForce(
                "k_rep_lj * (((sigma_particle1 + sigma_particle2) * 0.5 / r)^12)"
            )
            k_lj_idx = lj_rep_force.addGlobalParameter("k_rep_lj", 0.0)
            lj_rep_force.addPerParticleParameter("sigma_particle")
            for s in original_sigmas:
                lj_rep_force.addParticle([s])
            # Match the NonbondedMethod of the main force so all platforms accept the system
            if nb_force_idx != -1:
                nb_f = system.getForce(nb_force_idx)
                nb_method = nb_f.getNonbondedMethod()
                if nb_method == _openmm.NonbondedForce.NoCutoff:
                    lj_rep_force.setNonbondedMethod(_openmm.CustomNonbondedForce.NoCutoff)
                else:
                    lj_rep_force.setNonbondedMethod(_openmm.CustomNonbondedForce.CutoffNonPeriodic)
                    lj_rep_force.setCutoffDistance(nb_f.getCutoffDistance())
                for ex_i in range(nb_f.getNumExceptions()):
                    p1, p2, *_ = nb_f.getExceptionParameters(ex_i)
                    lj_rep_force.addExclusion(p1, p2)
            else:
                lj_rep_force.setNonbondedMethod(_openmm.CustomNonbondedForce.NoCutoff)
            lj_rep_force.setForceGroup(2)
            system.addForce(lj_rep_force)

        # ── Backbone harmonic restraint force (force group 1) ─────────────
        restraint_force = None
        k_r_idx = -1
        if restraint_k > 0:
            restraint_force = _openmm.CustomExternalForce(
                "0.5 * k_restraint * ((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0))"
            )
            k_r_idx = restraint_force.addGlobalParameter("k_restraint", _k_kj_per_nm2(restraint_k))
            restraint_force.addPerParticleParameter("x0")
            restraint_force.addPerParticleParameter("y0")
            restraint_force.addPerParticleParameter("z0")
            bb_atoms = {"N", "CA", "C", "O"}
            for atom in fixer.topology.atoms():
                if atom.name in bb_atoms:
                    xyz = fixer.positions[atom.index].value_in_unit(_openmm_unit.nanometer)
                    restraint_force.addParticle(atom.index, [xyz[0], xyz[1], xyz[2]])
            restraint_force.setForceGroup(1)
            system.addForce(restraint_force)

        # ── Integrator and simulation ──────────────────────────────────────
        if platform_name == "CPU":
            plat_order = ["CPU"]
        elif platform_name == "CUDA":
            plat_order = ["CUDA", "OpenCL", "CPU"]
        else:  # OpenCL or default
            plat_order = ["OpenCL", "CUDA", "CPU"]

        # A fresh integrator must be created for each attempt: once passed to a
        # Simulation constructor that fails, the integrator is in an invalid state.
        simulation = None
        integrator = None
        for p_name in plat_order:
            try:
                plat = _openmm_Platform.getPlatformByName(p_name)
                props = {}
                if p_name == "CUDA":
                    props = {"CudaPrecision": "mixed"}
                elif p_name == "OpenCL":
                    props = {"OpenCLPrecision": "single"}
                integrator = _openmm.LangevinMiddleIntegrator(
                    300 * _openmm_unit.kelvin,
                    1.0 / _openmm_unit.picosecond,
                    0.002 * _openmm_unit.picoseconds,
                )
                simulation = _openmm_app.Simulation(fixer.topology, system, integrator, plat, props)
                print(f"[OpenMM] Using platform: {p_name}")
                break
            except Exception as _plat_err:
                print(f"[OpenMM] Platform {p_name} unavailable: {_plat_err}")
                integrator = None
                continue
        if simulation is None:
            raise _OpenMMException("No suitable OpenMM platform found")

        simulation.context.setPositions(fixer.positions)

        # Log initial energy
        try:
            e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            print(f"[OpenMM] Initial energy: {e0.value_in_unit(_openmm_unit.kilojoule_per_mole):.2f} kJ/mol")
        except Exception:
            pass

        # ── Pre-ramp minimisation (full restraint, LJ-rep=0) ──────────────
        if restraint_force is not None:
            restraint_force.setGlobalParameterDefaultValue(k_r_idx, _k_kj_per_nm2(restraint_k))
            restraint_force.updateParametersInContext(simulation.context)
        if lj_rep_force is not None:
            lj_rep_force.setGlobalParameterDefaultValue(k_lj_idx, 0.0)
            lj_rep_force.updateParametersInContext(simulation.context)
        print("[OpenMM] Pre-ramp minimisation (full restraint, LJ-rep=0)")
        pre_tol = RAMP_TOL * _openmm_unit.kilojoule_per_mole / _openmm_unit.nanometer
        simulation.minimizeEnergy(tolerance=pre_tol, maxIterations=max_iterations)
        try:
            e_pre = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            print(f"[OpenMM] Pre-ramp complete: {e_pre.value_in_unit(_openmm_unit.kilojoule_per_mole):.2f} kJ/mol")
        except Exception:
            pass

        # ── 3-stage ramp ──────────────────────────────────────────────────
        best_energy = float('inf') * _openmm_unit.kilojoule_per_mole
        best_positions = None
        best_stage = None
        ramp_pairs = list(zip_longest(RESTRAINT_RAMP, LJ_REP_RAMP, fillvalue=0.0))
        num_stages = len(ramp_pairs)
        print(f"[OpenMM] Starting {num_stages}-stage ramp "
              f"(restraint factors={list(RESTRAINT_RAMP)}, LJ-rep factors={list(LJ_REP_RAMP)})")
        print("[OpenMM] Accept-to-best compares physical energy only (groups 0+1, LJ-rep excluded)")

        for i_stage, (k_factor_r, k_factor_lj) in enumerate(ramp_pairs):
            stage_num = i_stage + 1

            # Reset to best positions before each stage after the first
            if i_stage > 0 and best_positions is not None:
                simulation.context.setPositions(best_positions)

            # Update force parameters for this stage
            if restraint_force is not None:
                restraint_force.setGlobalParameterDefaultValue(
                    k_r_idx, _k_kj_per_nm2(restraint_k * k_factor_r)
                )
                restraint_force.updateParametersInContext(simulation.context)
            if lj_rep_force is not None:
                lj_rep_force.setGlobalParameterDefaultValue(k_lj_idx, LJ_REP_BASE * k_factor_lj)
                lj_rep_force.updateParametersInContext(simulation.context)

            # Stage header log
            try:
                e_start = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                e_start_str = f"{e_start.value_in_unit(_openmm_unit.kilojoule_per_mole):.2f}"
            except Exception:
                e_start_str = "N/A"
            stage_tol = FINAL_TOL if i_stage == num_stages - 1 else RAMP_TOL
            print(f"[OpenMM] Stage {stage_num}/{num_stages}: "
                  f"restraint_k={restraint_k * k_factor_r:.3f} kcal/mol/A2, "
                  f"LJ-rep_k={LJ_REP_BASE * k_factor_lj:.2f} kJ/mol, "
                  f"tol={stage_tol} kJ/mol/nm  |  start_E={e_start_str} kJ/mol")

            # MD shake (stages 1 and 2 only)
            if MD_STEPS > 0 and i_stage < 2:
                simulation.context.setVelocitiesToTemperature(300 * _openmm_unit.kelvin)
                simulation.step(MD_STEPS)
                try:
                    e_sh = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                    print(f"[OpenMM] Stage {stage_num}: post-shake "
                          f"E={e_sh.value_in_unit(_openmm_unit.kilojoule_per_mole):.2f} kJ/mol")
                except Exception:
                    pass

            # Chunked L-BFGS minimisation with early-stop on negligible improvement
            force_tol = stage_tol * _openmm_unit.kilojoule_per_mole / _openmm_unit.nanometer
            per_chunk = min(200, max_iterations) if max_iterations > 0 else 200
            remaining = max_iterations
            streak = 0
            last_e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            t_min = time.time()
            while True:
                simulation.minimizeEnergy(tolerance=force_tol, maxIterations=per_chunk)
                cur_e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                try:
                    streak = (streak + 1) if (last_e - cur_e) < (0.1 * _openmm_unit.kilojoule_per_mole) else 0
                except Exception:
                    streak = 3
                last_e = cur_e
                if max_iterations > 0:
                    remaining -= per_chunk
                    if remaining <= 0:
                        break
                if streak >= 3:
                    break

            # Accept-to-best: physical energy only (groups 0+1, excluding LJ-rep group 2)
            try:
                phys_state = simulation.context.getState(getEnergy=True, groups={0, 1})
                phys_e = phys_state.getPotentialEnergy()
                phys_val = phys_e.value_in_unit(_openmm_unit.kilojoule_per_mole)
            except Exception:
                phys_e = last_e
                phys_val = last_e.value_in_unit(_openmm_unit.kilojoule_per_mole)

            if phys_e < best_energy:
                best_energy = phys_e
                best_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
                best_stage = stage_num
                print(f"[OpenMM] Stage {stage_num}/{num_stages}: ACCEPTED as new best  "
                      f"E={phys_val:.2f} kJ/mol ({time.time() - t_min:.1f}s min)")
            else:
                prev_val = best_energy.value_in_unit(_openmm_unit.kilojoule_per_mole)
                print(f"[OpenMM] Stage {stage_num}/{num_stages}: REJECTED "
                      f"(E={phys_val:.2f} >= best={prev_val:.2f} kJ/mol); keeping stage {best_stage}")

        # Restore best positions
        if best_positions is not None:
            simulation.context.setPositions(best_positions)
            print(f"[OpenMM] All stages complete. Restoring from stage {best_stage} "
                  f"(best E={best_energy.value_in_unit(_openmm_unit.kilojoule_per_mole):.2f} kJ/mol)")

        # Write relaxed structure to in-memory buffer
        positions = simulation.context.getState(getPositions=True).getPositions()
        out_buf = io.StringIO()
        _openmm_app.PDBFile.writeFile(simulation.topology, positions, out_buf, keepIds=True)
        result = out_buf.getvalue()

        # Restore original B-factors (AF2 pLDDT) — OpenMM writes zeros by default
        result = _apply_residue_bfactors(result, _input_bfactors)

        # Release OpenMM objects to avoid memory accumulation over many cycles
        try:
            del simulation, integrator, system, restraint_force, lj_rep_force, fixer
        except Exception:
            pass
        gc.collect()

        return result

    except _OpenMMException:
        raise  # platform / context failures must propagate — do not silently skip relaxation
    except Exception as e:
        print(f"[OpenMM] ERROR during relaxation: {e}; returning input unchanged")
        return pdb_text
    finally:
        try:
            os.remove(in_path)
        except Exception:
            pass


def _kabsch_rmsd(P, Q):
    if P.shape != Q.shape or P.shape[0] == 0:
        return float('nan')

    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    P_aligned = P_centered @ R
    diff = P_aligned - Q_centered
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def _extract_ca_coords(pdb_lines):
    coords = {}
    order = []
    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue

        key = (line[21], line[22:26], line[26])
        try:
            xyz = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ], dtype=np.float64)
        except ValueError:
            continue

        if key not in coords:
            order.append(key)
        coords[key] = xyz

    return coords, order


def ca_rmsd_from_pdb_lines(pdb_lines_a, pdb_lines_b):
    coords_a, order_a = _extract_ca_coords(pdb_lines_a)
    coords_b, _ = _extract_ca_coords(pdb_lines_b)

    common = [key for key in order_a if key in coords_b]
    if not common:
        return float('nan')

    P = np.stack([coords_a[key] for key in common], axis=0)
    Q = np.stack([coords_b[key] for key in common], axis=0)
    return _kabsch_rmsd(P, Q)


#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()
script_dir = os.path.dirname(os.path.realpath(__file__))

# I/O Arguments
parser.add_argument("-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model')
parser.add_argument("-outpdbdir", type=str, default="outputs", help='The directory to which the output PDB files will be written, used if the -pdbdir arg is active')
parser.add_argument("-runlist", type=str, default='', help="The path of a list of pdb tags to run, only active when the -pdbdir arg is active (default: ''; Run all PDBs)")
parser.add_argument("-checkpoint_name", type=str, default='check.point', help="The name of a file where tags which have finished will be written (default: check.point)")

parser.add_argument("-debug", action="store_true", default=False, help='When active, errors will cause the script to crash and the error message to be printed out (default: False)')

# Design Arguments
parser.add_argument("-relax_max_cycles", type=int, default=1, help="The number of relax cycles to perform on each structure (default: 1)")
parser.add_argument("-output_intermediates", action="store_true", help='Whether to write all intermediate sequences from the relax cycles to disk (default: False)')
parser.add_argument("-seqs_per_struct", type=int, default=1, help="The number of sequences to generate for each input structure (default: 1)")
parser.add_argument("-relax_output", action="store_true", default=False, help='Whether to run relaxation before saving output. (default: False)')
parser.add_argument("-relax_seqs_per_cycle", type=int, default=1, help="Number of intermediate sequences to generate per relax cycle. The sequence with the lowest (best) score is kept (default: 1)")
parser.add_argument("-relax_convergence_rmsd", type=float, default=0.2, help="Convergence criteria 1 of 2. Design is considered converged if the C-alpha RMSD (A) between cycles is <= this threshold (default: 0.2)")
parser.add_argument("-relax_convergence_score", type=float, default=0.1, help="Convergence criteria 2 of 2. Design is considered converged if the improvement in score between cycles is <= this threshold (default: 0.1)")
parser.add_argument("-relax_convergence_max_cycles", type=int, default=1, help="Design is considered converged if it meets both convergence criteria for n consecutive cycles (default: 1)")
parser.add_argument("-relax_max_iterations", type=int, default=1000, help="OpenMM minimization max iterations per relax cycle (default: 1000)")
parser.add_argument("-relax_restraint_k", type=float, default=3.0, help="Backbone harmonic restraint strength in kcal/mol/A^2 (default: 3.0)")
parser.add_argument("-relax_platform", type=str, default="CPU", choices=["CPU", "CUDA", "OpenCL"], help="OpenMM platform to use for minimization (default: CPU)")

# ProteinMPNN-Specific Arguments
parser.add_argument("-checkpoint_path", type=str, default=os.path.join(script_dir, 'ProteinMPNN/vanilla_model_weights/v_48_020.pt'), help=f"The path to the ProteinMPNN weights you wish to use, default {os.path.join(script_dir, 'ProteinMPNN/vanilla_model_weights/v_48_020.pt')}")
parser.add_argument("-temperature", type=float, default=0.1, help='The sampling temperature to use when running ProteinMPNN (default: 0.1)')
parser.add_argument("-augment_eps", type=float, default=0, help='The variance of random noise to add to the atomic coordinates (default 0)')
parser.add_argument("-protein_features", type=str, default='full', help='What type of protein features to input to ProteinMPNN (default: full)')
parser.add_argument("-omit_AAs", type=str, default='CX', help='A string of all residue types (one letter case-insensitive) that you would not like to use for design. Letters not corresponding to residue types will be ignored (default: CX)')
parser.add_argument("-bias_AA_jsonl", type=str, default='', help='The path to a JSON file containing a dictionary mapping residue one-letter names to the bias for that residue eg. {A: -1.1, F: 0.7} (default: \'\'; no bias)')
parser.add_argument("-num_connections", type=int, default=48, help='Number of neighbors each residue is connected to. Do not mess around with this argument unless you have a specific set of ProteinMPNN weights which expects a different number of connections. (default: 48)')

args = parser.parse_args(sys.argv[1:])


class sample_features:
    """
    Struct holding mutable PDB text and parsed sample metadata.
    """

    BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

    def __init__(self, pdb_lines, tag):
        self.pdb_lines = list(pdb_lines)
        self.tag = os.path.basename(tag).split('.')[0]
        self.chains = self._parse_chain_order()
        self.fixed_res = {}

    def clone(self):
        cloned = sample_features(list(self.pdb_lines), self.tag)
        cloned.chains = list(self.chains)
        cloned.fixed_res = {k: list(v) for k, v in self.fixed_res.items()}
        return cloned

    def _iter_atom_records(self):
        for line in self.pdb_lines:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resseq = line[22:26]
            icode = line[26]
            yield line, (chain, resseq, icode)

    def _parse_chain_order(self):
        chains = []
        seen = set()
        for _, (chain, _, _) in self._iter_atom_records():
            if chain in seen:
                continue
            seen.add(chain)
            chains.append(chain)
        return chains

    def _chain_residue_keys(self, chain_id):
        ordered = []
        seen = set()
        for _, key in self._iter_atom_records():
            chain, _, _ = key
            if chain != chain_id or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    def parse_fixed_res(self):
        """
        Parse fixed residues from Rosetta PDB info labels if present.
        The FIXED labels contain PDB residue sequence numbers. We need to map
        them to chain-relative 1-based indices for ProteinMPNN.
        """
        # Collect all residue numbers marked as FIXED
        fixed_pdb_resnums = []
        fixed_re = re.compile(r"PDBinfo-LABEL:\s*(\d+)\s+FIXED")
        for line in self.pdb_lines:
            if "PDBinfo-LABEL" not in line or "FIXED" not in line:
                continue
            match = fixed_re.search(line)
            if match:
                fixed_pdb_resnums.append(int(match.group(1)))

        fixed_pdb_resnums = sorted(set(fixed_pdb_resnums))

        if not self.chains:
            self.fixed_res = {}
            return

        # Build mapping from PDB residue sequence numbers to chain-relative indices
        # For each chain, map: PDB resseq -> 1-based index within that chain
        chain_resnum_to_index = {}
        for chain_id in self.chains:
            chain_resnum_to_index[chain_id] = {}
            chain_keys = self._chain_residue_keys(chain_id)
            for idx, key in enumerate(chain_keys, start=1):
                _, resseq_str, icode = key
                try:
                    pdb_resnum = int(resseq_str.strip())
                    chain_resnum_to_index[chain_id][pdb_resnum] = idx
                except ValueError:
                    continue

        # Map the fixed residue numbers to chain-relative 1-based indices
        self.fixed_res = {chain_id: [] for chain_id in self.chains}
        for pdb_resnum in fixed_pdb_resnums:
            for chain_id in self.chains:
                if pdb_resnum in chain_resnum_to_index[chain_id]:
                    self.fixed_res[chain_id].append(chain_resnum_to_index[chain_id][pdb_resnum])
                    break

    def thread_mpnn_seq(self, binder_seq):
        """
        Thread sequence onto chain 0 by residue renaming and sidechain stripping.
        Sidechains are rebuilt before OpenMM minimization with PDBFixer.
        """
        if not self.chains:
            raise ValueError("No ATOM records found in input PDB")

        target_chain = self.chains[0]
        chain_keys = self._chain_residue_keys(target_chain)

        if len(chain_keys) != len(binder_seq):
            raise ValueError(
                f"Threading length mismatch for chain {target_chain}: "
                f"expected {len(chain_keys)} residues, got sequence of {len(binder_seq)}"
            )

        seq_map = {}
        for key, aa in zip(chain_keys, binder_seq):
            if aa not in mpnn_util.aa_1_3:
                raise ValueError(f"Unsupported residue in MPNN sequence: {aa}")
            seq_map[key] = mpnn_util.aa_1_3[aa]

        new_lines = []
        for line in self.pdb_lines:
            if not line.startswith("ATOM"):
                new_lines.append(line)
                continue

            atom_name = line[12:16].strip()
            key = (line[21], line[22:26], line[26])
            if key not in seq_map:
                new_lines.append(line)
                continue

            if atom_name not in self.BACKBONE_ATOMS:
                continue

            new_resname = seq_map[key]
            new_line = f"{line[:17]}{new_resname}{line[20:]}"
            new_lines.append(new_line)

        self.pdb_lines = new_lines

    def relax_with_openmm(self, max_iterations, restraint_k, platform_name):
        pdb_text = ''.join(self.pdb_lines)
        relaxed_text = _openmm_relax_pdb_text(
            pdb_text,
            max_iterations=max_iterations,
            restraint_k=restraint_k,
            platform_name=platform_name,
        )
        self.pdb_lines = [f"{line}\n" for line in relaxed_text.splitlines()]
        self.chains = self._parse_chain_order()


class ProteinMPNN_runner:
    """
    Runs ProteinMPNN and optional OpenMM relaxation on one input structure.
    """

    def __init__(self, args, struct_manager):
        self.struct_manager = struct_manager

        if torch.cuda.is_available():
            print('Found GPU will run ProteinMPNN on GPU')
            self.device = "cuda:0"
        else:
            print('No GPU found, running ProteinMPNN on CPU')
            self.device = "cpu"

        self.mpnn_model = mpnn_util.init_seq_optimize_model(
            self.device,
            hidden_dim=128,
            num_layers=3,
            backbone_noise=args.augment_eps,
            num_connections=args.num_connections,
            checkpoint_path=args.checkpoint_path,
        )

        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

        self.debug = args.debug

        self.temperature = args.temperature
        self.seqs_per_struct = args.seqs_per_struct
        self.relax_seqs_per_cycle = args.relax_seqs_per_cycle
        self.omit_AAs = [letter for letter in args.omit_AAs.upper() if letter in list(alphabet)]

        if os.path.isfile(args.bias_AA_jsonl):
            print(f'Found AA bias json file at {args.bias_AA_jsonl}')
            with open(args.bias_AA_jsonl, 'r') as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                bias_AA_dict = json.loads(json_str)

            self.bias_AAs_np = np.zeros(len(alphabet))
            for n, AA in enumerate(alphabet):
                if AA in list(bias_AA_dict.keys()):
                    self.bias_AAs_np[n] = bias_AA_dict[AA]
        else:
            self.bias_AAs_np = np.zeros(len(alphabet))

        self.relax_max_cycles = args.relax_max_cycles
        self.relax_max_iterations = args.relax_max_iterations
        self.relax_restraint_k = args.relax_restraint_k
        self.relax_platform = args.relax_platform

    def rebuild_sidechains(self, sample_feats):
        """
        Rebuild missing heavy sidechain atoms from in-memory PDB text.
        """
        pdb_text, _ = common_util.complete_pdb_backbone_to_all_atom_text_from_text(''.join(sample_feats.pdb_lines))
        pdb_text = common_util.optimize_rotamers_with_pyfaspr(pdb_text)
        sample_feats.pdb_lines = [f"{line}\n" for line in pdb_text.splitlines()]
        sample_feats.chains = sample_feats._parse_chain_order()

    def relax_pose(self, sample_feats):
        """
        Run one OpenMM restrained minimization cycle on the current structure.
        """
        relax_t0 = time.time()
        print('Running OpenMM minimization')
        sample_feats.relax_with_openmm(
            max_iterations=self.relax_max_iterations,
            restraint_k=self.relax_restraint_k,
            platform_name=self.relax_platform,
        )
        print(f"Completed one cycle of OpenMM minimization in {int(time.time() - relax_t0)} seconds")

    def sequence_optimize(self, sample_feats, num_seqs=None):
        if num_seqs is None:
            num_seqs = self.seqs_per_struct

        mpnn_t0 = time.time()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp.writelines(sample_feats.pdb_lines)
            pdbfile = tmp.name

        try:
            feature_dict = mpnn_util.generate_seqopt_features(pdbfile, sample_feats.chains)
        finally:
            if os.path.exists(pdbfile):
                os.remove(pdbfile)

        arg_dict = mpnn_util.set_default_args(num_seqs, omit_AAs=self.omit_AAs)
        arg_dict['temperature'] = self.temperature

        if len(sample_feats.chains) == 1:
            masked_chains = sample_feats.chains
            visible_chains = []
        else:
            masked_chains = sample_feats.chains[:-1]
            visible_chains = [sample_feats.chains[-1]]

        fixed_positions_dict = {feature_dict['name']: sample_feats.fixed_res}

        if self.debug:
            print(f'Fixed positions dict: {fixed_positions_dict}')

        sequences = mpnn_util.generate_sequences(
            self.mpnn_model,
            self.device,
            feature_dict,
            arg_dict,
            masked_chains,
            visible_chains,
            bias_AAs_np=self.bias_AAs_np,
            fixed_positions_dict=fixed_positions_dict,
        )

        print(f"ProteinMPNN generated {len(sequences)} sequences in {int(time.time() - mpnn_t0)} seconds")
        for i, (seq, score) in enumerate(sequences):
            print(f"Seq {i + 1}: {seq} (Score: {score:.2f})")

        return sequences

    def proteinmpnn(self, sample_feats):
        """
        Run ProteinMPNN sequence optimization only (no relaxation cycles).
        """
        seqs_scores = self.sequence_optimize(sample_feats)
        prefix = f"{sample_feats.tag}_seq"
        base_sample = sample_feats.clone()

        for idx, (seq, score) in enumerate(seqs_scores):
            working = base_sample.clone()
            working.thread_mpnn_seq(seq)
            self.rebuild_sidechains(working)

            if args.relax_output:
                print(f"Relaxing output for Seq {idx}")
                self.relax_pose(working)

            print(f"Final Sequence {idx} for {sample_feats.tag}: {seq} (Score: {score:.2f})")
            outtag = f"{prefix}_{idx}"
            self.struct_manager.dump_pose(working.pdb_lines, outtag, seq, score)

    def proteinmpnn_fastrelax(self, sample_feats):
        """
        Run ProteinMPNN plus OpenMM relaxation on the structure being designed.
        """
        base_sample = sample_feats.clone()
        seqs_scores = self.sequence_optimize(sample_feats)

        for seq_idx, (initial_seq, initial_score) in enumerate(seqs_scores):
            print(f"Performing optimisation of Seq {seq_idx}: {initial_seq} (Score: {initial_score:.2f})")

            working = base_sample.clone()
            working.thread_mpnn_seq(initial_seq)
            self.rebuild_sidechains(working)

            previous_best_score = None
            convergence_counter = 0
            last_best_seq = initial_seq
            best_score = initial_score
            previous_relaxed_lines = list(working.pdb_lines)

            for cycle in range(args.relax_max_cycles):
                self.relax_pose(working)
                current_relaxed_lines = list(working.pdb_lines)
                rmsd = ca_rmsd_from_pdb_lines(current_relaxed_lines, previous_relaxed_lines)
                print(f"Cycle {cycle} Ca-RMSD: {rmsd:.2f}A")

                new_seqs = self.sequence_optimize(working, num_seqs=self.relax_seqs_per_cycle)
                best_seq, best_score = sorted(new_seqs, key=lambda x: x[1])[0]
                print(f"Cycle {cycle} best sequence: {best_seq} (Score: {best_score:.2f})")

                last_best_seq = best_seq
                working.thread_mpnn_seq(best_seq)
                self.rebuild_sidechains(working)
                previous_relaxed_lines = current_relaxed_lines

                rmsd_threshold = args.relax_convergence_rmsd
                score_delta_threshold = args.relax_convergence_score
                convergence_cycles = args.relax_convergence_max_cycles

                if previous_best_score is not None:
                    score_delta = previous_best_score - best_score
                    if (rmsd <= rmsd_threshold and abs(score_delta) < score_delta_threshold):
                        convergence_counter += 1
                    else:
                        convergence_counter = 0

                    if convergence_counter >= convergence_cycles:
                        print(
                            f"Convergence at cycle {cycle}: "
                            f"RMSD <= {rmsd_threshold}A ({rmsd:.2f}A) "
                            f"and score delta < {score_delta_threshold} ({score_delta:.2f}) "
                            f"for {convergence_cycles} cycles"
                        )
                        break

                previous_best_score = best_score

                if args.output_intermediates:
                    tag = f"{sample_feats.tag}_seq_{seq_idx}_cycle{cycle}"
                    self.struct_manager.dump_pose(current_relaxed_lines, tag, best_seq, best_score)

            if args.relax_output:
                print(f"Relaxing output for Seq {seq_idx}")
                self.relax_pose(working)

            print(f"Final Sequence {seq_idx} for {sample_feats.tag}: {last_best_seq} (Score: {best_score:.2f})")
            final_tag = f"{sample_feats.tag}_seq_{seq_idx}"
            self.struct_manager.dump_pose(working.pdb_lines, final_tag, last_best_seq, best_score)

    def run_model(self, tag, args):
        t0 = time.time()

        print(f"Attempting pose: {tag}")

        pdb_lines = self.struct_manager.load_pose(tag)
        sample_feats = sample_features(pdb_lines, tag)
        sample_feats.parse_fixed_res()

        if args.relax_max_cycles > 0:
            self.proteinmpnn_fastrelax(sample_feats)
        else:
            self.proteinmpnn(sample_feats)

        seconds = int(time.time() - t0)
        print(f"Struct: {tag} reported success in {seconds} seconds")


class StructManager:
    """
    Handles input/output and checkpointing for PDB workflows.
    """

    def __init__(self, args):
        self.args = args

        self.pdb = bool(args.pdbdir)
        if not self.pdb:
            raise ValueError("Only -pdbdir input is supported in this OpenMM/PDB workflow.")

        self.pdbdir = args.pdbdir
        self.outpdbdir = args.outpdbdir

        self.struct_iterator = glob.glob(os.path.join(args.pdbdir, '*.pdb'))

        if args.runlist:
            with open(args.runlist, 'r') as f:
                self.runlist = set([line.strip() for line in f])
            self.struct_iterator = [
                struct for struct in self.struct_iterator
                if os.path.basename(struct).split('.')[0] in self.runlist
            ]
            print(f'After filtering by runlist, {len(self.struct_iterator)} structures remain')

        self.chkfn = args.checkpoint_name
        self.finished_structs = set()

        if os.path.isfile(self.chkfn):
            with open(self.chkfn, 'r') as f:
                for line in f:
                    self.finished_structs.add(line.strip())

    def record_checkpoint(self, tag):
        with open(self.chkfn, 'a') as f:
            f.write(f'{tag}\n')

    def iterate(self):
        for struct in self.struct_iterator:
            tag = os.path.basename(struct).split('.')[0]
            if tag in self.finished_structs:
                print(f'{tag} has already been processed. Skipping')
                continue
            yield struct

    def dump_pose(self, pdb_lines, tag, sequence=None, score=None):
        json_data = None
        if sequence is not None and score is not None:
            json_data = {
                "design": tag,
                "sequence": sequence,
                "score": f"{score:.2f}",
            }

        if not os.path.exists(self.outpdbdir):
            os.makedirs(self.outpdbdir)

        pdbfile = os.path.join(self.outpdbdir, tag + '.pdb')
        with open(pdbfile, 'w') as f:
            f.writelines(pdb_lines)

        if json_data:
            json_path = os.path.join(self.outpdbdir, f"{tag}.json")
            with open(json_path, 'w') as f:
                f.write(json.dumps(json_data) + '\n')

    def load_pose(self, tag):
        with open(tag, 'r') as f:
            return f.readlines()


####################
####### Main #######
####################

struct_manager = StructManager(args)
proteinmpnn_runner = ProteinMPNN_runner(args, struct_manager)

for pdb in struct_manager.iterate():
    if args.debug:
        proteinmpnn_runner.run_model(pdb, args)
    else:
        t0 = time.time()
        try:
            proteinmpnn_runner.run_model(pdb, args)
            struct_manager.record_checkpoint(os.path.basename(pdb).split('.')[0])
        except Exception as e:
            if args.debug:
                raise
            print(f"Error processing {pdb}: {e}")
        print(f"Finished {pdb} in {int(time.time() - t0)} seconds")
