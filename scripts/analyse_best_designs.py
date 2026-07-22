#!/usr/bin/env python3

import copy
import json
import os
import re
import shutil
import subprocess
import traceback
import logging
import numpy as np
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from Bio import PDB
from Bio.PDB import PDBParser, DSSP, Polypeptide
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.SeqUtils import seq1
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# Chothia/NACCESS-like atomic radii for SASA calculation
R_CHOTHIA = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}

# Hydrophobic amino acids (matches FreeBindCraft HYDROPHOBIC_AA_SET)
HYDROPHOBIC_AA = frozenset("ACFGILMPVWY")

# 8-state DSSP to 3-state mapping
DSSP_HELIX = frozenset(('H', 'G', 'I'))
DSSP_STRAND = frozenset(('E', 'B'))


# ---------------------------------------------------------------------------
# Core structure utilities
# ---------------------------------------------------------------------------

def parse_structure(pdb_path):
    """Parse PDB file and return (structure, model)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    return structure, structure[0]


def isolate_chain(model, chain_id):
    """Create a new structure containing only the specified chain.
    Returns (isolated_structure, isolated_model, isolated_chain).
    """
    iso_struct = Structure("iso")
    iso_model = Model(0)
    iso_chain = copy.deepcopy(model[chain_id])
    iso_model.add(iso_chain)
    iso_struct.add(iso_model)
    return iso_struct, iso_model, iso_chain


def get_chain_ids(model):
    """Get ordered list of chain IDs from a BioPython model."""
    return [chain.get_id() for chain in model.get_chains()]


def derive_ids_from_filename(filename):
    """Extract fold_id and seq_id from filename format: fold_X_seq_Y_*.pdb"""
    basename = Path(filename).stem
    fold_match = re.search(r'fold_(\d+)', basename)
    seq_match = re.search(r'seq_(\d+)', basename)

    fold_id = int(fold_match.group(1)) if fold_match else None
    seq_id = int(seq_match.group(1)) if seq_match else None

    return fold_id, seq_id


# ---------------------------------------------------------------------------
# Secondary structure and radius of gyration
# ---------------------------------------------------------------------------

def count_secondary_structures(model, pdb_path, chain_id=None):
    """Count secondary structure elements using BioPython DSSP + mkdssp.
    If chain_id is provided, only counts SS for that chain.
    """
    dssp_obj = DSSP(model, str(pdb_path), dssp="mkdssp")

    if chain_id is not None:
        target_chains = {chain_id}
    else:
        target_chains = {c.get_id() for c in model.get_chains()}

    dssp_chars = []
    for key in dssp_obj.keys():
        if key[0] in target_chains:
            ss = dssp_obj[key][2]
            if ss in DSSP_HELIX:
                dssp_chars.append('H')
            elif ss in DSSP_STRAND:
                dssp_chars.append('E')
            else:
                dssp_chars.append('L')

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

    return {
        'pr_helices': helix_count,
        'pr_strands': strand_count,
        'pr_total_ss': helix_count + strand_count
    }


def calculate_rog(chain):
    """Calculate mass-weighted radius of gyration for a chain or model."""
    coords = np.array([atom.get_coord() for atom in chain.get_atoms()])
    masses = np.array([atom.mass for atom in chain.get_atoms()])

    if len(coords) == 0:
        return {'pr_RoG': 0.0}

    com = np.average(coords, axis=0, weights=masses)
    diff = coords - com
    rg = round(float(np.sqrt(np.sum(masses * np.sum(diff**2, axis=1)) / masses.sum())), 2)
    return {'pr_RoG': rg}


# ---------------------------------------------------------------------------
# SASA and surface chemistry
# ---------------------------------------------------------------------------

def compute_sasa(model):
    """Compute atom-level SASA using BioPython ShrakeRupley with Chothia radii."""
    sr = ShrakeRupley(probe_radius=1.40, n_points=960, radii_dict=R_CHOTHIA)
    sr.compute(model, level='A')


def calculate_surface_chemistry(model, chain_id):
    """Calculate surface hydrophobicity using BioPython ShrakeRupley SASA.

    Computes the percentage of SASA contributed by hydrophobic residues.
    Note: this differs from PyRosetta LayerSelector (which classifies residues
    as surface/core then counts) but captures equivalent intent.
    """
    _, iso_model, iso_chain = isolate_chain(model, chain_id)
    compute_sasa(iso_model)

    hydrophobic_sasa = 0.0
    total_sasa = 0.0

    for residue in iso_chain:
        if Polypeptide.is_aa(residue, standard=True):
            res_sasa = sum(getattr(atom, 'sasa', 0.0) for atom in residue.get_atoms())
            total_sasa += res_sasa
            try:
                aa1 = seq1(residue.get_resname()).upper()
            except Exception:
                aa1 = ''
            if aa1 in HYDROPHOBIC_AA:
                hydrophobic_sasa += res_sasa

    if total_sasa > 0:
        pct = round(hydrophobic_sasa / total_sasa * 100.0, 1)
    else:
        pct = 0.0

    return {'pr_surfhphobics': pct}


# ---------------------------------------------------------------------------
# Placeholders (no open-source equivalent)
# ---------------------------------------------------------------------------

def calculate_tem():
    """Placeholder for total energy metric (requires Rosetta force field)."""
    return {'pr_TEM': 0}


def calculate_sap_scores(num_chains, chain_id='A'):
    """Placeholder for SAP scores (requires PyRosetta PerResidueSapScoreMetric)."""
    metrics = {'pr_SAP': 0.0}
    if num_chains >= 2:
        metrics['pr_SAP_complex'] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Interface metrics
# ---------------------------------------------------------------------------

def chain_total_sasa(chain):
    """Sum SASA over all atoms in a chain."""
    return sum(getattr(atom, "sasa", 0.0) for atom in chain.get_atoms())


def resolve_sc_binary():
    """Find the sc-rs shape complementarity binary.

    Resolution order (matches FreeBindCraft):
    1. Environment variables SC_RS_BIN, SC_BIN
    2. PATH lookup for sc, sc-rs, shape-complementarity, sc_rs
    """
    env_candidates = [os.environ.get('SC_RS_BIN'), os.environ.get('SC_BIN')]
    path_candidates = [
        shutil.which('sc'),
        shutil.which('sc-rs'),
        shutil.which('shape-complementarity'),
        shutil.which('sc_rs'),
    ]

    for candidate in env_candidates + path_candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def calculate_shape_complementarity(pdb_path, chain1, chain2):
    """Calculate shape complementarity using sc-rs CLI.

    Adapted from FreeBindCraft _calculate_shape_complementarity().
    Falls back to 0.70 if binary not found or fails.
    """
    sc_bin = resolve_sc_binary()
    if sc_bin is None:
        logger.warning("sc-rs binary not found; using placeholder 0.70")
        return 0.70

    try:
        cmd = [sc_bin, str(pdb_path), str(chain1), str(chain2), '--json']
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        stdout = (proc.stdout or '').strip()
        if not stdout:
            return 0.70

        try:
            payload = json.loads(stdout)
        except Exception:
            payload = None
            # Try extracting JSON from mixed output
            s_idx = stdout.rfind('{')
            e_idx = stdout.rfind('}')
            if s_idx != -1 and e_idx > s_idx:
                try:
                    payload = json.loads(stdout[s_idx:e_idx + 1])
                except Exception:
                    pass

        if isinstance(payload, dict):
            sc_key = 'sc' if 'sc' in payload else ('sc_value' if 'sc_value' in payload else None)
            if sc_key is not None:
                sc_val = float(payload[sc_key])
                if 0.0 <= sc_val <= 1.0:
                    return sc_val

    except subprocess.TimeoutExpired:
        logger.error(f"sc-rs timed out for {pdb_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"sc-rs failed for {pdb_path}: {e}")
    except Exception as e:
        logger.warning(f"sc-rs error for {pdb_path}: {e}")

    return 0.70


def compute_interface_bsa(model, chain1, chain2):
    """Compute buried surface area at the interface using BioPython ShrakeRupley.

    Adapted from FreeBindCraft _compute_sasa_metrics().
    Returns dSASA / 2 (to match PyRosetta InterfaceAnalyzerMover convention).
    """
    # SASA of chains within the full complex
    compute_sasa(model)
    chain1_sasa_complex = chain_total_sasa(model[chain1])
    chain2_sasa_complex = chain_total_sasa(model[chain2])

    # SASA of chain1 as isolated monomer
    _, iso_model1, iso_chain1 = isolate_chain(model, chain1)
    compute_sasa(iso_model1)
    chain1_sasa_mono = chain_total_sasa(iso_chain1)

    # SASA of chain2 as isolated monomer
    _, iso_model2, iso_chain2 = isolate_chain(model, chain2)
    compute_sasa(iso_model2)
    chain2_sasa_mono = chain_total_sasa(iso_chain2)

    # dSASA = buried area upon complex formation
    dsasa = max(chain1_sasa_mono - chain1_sasa_complex, 0.0) + \
            max(chain2_sasa_mono - chain2_sasa_complex, 0.0)

    return round(dsasa / 2)


def compute_prodigy_scores(pdb_path, chain1, chain2):
    """Predict binding affinity using Prodigy (IC-NIS model).

    Returns dict with deltaG (kcal/mol) and Kd (M).
    Falls back to neutral placeholders (dG=0.0) on failure.
    """
    try:
        from prodigy_prot.modules.parsers import parse_structure as prodigy_parse
        from prodigy_prot.modules.prodigy import Prodigy as ProdigyPredictor

        models, _, _ = prodigy_parse(str(pdb_path))
        prodigy = ProdigyPredictor(model=models[0], selection=[chain1, chain2])
        prodigy.predict()
        return {
            'dg': prodigy.ba_val,
            'kd': prodigy.kd_val,
        }
    except Exception as e:
        logger.warning(f"Prodigy scoring failed for {pdb_path}: {e}. Using placeholders.")
        return {
            'dg': 0.0, 'kd': 0.0,
        }


# ---------------------------------------------------------------------------
# H-bond metrics
# Distance-based approximation using heavy-atom D-A < 3.5 A cutoff.
# Overcounts relative to PyRosetta's angle-dependent method.
# ---------------------------------------------------------------------------

# H-bond donor/acceptor atom names by residue type (heavy atoms only).
# Backbone N is donor, backbone O is acceptor for all residues.
SIDECHAIN_DONORS = {
    'ARG': ('NE', 'NH1', 'NH2'), 'ASN': ('ND2',), 'GLN': ('NE2',),
    'HIS': ('ND1', 'NE2'), 'LYS': ('NZ',), 'SER': ('OG',),
    'THR': ('OG1',), 'TRP': ('NE1',), 'TYR': ('OH',),
}
SIDECHAIN_ACCEPTORS = {
    'ASN': ('OD1',), 'ASP': ('OD1', 'OD2'), 'GLN': ('OE1',),
    'GLU': ('OE1', 'OE2'), 'HIS': ('ND1', 'NE2'), 'SER': ('OG',),
    'THR': ('OG1',), 'TYR': ('OH',),
}


def get_interface_polar_atoms(model, chain1, chain2, contact_dist=4.0):
    """Identify polar atoms at the interface between two chains.

    Returns (donors_c1, acceptors_c1, donors_c2, acceptors_c2) where each
    is a list of Atom objects from the respective chain that are within
    contact_dist of any atom on the other chain.
    """
    atoms_c1 = list(model[chain1].get_atoms())
    atoms_c2 = list(model[chain2].get_atoms())

    # Build neighbor search on each chain to find interface atoms
    ns_c2 = NeighborSearch(atoms_c2)
    ns_c1 = NeighborSearch(atoms_c1)

    def is_interface_atom(atom, ns_other, dist):
        """Check if atom is within dist of any atom in the other chain."""
        return len(ns_other.search(atom.get_vector().get_array(), dist, 'A')) > 0

    def collect_polar(chain_obj, ns_other, dist):
        """Collect donor and acceptor atoms at the interface for a chain."""
        donors = []
        acceptors = []
        for residue in chain_obj:
            if not Polypeptide.is_aa(residue, standard=True):
                continue
            resname = residue.get_resname()
            for atom in residue:
                aname = atom.get_name()
                if not is_interface_atom(atom, ns_other, dist):
                    continue
                # Backbone
                if aname == 'N':
                    donors.append(atom)
                elif aname == 'O':
                    acceptors.append(atom)
                # Sidechain donors
                elif aname in SIDECHAIN_DONORS.get(resname, ()):
                    donors.append(atom)
                # Sidechain acceptors
                if aname in SIDECHAIN_ACCEPTORS.get(resname, ()):
                    acceptors.append(atom)
        return donors, acceptors

    donors_c1, acceptors_c1 = collect_polar(model[chain1], ns_c2, contact_dist)
    donors_c2, acceptors_c2 = collect_polar(model[chain2], ns_c1, contact_dist)

    return donors_c1, acceptors_c1, donors_c2, acceptors_c2


def compute_interface_hbonds(model, chain1, chain2):
    """Count intermolecular hydrogen bonds using distance criterion.

    Uses heavy-atom distance D-A < 3.5 A (standard for PDBs without hydrogens).
    Counts unique donor-acceptor pairs across the interface.
    """
    donors_c1, acceptors_c1, donors_c2, acceptors_c2 = get_interface_polar_atoms(
        model, chain1, chain2, contact_dist=4.0)

    hbond_cutoff = 3.5
    hbond_count = 0

    # Donors on chain1 -> acceptors on chain2
    for donor in donors_c1:
        d_coord = donor.get_vector().get_array()
        for acceptor in acceptors_c2:
            dist = np.linalg.norm(d_coord - acceptor.get_vector().get_array())
            if dist <= hbond_cutoff:
                hbond_count += 1

    # Donors on chain2 -> acceptors on chain1
    for donor in donors_c2:
        d_coord = donor.get_vector().get_array()
        for acceptor in acceptors_c1:
            dist = np.linalg.norm(d_coord - acceptor.get_vector().get_array())
            if dist <= hbond_cutoff:
                hbond_count += 1

    return hbond_count


def compute_unsat_hbonds(model, chain1, chain2):
    """Approximate buried unsatisfied hydrogen bonds at the interface.

    Identifies polar atoms (N, O) at the interface that are:
    1. Buried (atom SASA < 2.0 A^2 in the complex)
    2. Lacking any H-bond partner (no complementary polar atom within 3.5 A)
    """
    # Compute SASA for the complex
    compute_sasa(model)

    donors_c1, acceptors_c1, donors_c2, acceptors_c2 = get_interface_polar_atoms(
        model, chain1, chain2, contact_dist=5.0)

    burial_cutoff = 2.0  # A^2 -- atom is considered buried below this SASA
    hbond_cutoff = 3.5

    unsat_count = 0

    # Check each buried donor: does it have an acceptor partner on the other chain?
    for donor in donors_c1 + donors_c2:
        if getattr(donor, 'sasa', 999.0) >= burial_cutoff:
            continue  # not buried, skip
        donor_chain = donor.get_parent().get_parent().get_id()
        d_coord = donor.get_vector().get_array()
        has_partner = False
        partner_acceptors = acceptors_c2 if donor_chain == chain1 else acceptors_c1
        for acc in partner_acceptors:
            if np.linalg.norm(d_coord - acc.get_vector().get_array()) <= hbond_cutoff:
                has_partner = True
                break
        if not has_partner:
            unsat_count += 1

    # Check each buried acceptor: does it have a donor partner on the other chain?
    for acceptor in acceptors_c1 + acceptors_c2:
        if getattr(acceptor, 'sasa', 999.0) >= burial_cutoff:
            continue
        acc_chain = acceptor.get_parent().get_parent().get_id()
        a_coord = acceptor.get_vector().get_array()
        has_partner = False
        partner_donors = donors_c2 if acc_chain == chain1 else donors_c1
        for don in partner_donors:
            if np.linalg.norm(a_coord - don.get_vector().get_array()) <= hbond_cutoff:
                has_partner = True
                break
        if not has_partner:
            unsat_count += 1

    return unsat_count


def calculate_interface_metrics(model, pdb_path, chain1='A', chain2='B'):
    """Calculate interface metrics between specified chains.

    Computes BSA, shape complementarity (sc-rs), binding deltaG (Prodigy),
    H-bonds (distance-based), and unsatisfied H-bonds (buried polar atoms).
    PackStat has no open-source equivalent and uses a placeholder.
    """
    # Add suffix for metrics if not the default A_B interface
    suffix = '' if (chain1 == 'A' and chain2 == 'B') else f'_{chain1}_{chain2}'

    # Geometry metrics
    bsa = compute_interface_bsa(model, chain1, chain2)
    sc = round(calculate_shape_complementarity(pdb_path, chain1, chain2), 3)

    # Binding affinity (Prodigy IC-NIS model)
    pg = compute_prodigy_scores(pdb_path, chain1, chain2)
    dg_to_bsa = round(pg['dg'] / bsa, 4) if bsa > 0 else 0.0

    # Hydrogen bonds
    hbonds = compute_interface_hbonds(model, chain1, chain2)

    # Buried unsatisfied H-bonds
    unsat = compute_unsat_hbonds(model, chain1, chain2)

    return {
        f'pr_intface_BSA{suffix}': bsa,
        f'pr_intface_shpcomp{suffix}': sc,
        f'pr_intface_deltaG{suffix}': round(pg['dg'], 2),
        f'pr_intface_deltaGtoBSA{suffix}': dg_to_bsa,
        f'pr_intface_hbonds{suffix}': hbonds,
        f'pr_intface_unsat_hbonds{suffix}': unsat,
        f'pr_intface_packstat{suffix}': 0.65,  # no non-Rosetta equivalent
    }


# ---------------------------------------------------------------------------
# Sequence metrics
# ---------------------------------------------------------------------------

def get_chain_sequence(pdb_path, chain_id):
    """Extract sequence for a specific chain from PDB file."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('temp', str(pdb_path))

        # Convert numeric chain IDs to letters if needed
        if chain_id.isdigit():
            chain_id = chr(64 + int(chain_id))  # Convert 1->A, 2->B etc

        for chain in structure.get_chains():
            if chain.id == chain_id:
                residues = [r for r in chain.get_residues() if PDB.is_aa(r)]
                return ''.join([Polypeptide.protein_letters_3to1[r.get_resname()] for r in residues])
    except Exception as e:
        logger.error(f"Error getting sequence: {str(e)}")
    return None


def calculate_seq_metrics(sequence):
    """Calculate metrics for a protein sequence."""
    analysis = ProteinAnalysis(sequence)
    extinction_coef = analysis.molar_extinction_coefficient()[0]
    molecular_weight = int(analysis.molecular_weight())

    return {
        'sequence': sequence,
        'seq_ext_coef': extinction_coef,
        'seq_length': len(sequence),
        'seq_MW': molecular_weight,
        'seq_pI': round(analysis.isoelectric_point(), 2)
    }


# ---------------------------------------------------------------------------
# Per-chain and whole-pose metric aggregation
# ---------------------------------------------------------------------------

def calculate_chain_metrics(model, chain_id, pdb_path):
    """Calculate all metrics for a single chain."""
    # Add suffix for metrics if not chain A
    suffix = '' if chain_id == 'A' else f'_{chain_id}'

    # Calculate secondary structure metrics
    ss_metrics = count_secondary_structures(model, pdb_path, chain_id)
    ss_metrics = {f'{k}{suffix}': v for k, v in ss_metrics.items()}

    # Calculate RoG
    rog_metrics = calculate_rog(model[chain_id])
    rog_metrics = {f'{k}{suffix}': v for k, v in rog_metrics.items()}

    # Calculate surface chemistry
    surface_metrics = calculate_surface_chemistry(model, chain_id)
    surface_metrics = {f'{k}{suffix}': v for k, v in surface_metrics.items()}

    # Calculate total energy metric
    tem_metrics = calculate_tem()
    tem_metrics = {f'{k}{suffix}': v for k, v in tem_metrics.items()}

    # Get sequence metrics
    seq_metrics = {}
    sequence = get_chain_sequence(pdb_path, chain_id)
    if sequence:
        seq_metrics = calculate_seq_metrics(sequence)
        seq_metrics = {f'{k}{suffix}': v for k, v in seq_metrics.items()}

    # Combine all metrics
    metrics = {}
    metrics.update(ss_metrics)
    metrics.update(rog_metrics)
    metrics.update(surface_metrics)
    metrics.update(tem_metrics)
    metrics.update(seq_metrics)

    return metrics


def calculate_whole_pose_metrics(model, pdb_path):
    """Calculate metrics for the entire structure (all chains combined)."""
    rog_all = calculate_rog(model)

    compute_sasa(model)
    hydrophobic_sasa = 0.0
    total_sasa = 0.0
    for chain in model.get_chains():
        for residue in chain:
            if Polypeptide.is_aa(residue, standard=True):
                res_sasa = sum(getattr(atom, 'sasa', 0.0) for atom in residue.get_atoms())
                total_sasa += res_sasa
                try:
                    aa1 = seq1(residue.get_resname()).upper()
                except Exception:
                    aa1 = ''
                if aa1 in HYDROPHOBIC_AA:
                    hydrophobic_sasa += res_sasa

    surfhphobics = round(hydrophobic_sasa / total_sasa * 100.0, 1) if total_sasa > 0 else 0.0

    return {
        'pr_RoG_total': rog_all['pr_RoG'],
        'pr_surfhphobics_total': surfhphobics,
        **calculate_tem()
    }


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_single_pdb(pdb_path):
    """Process a single PDB file and return all calculated metrics."""

    logger.info(f"Processing PDB file: {pdb_path}")
    try:
        structure, model = parse_structure(pdb_path)
        chain_ids = get_chain_ids(model)

        metrics = {'description': pdb_path.stem}

        # Handle metrics calculation based on design type
        if len(chain_ids) == 1:
            # Monomer design
            primary_chain = chain_ids[0]
            chain_metrics = calculate_chain_metrics(model, primary_chain, pdb_path)
            metrics.update(chain_metrics)

            # Calculate SAP scores for monomer
            sap_metrics = calculate_sap_scores(1, primary_chain)
            metrics.update(sap_metrics)

        elif len(chain_ids) == 2:
            # Calculate binder chain (A) sequence metrics
            sequence = get_chain_sequence(pdb_path, 'A')
            if sequence:
                seq_metrics = calculate_seq_metrics(sequence)
                metrics.update(seq_metrics)

            # Calculate interface metrics for A-B
            interface_metrics = calculate_interface_metrics(model, pdb_path, 'A', 'B')
            metrics.update(interface_metrics)

            # Calculate chain metrics for binder chain only
            chain_metrics = calculate_chain_metrics(model, 'A', pdb_path)
            metrics.update(chain_metrics)

            # Calculate SAP scores for binder (alone and complex)
            sap_metrics = calculate_sap_scores(2, 'A')
            metrics.update(sap_metrics)

        elif len(chain_ids) >= 3:
            # Oligomer design: calculate all pairwise interfaces
            for i, c1 in enumerate(chain_ids):
                for c2 in chain_ids[i+1:]:
                    interface_metrics = calculate_interface_metrics(model, pdb_path, c1, c2)
                    metrics.update(interface_metrics)

            # Calculate per-chain metrics
            all_chain_metrics = {}
            total_helices = 0
            total_strands = 0
            total_ss = 0

            for chain_id in chain_ids:
                chain_metrics = calculate_chain_metrics(model, chain_id, pdb_path)

                # Track secondary structure aggregates
                suffix = '' if chain_id == 'A' else f'_{chain_id}'
                total_helices += chain_metrics.get(f'pr_helices{suffix}', 0)
                total_strands += chain_metrics.get(f'pr_strands{suffix}', 0)
                total_ss += chain_metrics.get(f'pr_total_ss{suffix}', 0)

                all_chain_metrics.update(chain_metrics)

            # Add whole-pose metrics
            whole_pose_metrics = calculate_whole_pose_metrics(model, pdb_path)
            all_chain_metrics.update(whole_pose_metrics)

            # Calculate SAP for each chain
            for chain_id in chain_ids:
                sap_metrics = calculate_sap_scores(len(chain_ids), chain_id)
                # Add suffix if not chain A
                if chain_id != 'A':
                    sap_metrics = {f'{k}_{chain_id}': v for k, v in sap_metrics.items()}
                all_chain_metrics.update(sap_metrics)

            # Add aggregated secondary structure metrics
            all_chain_metrics.update({
                'pr_helices_allchains': total_helices,
                'pr_strands_allchains': total_strands,
                'pr_total_ss_allchains': total_ss
            })

            metrics.update(all_chain_metrics)

        return metrics
    except Exception as e:
        logger.error(f"Error processing {pdb_path}: {str(e)}\n{traceback.format_exc()}")
        return {}


# ---------------------------------------------------------------------------
# JSONL I/O and batch processing
# ---------------------------------------------------------------------------

def read_jsonl(file_path):
    """Read JSONL file and return list of records."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSONL line: {e}")
    return records


def write_jsonl(file_path, records):
    """Write records to JSONL file."""
    with open(file_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def process_pdbs(pdb_dir=None, output_path=None, num_processes=4):
    """Main function to process multiple PDB files in parallel and create enriched JSONL."""

    logger.info(f"Starting processing of PDB files in directory: {pdb_dir}")
    try:
        # Get all PDB files in the directory
        pdb_paths = list(Path(pdb_dir).glob('*.pdb'))
        logger.info(f"Found {len(pdb_paths)} PDB files to process")

        if not pdb_paths:
            logger.error(f"No PDB files found in directory: {pdb_dir}")
            return []

        # Process PDB files in parallel
        logger.info(f"Processing PDB files with {num_processes} processes")
        process_func = partial(process_single_pdb)

        with Pool(processes=num_processes) as pool:
            results = pool.map(process_func, pdb_paths)

        # Create enriched records with fold_id and seq_id derived from filenames
        enriched_records = []
        for i, pdb_path in enumerate(pdb_paths):
            result = results[i]
            if result:  # Skip failed processing results
                # Extract fold_id and seq_id from filename
                fold_id, seq_id = derive_ids_from_filename(pdb_path.name)

                # Create record with derived IDs and calculated metrics
                record = {
                    'description': pdb_path.stem,
                    'fold_id': fold_id,
                    'seq_id': seq_id
                }

                # Add all calculated metrics (excluding duplicate description)
                for key, value in result.items():
                    if key != 'description':
                        record[key] = value

                enriched_records.append(record)
            else:
                logger.warning(f"Failed to process {pdb_path}, skipping")

        # Save enriched records to output JSONL
        logger.info(f"Saving enriched JSONL to {output_path}")
        write_jsonl(output_path, enriched_records)
        logger.info(f"Processing completed successfully. Processed {len(enriched_records)} files")

        return enriched_records

    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}\n{traceback.format_exc()}")
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process PDB files and create enriched JSONL')
    parser.add_argument('--pdb_dir', required=True, help='Directory containing PDB files')
    parser.add_argument('--output', required=True, help='Path for output enriched JSONL file')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes to use')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Process files and save to specified output path
    enriched_records = process_pdbs(
        pdb_dir=args.pdb_dir,
        output_path=args.output,
        num_processes=args.num_processes
    )
