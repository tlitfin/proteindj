#!/usr/bin/env python3
import sys
import os
import re
import json
import logging
from pathlib import Path
import argparse
from multiprocessing import Pool
from Bio.PDB import PDBParser, PDBIO, Superimposer
from Bio.PDB.Selection import unfold_entities
from copy import deepcopy

def setup_logging():
    """Configure logging"""
    logger = logging.getLogger(__name__)
    log_file = "alignment.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logger

logger = setup_logging()

def get_all_ca_atoms(structure):
    """Collect all CA atoms from all chains in structure"""
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'])
    if not ca_atoms:
        raise ValueError("No CA atoms found in structure")
    return ca_atoms

def get_chain_ca_atoms(structure, chain_id):
    """Collect CA atoms from specific chain in structure"""
    ca_atoms = []
    for model in structure:
        # Get all chains in the model
        for chain in model.get_chains():
            if chain.id == chain_id:
                for residue in chain:
                    if 'CA' in residue:
                        ca_atoms.append(residue['CA'])
    if not ca_atoms:
        raise ValueError(f"No CA atoms found in chain {chain_id}")
    return ca_atoms

def get_target_ca_atoms(structure, binder_chain='A'):
    """Collect CA atoms from all non-binder chains, in chain order.

    Handles split multi-chain targets (B, C, D...) produced by Boltz-2 when
    chain breaks are detected in prep_boltz_yaml.py, as well as the original
    merged single-chain target in the design PDB.
    """
    ca_atoms = []
    for model in structure:
        for chain in model:
            if chain.id != binder_chain:
                for residue in chain:
                    if 'CA' in residue:
                        ca_atoms.append(residue['CA'])
    if not ca_atoms:
        raise ValueError(f"No CA atoms found in target chains (excluding '{binder_chain}')")
    return ca_atoms

def align_structures(args):
    """Align Boltz structure to Design template with chain-specific handling"""
    (design_path, boltz_path, out_pdb, src_json, dst_json, 
     fold_id, seq_id, design_type) = args  # Added design_type
    
    try:
        parser = PDBParser(QUIET=True)
        ref_structure = parser.get_structure("design", design_path)
        boltz_structure = parser.get_structure("boltz", boltz_path)

        if design_type == 'binder':
            # 1. Align all target chains (B, C, D... if split) for final structure
            ref_target = get_target_ca_atoms(ref_structure, binder_chain='A')
            boltz_target = get_target_ca_atoms(boltz_structure, binder_chain='A')

            superimposer = Superimposer()
            superimposer.set_atoms(ref_target, boltz_target)
            superimposer.apply(boltz_structure.get_atoms())
            rmsd_target = superimposer.rms

            # 2. Calculate overall RMSD (all CA atoms)
            ref_all_ca = get_all_ca_atoms(ref_structure)
            boltz_all_ca = get_all_ca_atoms(boltz_structure)
            superimposer_all = Superimposer()
            superimposer_all.set_atoms(ref_all_ca, boltz_all_ca)
            rmsd_overall = superimposer_all.rms

            # 3. Calculate binder RMSD (chain A)
            ref_chainA = get_chain_ca_atoms(ref_structure, 'A')
            boltz_chainA = get_chain_ca_atoms(boltz_structure, 'A')
            superimposer_a = Superimposer()
            superimposer_a.set_atoms(ref_chainA, boltz_chainA)
            rmsd_binder = superimposer_a.rms

            rmsd_data = {
                "boltz_rmsd_overall": round(rmsd_overall, 2),
                "boltz_rmsd_target": round(rmsd_target, 2),
                "boltz_rmsd_binder": round(rmsd_binder, 2)
            }

        elif design_type == 'unbound_binder':
            # Reference is still the full (binder+target) design; only chain A (binder) is compared
            # against the target-free prediction, which contains just that one chain.
            ref_chainA = get_chain_ca_atoms(ref_structure, 'A')
            boltz_all_ca = get_all_ca_atoms(boltz_structure)

            superimposer = Superimposer()
            superimposer.set_atoms(ref_chainA, boltz_all_ca)
            superimposer.apply(boltz_structure.get_atoms())

            rmsd_data = {
                "boltz_unbound_rmsd": round(superimposer.rms, 2)
            }

        else:  # Monomer design
            ref_atoms = get_all_ca_atoms(ref_structure)
            boltz_atoms = get_all_ca_atoms(boltz_structure)
            
            superimposer = Superimposer()
            superimposer.set_atoms(ref_atoms, boltz_atoms)
            superimposer.apply(boltz_structure.get_atoms())
            
            rmsd_data = {
                "boltz_rmsd_overall": round(superimposer.rms, 2)
            }

        # Save aligned structure (always chain B aligned for binder)
        io = PDBIO()
        io.set_structure(boltz_structure)
        io.save(str(out_pdb))

        # Write new JSON with only the requested fields
        if src_json.exists():
            with open(src_json, 'r') as f:
                data = json.load(f)

            # Build output dictionary
            if design_type == 'binder':
                # Mean ptm across all non-binder chains (key "0" = binder)
                chains_ptm = data.get("chains_ptm", {})
                target_ptm_values = [v for k, v in chains_ptm.items() if k != "0"]
                boltz_ptm_target = round(sum(target_ptm_values) / len(target_ptm_values), 3) if target_ptm_values else 0.0

                # Mean iptm from binder (chain 0) toward all target chains
                binder_iptm_row = data.get("pair_chains_iptm", {}).get("0", {})
                binder_to_target_iptm = [v for k, v in binder_iptm_row.items() if k != "0"]
                boltz_iptm = round(sum(binder_to_target_iptm) / len(binder_to_target_iptm), 3) if binder_to_target_iptm else 0.0

                out_json = {
                    "fold_id": fold_id,
                    "seq_id": seq_id,
                    "description": boltz_path.stem,
                    "boltz_rmsd_overall": round(data.get("boltz_rmsd_overall", rmsd_data.get("boltz_rmsd_overall", 0)), 2),
                    "boltz_rmsd_target": round(rmsd_data.get("boltz_rmsd_target", 0), 2),
                    "boltz_rmsd_binder": round(rmsd_data.get("boltz_rmsd_binder", 0), 2),
                    "boltz_conf_score": round(data.get("confidence_score", 0), 3),
                    "boltz_ipSAE_min": round(data.get("ipSAE_min", 0), 3),
                    "boltz_LIS": round(data.get("LIS", 0), 3),
                    "boltz_pDockQ2_min": round(data.get("pDockQ2_min", 0), 3),
                    "boltz_pae_interaction": round(data.get("ipae", 0), 2),
                    "boltz_ptm": round(data.get("ptm", 0), 3),
                    "boltz_ptm_binder": round(chains_ptm.get("0", 0), 3),
                    "boltz_ptm_target": boltz_ptm_target,
                    "boltz_iptm": boltz_iptm,
                    "boltz_plddt": round(data.get("complex_plddt", 0), 3),
                    "boltz_iplddt": round(data.get("complex_iplddt", 0), 3),
                    "boltz_pde": round(data.get("complex_pde", 0), 2),
                    "boltz_ipde": round(data.get("complex_ipde", 0), 2)
                }
            elif design_type == 'unbound_binder':
                out_json = {
                    "fold_id": fold_id,
                    "seq_id": seq_id,
                    "description": boltz_path.stem,
                    "boltz_unbound_rmsd": round(rmsd_data.get("boltz_unbound_rmsd", 0), 2),
                    "boltz_unbound_conf_score": round(data.get("confidence_score", 0), 3),
                    "boltz_unbound_ptm": round(data.get("ptm", 0), 3),
                    "boltz_unbound_plddt": round(data.get("complex_plddt", 0), 3),
                    "boltz_unbound_pde": round(data.get("complex_pde", 0), 2),
                }
            else: # Monomer design
                out_json = {
                    "fold_id": fold_id,
                    "seq_id": seq_id,
                    "description": boltz_path.stem,
                    "boltz_rmsd_overall": round(data.get("boltz_rmsd_overall", rmsd_data.get("boltz_rmsd_overall", 0)), 2),
                    "boltz_conf_score": round(data.get("confidence_score", 0), 3),
                    "boltz_ptm": round(data.get("ptm", 0), 3),
                    "boltz_plddt": round(data.get("complex_plddt", 0), 3),
                    "boltz_pde": round(data.get("complex_pde", 0), 2),
                }

            with open(dst_json, 'w') as f:
                json.dump(out_json, f, indent=2)

        return (boltz_path.name, rmsd_data.get('boltz_rmsd_overall', rmsd_data.get('boltz_unbound_rmsd')), None)

    except Exception as e:
        logger.error(f"Failed {boltz_path.name}: {str(e)}")
        return (boltz_path.name, None, str(e))

def main():
    parser = argparse.ArgumentParser(description="Align Boltz predictions to designs")
    parser.add_argument("--design_dir", type=Path, required=True, 
                      help="Directory with Design PDBs (fold_*_seq_*.pdb)")
    parser.add_argument("--boltz_dir", type=Path, required=True,
                      help="Directory with Boltz PDBs and JSONs (fold_*_seq_*_boltzpred.pdb)")
    parser.add_argument("--output_dir", type=Path, default="aligned",
                      help="Output directory for results")
    parser.add_argument("--design_type", choices=['binder', 'monomer', 'unbound_binder'], required=True,
                      help="Design type: 'binder' (A/B chains), 'monomer' (A chain), or "
                           "'unbound_binder' (target-free binder-only prediction aligned to chain A)")
    parser.add_argument("--ncpus", type=int, default=1,
                      help="Number of CPUs for parallel processing")
    args = parser.parse_args()
    
    # Validate input directories
    if not args.design_dir.exists():
        logger.error(f"Design directory not found: {args.design_dir}")
        sys.exit(1)
        
    if not args.boltz_dir.exists():
        logger.error(f"Boltz directory not found: {args.boltz_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map Design files
    design_files = {}
    for design_file in args.design_dir.glob("fold_*_seq_*.pdb"):
        match = re.match(r"fold_(\d+)_seq_(\d+)\.pdb", design_file.name)
        if match:
            fold_id = int(match.group(1))
            seq_id = int(match.group(2))
            design_files[(fold_id, seq_id)] = design_file
    
    # Prepare processing tasks
    tasks = []
    for boltz_file in args.boltz_dir.glob("fold_*_seq_*_boltzpred.pdb"):
        match = re.match(r"fold_(\d+)_seq_(\d+)_.*\.pdb", boltz_file.name)
        if not match:
            continue
            
        fold_id = int(match.group(1))
        seq_id = int(match.group(2))
        key = (fold_id, seq_id)
        
        if key not in design_files:
            logger.warning(f"No design file for fold {fold_id} seq {seq_id}, skipping {boltz_file.name}")
            continue
            
        # Generate paths. For unbound_binder, rename outputs so they never collide with the
        # bound-run outputs when both are later staged into the same downstream task directory.
        base_name = boltz_file.stem  # fold_X_seq_Y_boltzpred
        src_json = args.boltz_dir / f"{base_name}.json"
        if args.design_type == 'unbound_binder':
            out_base_name = base_name.replace('_boltzpred', '_unbound_boltzpred')
        else:
            out_base_name = base_name
        out_pdb = args.output_dir / f"{out_base_name}.pdb"
        dst_json = args.output_dir / f"{out_base_name}.json"
        
        tasks.append((
            design_files[key],
            boltz_file,
            out_pdb,
            src_json,
            dst_json,
            fold_id,
            seq_id,
            args.design_type
        ))
    
    # Log processing start
    logger.info(f"Starting alignment of {len(tasks)} Boltz structures")
    logger.info(f"Using design directory: {args.design_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Process tasks in parallel
    with Pool(args.ncpus) as pool:
        results = pool.map(align_structures, tasks)
    
    # Report summary
    successes = sum(1 for r in results if r[1] is not None)
    failures = len(results) - successes
    
    logger.info("\n=== Alignment Summary ===")
    logger.info(f"Total structures processed: {len(tasks)}")
    logger.info(f"Successful alignments: {successes}")
    logger.info(f"Failed alignments: {failures}")
    
    if failures > 0:
        logger.info("\nFailed cases:")
        for name, _, error in results:
            if error:
                logger.info(f"  {name}: {error}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
