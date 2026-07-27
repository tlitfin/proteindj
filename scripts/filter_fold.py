import argparse
from pathlib import Path
from multiprocessing import Pool
import logging
import os
import sys
import tempfile
import uuid
import json
import shutil
import re

import numpy as np
from Bio.PDB import PDBParser, DSSP

# 8-state DSSP to 3-state mapping
DSSP_HELIX = frozenset(('H', 'G', 'I'))
DSSP_STRAND = frozenset(('E',))

def setup_logger():
    """Configure logging to output to both file and stdout"""
    logger = logging.getLogger('filter_fold')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Create unique log filename
    unique_id = str(uuid.uuid4())[:8]
    log_filename = f'filter_fold_{unique_id}.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def extract_fold_id(pdb_filename):
    """Extract fold ID from PDB filename"""
    match = re.search(r'fold_(\d+)', pdb_filename.name)
    if match:
        return int(match.group(1))
    else:
        return None

def parse_structure(pdb_path):
    """Parse PDB file and return (structure, model)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    return structure, structure[0]


def get_chain_ids(model):
    """Get ordered list of chain IDs from a BioPython model."""
    return [chain.get_id() for chain in model.get_chains()]


def _prepare_dssp_input(pdb_path):
    """Ensure a PDB file is recognized as PDB format by mkdssp (>=4.0).

    mkdssp>=4.0 decides whether to parse a file as PDB or mmCIF based on
    whether it starts with a HEADER record; without one it assumes mmCIF and
    fails. RFdiffusion output PDBs have no HEADER line, so we prepend one to
    a temporary copy when needed.

    Returns a (path, tmp_path) tuple where path is what should be passed to
    DSSP and tmp_path is the temporary file to clean up afterwards (None if
    no temporary file was created).
    """
    pdb_path = Path(pdb_path)
    with open(pdb_path) as f:
        first_line = f.readline()

    if first_line.startswith('HEADER'):
        return str(pdb_path), None

    fd, tmp_path = tempfile.mkstemp(suffix='.pdb', dir=pdb_path.parent)
    with os.fdopen(fd, 'w') as dst, open(pdb_path) as src:
        dst.write('HEADER\n')
        shutil.copyfileobj(src, dst)

    return tmp_path, tmp_path


def count_secondary_structures(model, pdb_path, chain_id=None):
    """Count secondary structure elements using BioPython DSSP + mkdssp.
    If chain_id is provided, only counts SS for that chain.
    """
    dssp_path, tmp_path = _prepare_dssp_input(pdb_path)
    try:
        dssp_obj = DSSP(model, dssp_path, dssp="mkdssp")

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
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)

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

    return helix_count, strand_count


def calculate_rog(chain_or_model):
    """Calculate mass-weighted radius of gyration for a chain or model."""
    coords = np.array([atom.get_coord() for atom in chain_or_model.get_atoms()])
    masses = np.array([atom.mass for atom in chain_or_model.get_atoms()])

    if len(coords) == 0:
        return 0.0

    com = np.average(coords, axis=0, weights=masses)
    diff = coords - com
    rg = round(float(np.sqrt(np.sum(masses * np.sum(diff**2, axis=1)) / masses.sum())), 2)
    return rg

def analyze_structure(args):
    """Analyze structure with automatic chain detection"""
    (pdb_file, fold_min_ss, fold_max_ss, fold_min_helices, fold_max_helices, 
     fold_min_strands, fold_max_strands, fold_min_rog, fold_max_rog, 
     output_dir, json_dir) = args

    logger = logging.getLogger('filter_fold')

    try:
        # Load structure with BioPython PDB parser
        structure, model = parse_structure(pdb_file)

        # Autodetect chain structure
        chain_ids = get_chain_ids(model)
        num_chains = len(chain_ids)

        if num_chains == 1:
            # Monomer - analyze entire structure (single chain)
            primary_chain = chain_ids[0]
            helix_count, strand_count = count_secondary_structures(model, pdb_file, primary_chain)
            rog = calculate_rog(model[primary_chain])
            logger.info(f"{pdb_file.name}: Single chain detected - treating as monomer")
        elif num_chains == 2:
            # Binder - analyze first chain only
            first_chain_id = chain_ids[0]
            helix_count, strand_count = count_secondary_structures(model, pdb_file, first_chain_id)
            rog = calculate_rog(model[first_chain_id])
            logger.info(f"{pdb_file.name}: Two chains detected ({num_chains}) - treating as binder, analyzing first chain only")
        elif num_chains >= 3:
            # Oligomer - aggregate per-chain SS counts and whole-structure RoG
            helix_count = 0
            strand_count = 0
            for chain_id in chain_ids:
                chain_helices, chain_strands = count_secondary_structures(model, pdb_file, chain_id)
                helix_count += chain_helices
                strand_count += chain_strands
            rog = calculate_rog(model)
            logger.info(f"{pdb_file.name}: More than two chains detected - treating as oligomer, analysing all chains")
        else:
            logger.error(f"{pdb_file.name}: No chains found, skipping")
            return None

        total_ss = helix_count + strand_count
        fold_id = extract_fold_id(pdb_file)

        # Helper function for cleaner filter checking
        def passes_filter(value, min_val, max_val):
            if min_val is not None and max_val is not None:
                return min_val <= value <= max_val
            if min_val is not None:
                return value >= min_val
            if max_val is not None:
                return value <= max_val
            return True

        # Apply all filters
        passes_ss_filter = passes_filter(total_ss, fold_min_ss, fold_max_ss)
        passes_helix_filter = passes_filter(helix_count, fold_min_helices, fold_max_helices)
        passes_strand_filter = passes_filter(strand_count, fold_min_strands, fold_max_strands)
        passes_rog_filter = passes_filter(rog, fold_min_rog, fold_max_rog)

        passes_all_filters = (passes_ss_filter and passes_helix_filter and 
                             passes_strand_filter and passes_rog_filter)

        # Create filter description for logging
        applied_filters = []
        if fold_min_ss is not None or fold_max_ss is not None:
            applied_filters.append(f"SS: {fold_min_ss or 'None'}-{fold_max_ss or 'None'}")
        if fold_min_helices is not None or fold_max_helices is not None:
            applied_filters.append(f"Helix: {fold_min_helices or 'None'}-{fold_max_helices or 'None'}")
        if fold_min_strands is not None or fold_max_strands is not None:
            applied_filters.append(f"Strand: {fold_min_strands or 'None'}-{fold_max_strands or 'None'}")
        if fold_min_rog is not None or fold_max_rog is not None:
            applied_filters.append(f"RoG: {fold_min_rog or 'None'}-{fold_max_rog or 'None'}")

        filters_str = ", ".join(applied_filters) if applied_filters else "No filters applied"

        logger.info(
            f"{pdb_file.name}: Analysis complete - "
            f"Helices={helix_count}, Strands={strand_count}, "
            f"Total SS={total_ss}, RoG={rog} Å, "
            f"Filters=[{filters_str}], Passed={passes_all_filters}"
        )

        # Copy files if they pass filters
        if passes_all_filters:
            # Copy PDB file
            output_path = output_dir / pdb_file.name
            shutil.copy2(pdb_file, output_path)

            # Copy corresponding JSON file if it exists
            if json_dir:
                json_file = Path(json_dir) / f"{pdb_file.stem}.json"
                if json_file.exists():
                    json_output_path = output_dir / json_file.name
                    shutil.copy2(json_file, json_output_path)
                    logger.info(f"{json_file.name}: Corresponding JSON copied to output directory")
                else:
                    logger.warning(f"{json_file.name}: Corresponding JSON file not found in {json_dir}")

            logger.info(f"{pdb_file.name}: Copied to output directory")

        return {
            "fold_id": fold_id,
            "fold_helices": helix_count,
            "fold_strands": strand_count,
            "fold_total_ss": total_ss,
            "fold_RoG": rog,
        }

    except Exception as e:
        logger.error(f"{pdb_file.name}: Failed - {str(e)}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(description='Filter PDB files based on secondary structure content')
    parser.add_argument('--input-dir', type=str, help='Directory containing PDB files')
    parser.add_argument('--json-dir', type=str, default=None, 
                       help='Directory containing JSON files corresponding to PDB files (default: same as input-dir)')
    parser.add_argument('--fold-min-ss', type=int, default=None, help='Minimum total secondary structure elements')
    parser.add_argument('--fold-max-ss', type=int, default=None, help='Maximum total secondary structure elements')
    parser.add_argument('--fold-min-helices', type=int, default=None, help='Minimum number of alpha helices')
    parser.add_argument('--fold-max-helices', type=int, default=None, help='Maximum number of alpha helices')
    parser.add_argument('--fold-min-strands', type=int, default=None, help='Minimum number of beta strands')
    parser.add_argument('--fold-max-strands', type=int, default=None, help='Maximum number of beta strands')
    parser.add_argument('--fold-min-rog', type=float, default=None, help='Minimum radius of gyration')
    parser.add_argument('--fold-max-rog', type=float, default=None, help='Maximum radius of gyration')
    parser.add_argument('--ncpus', type=int, default=1, help='Number of CPU cores to use')
    parser.add_argument('--output-dir', type=Path, default="filtered", 
        help="Directory to save filtered PDB files (default: filtered)")
    args = parser.parse_args()
    
    # setup logging
    logger = setup_logger()
    logger.info(f"Starting analysis with parameters: {vars(args)}")
    
    # create output directory
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Set json_dir to input_dir if not specified
    json_dir = args.json_dir if args.json_dir else args.input_dir
    
    pdb_files = list(Path(args.input_dir).glob('*.pdb'))
    if not pdb_files:
        logger.error(f"No PDB files found in {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(pdb_files)} PDB files to analyze")
    logger.info(f"Looking for JSON files in {json_dir}")
    
    process_args = [(
        pdb, 
        args.fold_min_ss, 
        args.fold_max_ss,
        args.fold_min_helices, 
        args.fold_max_helices,
        args.fold_min_strands, 
        args.fold_max_strands,
        args.fold_min_rog,
        args.fold_max_rog,
        args.output_dir,
        json_dir
    ) for pdb in pdb_files]
    
    with Pool(processes=args.ncpus) as pool:
        results = pool.map(analyze_structure, process_args)
    
    # filter out None results
    valid_results = [result for result in results if result is not None]
    
    if not valid_results:
        logger.error(
            f"All {len(pdb_files)} PDB files failed analysis"
        )
        sys.exit(1)
    
    # save analysis data to JSONL
    output_filename = f'fold_data_{str(uuid.uuid4())[:8]}.jsonl'
    with open(output_filename, 'w') as f:
        for result in valid_results:
            json.dump(result, f)
            f.write('\n')
    
    logger.info(f"Analysis complete. Results saved to {output_filename}")
    
    # Print summary statistics
    passed_pdbs = len(list(output_dir.glob('*.pdb')))
    passed_jsons = len(list(output_dir.glob('*.json')))
    logger.info(f"Summary: {passed_pdbs} out of {len(valid_results)} structures passed all filters")
    logger.info(f"Copied {passed_jsons} corresponding JSON files to the output directory")

if __name__ == "__main__":
    main()
