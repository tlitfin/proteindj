#!/usr/bin/env python3

import argparse
import sys
import os
import json
import re
from pathlib import Path
import shutil
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description='Filter designs and copy top hits from PDB files.')
    parser.add_argument('--json-directory', required=True, help='Path to directory containing JSON score files')
    parser.add_argument('--boltz-max-rmsd-overall', type=float,
                    help='Maximum allowed RMSD value')
    parser.add_argument('--boltz-max-rmsd-binder', type=float,
                        help='Maximum allowed binder chain RMSD value')
    parser.add_argument('--boltz-max-rmsd-target', type=float,
                        help='Maximum allowed target chain RMSD value')
    parser.add_argument('--boltz-min-conf-score', type=float, 
                    help='Minimum confidence score')
    parser.add_argument('--boltz-min-ptm', type=float, 
                    help='Minimum pTM score')
    parser.add_argument('--boltz-min-ptm-binder', type=float,
                    help='Minimum pTM score for binder chain')
    parser.add_argument('--boltz-min-ptm-target', type=float,
                    help='Minimum pTM score for target chain')
    parser.add_argument('--boltz-min-iptm', type=float,
                    help='Minimum interface pTM score')
    parser.add_argument('--boltz-min-plddt', type=float,
                    help='Minimum complex pLDDT score')
    parser.add_argument('--boltz-min-iplddt', type=float,
                    help='Minimum interface-weighted pLDDT')
    parser.add_argument('--boltz-max-pde', type=float,
                    help='Maximum complex PDE score')
    parser.add_argument('--boltz-max-ipde', type=float,
                    help='Maximum interface-weighted PDE')
    parser.add_argument('--boltz-min-ipSAE-min', type=float,
                    help='Minimum interaction Prediction Score from Aligned Errors (ipSAE)')
    parser.add_argument('--boltz-min-LIS', type=float,
                    help='Minimum Local Interaction Score (LIS)')
    parser.add_argument('--boltz-min-pDockQ2-min', type=float,
                    help='Minimum predicted DockQ Score v2')
    parser.add_argument('--boltz-max-pae-interaction', type=float,
                    help='Maximum predicted aligned error at interface')
    parser.add_argument('--boltz-max-unbound-rmsd', type=float,
                    help='Maximum allowed C-alpha RMSD between the unbound (target-free) binder prediction and the design')
    parser.add_argument('--boltz-min-unbound-conf-score', type=float,
                    help='Minimum confidence score for the unbound (target-free) binder prediction')
    parser.add_argument('--boltz-min-unbound-ptm', type=float,
                    help='Minimum pTM score for the unbound (target-free) binder prediction')
    parser.add_argument('--boltz-min-unbound-plddt', type=float,
                    help='Minimum pLDDT score for the unbound (target-free) binder prediction')
    parser.add_argument('--boltz-max-unbound-pde', type=float,
                    help='Maximum PDE score for the unbound (target-free) binder prediction')
    parser.add_argument('--unbound-json-directory',
                    help='Path to directory containing unbound (target-free) binder prediction JSON files '
                         '(fold_*_seq_*_unbound_boltzpred.json), merged into each matching entry before filtering')
    parser.add_argument('--output-directory', default='output', help='Directory to copy passing PDB files to')
    parser.add_argument('--output-score-file', default='filtered.jsonl')
    parser.add_argument('--num-to-extract', type=int, help='Number of designs to extract (extracts all if not specified)')
    parser.add_argument('--json-pattern', default='*.json', help='Pattern to match JSON files in the directory')
    return parser.parse_args()

def read_data_from_directory(directory_path, pattern='*.json'):
    """
    Read all JSON files from a directory.
    
    Args:
        directory_path: Path to directory containing JSON files
        pattern: Glob pattern to match JSON files
        
    Returns:
        List of data entries from all JSON files
    """
    try:
        data = []
        json_files = glob.glob(os.path.join(directory_path, pattern))
        
        if not json_files:
            print(f"No JSON files found in {directory_path} matching pattern {pattern}", file=sys.stderr)
            sys.exit(1)
            
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            try:
                # Extract metadata from filename
                filename = Path(json_file).stem
                if filename.endswith('_unbound_boltzpred'):
                    # Unbound (target-free) binder predictions are merged in separately via
                    # --unbound-json-directory, not treated as independent bound-design entries
                    continue
                match = re.match(r'fold_(\d+)_seq_(\d+)_.*', filename)
                if not match:
                    print(f"Skipping invalid filename format: {filename}")
                    continue
                    
                fold_id = int(match.group(1))
                seq_id = int(match.group(2))
                
                with open(json_file, 'r') as f:
                    file_content = f.read().strip()
                    if not file_content:
                        print(f"Skipping empty file: {json_file}")
                        continue

                    # Add filename metadata to all entries
                    if file_content.startswith('['):  # JSON array
                        file_data = json.loads(file_content)
                        for entry in file_data:
                            entry.update({
                                "description": filename,
                                "fold_id": fold_id,
                                "seq_id": seq_id
                            })
                        data.extend(file_data)
                    elif file_content.startswith('{'):  # Single JSON object
                        entry = json.loads(file_content)
                        entry.update({
                            "description": filename,
                            "fold_id": fold_id,
                            "seq_id": seq_id
                        })
                        data.append(entry)
                    else:  # JSONL format
                        for line in file_content.splitlines():
                            if line.strip():
                                entry = json.loads(line)
                                entry.update({
                                    "description": filename,
                                    "fold_id": fold_id,
                                    "seq_id": seq_id
                                })
                                data.append(entry)
                                
                print(f"Processed {json_file}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {json_file}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing file {json_file}: {e}", file=sys.stderr)
                
        return data
    except Exception as e:
        print(f"Error reading JSON files from directory: {e}", file=sys.stderr)
        sys.exit(1)

UNBOUND_METRIC_KEYS = ('boltz_unbound_rmsd', 'boltz_unbound_conf_score', 'boltz_unbound_ptm', 'boltz_unbound_plddt', 'boltz_unbound_pde')

def load_unbound_metadata(directory):
    """
    Load unbound (target-free) binder prediction JSON files and index by (fold_id, seq_id).

    Only the boltz_unbound_* metric keys are kept, so merging can never clobber unrelated
    fields (e.g. description) on the matching bound entry.

    Returns:
        Dict mapping (fold_id, seq_id) -> dict of boltz_unbound_* metrics
    """
    lookup = {}
    json_files = glob.glob(os.path.join(directory, 'fold_*_seq_*_unbound_boltzpred.json'))
    for json_file in json_files:
        match = re.match(r'fold_(\d+)_seq_(\d+)_unbound_boltzpred', Path(json_file).stem)
        if not match:
            continue
        fold_id = int(match.group(1))
        seq_id = int(match.group(2))
        with open(json_file, 'r') as f:
            entry = json.load(f)
        lookup[(fold_id, seq_id)] = {k: v for k, v in entry.items() if k in UNBOUND_METRIC_KEYS}
    print(f"Loaded {len(lookup)} unbound binder prediction entries from {directory}")
    return lookup

def filter_data(data, args):
    passed_designs = []
    passed_entries = []
    
    for entry in data:
        try:
            failures = []
            
            # Overall RMSD check
            if args.boltz_max_rmsd_overall is not None:
                rmsd = entry.get('boltz_rmsd_overall', 1000)
                if rmsd > args.boltz_max_rmsd_overall:
                    failures.append(f"rmsd_overall {rmsd:.2f} > {args.boltz_max_rmsd_overall}")
            
            # Binder RMSD check
            if args.boltz_max_rmsd_binder is not None:
                rmsd_binder = entry.get('boltz_rmsd_binder', 1000)
                if rmsd_binder > args.boltz_max_rmsd_binder:
                    failures.append(f"rmsd_binder {rmsd_binder:.2f} > {args.boltz_max_rmsd_binder}")
            
            # Target RMSD check
            if args.boltz_max_rmsd_target is not None:
                rmsd_target = entry.get('boltz_rmsd_target', 1000)
                if rmsd_target > args.boltz_max_rmsd_target:
                    failures.append(f"rmsd_target {rmsd_target:.2f} > {args.boltz_max_rmsd_target}")

            # Confidence score check
            if args.boltz_min_conf_score is not None:
                score = entry.get('boltz_conf_score', 0)
                if score < args.boltz_min_conf_score:
                    failures.append(f"conf_score {score:.3f} < {args.boltz_min_conf_score}")
            
            # PTM/iPTM checks
            if args.boltz_min_ptm is not None:
                ptm = entry.get('boltz_ptm', 0)
                if ptm < args.boltz_min_ptm:
                    failures.append(f"ptm {ptm:.3f} < {args.boltz_min_ptm}")
            
            if args.boltz_min_ptm_binder is not None:
                ptm_binder = entry.get('boltz_ptm_binder', 0)
                if ptm_binder < args.boltz_min_ptm_binder:
                    failures.append(f"ptm_binder {ptm_binder:.3f} < {args.boltz_min_ptm_binder}")
            
            if args.boltz_min_ptm_target is not None:
                ptm_target = entry.get('boltz_ptm_target', 0)
                if ptm_target < args.boltz_min_ptm_target:
                    failures.append(f"ptm_target {ptm_target:.3f} < {args.boltz_min_ptm_target}")
            
            if args.boltz_min_iptm is not None:
                iptm = entry.get('boltz_iptm', 0)
                if iptm < args.boltz_min_iptm:
                    failures.append(f"iptm {iptm:.3f} < {args.boltz_min_iptm}")
            
            # Structure quality checks
            if args.boltz_min_plddt is not None:
                plddt = entry.get('boltz_plddt', 0)
                if plddt < args.boltz_min_plddt:
                    failures.append(f"plddt {plddt:.3f} < {args.boltz_min_plddt}")
            
            if args.boltz_min_iplddt is not None:
                iplddt = entry.get('boltz_iplddt', 0)
                if iplddt < args.boltz_min_iplddt:
                    failures.append(f"iplddt {iplddt:.3f} < {args.boltz_min_iplddt}")
            
            # Energy metric checks
            if args.boltz_max_pde is not None:
                pde = entry.get('boltz_pde', 0)
                if pde > args.boltz_max_pde:
                    failures.append(f"pde {pde:.2f} > {args.boltz_max_pde}")
            
            if args.boltz_max_ipde is not None:
                ipde = entry.get('boltz_ipde', 0)
                if ipde > args.boltz_max_ipde:
                    failures.append(f"ipde {ipde:.2f} > {args.boltz_max_ipde}")
            
            # Interface interaction metrics
            if args.boltz_min_ipSAE_min is not None:
                ipsae_min = entry.get('ipSAE_min', 0)
                if ipsae_min < args.boltz_min_ipSAE_min:
                    failures.append(f"ipSAE_min {ipsae_min:.3f} < {args.boltz_min_ipSAE_min}")
            
            if args.boltz_min_LIS is not None:
                lis = entry.get('LIS', 0)
                if lis < args.boltz_min_LIS:
                    failures.append(f"LIS {lis:.3f} < {args.boltz_min_LIS}")
            
            if args.boltz_min_pDockQ2_min is not None:
                pdockq2_min = entry.get('pDockQ2_min', 0)
                if pdockq2_min < args.boltz_min_pDockQ2_min:
                    failures.append(f"pDockQ2_min {pdockq2_min:.3f} < {args.boltz_min_pDockQ2_min}")
            
            if args.boltz_max_pae_interaction is not None:
                pae_interaction = entry.get('boltz_pae_interaction', 1000)
                if pae_interaction > args.boltz_max_pae_interaction:
                    failures.append(f"pae_interaction {pae_interaction:.2f} > {args.boltz_max_pae_interaction}")

            # Unbound (target-free) binder prediction checks
            if args.boltz_max_unbound_rmsd is not None:
                unbound_rmsd = entry.get('boltz_unbound_rmsd', 1000)
                if unbound_rmsd > args.boltz_max_unbound_rmsd:
                    failures.append(f"unbound_rmsd {unbound_rmsd:.2f} > {args.boltz_max_unbound_rmsd}")

            if args.boltz_min_unbound_conf_score is not None:
                unbound_conf_score = entry.get('boltz_unbound_conf_score', 0)
                if unbound_conf_score < args.boltz_min_unbound_conf_score:
                    failures.append(f"unbound_conf_score {unbound_conf_score:.3f} < {args.boltz_min_unbound_conf_score}")

            if args.boltz_min_unbound_ptm is not None:
                unbound_ptm = entry.get('boltz_unbound_ptm', 0)
                if unbound_ptm < args.boltz_min_unbound_ptm:
                    failures.append(f"unbound_ptm {unbound_ptm:.3f} < {args.boltz_min_unbound_ptm}")

            if args.boltz_min_unbound_plddt is not None:
                unbound_plddt = entry.get('boltz_unbound_plddt', 0)
                if unbound_plddt < args.boltz_min_unbound_plddt:
                    failures.append(f"unbound_plddt {unbound_plddt:.3f} < {args.boltz_min_unbound_plddt}")

            if args.boltz_max_unbound_pde is not None:
                unbound_pde = entry.get('boltz_unbound_pde', 0)
                if unbound_pde > args.boltz_max_unbound_pde:
                    failures.append(f"unbound_pde {unbound_pde:.2f} > {args.boltz_max_unbound_pde}")

            if not failures:
                passed_designs.append(entry['description'])
                passed_entries.append(entry)
            else:
                print(f"Rejected {entry['description']}: {', '.join(failures)}")
                
        except KeyError as e:
            print(f"Invalid entry missing key {e}")
    
    return passed_designs, passed_entries

def copy_pdb_files(passed_designs, output_dir):
    """
    Copy PDB files from the current directory to the output directory.
    
    Args:
        passed_designs: List of design names that passed filtering
        output_dir: Directory to copy files to
        
    Returns:
        List of successfully copied design names
    """
    successfully_copied = []
    
    for design_name in passed_designs:
        # Append .pdb to the description to get the filename
        source_path = Path(f"{design_name}.pdb")
        
        if source_path.exists():
            dest_path = Path(output_dir) / source_path.name
            shutil.copy2(source_path, dest_path)
            print(f"Copied {source_path} to {dest_path}")
            successfully_copied.append(design_name)
        else:
            print(f"Warning: Could not find PDB file for design '{design_name}'")
    
    return successfully_copied

def write_filtered_score_file(passed_entries, successfully_copied, output_score_file):
    with open(output_score_file, 'w') as outfile:
        for entry in passed_entries:
            if entry['description'] in successfully_copied:
                outfile.write(json.dumps(entry) + '\n')
    
    print(f"filtered score file created: {output_score_file}")

def main():
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output_directory)
    output_dir.mkdir(exist_ok=True)
    
    # Read and filter data from directory
    data = read_data_from_directory(args.json_directory, args.json_pattern)

    if not data:
        print("No data found in the JSON files. Exiting.")
        sys.exit(0)

    print(f"Total entries loaded from all JSON files: {len(data)}")

    # Merge in unbound (target-free) binder prediction metrics, keyed by (fold_id, seq_id)
    if args.unbound_json_directory:
        unbound_lookup = load_unbound_metadata(args.unbound_json_directory)
        for entry in data:
            entry.update(unbound_lookup.get((entry.get('fold_id'), entry.get('seq_id')), {}))

    # Check if any filter is applied
    any_filter_applied = (
        args.boltz_max_rmsd_overall is not None or
        args.boltz_max_rmsd_binder is not None or
        args.boltz_max_rmsd_target is not None or
        args.boltz_min_conf_score is not None or
        args.boltz_min_ptm is not None or
        args.boltz_min_ptm_binder is not None or
        args.boltz_min_ptm_target is not None or
        args.boltz_min_iptm is not None or
        args.boltz_min_plddt is not None or
        args.boltz_min_iplddt is not None or
        args.boltz_max_pde is not None or
        args.boltz_max_ipde is not None or
        args.boltz_max_unbound_rmsd is not None or
        args.boltz_min_unbound_conf_score is not None or
        args.boltz_min_unbound_ptm is not None or
        args.boltz_min_unbound_plddt is not None or
        args.boltz_max_unbound_pde is not None
    )
    
    if not any_filter_applied:
        print("Warning: No filter parameters specified. All designs will pass.")
    
    passed_designs, passed_entries = filter_data(data, args)

    if not passed_designs:
        print("no designs passed filtering.")
        sys.exit(0)
        
    if args.num_to_extract is not None:
        if args.num_to_extract < len(passed_designs):
            print(f"Limiting to top {args.num_to_extract} designs out of {len(passed_designs)} that passed filters")
            passed_designs = passed_designs[:args.num_to_extract]
            passed_entries = passed_entries[:args.num_to_extract]
        else:
            print(f"Requested {args.num_to_extract} designs but only {len(passed_designs)} passed filters")

    # Copy PDB files sequentially
    print(f"\nCopying PDB files...")
    print(f"Total designs to process: {len(passed_designs)}")
    
    successfully_copied_designs = copy_pdb_files(passed_designs, args.output_directory)
    
    # Print summary
    print(f"\nFile Copy Summary:")
    print(f"Total designs that passed filtering: {len(passed_designs)}")
    print(f"Successfully copied PDB files: {len(successfully_copied_designs)}")
    
    # Create filtered score file
    write_filtered_score_file(passed_entries, successfully_copied_designs, args.output_score_file)

if __name__ == '__main__':
    main()
