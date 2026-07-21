#!/usr/bin/env python3
import os
import re
import argparse
import json
import shutil
from Bio.SeqUtils.ProtParam import ProteinAnalysis

SCORE_FIELDS = {
    'mpnn': 'score',
    'fampnn': 'fampnn_avg_psce',
}

def extract_designed_sequence(sequence_field):
    """Extract the first designed chain's sequence.
    Handles plain sequences and FAMPNN multi-chain format (e.g., 'A:SEQ|B:SEQ').
    """
    first = sequence_field.split('|')[0]  # Take first chain if multi-chain
    if ':' in first:
        return first.split(':', 1)[1]     # Strip chain ID prefix if present
    return first

def calculate_seq_metrics(design, sequence_field):
    """Calculate sequence metrics for a design using BioPython ProteinAnalysis."""
    sequence = extract_designed_sequence(sequence_field)
    if not sequence:
        return None
    try:
        analysis = ProteinAnalysis(sequence)
        match = re.match(r'fold_(\d+)_seq_(\d+)', design)
        return {
            'fold_id': int(match.group(1)) if match else None,
            'seq_id': int(match.group(2)) if match else None,
            'seq_ext_coef': analysis.molar_extinction_coefficient()[0],
            'seq_length': len(sequence),
            'seq_MW': int(analysis.molecular_weight()),
            'seq_pI': round(analysis.isoelectric_point(), 2),
        }
    except Exception as e:
        print(f"Warning: Could not calculate seq metrics for {design}: {e}")
        return None

def load_json_data(json_dir, method):
    """Load sequence design scores and sequences from JSON metadata files."""
    data_map = {}
    score_field = SCORE_FIELDS[method]

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            with open(os.path.join(json_dir, json_file)) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        design_name = data['design']
                        data_map[design_name] = {
                            'score': float(data[score_field]),
                            'sequence': data.get('sequence', ''),
                        }
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Error parsing line in {json_file}: {e}")
    return data_map

def filter_by_score(data_map, max_score):
    """Filter designs by score threshold."""
    if max_score is None:
        return data_map
    return {d: v for d, v in data_map.items() if v['score'] <= max_score}

def filter_by_seq_metrics(data_map, metrics_map, args):
    """Filter designs by sequence metrics (ext_coef and pI)."""
    passed = {}
    for design, data in data_map.items():
        m = metrics_map.get(design)
        if m is None:
            print(f"Warning: No seq metrics for {design}, skipping seq metric filters")
            passed[design] = data
            continue
        if args.seq_min_ext_coef is not None and m['seq_ext_coef'] < args.seq_min_ext_coef:
            print(f"  {design}: rejected (seq_ext_coef {m['seq_ext_coef']} < {args.seq_min_ext_coef})")
            continue
        if args.seq_max_ext_coef is not None and m['seq_ext_coef'] > args.seq_max_ext_coef:
            print(f"  {design}: rejected (seq_ext_coef {m['seq_ext_coef']} > {args.seq_max_ext_coef})")
            continue
        if args.seq_min_pi is not None and m['seq_pI'] < args.seq_min_pi:
            print(f"  {design}: rejected (seq_pI {m['seq_pI']} < {args.seq_min_pi})")
            continue
        if args.seq_max_pi is not None and m['seq_pI'] > args.seq_max_pi:
            print(f"  {design}: rejected (seq_pI {m['seq_pI']} > {args.seq_max_pi})")
            continue
        passed[design] = data
    return passed

def copy_filtered_designs(filtered_designs, pdb_dir, json_dir, output_dir):
    """Copy matching PDBs and JSONs to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    copied_count = 0

    for design in filtered_designs:
        # Copy PDB file
        pdb_file = os.path.join(pdb_dir, f"{design}.pdb")
        if os.path.exists(pdb_file):
            shutil.copy2(pdb_file, output_dir)
            copied_count += 1

        # Copy JSON metadata
        json_file = os.path.join(json_dir, f"{design}.json")
        if os.path.exists(json_file):
            shutil.copy2(json_file, output_dir)
        else:
            print(f"Warning: JSON file for {design} not found")

    return copied_count

def write_seq_metrics_jsonl(metrics_map, output_path):
    """Write sequence metrics for passing designs to a JSONL file for the metadata topic channel."""
    with open(output_path, 'w') as f:
        for m in metrics_map.values():
            if m is not None:
                f.write(json.dumps(m) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Filter sequence design outputs (MPNN or FAMPNN)")
    parser.add_argument("--method", required=True, choices=["mpnn", "fampnn"],
                        help="Sequence design method")
    parser.add_argument("--jsons", required=True, help="Directory containing JSON metadata files")
    parser.add_argument("--pdbs", required=True, help="Directory containing PDB files")
    parser.add_argument("--max-score", type=float,
                        help="Maximum score threshold (MPNN score or FAMPNN avg PSCE; copies all if not provided)")
    # Sequence metric filters
    parser.add_argument("--seq-min-ext-coef", type=float, help="Minimum extinction coefficient")
    parser.add_argument("--seq-max-ext-coef", type=float, help="Maximum extinction coefficient")
    parser.add_argument("--seq-min-pi", type=float, help="Minimum isoelectric point")
    parser.add_argument("--seq-max-pi", type=float, help="Maximum isoelectric point")
    parser.add_argument("--seq-metrics-jsonl", default="seq_metrics.jsonl",
                        help="Output JSONL file for sequence metrics metadata (default: seq_metrics.jsonl)")
    parser.add_argument("--output-dir", default="filtered_output",
                        help="Output directory for filtered designs (default: filtered_output)")

    args = parser.parse_args()

    method_label = args.method.upper()
    if args.max_score is not None:
        print(f"Filtering {method_label} designs with score ≤ {args.max_score}")
    else:
        print(f"No {method_label} score filtering applied.")

    data_map = load_json_data(args.jsons, args.method)
    print(f"Pre-filter designs: {len(data_map)}")

    # Calculate seq metrics for ALL designs before any filtering
    metrics_map = {}
    for design, data in data_map.items():
        if data['sequence']:
            metrics_map[design] = calculate_seq_metrics(design, data['sequence'])

    filtered = filter_by_score(data_map, args.max_score)
    print(f"Post-score-filter designs: {len(filtered)}")

    # Apply seq metric filters if any are set
    seq_filters_active = any(v is not None for v in [
        args.seq_min_ext_coef, args.seq_max_ext_coef,
        args.seq_min_pi, args.seq_max_pi,
    ])
    if seq_filters_active:
        print("Applying sequence metric filters...")
        filtered = filter_by_seq_metrics(filtered, metrics_map, args)
        print(f"Post-seq-filter designs: {len(filtered)}")

    # Write seq metrics for all designs to JSONL for the metadata topic channel
    write_seq_metrics_jsonl(metrics_map, args.seq_metrics_jsonl)
    print(f"Wrote sequence metrics for {len(metrics_map)} designs to {args.seq_metrics_jsonl}")

    copied_count = copy_filtered_designs(filtered.keys(), args.pdbs, args.jsons, args.output_dir)
    print(f"\nResults: {len(filtered)} designs found, {copied_count} PDB files copied to {args.output_dir}")

if __name__ == "__main__":
    main()
