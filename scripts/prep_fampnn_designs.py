import argparse
import os
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
import pyfaspr
import numpy as np
import pdbutil

def restore_sidechains(pdb_fn: str, faspr_bin: str) -> str:
    """Use FASPR to add/rebuild side-chains from a backbone PDB file, returning PDB text."""

    pdb_text = Path(pdb_fn).read_text()

    # Renumber residues sequentially (required by FASPR)
    data_org = pdbutil.read_pdb(pdb_text)
    data = deepcopy(data_org)
    data['resnum'] = np.arange(1, len(data['resnum']) + 1)
    pdb_text_in = pdbutil.write_pdb(**data)

    with tempfile.TemporaryDirectory() as tmp:
        input_pdb = os.path.join(tmp, 'input.pdb')
        output_pdb = os.path.join(tmp, 'output.pdb')
        Path(input_pdb).write_text(pdb_text_in)
        result = subprocess.run(
            [faspr_bin, '-i', input_pdb, '-o', output_pdb],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode != 0 or not os.path.exists(output_pdb):
            raise RuntimeError(
                f"FASPR failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout.decode()}\n"
                f"stderr: {result.stderr.decode()}"
            )
        
        pdb_text_out = Path(output_pdb).read_text()

    # Restore original residue numbering, b-factors, and occupancies
    data_out = pdbutil.read_pdb(pdb_text_out)
    data_out['resnum'] = data_org['resnum']
    data_out['bfactor'] = data_org['bfactor']
    data_out['occupancy'] = data_org['occupancy']
    return pdbutil.write_pdb(**data_out)

def main():
    parser = argparse.ArgumentParser(description='Restores side-chains to PDB files after RFdiffusion processing')
    parser.add_argument('--input_dir', required=True, help='Input directory containing PDB files')
    parser.add_argument('--out_dir', default='./outputpdbs', help='Output directory for updated PDB files')
    args = parser.parse_args()

    faspr_bin = os.path.join(os.path.dirname(pyfaspr.__file__), 'bin', 'FASPR_x86_64_linux')

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for pdb_file in input_dir.glob("*.pdb"):
        pdb_text = restore_sidechains(str(pdb_file), faspr_bin)
        output_path = out_dir / pdb_file.name
        print(f"Rebuilt sidechains for PDB file: {output_path}")
        output_path.write_text(pdb_text)

if __name__ == "__main__":
    main()
