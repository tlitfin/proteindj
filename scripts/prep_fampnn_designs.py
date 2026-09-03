import argparse
import io
import os
import subprocess
import tempfile
from pathlib import Path
import pyfaspr
import numpy as np
import pdbutil
from Bio.PDB import PDBParser
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def restore_backbone_atoms(pdb_text: str) -> str:
    """Use PDBFixer to restore any missing backbone (and other standard heavy) atoms before side-chain packing."""

    fixer = PDBFixer(pdbfile=io.StringIO(pdb_text))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    out_buf = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, out_buf, keepIds=True)
    return out_buf.getvalue()

def get_residue_metadata(pdb_text: str) -> dict:
    """Map (chain_id, resnum) -> (bfactor, occupancy) using each residue's CA atom.

    Unlike pdbutil.read_pdb, this does not drop residues that are missing other
    backbone atoms, so metadata can still be recovered for residues that
    restore_backbone_atoms() repairs later on.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', io.StringIO(pdb_text))
    metadata = {}
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                if atom.name == 'CA':
                    _, resnum, _ins = residue.get_id()
                    metadata[(chain.get_id(), int(resnum))] = (atom.get_bfactor(), atom.get_occupancy())
                    break
    return metadata

def restore_sidechains(pdb_fn: str, faspr_bin: str) -> str:
    """Use PDBFixer to restore missing backbone atoms, then FASPR to add/rebuild side-chains, returning PDB text."""

    pdb_text = Path(pdb_fn).read_text()

    # Capture the original per-residue metadata (bfactor, occupancy) keyed by (chain, resnum)
    # before PDBFixer/FASPR touch the structure. pdbutil.read_pdb can't be used for this since
    # it silently drops residues with missing backbone atoms (exactly what we're about to fix).
    orig_metadata = get_residue_metadata(pdb_text)

    # Restore any missing backbone atoms (e.g. dropped carbonyl O) prior to side-chain packing
    pdb_text = restore_backbone_atoms(pdb_text)

    # Renumber residues sequentially (required by FASPR)
    data_org = pdbutil.read_pdb(pdb_text)
    data = data_org.copy()
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

    # Restore original residue numbering, b-factors, and occupancies.
    # Looked up by (chain, resnum) identity rather than array position/order, since PDBFixer
    # may have added atoms/residues not present in the original file.
    data_out = pdbutil.read_pdb(pdb_text_out)
    data_out['resnum'] = data_org['resnum']
    lookup = [orig_metadata.get((c, r), (0.0, 1.0)) for c, r in zip(data_org['chain'], data_org['resnum'])]
    data_out['bfactor'] = np.array([bfac for bfac, _ in lookup])
    data_out['occupancy'] = np.array([occu for _, occu in lookup])
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
