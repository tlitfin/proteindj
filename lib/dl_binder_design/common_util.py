import io
from typing import Tuple


def _init_pdbfixer_from_text(pdb_text: str):
    from pdbfixer import PDBFixer
    try:
        return PDBFixer(pdbfile=io.StringIO(pdb_text))
    except TypeError:
        # Older/newer PDBFixer variants may not expose `pdbfile=` constructor.
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
            tmp.write(pdb_text)
            tmp_path = tmp.name
        try:
            return PDBFixer(filename=tmp_path)
        finally:
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def complete_pdb_backbone_to_all_atom_text_from_text(pdb_text: str) -> Tuple[str, int]:
    """
    Use PDBFixer to add missing heavy atoms from in-memory PDB text.

    Returns:
        tuple[str, int]: (completed_pdb_text, number_of_missing_atoms_identified)
    """
    try:
        from pdbfixer import PDBFixer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PDBFixer is required for atom completion. Install it (e.g. `conda install -c conda-forge pdbfixer`)."
        ) from exc

    try:
        from openmm.app import PDBFile
    except ImportError:
        from simtk.openmm.app import PDBFile

    fixer = _init_pdbfixer_from_text(pdb_text)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    missing_atom_count = sum(len(atom_list) for atom_list in fixer.missingAtoms.values())
    fixer.addMissingAtoms()

    out_buf = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, out_buf, keepIds=True)
    return out_buf.getvalue(), missing_atom_count

def complete_pdb_backbone_to_all_atom_text(pdb_fn: str) -> Tuple[str, int]:
    """
    Use PDBFixer to add missing heavy atoms, returning completed PDB text.

    Returns:
        tuple[str, int]: (completed_pdb_text, number_of_missing_atoms_identified)
    """
    try:
        from pdbfixer import PDBFixer
    except ImportError as exc:
        raise ImportError(
            "PDBFixer is required for atom completion. Install it (e.g. `conda install -c conda-forge pdbfixer`)."
        ) from exc

    try:
        from openmm.app import PDBFile
    except ImportError:
        from simtk.openmm.app import PDBFile

    fixer = PDBFixer(filename=pdb_fn)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    missing_atom_count = sum(len(atom_list) for atom_list in fixer.missingAtoms.values())
    fixer.addMissingAtoms()

    out_buf = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, out_buf, keepIds=True)
    return out_buf.getvalue(), missing_atom_count

def optimize_rotamers_with_pyfaspr(pdb_text: str) -> str:
    """
    Optimize side-chain rotamers using pyfaspr.

    Args:
        pdb_text (str): Input structure as PDB text.

    Returns:
        str: Rotamer-optimized PDB text.
    """
    try:
        import pyfaspr
    except ImportError as exc:
        raise ImportError(
            "pyfaspr is required for rotamer optimization. Install it (e.g. `pip install pyfaspr`)."
        ) from exc

    if not hasattr(pyfaspr, "run_FASPR"):
        raise AttributeError("pyfaspr module does not expose `run_FASPR`.")

    result = pyfaspr.run_FASPR(pdb=pdb_text)

    if isinstance(result, bytes):
        return result.decode("utf-8")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("pdb", "pdb_out", "output_pdb", "result_pdb"):
            if key in result and isinstance(result[key], str):
                return result[key]
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, str):
                return item

    raise TypeError(f"Unexpected return type from pyfaspr.run_FASPR: {type(result)}")
