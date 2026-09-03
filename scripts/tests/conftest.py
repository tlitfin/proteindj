"""Shared pytest fixtures for scripts/ Tier A unit tests.

scripts/*.py are flat CLI scripts (no package/__init__.py), so we add scripts/ to
sys.path here to allow tests to `import module_name` directly.
"""
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def write_json_files(tmp_path):
    """Factory fixture: write a dict of {filename: dict_or_list} as JSON files in tmp_path."""
    def _write(files, directory=None):
        target_dir = directory or tmp_path
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (target_dir / filename).write_text(json.dumps(content))
        return target_dir
    return _write


@pytest.fixture
def write_jsonl_file(tmp_path):
    """Factory fixture: write a list of dicts as a JSONL file."""
    def _write(filename, entries):
        path = tmp_path / filename
        with open(path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        return path
    return _write


@pytest.fixture
def pdb_atom_line():
    """Factory fixture: build a fixed-width ATOM record matching the PDB format spec,
    so both raw column-slicing code (line[21], line[22:26], line[60:66], ...) and
    Bio.PDB's PDBParser can read it correctly."""
    def _line(serial=1, name='CA', resname='ALA', chain='A', resnum=1,
              x=0.0, y=0.0, z=0.0, occ=1.0, temp=0.0, element='C'):
        return (
            f"ATOM  {serial:>5} {name:<4} {resname:>3} {chain:1}{resnum:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{temp:>6.2f}          {element:>2}"
        )
    return _line
