#!/bin/bash
# Regenerates the per-mode Nextflow schemas and bindsweeper's binder_schema.json together,
# so the two derived-schema generators can never drift out of sync with each other.

set -e

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

python3 scripts/generate_mode_schemas.py -v
python3 scripts/generate_bindsweeper_schema.py -v

echo "Schemas regenerated successfully."
