#!/usr/bin/env python3
"""
Generate a BoltzGen design-spec YAML for ProteinDJ's boltzgen_denovo/boltzgen_redesign modes.

- boltzgen_denovo: designs a new binder (entity 1, chain A) against a fixed target
  (entity 2, taken from input_pdb), with optional hotspot/anti-hotspot residues and/or
  target structural flexibility.
- boltzgen_redesign: reworks an existing chain A binder against the remaining fixed
  chain(s) of input_pdb - optionally changing its architecture (keeping/deleting/
  inserting residues via bg_redesign_spec), redesigning the sequence of kept residues
  (bg_redesign_inpaint_seq), and/or freeing up the structure of residues on any chain
  (bg_flexible_residues).

Entities are emitted binder-then-target so BoltzGen's generated CIF naturally places
the binder as chain A (matches ProteinDJ's chain A=binder/chain B=target convention).
"""

import argparse
import re
from pathlib import Path

import yaml
from Bio.PDB import PDBParser


def get_protein_chains(pdb_file):
    """Return sorted list of chain IDs containing at least one standard residue."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    chains = []
    for model in structure:
        for chain in model:
            if chain.id not in chains and any(res.id[0] == ' ' for res in chain):
                chains.append(chain.id)
        break  # Only need the first model
    return sorted(chains)


def get_chain_length(pdb_file, chain_id):
    """Return the number of standard residues in a given chain."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return len([res for res in chain if res.id[0] == ' '])
        break
    return 0


def get_residue_rank_map(pdb_file):
    """
    Return {chain_id: {auth_resnum: rank}} mapping each chain's PDB author residue
    numbers (as used in ProteinDJ's hotspot_residues/bg_* residue specs, e.g. 'A56')
    to their 1-based positional rank within that chain's standard-residue sequence.

    BoltzGen's design-spec 'res_index'/'binding'/'not_binding' fields are positional
    (1 = first residue present in the chain, regardless of its PDB numbering), so
    author residue numbers must be translated through this map before being written
    into the YAML - passing raw PDB numbers directly causes BoltzGen to misinterpret
    them as positions and raise spurious 'end is higher than length of chain' errors.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    rank_maps = {}
    for model in structure:
        for chain in model:
            resnums = [res.id[1] for res in chain if res.id[0] == ' ']
            rank_maps[chain.id] = {resnum: rank for rank, resnum in enumerate(resnums, start=1)}
        break  # Only need the first model
    return rank_maps


def _resolve_rank(chain, resnum, rank_map):
    """Look up the 1-based positional rank of a PDB author residue number."""
    chain_map = rank_map.get(chain)
    if chain_map is None:
        raise ValueError(f"Chain '{chain}' not found in input_pdb")
    rank = chain_map.get(resnum)
    if rank is None:
        raise ValueError(f"Residue {chain}{resnum} not found in input_pdb (check residue numbering)")
    return rank


# PDB header record types that only carry sequence/molecule metadata (as opposed to atomic
# coordinates/geometry). BoltzGen's PDB parser uses gemmi, whose automatic entity detection
# reads these records to infer polymer entities. If a PDB has been cropped to fewer chains
# without updating these records (e.g. our benchmark/example PDBs, or any real-world structure
# where a partner chain was deleted), gemmi can fabricate a phantom Polymer entity for the
# stale chain with zero actual subchains, crashing BoltzGen with "IndexError: list index out
# of range" in parse_pdb (entity.subchains[0]). We strip these records and keep only what's
# needed for coordinates/geometry; gemmi/BoltzGen fall back to deriving sequence from atoms.
_PDB_HEADER_METADATA_PREFIXES = (
    'SEQRES', 'COMPND', 'DBREF', 'SEQADV', 'SOURCE', 'MODRES',
    'HET ', 'HETNAM', 'HETSYN', 'FORMUL',
)


def clean_pdb_for_boltzgen(input_path, output_path):
    """Write a copy of input_path with stale sequence/molecule header records stripped."""
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            if line.startswith(_PDB_HEADER_METADATA_PREFIXES):
                continue
            fout.write(line)


def parse_design_length(design_length):
    """Convert ProteinDJ's design_length ('80' or '60-100') to BoltzGen sequence range syntax."""
    design_length = design_length.strip()
    if '-' in design_length:
        min_len, max_len = design_length.split('-')
        return f"{int(min_len)}..{int(max_len)}"
    return str(int(design_length))


def parse_residue_ranges(spec_str, rank_map):
    """
    Parse a hotspot/anti-hotspot/inpaint-seq/flexible-residue style string, e.g.
    'A56,A115-120,B10', into {chain: 'res_index string'}. This is the shared token
    grammar for all of ProteinDJ's bg_* residue-spec params (bg_not_binding_residues,
    bg_redesign_inpaint_seq, bg_flexible_residues) as well as hotspot_residues. PDB
    author residue numbers are translated to BoltzGen's 1-based positional rank via
    rank_map (see get_residue_rank_map), and emitted as an explicit comma-separated
    list of positions (not a '..' range) so gaps in PDB numbering don't get
    misinterpreted as contiguous positional ranges.
    Chain-only tokens (e.g. 'A') mean the whole chain (BoltzGen's 'all' keyword).
    """
    per_chain = {}
    whole_chains = []
    for token in (t.strip() for t in spec_str.split(',') if t.strip()):
        match = re.match(r'^([A-Za-z]+)(\d+)(?:-(\d+))?$', token)
        chain_only = re.match(r'^([A-Za-z]+)$', token)
        if match:
            chain, start, end = match.groups()
            start = int(start)
            end = int(end) if end else start
            ranks = [_resolve_rank(chain, resnum, rank_map) for resnum in range(start, end + 1)]
            per_chain.setdefault(chain, []).extend(str(rank) for rank in ranks)
        elif chain_only:
            whole_chains.append(chain_only.group(1))
        else:
            raise ValueError(f"Could not parse residue token: '{token}' (expected e.g. 'A56', 'A115-120', or 'A')")

    ranges = {chain: ','.join(vals) for chain, vals in per_chain.items()}
    for chain in whole_chains:
        ranges[chain] = 'all'
    return ranges


def parse_flexible_spec(spec_str, rank_map):
    """
    Parse a bg_flexible_residues style string, e.g. 'A10-13,A16,B', into
    {chain: 'res_index string' or None}. Uses the same token grammar as
    parse_residue_ranges (see above) - a value of None means the entire chain is
    flexible (res_index omitted), mapped from a bare chain-ID token.
    """
    parsed = parse_residue_ranges(spec_str, rank_map)
    return {chain: (None if res_index == 'all' else res_index) for chain, res_index in parsed.items()}



def _res_index_to_ranks(res_index, chain_length):
    """
    Expand a BoltzGen res_index value - as produced by parse_residue_ranges/
    parse_flexible_spec, i.e. 'all', None, or a comma-separated list of 1-based
    positional ranks - into a Python set of ints for containment checks.
    """
    if res_index is None or res_index == 'all':
        return set(range(1, chain_length + 1))
    return {int(r) for r in res_index.split(',')}


def parse_architecture_spec(spec_str, rank_map, chain_a_length):
    """
    Parse a bg_redesign_spec string, e.g. '7-10,A1-60,5,A70-100,10', describing the
    new chain A architecture as an ordered sequence of tokens:
      - Keep token 'A<n>' or 'A<start>-<end>': PDB-author-numbered (rank_map-translated)
        contiguous run of ORIGINAL chain A residues to retain as-is (sequence +
        structure) at this position in the new architecture. Must be strictly
        ascending / non-overlapping relative to the previous keep token.
      - Insert token '<n>' or '<min>-<max>': bare digit(s), no chain letter. Adds that
        many brand-new (BoltzGen design_insertions, fully designed) residues at this
        position.
    Any original chain A residue not covered by a keep token (including a leading gap
    before the first keep token, a gap between keep tokens, or a trailing gap after the
    last keep token) is implicitly deleted (added to BoltzGen `exclude`).

    Returns (exclude_ranks, insertions, kept_ranks):
      - exclude_ranks: list of (start_rank, end_rank) 1-based inclusive tuples (original
        chain A numbering) to drop.
      - insertions: list of (anchor_rank, num_residues_spec) tuples for BoltzGen
        `design_insertions` (anchor_rank is the 1-based original rank to insert before;
        num_residues_spec is already converted to BoltzGen's '<n>'/'<min>..<max>' syntax).
      - kept_ranks: sorted list of 1-based original ranks retained.
    """
    keep_re = re.compile(r'^A(\d+)(?:-(\d+))?$')
    insert_re = re.compile(r'^(\d+)(?:-(\d+))?$')

    exclude_ranks = []
    insertions = []
    kept_ranks = []
    last_kept_end_rank = 0

    for token in (t.strip() for t in spec_str.split(',') if t.strip()):
        keep_match = keep_re.match(token)
        insert_match = None if keep_match else insert_re.match(token)
        if keep_match:
            start, end = keep_match.groups()
            start, end = int(start), int(end) if end else int(start)
            if end < start:
                raise ValueError(f"Invalid bg_redesign_spec keep token '{token}': end before start")
            start_rank = _resolve_rank('A', start, rank_map)
            end_rank = _resolve_rank('A', end, rank_map)
            if start_rank <= last_kept_end_rank:
                raise ValueError(
                    f"bg_redesign_spec keep token '{token}' is out of order or overlaps a "
                    "previous keep token - keep tokens must be strictly ascending"
                )
            if start_rank > last_kept_end_rank + 1:
                exclude_ranks.append((last_kept_end_rank + 1, start_rank - 1))
            kept_ranks.extend(range(start_rank, end_rank + 1))
            last_kept_end_rank = end_rank
        elif insert_match:
            count_start, count_end = insert_match.groups()
            if count_end and int(count_end) < int(count_start):
                raise ValueError(f"Invalid bg_redesign_spec insert token '{token}': end before start")
            num_residues_spec = count_start if not count_end else f"{count_start}..{count_end}"
            insertions.append((last_kept_end_rank + 1, num_residues_spec))
        else:
            raise ValueError(
                f"Could not parse bg_redesign_spec token: '{token}' "
                "(expected e.g. 'A1-60', 'A56', '10', or '7-10')"
            )

    if last_kept_end_rank < chain_a_length:
        exclude_ranks.append((last_kept_end_rank + 1, chain_a_length))

    return exclude_ranks, insertions, kept_ranks


def build_structure_groups(flexible_by_chain):
    """Build a BoltzGen `structure_groups` list from a {chain: res_index-or-None} dict."""
    groups = []
    for chain, res_index in flexible_by_chain.items():
        # Explicit visibility:1 group first (BoltzGen's implicit default), makes the
        # generated spec self-documenting and robust to any future default changes.
        groups.append({'group': {'visibility': 1, 'id': chain}})
        flexible_group = {'visibility': 0, 'id': chain}
        if res_index is not None:
            flexible_group['res_index'] = res_index
        groups.append({'group': flexible_group})
    return groups


def build_binding_types(binding_by_chain, not_binding_by_chain):
    """Build a BoltzGen `binding_types` list from binding/not_binding {chain: res_index} dicts."""
    chains = sorted(set(binding_by_chain) | set(not_binding_by_chain))
    binding_types = []
    for chain in chains:
        entry = {'id': chain}
        if chain in binding_by_chain:
            entry['binding'] = binding_by_chain[chain]
        if chain in not_binding_by_chain:
            entry['not_binding'] = not_binding_by_chain[chain]
        binding_types.append({'chain': entry})
    return binding_types


def apply_binding_types(file_entity, hotspot_residues, bg_not_binding_residues, context_chains, rank_map):
    """
    Parse hotspot_residues/bg_not_binding_residues and attach a `binding_types` block to
    file_entity if either is set. context_chains are the chains binding_types may reference:
    the fixed target in boltzgen_denovo, or the fixed non-A chain(s) in boltzgen_redesign.
    """
    binding_by_chain = parse_residue_ranges(hotspot_residues, rank_map) if hotspot_residues else {}
    not_binding_by_chain = (
        parse_residue_ranges(bg_not_binding_residues, rank_map) if bg_not_binding_residues else {}
    )
    if not (binding_by_chain or not_binding_by_chain):
        return
    invalid_chains = (set(binding_by_chain) | set(not_binding_by_chain)) - set(context_chains)
    if invalid_chains:
        raise ValueError(
            f"hotspot_residues/bg_not_binding_residues reference chain(s) not in target: {sorted(invalid_chains)}"
        )
    file_entity['binding_types'] = build_binding_types(binding_by_chain, not_binding_by_chain)


def apply_structure_groups(file_entity, bg_flexible_residues, context_chains, rank_map):
    """Parse bg_flexible_residues and attach/extend the `structure_groups` block on file_entity if set."""
    if not bg_flexible_residues:
        return
    flexible_by_chain = parse_flexible_spec(bg_flexible_residues, rank_map)
    invalid_chains = set(flexible_by_chain) - set(context_chains)
    if invalid_chains:
        raise ValueError(f"bg_flexible_residues references chain(s) not in target: {sorted(invalid_chains)}")
    file_entity.setdefault('structure_groups', []).extend(build_structure_groups(flexible_by_chain))


def build_denovo_spec(args, target_chains, rank_map):
    design_entity = {
        'protein': {
            'id': 'A',
            'sequence': parse_design_length(args.design_length),
        }
    }

    target_file_entity = {
        'path': Path(args.input_pdb).name,
        'include': [{'chain': {'id': chain}} for chain in target_chains],
    }

    apply_binding_types(
        target_file_entity, args.hotspot_residues, args.bg_not_binding_residues, target_chains, rank_map
    )
    apply_structure_groups(target_file_entity, args.bg_flexible_residues, target_chains, rank_map)

    return {'entities': [design_entity, {'file': target_file_entity}]}


def build_redesign_spec(args, all_chains, rank_map):
    if 'A' not in all_chains:
        raise ValueError("boltzgen_redesign requires input_pdb to contain a chain A (binder)")

    if not (args.bg_redesign_spec or args.bg_redesign_inpaint_seq or args.bg_flexible_residues):
        raise ValueError(
            "boltzgen_redesign mode requires at least one of bg_redesign_spec, "
            "bg_redesign_inpaint_seq, or bg_flexible_residues to be set - otherwise chain A "
            "would be left completely unchanged (wasted computation)."
        )

    other_chains = [c for c in all_chains if c != 'A']

    chain_a_length = get_chain_length(args.input_pdb, 'A')
    if chain_a_length == 0:
        raise ValueError("Chain A in input_pdb has no standard residues")

    if args.bg_redesign_spec:
        exclude_ranks, insertions, kept_ranks = parse_architecture_spec(
            args.bg_redesign_spec, rank_map, chain_a_length
        )
    else:
        exclude_ranks, insertions, kept_ranks = [], [], list(range(1, chain_a_length + 1))
    kept_ranks_set = set(kept_ranks)

    file_entity = {
        'path': Path(args.input_pdb).name,
        'include': [{'chain': {'id': chain}} for chain in all_chains],
    }

    if exclude_ranks:
        file_entity['exclude'] = [
            {'chain': {'id': 'A', 'res_index': str(start) if start == end else f"{start}..{end}"}}
            for start, end in exclude_ranks
        ]

    if insertions:
        file_entity['design_insertions'] = [
            {'insertion': {'id': 'A', 'res_index': anchor, 'num_residues': num_residues_spec}}
            for anchor, num_residues_spec in insertions
        ]

    if exclude_ranks or insertions:
        # Chain A's residue numbering shifts once residues are excluded/inserted; reset it to
        # a clean 1..N sequence so downstream tooling sees contiguous chain A numbering in
        # BoltzGen's output structure.
        file_entity['reset_res_index'] = [{'chain': {'id': 'A'}}]

    if args.bg_redesign_inpaint_seq:
        inpaint_by_chain = parse_residue_ranges(args.bg_redesign_inpaint_seq, rank_map)
        invalid_chains = set(inpaint_by_chain) - {'A'}
        if invalid_chains:
            raise ValueError(
                f"bg_redesign_inpaint_seq must only reference chain A, got: {sorted(invalid_chains)}"
            )
        inpaint_res_index = inpaint_by_chain.get('A', 'all')
        invalid_ranks = _res_index_to_ranks(inpaint_res_index, chain_a_length) - kept_ranks_set
        if invalid_ranks:
            raise ValueError(
                "bg_redesign_inpaint_seq references chain A residue(s) removed by "
                f"bg_redesign_spec: {sorted(invalid_ranks)}"
            )
        # design:True keeps the residue's original identity/coordinates as a template (BoltzGen's
        # design step only diffuses structure) but marks it for downstream sequence redesign via
        # MPNN/FAMPNN (--skip_inverse_folding); structure_groups visibility stays at the default
        # (1, conditioned) so the backbone itself is not freed up here - only bg_flexible_residues
        # does that.
        file_entity.setdefault('design', []).append({'chain': {'id': 'A', 'res_index': inpaint_res_index}})

    if args.bg_flexible_residues:
        flexible_by_chain = parse_flexible_spec(args.bg_flexible_residues, rank_map)
        if 'A' in flexible_by_chain:
            invalid_ranks = _res_index_to_ranks(flexible_by_chain['A'], chain_a_length) - kept_ranks_set
            if invalid_ranks:
                raise ValueError(
                    "bg_flexible_residues references chain A residue(s) removed by "
                    f"bg_redesign_spec: {sorted(invalid_ranks)}"
                )

    # hotspot_residues/bg_not_binding_residues apply to the fixed non-A context chain(s), same
    # as the target in boltzgen_denovo. bg_flexible_residues may target chain A's kept residues
    # in addition to the fixed non-A context chain(s).
    apply_binding_types(file_entity, args.hotspot_residues, args.bg_not_binding_residues, other_chains, rank_map)
    apply_structure_groups(file_entity, args.bg_flexible_residues, all_chains, rank_map)

    return {'entities': [{'file': file_entity}]}


def main():
    parser = argparse.ArgumentParser(description='Generate a BoltzGen design-spec YAML')
    parser.add_argument('--input_pdb', required=True, help='Path to input PDB file')
    parser.add_argument('--design_mode', required=True, choices=['boltzgen_denovo', 'boltzgen_redesign'])
    parser.add_argument('--design_length', default='', help="Binder length, e.g. '80' or '60-100' (denovo only)")
    parser.add_argument('--hotspot_residues', default='', help="Target hotspot residues, e.g. 'A56,A115-120'")
    parser.add_argument('--bg_not_binding_residues', default='', help="Target anti-hotspot residues")
    parser.add_argument(
        '--bg_redesign_spec', default='',
        help=(
            "Chain A architecture spec (redesign only), e.g. '7-10,A1-60,5,A70-100,10': "
            "ordered comma-separated keep ('A<start>-<end>') and insert ('<n>' or "
            "'<min>-<max>') tokens. Residues not covered by a keep token are deleted."
        ),
    )
    parser.add_argument(
        '--bg_redesign_inpaint_seq', default='',
        help="Chain A residues (within kept residues) whose sequence is allowed to change (redesign only)",
    )
    parser.add_argument('--bg_flexible_residues', default='', help='Residues with unconditioned/free structure')
    parser.add_argument('--output', required=True, help='Output YAML file path')
    args = parser.parse_args()

    rank_map = get_residue_rank_map(args.input_pdb)

    if args.design_mode == 'boltzgen_denovo':
        target_chains = get_protein_chains(args.input_pdb)
        if not target_chains:
            raise ValueError(
                "No protein chains found in input_pdb. Please crop input_pdb to only the desired target chain(s)."
            )
        spec = build_denovo_spec(args, target_chains, rank_map)
        print(f"Designing a new binder of length {args.design_length} against target chain(s): {target_chains}")
    else:
        all_chains = get_protein_chains(args.input_pdb)
        spec = build_redesign_spec(args, all_chains, rank_map)
        print(f"Redesigning residues in chain A against fixed chain(s): {[c for c in all_chains if c != 'A']}")

    # Write a cleaned copy of the input PDB (stripped of stale sequence/molecule header
    # records) and point the spec at it instead of the raw input_pdb.
    cleaned_pdb_path = Path(args.output).parent / f"{Path(args.input_pdb).stem}_boltzgen.pdb"
    clean_pdb_for_boltzgen(args.input_pdb, cleaned_pdb_path)
    for entity in spec['entities']:
        if 'file' in entity:
            entity['file']['path'] = cleaned_pdb_path.name

    with open(args.output, 'w') as f:
        yaml.dump(spec, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote BoltzGen design spec: {args.output}")


if __name__ == '__main__':
    main()
