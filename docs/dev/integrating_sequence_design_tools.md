# Integrating a New Sequence Design Tool into ProteinDJ

This document reverse-engineers how **ProteinMPNN-OpenMMRelax (MPNN)** and **Full-Atom MPNN
(FAMPNN)** were integrated into ProteinDJ as interchangeable engines for the "Sequence Design"
stage. It is intended as a template/checklist for integrating other sequence design tools (e.g.
LigandMPNN, ESM-IF, another inverse-folding model, etc.).

Having *two* existing implementations to compare is useful here (unlike the single BindCraft
case study in [integrating_fold_design_tools.md](integrating_fold_design_tools.md)) because MPNN
and FAMPNN diverge in a couple of important ways — MPNN is backbone-only and needs no extra prep,
while FAMPNN is full-atom and needs a side-chain-restoration + fixed-position-CSV adapter step.
Seeing both makes it clear which parts of the contract are truly fixed and which parts a new tool
has freedom to implement differently.

---

## 1. Pipeline architecture recap

See [integrating_fold_design_tools.md](integrating_fold_design_tools.md) §1 for the full 4-stage
pipeline diagram. The Sequence Design stage sits between Fold Design and Structure Prediction:

```mermaid
graph LR
    A[Fold Design] --> B[Sequence Design]
    B --> C[Structure Prediction]
    C --> D[Analysis]
```

The key architectural difference from Fold Design is **how the engine is selected**. Fold-design
tools are selected via `params.design_mode` (e.g. `bindcraft_denovo` vs `rfd_denovo`), because
each tool has its own mode-specific required parameters (contigs, hotspots, target PDBs, etc.).
Sequence design tools, by contrast, are selected via a single dedicated switch,
`params.seq_method` (`'mpnn'` or `'fampnn'`), which is **orthogonal to `design_mode`** — any
`design_mode` can be paired with either `seq_method`. This is possible because, by the time
Sequence Design runs, Fold Design has already normalized everything into the tool-agnostic
`fold_N.pdb` / `fold_N.json` convention (see the Fold Design contract), so a sequence design tool
never needs to know which upstream fold-design engine produced its input — only whether the
overall run is a binder or monomer design (see §3.3 below).

The `if/else` dispatch lives in [main.nf](../main.nf) under the `SEQUENCE DESIGN STAGE` comment
block, and both branches converge on the same `seq_tuple` variable/shape, consumed by the shared
`FilterSeq` process — directly analogous to how Fold Design's branches converge on `fold_tuples`
before the shared `FilterFold` process.

---

## 2. The "Sequence Design" stage contract

Any tool plugged in as a sequence-design engine consumes the Fold Design stage's output
(`filt_fold_pdbs_jsons`: a `tuple(path(pdb_files), path(json_files))` of `fold_N.pdb`/`fold_N.json`
pairs) and must ultimately deliver, into the variable `seq_tuple` (consumed by the shared
`FilterSeq` process), a Nextflow tuple of the same shape:

```
tuple( path(pdb_files), path(json_files) )
```

with these hard requirements:

| Requirement | Details |
|---|---|
| **Naming convention: `fold_<fold_id>_seq_<seq_id>`** | Every output PDB/JSON pair must be named `fold_N_seq_M.pdb` / `fold_N_seq_M.json`, where `N` is the (already-assigned, globally unique) `fold_id` from the input and `M` is a sequence index from `0` to `params.seqs_per_design - 1`. Downstream code (`filter_seq.py`, `metadata_converter.py`'s `MPNNMetadataConverter`/`FAMPNNMetadataConverter`) parses these IDs back out with the regex `fold_(\d+)_seq_(\d+)`. |
| **`seqs_per_design` designs per fold** | `params.seqs_per_design` is a tool-agnostic, already-existing param (not `mpnn_*`/`fampnn_*` prefixed) — a new tool should honor it as "number of sequences to sample per input backbone" (MPNN: `-seqs_per_struct`, FAMPNN: `num_seqs_per_pdb=`). |
| **A `sequence` field in the output JSON** | The per-design JSON metadata must include the designed sequence under the key `sequence`. Plain single-chain sequences are a bare string; if a tool designs multiple chains, use FAMPNN's convention of pipe/colon-delimited `'A:SEQ_A|B:SEQ_B'` so that the shared `extract_designed_sequence()` helper in [scripts/filter_seq.py](../scripts/filter_seq.py) (which always takes the *first* chain, i.e. the binder/chain A) continues to work unmodified. |
| **A tool-specific score field, registered for filtering** | The shared `FilterSeq` process ([modules/proteinmpnn.nf](../modules/proteinmpnn.nf)) filters by a single per-design score threshold looked up via the `SCORE_FIELDS` dict in [scripts/filter_seq.py](../scripts/filter_seq.py) (`{'mpnn': 'score', 'fampnn': 'fampnn_avg_psce'}`) and a matching `<tool>_max_score`-style param (`mpnn_max_score`, `fampnn_max_psce`). A new tool needs one new entry in `SCORE_FIELDS` plus a new `params.seq_method` branch value; there is no requirement that the score have any particular meaning/direction beyond "lower is better, threshold via `<=`" — pick whatever the tool's native confidence/energy metric is. |
| **Respect fixed/inpainted positions from Fold Design** | Fold-design tools mark residues that must **not** have their sequence changed via a boolean array under one of `rfd_inpaint_seq` / `bc_inpaint_seq` / `bg_inpaint_seq` in the input JSON (only one key present, depending on which fold-design tool produced the input — see [integrating_fold_design_tools.md](integrating_fold_design_tools.md) §2). A sequence design tool must honor this mask (or its own equivalent, e.g. a B-factor-based fallback) so that motif-scaffolding/partial-diffusion/interface-fixing use cases work regardless of `seq_method`. See §3.1/§3.2 for how MPNN and FAMPNN each adapt this into their own native "fixed positions" mechanism. |
| **Binder vs monomer chain awareness** | Unlike Fold Design tools (which are fully oblivious once `fold_tuples` exists), a Sequence Design tool typically *does* need to know whether the run is a binder design (only chain A, the binder, should be redesigned/scored; target chain(s) must be fixed and excluded from scoring) or a monomer design (all chains are fair game). MPNN gets this implicitly via the fixed-residue mask; FAMPNN takes it as an explicit `analysis_chain_id` argument (`'A'` for binder modes, `'all_chains'` for monomer modes) computed in `main.nf` from `params.design_mode` membership in the binder-mode list. |
| **JSONL metadata emitted on the `metadata_ch_fold_seq` topic** | Any process that generates fold+seq metadata publishes a `path(..._metadata_*.jsonl), topic: metadata_ch_fold_seq` output (see `RunMPNN`'s `mpnn_metadata_${batch_id}.jsonl` and `RunFAMPNN`'s `fampnn_metadata_${batch_id}.jsonl`). This is a **different** topic from Fold Design's `metadata_ch_fold` — sequence-design metadata is per-`(fold_id, seq_id)` pair rather than per-`fold_id`, and `metadata_converter.py::merge_all()` joins the two topics on `fold_id` (broadcasting fold-only fields across every seq_id sharing that fold). |
| **Metadata field naming convention** | Fields are prefixed by tool: `mpnn_*` for ProteinMPNN, `fampnn_*` for Full-Atom MPNN. A new tool should use its own short prefix (e.g. `lmpnn_*` for LigandMPNN) to avoid clashes; see [scripts/metadata_converter.py](../scripts/metadata_converter.py)'s `metadata_field_names` list in `merge_all()`, which must be updated to include any new prefixed fields (and the `*_time` timing field, following `mpnn_time`) so they survive the CSV merge and column ordering step. |
| **GPU batching** | The sequence design channel must be built via `Utils.rebatchGPU(pdbs_channel, params.gpus)` so batches can run in parallel across `params.gpus` GPUs (see §4 below), then re-flattened with `Utils.rebatchTuples(...)` for the downstream CPU-based `FilterSeq` step. |
| **`FilterSeq` compatibility** | `FilterSeq` ([modules/proteinmpnn.nf](../modules/proteinmpnn.nf)) is shared/generic — it filters by the tool's own score field plus tool-agnostic sequence property filters (extinction coefficient, pI, computed directly from the `sequence` field via BioPython). It doesn't care which tool produced the designs, only that the naming/score-field/sequence-field conventions above are followed. |

---

## 3. Case study: how MPNN and FAMPNN satisfy the contract

### 3.1 Container & process labels

- MPNN reuses the existing `dl_binder_design` container (it's bundled alongside AF2-Initial-Guess
  in the same upstream repo) and binds the externally-downloaded weights directory:
  ```groovy
  withLabel: MPNN {
      container = "${params.container_registry}/dl_binder_design:${params.container_version}"
      containerOptions = """--nv \
          --bind ${params.mpnn_models}:/ProteinMPNN/ \
          --bind ${projectDir}/scripts:/scripts \
          --bind ${projectDir} \
          """
  }
  ```
- FAMPNN has its own dedicated container/def file
  ([apptainer/fampnn.def](../apptainer/fampnn.def)) with model weights **baked into the container
  image** at a fixed path (`/app/fampnn/weights/fampnn_0_3.pt`) rather than bind-mounted — so there
  is no `fampnn_models` param at all:
  ```groovy
  withLabel: FAMPNN {
      container = "${params.container_registry}/fampnn:${params.container_version}"
      containerOptions = """--nv \
          --bind ${projectDir}/scripts:/scripts \
          --bind ${projectDir} \
          """
  }
  ```
  Both approaches are valid — bind-mount large/updatable weights (MPNN, RFdiffusion, AF2, Boltz,
  BindCraft, BoltzGen), or bake in small/stable weights (FAMPNN). Decide per-tool based on weight
  file size and how often they're expected to change.
- Both processes carry `label 'gpu'` (for scheduler resource requests, see `withLabel: gpu` in
  [nextflow.config](../nextflow.config)) in addition to their tool-specific label.

### 3.2 Module structure

**MPNN** ([modules/proteinmpnn.nf](../modules/proteinmpnn.nf)) — two processes, since MPNN needs
no side-chain restoration:

1. **`PrepMPNN`** (`python_tools`, CPU) — runs
   [scripts/prep_mpnn_designs.py](../scripts/prep_mpnn_designs.py), which determines fixed
   residues per PDB from the `rfd_inpaint_seq`/`bc_inpaint_seq`/`bg_inpaint_seq` JSON mask (falling
   back to a B-factor-based heuristic if no JSON is available) and writes MPNN's own native
   "fixed positions" encoding (B-factor tagging, consumed internally by the `dl_binder_design`
   MPNN fork) into the PDB files themselves — i.e. the *fixed-position adapter* lives entirely
   inside the PDB file, no side-channel CSV needed.
2. **`RunMPNN`** (`MPNN`, `gpu`) — runs `dl_interface_design_multi.py` for one GPU-batch of PDBs,
   producing `results/*.pdb` + `results/*.json` already named `fold_N_seq_M.*` (MPNN's own script
   handles this numbering directly, since it processes one fold's PDB at a time and knows
   `seqs_per_design`), then runs `metadata_converter.py --converter mpnn` to emit
   `mpnn_metadata_${batch_id}.jsonl` on the `metadata_ch_fold_seq` topic. Also declares `maxRetries
   3` — a pragmatic accommodation for a known memory-leak flakiness in the underlying tool, not
   part of the general contract, but worth copying if a new tool has similar instability.

**FAMPNN** ([modules/fampnn.nf](../modules/fampnn.nf)) — three processes, since FAMPNN is a
full-atom model and needs real side-chains plus an explicit fixed-positions file:

1. **`PrepFAMPNN`** (`python_tools`, CPU) — two adapter steps in one process:
   - [scripts/prep_fampnn_designs.py](../scripts/prep_fampnn_designs.py) restores missing
     side-chain atoms that fold-design tools (which only emit backbone coordinates) don't provide,
     since FAMPNN operates on full-atom structures.
   - [scripts/prep_fampnn_csv.py](../scripts/prep_fampnn_csv.py) converts the same
     `rfd_inpaint_seq`/`bc_inpaint_seq`/`bg_inpaint_seq` boolean mask into FAMPNN's native
     "fixed positions" CSV format (chain-qualified residue ranges, e.g. `A10-50,A60`) — the same
     underlying information as MPNN's B-factor encoding, just expressed as a side-channel CSV file
     instead of being embedded in the PDB, because that's the interface FAMPNN's own inference
     script expects (`fixed_pos_csv=`).
2. **`RunFAMPNN`** (`FAMPNN`, `gpu`) — takes an extra `analysis_chain_id` input value (`'A'` or
   `'all_chains'`, decided in `main.nf` — see §3.3) in addition to `tuple(batch_id, pdbs, csv)`.
   Runs FAMPNN's inference script, renames its native `fold_X_sampleY.pdb` output to the
   pipeline's `fold_X_seq_Y.pdb` convention with a `sed`, then calls
   [scripts/analyse_fampnn.py](../scripts/analyse_fampnn.py) to compute `fampnn_avg_psce`
   (average per-residue side-chain confidence error, averaged over the specified chain(s)) from
   per-atom B-factors that FAMPNN writes into its output PDBs — this is FAMPNN's equivalent of
   MPNN's native `score` field, and becomes the value registered in `SCORE_FIELDS['fampnn']`.
   Finally runs `metadata_converter.py --converter fampnn` to emit the topic JSONL.
3. There's no separate "Analyse" process distinct from step 2's post-processing — `RunFAMPNN`
   does prep+run+analyse-equivalent work inline via the shell block, since (unlike BindCraft's
   fold-only output) FAMPNN's own PDB output already needs no chain reordering/renumbering.

Because FAMPNN needs a *single, merged* fixed-positions CSV covering every PDB across every prep
batch (not one CSV per PDB), `PrepFAMPNN.out.csv` is merged with `.collectFile(name:
'merged_results.csv', keepHeader: true)` before being `.combine()`d onto the GPU-batched PDB
channel — a batching wrinkle specific to FAMPNN's CSV-based interface, worth being aware of if a
new tool also uses a single side-channel config file rather than a per-PDB file.

### 3.3 `main.nf` integration point

Inside the `if (!params.skip_fold_seq & !params.skip_fold_seq_pred & !params.run_fold_only)`
block, `seq_method` is checked once:

```groovy
if (params.seq_method == "mpnn") {
    PrepMPNN(filt_fold_pdbs_jsons)
    Utils.rebatchGPU(PrepMPNN.out.pdbs, params.gpus).set { seq_input_pdbs }
    RunMPNN(seq_input_pdbs)
    CompressMPNN("mpnn", RunMPNN.out.pdbs_jsons.flatten().collect())
    Utils.rebatchTuples(RunMPNN.out.pdbs_jsons, 200).set { seq_tuple }
}
else if (params.seq_method == "fampnn") {
    Utils.rebatchTuples(filt_fold_pdbs_jsons, 10).set { fampnn_prep_input_tuple }
    PrepFAMPNN(fampnn_prep_input_tuple)
    PrepFAMPNN.out.csv.collectFile(name: 'merged_results.csv', keepHeader: true).set { mega_csv }
    Utils.rebatchGPU(PrepFAMPNN.out.pdbs, params.gpus).set { fampnn_pdbs }
    fampnn_pdbs.combine(mega_csv).set { fampnn_input }

    if (is_binder_mode) {
        RunFAMPNN(fampnn_input, 'A')          // binder mode: score/design chain A only
    } else {
        RunFAMPNN(fampnn_input, 'all_chains') // monomer mode: score/design all chains
    }

    CompressFAMPNN("fampnn", RunFAMPNN.out.pdbs_jsons.flatten().collect())
    Utils.rebatchTuples(RunFAMPNN.out.pdbs_jsons, 200).set { seq_tuple }
}
else {
    error("Not a valid sequence assignment method")
}

// FilterSeq(seq_tuple) runs identically regardless of branch taken above
```

Key point for future tools: **both branches converge on the same `seq_tuple` variable name and
shape**, so `FilterSeq` and everything downstream (Structure Prediction, Analysis) is completely
tool-agnostic. A new tool is a third `else if (params.seq_method == "x")` branch that ends by
setting `seq_tuple`. The binder-vs-monomer `design_mode in [...]` check is currently duplicated
logic that exists in a couple of places in `main.nf` (fold design also branches on similar mode
lists) — if a new sequence design tool needs this distinction, reuse the existing list rather than
inventing a new one, to avoid the two lists drifting apart.

### 3.4 Parameters, schemas & docs

A new tool's params must be threaded through the same places as a new fold-design tool, but note
one difference: **sequence design params are not split per-mode-schema-file** the way fold design
params are, because `seq_method` is orthogonal to `design_mode` — `mpnn_*`/`fampnn_*` params
appear identically in *every* mode's schema/CSV column (see
[schemas/mode_parameters.csv](../schemas/mode_parameters.csv) rows 40-52, where `mpnn_*`/`fampnn_*`
appear in every mode column) rather than being mode-specific.

1. **[nextflow.config](../nextflow.config)** — declare params with defaults + doc comments under a
   `//// <TOOL> ADVANCED PARAMETERS ////` section (see `mpnn_omitAAs`, `mpnn_temperature`, ...,
   `fampnn_psce_threshold`, `fampnn_exclude_cys`, ...). Also add the score-threshold param used by
   `FilterSeq` (`mpnn_max_score`, `fampnn_max_psce` — declared in the Filtering Parameters section,
   see [docs/parameters.md](parameters.md) `## Sequence Filtering Parameters`).
2. **[schemas/mode_parameters.csv](../schemas/mode_parameters.csv)** — add new rows (not columns —
   sequence-design tool params apply identically across all mode columns) for every new `x_*` param.
3. **Every `schemas/nextflow_schema_<mode>.json` file** — since `seq_method` applies to all modes,
   a new tool's params need to be added, under their own `<tool>_advanced_parameters` group, to
   *every* per-mode schema file (contrast with a new fold-design tool, which only needs its own new
   `schemas/nextflow_schema_<mode>.json` file). Don't hand-edit these — run
   `./scripts/regenerate_schemas.sh` from the repo root after updating `nextflow_schema.json` and
   `mode_parameters.csv`; it regenerates every per-mode schema plus bindsweeper's
   `binder_schema.json` in one step.
4. **[docs/parameters.md](parameters.md)** — add a `## <Tool> Advanced Parameters` table (see the
   existing `## ProteinMPNN-OpenMMRelax Advanced Parameters` / `## Full-Atom MPNN (FAMPNN) Advanced
   Parameters` sections) and a row for the new score-threshold param under `### Sequence Filtering
   Parameters`. Unlike fold design tools, a new sequence design tool generally does **not** need a
   new section in [docs/modes.md](modes.md), since it isn't a `design_mode`.

Also update `main.nf`'s `seq_method` validation (`error("Not a valid sequence assignment
method")`) to accept the new value, and — if the tool needs any tool-native settings files (like
BindCraft's advanced-settings JSONs) — add a `getXSettingsPath()`-style helper and wire required
input files into `collectInputFiles()`.

### 3.5 Metadata plumbing

- [scripts/metadata_converter.py](../scripts/metadata_converter.py) needs:
  - a new `<Tool>MetadataConverter(MetadataConverter)` subclass implementing `_parse_metadata()`.
    `MPNNMetadataConverter`/`FAMPNNMetadataConverter` are simple templates: read one JSON file,
    regex-parse `fold_id`/`seq_id` out of the `design` field, `yield` a dict with `fold_id`,
    `seq_id`, `sequence`, and the tool's score field(s). Contrast with `BCMetadataConverter`, which
    just passes through pre-formatted JSON — sequence design converters generally need the regex
    parsing step since MPNN/FAMPNN emit fairly raw per-design JSON rather than a fully-prefixed
    metadata dict.
  - the new converter registered in the `converters = {...}` dict and `choices=[...]` list in
    `main()`.
  - the new tool's field names (and any `<tool>_time` timing field) added to the master
    `metadata_field_names` ordering list in `merge_all()`, otherwise they'll be silently dropped
    from the final CSV.
- [scripts/filter_seq.py](../scripts/filter_seq.py) needs:
  - a new entry in `SCORE_FIELDS` mapping `params.seq_method` value → the JSON field name to filter
    on.
  - if the tool's designed sequence needs special extraction (e.g. multi-chain output), extend
    `extract_designed_sequence()` — but prefer reusing the existing `'A:SEQ|B:SEQ'` convention
    FAMPNN already established rather than inventing a third format.

---

## 4. Generalized checklist for adding a new sequence design tool

Use MPNN (simple, backbone-only) and FAMPNN (side-chain-aware, needs an extra prep step) as the
two templates. Concretely, to add tool `X` (prefix `x_`) as a new `seq_method` value (e.g. `'x'`):

1. **Container**
   - [ ] Add `apptainer/x.def` (or Dockerfile) building the tool + a way to invoke it headlessly/CLI.
   - [ ] Add a `withLabel: X { container = ...; containerOptions = ... }` block to
     [nextflow.config](../nextflow.config)'s `process {}` scope. Decide whether model weights are
     bind-mounted (add an `x_models` param, like `mpnn_models`) or baked into the container image
     (like FAMPNN's weights) based on size/update frequency.
   - [ ] Add build/download entries to `apptainer/build_containers.sh` / `download_containers.sh`.

2. **Nextflow module** (`modules/x.nf`)
   - [ ] `PrepX` (CPU, `python_tools` label): translate the input `fold_N.pdb`/`fold_N.json` pairs
     into whatever the tool natively needs — at minimum, convert the
     `rfd_inpaint_seq`/`bc_inpaint_seq`/`bg_inpaint_seq` boolean mask into the tool's native
     fixed-position mechanism (B-factor tagging like MPNN, a side-channel CSV like FAMPNN, or
     whatever the tool supports). If the tool needs full side-chains and fold-design tools only
     provide backbone atoms, add a side-chain restoration step here too (see
     `prep_fampnn_designs.py`).
   - [ ] `RunX` (GPU, `X` label): invoke the tool per-GPU-batch; ensure output PDB/JSON pairs are
     renamed to the `fold_N_seq_M.pdb`/`fold_N_seq_M.json` convention (`N` = input `fold_id`, `M` =
     0-indexed sequence number, up to `params.seqs_per_design`); emit metadata with `sequence` and
     the tool's own score field via `metadata_converter.py --converter x` on the
     `metadata_ch_fold_seq` topic.
   - [ ] If the tool needs binder/monomer chain-awareness (which chain(s) to design/score), take it
     as an explicit input value computed in `main.nf` from `params.design_mode` list membership
     (reuse the existing binder-mode list rather than duplicating it), following FAMPNN's
     `analysis_chain_id` pattern.

3. **`main.nf` wiring**
   - [ ] Add `include { PrepX; RunX } from './modules/x.nf'` and
     `include { Compress as CompressX } from './modules/compress'`.
   - [ ] Add a new `else if (params.seq_method == "x")` branch in the Sequence Design stage
     `if/else`, following the MPNN/FAMPNN shape: GPU-batch via `Utils.rebatchGPU`, run
     `PrepX -> RunX`, compress with `CompressX`, re-batch with `Utils.rebatchTuples(..., 200)`
     into `seq_tuple`. Update the trailing `error("Not a valid sequence assignment method")` to
     include the new value in its accepted set (implicitly, by adding this branch before it).
   - [ ] `FilterSeq(seq_tuple)` requires zero changes as long as `seq_tuple`'s shape matches.

4. **Params & schema**
   - [ ] Add `x_*` params (with default values and doc comments) to [nextflow.config](../nextflow.config)
     under a `//// X ADVANCED PARAMETERS ////` section, plus an `x_max_score`-style threshold param
     in the Filtering Parameters section.
   - [ ] Add new **rows** (not new columns) to [schemas/mode_parameters.csv](../schemas/mode_parameters.csv)
     for every new `x_*` param, since sequence design params apply to every mode.
   - [ ] Run `./scripts/regenerate_schemas.sh` to add the new params, under an
     `x_advanced_parameters` group, to **every** `schemas/nextflow_schema_<mode>.json` file (there
     is one per `design_mode`, not per `seq_method`).

5. **Metadata & filtering**
   - [ ] Add `XMetadataConverter` to [scripts/metadata_converter.py](../scripts/metadata_converter.py),
     register it in the `converters` dict and CLI `choices`, and add its field names (plus a
     `x_time` timing field if applicable) to the `metadata_field_names` list in `merge_all()`.
   - [ ] Add an entry to `SCORE_FIELDS` in [scripts/filter_seq.py](../scripts/filter_seq.py) mapping
     `'x'` → the JSON field name to threshold-filter on.

6. **Docs**
   - [ ] Add a `## X Advanced Parameters` table to [docs/parameters.md](parameters.md) (mirror the
     ProteinMPNN-OpenMMRelax / FAMPNN sections) and a row under `### Sequence Filtering Parameters`
     for the new score threshold param.
   - [ ] Update the `seq_method` description/enum in [nextflow.config](../nextflow.config) and
     [README.md](../README.md).
   - [ ] No [docs/modes.md](modes.md) changes are needed — `seq_method` isn't a `design_mode`.

7. **Testing**
   - [ ] Add an entry/profile exercising the new `seq_method`, following [docs/testing.md](testing.md)
     conventions (small `num_designs`/`seqs_per_design`, existing `benchmarkdata/` targets).

---

## 5. Key lessons / gotchas observed

- **`seq_method` is orthogonal to `design_mode`, unlike fold-design tool selection.** This makes
  the `main.nf` dispatch simpler (no per-mode validation needed for sequence design itself), but
  it also means new params must be threaded into *every* per-mode schema file rather than one new
  file, and any binder/monomer-specific behavior a new tool needs must be derived from
  `design_mode` at the call site (see FAMPNN's `analysis_chain_id`) rather than being implicit.
- **The fixed/inpainted-residue mask is the #1 shared contract to honor**, exactly mirroring the
  "chain order" lesson from Fold Design integration. Every sequence design tool must translate the
  fold-design tool's `rfd_inpaint_seq`/`bc_inpaint_seq`/`bg_inpaint_seq` boolean mask into its own
  native mechanism for preventing target/fixed-residue mutation — MPNN does this via B-factor
  tagging embedded in the PDB, FAMPNN via a side-channel fixed-positions CSV. Get this right early;
  it's what makes motif scaffolding, partial diffusion, and binder-interface-fixing work
  transparently regardless of which sequence design tool is chosen.
- **Full-atom vs backbone-only models need a side-chain restoration adapter.** Fold-design tools
  only emit backbone coordinates. A backbone-only sequence design tool (MPNN) needs no extra step;
  a full-atom tool (FAMPNN) needs an explicit side-chain-restoration pass before it can run at all
  (`prep_fampnn_designs.py`).
- **Naming convention (`fold_N_seq_M`) is the join key for everything downstream** — Structure
  Prediction, Analysis, and metadata merging all parse `fold_id`/`seq_id` back out of this filename
  pattern via the same regex. Get the renaming step right in `RunX` (or a dedicated post-process
  step) even if the underlying tool uses a different native output naming scheme (see FAMPNN's
  `sed` rename from `sampleY` to `seq_Y`).
- **A single merged side-channel config file (if needed) requires its own batching care.** FAMPNN's
  fixed-positions CSV must cover every PDB across every prep batch, so it's merged with
  `.collectFile(keepHeader: true)` before being `.combine()`d back onto the GPU-batched PDB
  channel — a wrinkle that wouldn't exist for a tool whose fixed-position info travels per-PDB
  (embedded in the PDB itself, like MPNN, or as one file per PDB).
- **Reuse the existing `seqs_per_design` param and binder-mode list** rather than inventing
  tool-specific equivalents — both are already tool-agnostic and consumed generically downstream
  (`FilterSeq`, ranking/analysis logic elsewhere in `main.nf`).
- **Metadata field prefixes must be unique and registered** in
  `metadata_converter.py::merge_all()`'s `metadata_field_names` list, or they will be silently
  dropped from the final `all_designs.csv` even if the JSONL topic channel captured them correctly
  — identical gotcha to Fold Design integration.
