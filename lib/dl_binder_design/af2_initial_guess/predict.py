#!/usr/bin/env python

import os
import numpy as np
import sys
import pickle

from timeit import default_timer as timer
import argparse
import glob

import jax

from jax.lib import xla_bridge

from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

import af2_util
import common_util


def _rewrite_chain_ids(pdb_text, binderlen):
    """Assign output chain IDs (A/B) by residue index and insert TER between chains."""
    if binderlen < 0:
        return pdb_text

    out_lines = []
    resid_to_idx = {}
    next_idx = 0
    inserted_ter = False

    for line in pdb_text.splitlines(True):
        if line.startswith("TER"):
            continue

        if line.startswith("ATOM") or line.startswith("HETATM"):
            chain_id = line[21]
            resseq = int(line[22:26])
            icode = line[26]
            resid = (chain_id, resseq, icode)

            if resid not in resid_to_idx:
                resid_to_idx[resid] = next_idx
                if next_idx == binderlen and not inserted_ter and binderlen > 0:
                    out_lines.append("TER\n")
                    inserted_ter = True
                next_idx += 1

            resid_idx = resid_to_idx[resid]
            new_chain = "A" if resid_idx < binderlen else "B"
            line = f"{line[:21]}{new_chain}{line[22:]}"

        out_lines.append(line)

    return "".join(out_lines)


#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()

# I/O Arguments
parser.add_argument("-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model')
parser.add_argument("-outpdbdir", type=str, default="outputs", help='The directory to which the output PDB files will be written. Only used when -pdbdir is active')
parser.add_argument("-runlist", type=str, default='', help="The path of a list of pdb tags to run. Only used when -pdbdir is active (default: ''; Run all PDBs)")
parser.add_argument("-checkpoint_name", type=str, default='check.point', help="The name of a file where tags which have finished will be written (default: check.point)")
parser.add_argument("-scorefilename", type=str, default='out.sc', help="The name of a file where scores will be written (default: out.sc)")
parser.add_argument("-maintain_res_numbering", action="store_true", default=False, help='Unused in no-PyRosetta mode')

parser.add_argument("-debug", action="store_true", default=False, help='When active, errors will cause the script to crash and the error message to be printed out (default: False)')
parser.add_argument("-debug_print", action="store_true", default=False, help='When active, print per-structure parser/feature diagnostics (default: False)')
parser.add_argument(
    "-feature_dump_dir",
    type=str,
    default="",
    help="Optional directory to write featurized AF2 feature dicts (.pkl) for diagnostics",
)

# AF2-Specific Arguments
parser.add_argument("-max_amide_dist", type=float, default=3.0, help='The maximum distance between an amide bond\'s carbon and nitrogen (default: 3.0)')
parser.add_argument("-recycle", type=int, default=3, help='The number of AF2 recycles to perform (default: 3)')
parser.add_argument("-no_initial_guess", action="store_true", default=False, help='When active, the model will not use an initial guess (default: False)')
parser.add_argument("-force_monomer", action="store_true", default=False, help='When active, predict only the first chain in a two-chain pdb as a monomer (default: False)')
parser.add_argument(
    "-use_pdbfixer",
    action="store_true",
    default=False,
    help="Disable PDBFixer atom completion (default: False; completion enabled).",
)
parser.add_argument(
    "-use_pyfaspr",
    action="store_true",
    default=False,
    help="Enable pyfaspr side-chain rotamer optimization in-memory after optional PDBFixer completion.",
)

args = parser.parse_args()


class FeatureHolder():
    """Hold model features and outputs for one structure."""

    def __init__(self, struct_data, monomer, binderlen, tag):
        self.tag = tag
        self.outtag = self.tag + '_af2pred'

        self.seq = struct_data["seq"]
        self.binderlen = binderlen
        self.monomer = monomer

        # Pre model features
        self.initial_all_atom_positions = struct_data["all_atom_positions"]
        self.initial_all_atom_masks = struct_data["all_atom_masks"]

        # Post model features
        self.plddt_array = None
        self.score_dict = None


class AF2_runner():
    """Handle feature generation, model execution, and output parsing."""

    def __init__(self, args, struct_manager):

        self.max_amide_dist = args.max_amide_dist

        # For timing
        self.t0 = None

        self.struct_manager = struct_manager

        # Other models may be run but their weights will also need to be downloaded
        self.model_name = "model_1_ptm"

        model_config = config.model_config(self.model_name)
        model_config.data.eval.num_ensemble = 1

        model_config.data.common.num_recycle = args.recycle
        model_config.model.num_recycle = args.recycle

        model_config.model.embeddings_and_evoformer.initial_guess = False if args.no_initial_guess else True

        model_config.data.common.max_extra_msa = 5
        model_config.data.eval.max_msa_clusters = 5

        params_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model_weights')

        model_params = data.get_model_haiku_params(model_name=self.model_name, data_dir=params_dir)

        self.model_runner = model.RunModel(model_config, model_params)

    def featurize(self, feat_holder):
        initial_guess = af2_util.parse_initial_guess(feat_holder.initial_all_atom_positions)

        # Determine which residues to template
        if feat_holder.monomer:
            # For monomers predict all residues
            feat_holder.residue_mask = [False for _ in range(len(feat_holder.seq))]
        else:
            # For interfaces fix the target and predict the binder
            feat_holder.residue_mask = [int(i) > feat_holder.binderlen for i in range(len(feat_holder.seq))]

        template_dict = af2_util.generate_template_features(
            feat_holder.seq,
            feat_holder.initial_all_atom_positions,
            feat_holder.initial_all_atom_masks,
            feat_holder.residue_mask,
        )

        feature_dict = {
            **pipeline.make_sequence_features(sequence=feat_holder.seq,
                                              description="none",
                                              num_res=len(feat_holder.seq)),
            **pipeline.make_msa_features(msas=[[feat_holder.seq]],
                                         deletion_matrices=[[[0] * len(feat_holder.seq)]]),
            **template_dict,
        }

        if feat_holder.monomer:
            breaks = []
        else:
            breaks = af2_util.check_residue_distances(
                feat_holder.initial_all_atom_positions,
                feat_holder.initial_all_atom_masks,
                self.max_amide_dist,
            )

        if self.struct_manager.debug_print:
            templated_res = int(np.sum(feat_holder.residue_mask))
            predicted_res = len(feat_holder.residue_mask) - templated_res
            print(
                f"[debug_print] tag={feat_holder.tag} "
                f"seq_len={len(feat_holder.seq)} "
                f"templated_res={templated_res} "
                f"predicted_res={predicted_res} "
                f"breaks={breaks}"
            )

        feature_dict['residue_index'] = af2_util.insert_truncations(feature_dict['residue_index'], breaks)

        feature_dict = self.model_runner.process_features(feature_dict, random_seed=0)

        return feature_dict, initial_guess

    def calculate_rmsd_target(self, feat_holder, pred_ca):
        """Calculate target-chain CA RMSD aligned on target residues."""
        if feat_holder.monomer or feat_holder.binderlen < 0:
            return float('nan')

        init_ca = feat_holder.initial_all_atom_positions[:, 1, :]
        target_indices = np.arange(feat_holder.binderlen, len(feat_holder.seq))

        return af2_util.subset_rmsd(
            xyz1=init_ca,
            align1=target_indices,
            calc1=target_indices,
            xyz2=pred_ca,
            align2=target_indices,
            calc2=target_indices,
        )

    def generate_scoredict(self, feat_holder, confidences, rmsds, pred_ca):
        """Collect confidence values and derive final score dict."""
        binderlen = feat_holder.binderlen
        plddt_array = confidences['plddt']
        plddt_overall = np.mean(plddt_array)
        pae = confidences['predicted_aligned_error']

        rmsd_overall = af2_util.subset_rmsd(
            xyz1=feat_holder.initial_all_atom_positions[:, 1, :],
            align1=np.arange(len(plddt_array)),
            calc1=np.arange(len(plddt_array)),
            xyz2=pred_ca,
            align2=np.arange(len(plddt_array)),
            calc2=np.arange(len(plddt_array)),
        )

        if feat_holder.monomer:
            score_dict = {
                "plddt_overall": plddt_overall,
                "pae_overall": np.mean(pae),
                "rmsd_overall": rmsd_overall,
                "time": timer() - self.t0,
            }
        else:
            rmsd_target = self.calculate_rmsd_target(feat_holder, pred_ca)
            plddt_binder = np.mean(plddt_array[:binderlen])
            plddt_target = np.mean(plddt_array[binderlen:])

            pae_overall = np.mean(pae)
            pae_binder = np.mean(pae[:binderlen, :binderlen])
            pae_target = np.mean(pae[binderlen:, binderlen:])

            pae_interaction1 = np.mean(pae[:binderlen, binderlen:])
            pae_interaction2 = np.mean(pae[binderlen:, :binderlen])
            pae_interaction = (pae_interaction1 + pae_interaction2) / 2

            score_dict = {
                "plddt_overall": plddt_overall,
                "plddt_binder": plddt_binder,
                "plddt_target": plddt_target,
                "pae_overall": pae_overall,
                "pae_binder": pae_binder,
                "pae_target": pae_target,
                "pae_interaction": pae_interaction,
                "rmsd_overall": rmsd_overall,
                "rmsd_binder_bndaln": rmsds['binder_aligned_rmsd'],
                "rmsd_binder_tgtaln": rmsds['target_aligned_rmsd'],
                "rmsd_target": rmsd_target,
                "time": timer() - self.t0,
            }

        feat_holder.score_dict = score_dict
        self.struct_manager.record_scores(feat_holder.outtag, score_dict, None)

        print(f"Tag: {feat_holder.outtag} scores: {score_dict}\\n")

    def process_output(self, feat_holder, feature_dict, prediction_result):
        """Parse AF2 output, score it, and write output pdb."""

        structure_module = prediction_result['structure_module']

        confidences = {}
        confidences['distogram'] = prediction_result['distogram']
        confidences['plddt'] = confidence.compute_plddt(
            prediction_result['predicted_lddt']['logits'][...])
        if 'predicted_aligned_error' in prediction_result:
            confidences.update(confidence.compute_predicted_aligned_error(
                prediction_result['predicted_aligned_error']['logits'][...],
                prediction_result['predicted_aligned_error']['breaks'][...]))

        feat_holder.plddt_array = confidences['plddt']

        b_factors = np.repeat(confidences['plddt'][:, None], residue_constants.atom_type_num, axis=1)
        this_protein = protein.Protein(
            aatype=feature_dict['aatype'][0],
            atom_positions=structure_module['final_atom_positions'][...],
            atom_mask=structure_module['final_atom_mask'][...],
            residue_index=feature_dict['residue_index'][0] + 1,
            b_factors=b_factors,
        )

        if feat_holder.monomer:
            rmsds = {
                'binder_aligned_rmsd': float('nan'),
                'target_aligned_rmsd': float('nan'),
            }
        else:
            target_mask = np.zeros(len(feat_holder.seq), dtype=bool)
            target_mask[feat_holder.binderlen:] = True
            rmsds = af2_util.calculate_rmsds(
                feat_holder.initial_all_atom_positions,
                this_protein.atom_positions,
                target_mask,
            )

        pred_ca = this_protein.atom_positions[:, 1, :]
        self.generate_scoredict(feat_holder, confidences, rmsds, pred_ca)

        unrelaxed_pdb_lines = protein.to_pdb(this_protein)
        self.struct_manager.dump_pdb(feat_holder, unrelaxed_pdb_lines)

    def process_struct(self, pdb_path):

        self.t0 = timer()

        struct_data, monomer, binderlen, usetag = self.struct_manager.load_pose(pdb_path)
        feat_holder = FeatureHolder(struct_data, monomer, binderlen, usetag)

        print(f'Processing struct with tag: {feat_holder.tag}')

        feature_dict, initial_guess = self.featurize(feat_holder)

        start = timer()
        print(f'Running {self.model_name}')

        prediction_result = self.model_runner.apply(
            self.model_runner.params,
            jax.random.PRNGKey(0),
            feature_dict,
            initial_guess,
        )

        print(f'Tag: {feat_holder.tag} finished AF2 prediction in {timer() - start} seconds')

        self.process_output(feat_holder, feature_dict, prediction_result)


class StructManager():
    """Handle input/output, runlist filtering, and checkpointing for pdb mode."""

    def __init__(self, args):
        self.args = args

        self.force_monomer = args.force_monomer
        self.use_pdbfixer = args.use_pdbfixer
        self.use_pyfaspr = args.use_pyfaspr
        self.debug_print = args.debug_print
        self.feature_dump_dir = args.feature_dump_dir
        self.score_fn = args.scorefilename

        if args.pdbdir == '':
            raise ValueError('Please provide `-pdbdir` (pdb mode only).')

        self.pdbdir = args.pdbdir
        self.outpdbdir = args.outpdbdir

        self.struct_iterator = glob.glob(os.path.join(args.pdbdir, '*.pdb'))

        if args.runlist != '':
            with open(args.runlist, 'r') as f:
                runlist = set(line.strip() for line in f)
            self.struct_iterator = [
                struct for struct in self.struct_iterator
                if '.'.join(os.path.basename(struct).split('.')[:-1]) in runlist
            ]
            print(f'After filtering by runlist, {len(self.struct_iterator)} structures remain')

        self.chkfn = args.checkpoint_name
        self.finished_structs = set()

        if os.path.isfile(self.chkfn):
            with open(self.chkfn, 'r') as f:
                for line in f:
                    self.finished_structs.add(line.strip())

    def _tag_from_path(self, pdb_path):
        return '.'.join(os.path.basename(pdb_path).split('.')[:-1])

    def record_checkpoint(self, pdb_path):
        tag = self._tag_from_path(pdb_path)
        with open(self.chkfn, 'a') as f:
            f.write(f'{tag}\\n')

    def iterate(self):
        for struct in self.struct_iterator:
            tag = self._tag_from_path(struct)
            if tag in self.finished_structs:
                print(f'{tag} has already been processed. Skipping')
                continue
            yield struct

    def record_scores(self, tag, score_dict, string_dict):
        write_header = not os.path.isfile(self.score_fn)
        af2_util.add2scorefile(tag, self.score_fn, write_header, score_dict, string_dict)

    def dump_pdb(self, feat_holder, pdb_text):
        if not os.path.exists(self.outpdbdir):
            os.makedirs(self.outpdbdir)

        if feat_holder.monomer:
            out_text = pdb_text
        else:
            out_text = _rewrite_chain_ids(pdb_text, feat_holder.binderlen)

        pdbfile = os.path.join(self.outpdbdir, feat_holder.outtag + '.pdb')
        with open(pdbfile, 'w') as handle:
            handle.write(out_text)

    def dump_feature_dict(self, feat_holder, feature_dict):
        if self.feature_dump_dir == "":
            return

        if not os.path.exists(self.feature_dump_dir):
            os.makedirs(self.feature_dump_dir)

        out_fn = os.path.join(self.feature_dump_dir, feat_holder.outtag + "_features.pkl")
        with open(out_fn, "wb") as handle:
            pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.debug_print:
            print(f"[debug_print] dumped featurized features to {out_fn}")

    def load_pose(self, pdb_path):
        fixed_missing_atom_count = 0
        pyfaspr_applied = False
        if self.use_pdbfixer:
            pdb_text, fixed_missing_atom_count = common_util.complete_pdb_backbone_to_all_atom_text(pdb_path)
        else:
            with open(pdb_path, "r") as handle:
                pdb_text = handle.read()

        if self.use_pyfaspr:
            pdb_text = common_util.optimize_rotamers_with_pyfaspr(pdb_text)
            pyfaspr_applied = True

        residues = af2_util.parse_pdb_residues_from_text(pdb_text)
        if len(residues) == 0:
            raise Exception(f'Pose {pdb_path} is empty. This is not supported by this script.')

        chain_order = []
        for res in residues:
            if res["chain"] not in chain_order:
                chain_order.append(res["chain"])

        if len(chain_order) > 2:
            raise Exception(f'Pose {pdb_path} has more than two chains. This is not supported by this script.')

        selected_chain_ids = set(chain_order)
        monomer = False
        binderlen = -1

        if len(chain_order) == 1:
            monomer = True
        elif self.force_monomer:
            print("/" * 60)
            print(f"Pose {pdb_path} has two chains. But force_monomer is set to True. Treating as monomer.")
            print("I am going to assume that the first chain is the binder and that is the chain I will predict")
            print("/" * 60)

            monomer = True
            selected_chain_ids = {chain_order[0]}
        else:
            binderlen = sum(1 for res in residues if res["chain"] == chain_order[0])

        all_atom_positions, all_atom_masks = af2_util.af2_get_atom_positions_from_pdb_text(pdb_text, selected_chain_ids)
        selected_residues = [res for res in residues if res["chain"] in selected_chain_ids]
        seq = af2_util.residues_to_sequence(selected_residues)

        if len(seq) != all_atom_positions.shape[0]:
            raise Exception(
                f"Residue/atom parsing mismatch for {pdb_path}: "
                f"sequence length {len(seq)} vs atoms length {all_atom_positions.shape[0]}"
            )

        usetag = self._tag_from_path(pdb_path)
        struct_data = {
            "seq": seq,
            "all_atom_positions": all_atom_positions,
            "all_atom_masks": all_atom_masks,
        }

        if self.debug_print:
            nonzero_atoms = int(np.sum(all_atom_masks))
            print(
                f"[debug_print] tag={usetag} pdb={pdb_path} "
                f"chain_order={chain_order} selected_chains={sorted(selected_chain_ids)} "
                f"monomer={monomer} binderlen={binderlen} "
                f"seq_len={len(seq)} atom_len={all_atom_positions.shape[0]} "
                f"nonzero_atom_entries={nonzero_atoms} "
                f"pdbfixer_enabled={self.use_pdbfixer} pdbfixer_missing_atoms={fixed_missing_atom_count} "
                f"pyfaspr_enabled={self.use_pyfaspr} pyfaspr_applied={pyfaspr_applied}"
            )

        return struct_data, monomer, binderlen, usetag


####################
####### Main #######
####################

device = xla_bridge.get_backend().platform
if device == 'gpu':
    print('/' * 60)
    print('/' * 60)
    print('Found GPU and will use it to run AF2')
    print('/' * 60)
    print('/' * 60)
    print('\\n')
else:
    print('/' * 60)
    print('/' * 60)
    print('WARNING! No GPU detected running AF2 on CPU')
    print('/' * 60)
    print('/' * 60)
    print('\\n')

struct_manager = StructManager(args)
af2_runner = AF2_runner(args, struct_manager)

for pdb in struct_manager.iterate():

    if args.debug:
        af2_runner.process_struct(pdb)

    else:
        t0 = timer()

        try:
            af2_runner.process_struct(pdb)

        except KeyboardInterrupt:
            sys.exit("Script killed by Control+C, exiting")

        except Exception:
            seconds = int(timer() - t0)
            print("Struct with tag %s failed in %i seconds with error: %s" % (pdb, seconds, sys.exc_info()[0]))

    struct_manager.record_checkpoint(pdb)
