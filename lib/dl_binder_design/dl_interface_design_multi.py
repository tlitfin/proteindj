#!/usr/bin/env python

import os, sys

from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.scoring import CA_rmsd

import numpy as np
from collections import OrderedDict
import time
import argparse
import subprocess
import time
import glob

import torch
import json

import util_protein_mpnn as mpnn_util

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent, 'include'))
from silent_tools import silent_tools

init( "-beta_nov16 -in:file:silent_struct_type binary -mute all" +
    " -use_terminal_residues true -mute basic.io.database core.scoring" )

def cmd(command, wait=True):
    the_command = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if (not wait):
        return
    the_stuff = the_command.communicate()
    return str( the_stuff[0]) + str(the_stuff[1] )

def range1( iterable ): return range( 1, iterable + 1 )

#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()
script_dir = os.path.dirname(os.path.realpath(__file__))

# I/O Arguments
parser.add_argument( "-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model' )
parser.add_argument( "-silent", type=str, default="", help='The name of a silent file to run through the model' )
parser.add_argument( "-outpdbdir", type=str, default="outputs", help='The directory to which the output PDB files will be written, used if the -pdbdir arg is active' )
parser.add_argument( "-outsilent", type=str, default="out.silent", help='The name of the silent file to which output structs will be written, used if the -silent arg is active' )
parser.add_argument( "-runlist", type=str, default='', help="The path of a list of pdb tags to run, only active when the -pdbdir arg is active (default: ''; Run all PDBs)" )
parser.add_argument( "-checkpoint_name", type=str, default='check.point', help="The name of a file where tags which have finished will be written (default: check.point)" )

parser.add_argument( "-debug", action="store_true", default=False, help='When active, errors will cause the script to crash and the error message to be printed out (default: False)')

# Design Arguments
parser.add_argument( "-relax_max_cycles", type=int, default=1, help="The number of relax cycles to perform on each structure (default: 1)" )
parser.add_argument( "-output_intermediates", action="store_true", help='Whether to write all intermediate sequences from the relax cycles to disk (default: False)' )
parser.add_argument( "-seqs_per_struct", type=int, default="1", help="The number of sequences to generate for each input structure (default: 1)" )
# New relaxation parameters
parser.add_argument( "-relax_output", action="store_true", default=False, help='Whether to run relaxation before saving output. (default: False)')
parser.add_argument( "-relax_seqs_per_cycle", type=int, default=1, help="Number of intermediate sequences to generate per relax cycle. The sequence with the lowest (best) score is kept (default: 1)")
parser.add_argument( "-relax_convergence_rmsd", type=float, default=0.2, help="Convergence criteria 1 of 2. Design is considered converged if the C-alpha RMSD (Å) between cycles is <= this threshold (default: 0.2)")
parser.add_argument( "-relax_convergence_score", type=float, default=0.1, help="Convergence criteria 2 of 2. Design is considered converged if the improvement in score between cycles is <= this threshold (default: 0.1)")
parser.add_argument( "-relax_convergence_max_cycles", type=int, default=1, help="Design is considered converged if it meets both convergence criteria for n consecutive cycles (default: 1)")

# ProteinMPNN-Specific Arguments
parser.add_argument( "-checkpoint_path", type=str, default=os.path.join(script_dir, 'ProteinMPNN/vanilla_model_weights/v_48_020.pt'), help=f"The path to the ProteinMPNN weights you wish to use, default {os.path.join(script_dir, 'ProteinMPNN/vanilla_model_weights/v_48_020.pt')}")
parser.add_argument( "-temperature", type=float, default=0.1, help='The sampling temperature to use when running ProteinMPNN (default: 0.1)' )
parser.add_argument( "-augment_eps", type=float, default=0, help='The variance of random noise to add to the atomic coordinates (default 0)' )
parser.add_argument( "-protein_features", type=str, default='full', help='What type of protein features to input to ProteinMPNN (default: full)' )
parser.add_argument( "-omit_AAs", type=str, default='CX', help='A string of all residue types (one letter case-insensitive) that you would not like to use for design. Letters not corresponding to residue types will be ignored (default: CX)' )
parser.add_argument( "-bias_AA_jsonl", type=str, default='', help='The path to a JSON file containing a dictionary mapping residue one-letter names to the bias for that residue eg. {A: -1.1, F: 0.7} (default: ''; no bias)' )
parser.add_argument( "-num_connections", type=int, default=48, help='Number of neighbors each residue is connected to. Do not mess around with this argument unless you have a specific set of ProteinMPNN weights which expects a different number of connections. (default: 48)' )

args = parser.parse_args( sys.argv[1:] )

class sample_features():
    '''
    This is a struct which keeps all the features related to a single sample together
    '''

    def __init__(self, pose, tag):
        self.pose = pose
        self.tag = os.path.basename(tag).split('.')[0]
    
    def parse_fixed_res(self):
        '''
            Parse which residues are fixed from the residue remarks
        '''
        # Iterate over the residue positions in this pose and check if they are fixed
        fixed_list = []

        # Get ordered unique chainIDs for the pose first
        self.chains = list( OrderedDict.fromkeys( [ self.pose.pdb_info().chain(i) for i in range1( self.pose.total_residue() ) ] ) )
        
        # Determine the range of residues to check
        if len(self.chains) == 1:
            # For single chain, check all residues
            endA = self.pose.total_residue()
        else:
            # For multi-chain, check only first chain
            endA = self.pose.split_by_chain()[1].total_residue()
        
        for resi in range1( endA ):
            reslabels = self.pose.pdb_info().get_reslabels( resi )

            if len(reslabels) == 0: continue

            if str(self.pose.pdb_info().get_reslabels( resi )[1]).strip() == 'FIXED':
                fixed_list.append( resi )

        # Create the fixed res dict, this will be input to ProteinMPNN
        if len(self.chains) == 1:
            self.fixed_res = {
                self.chains[0]: fixed_list
            }
        else:
            self.fixed_res = {
                self.chains[0]: fixed_list,
                self.chains[1]: []
            }

    def thread_mpnn_seq(self, binder_seq):
        '''
        Thread the binder sequence onto the pose being designed
        '''
        rsd_set = self.pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

        for resi, mut_to in enumerate( binder_seq ):
            resi += 1 # 1 indexing
            name3 = mpnn_util.aa_1_3[ mut_to ]
            new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
            self.pose.replace_residue( resi, new_res, True )
    

class ProteinMPNN_runner():
    '''
    This class is designed to run the ProteinMPNN model on a single input. This class handles
    the loading of the model, the loading of the input data, the FastRelax cycles, and the processing of the output
    '''

    def __init__(self, args, struct_manager):
        self.struct_manager = struct_manager

        if torch.cuda.is_available():
            print('Found GPU will run ProteinMPNN on GPU')
            self.device = "cuda:0"
        else:
            print('No GPU found, running ProteinMPNN on CPU')
            self.device = "cpu"

        self.mpnn_model = mpnn_util.init_seq_optimize_model(
            self.device,
            hidden_dim=128,
            num_layers = 3,
            backbone_noise = args.augment_eps,
            num_connections = args.num_connections,
            checkpoint_path = args.checkpoint_path
        )

        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

        self.debug = args.debug

        self.temperature = args.temperature
        self.seqs_per_struct = args.seqs_per_struct
        self.relax_seqs_per_cycle = args.relax_seqs_per_cycle
        self.omit_AAs = [ letter for letter in args.omit_AAs.upper() if letter in list(alphabet) ]

        # Parse AA bias settings from json
        if os.path.isfile(args.bias_AA_jsonl):
            print(f'Found AA bias json file at {args.bias_AA_jsonl}')
            with open(args.bias_AA_jsonl, 'r') as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                bias_AA_dict = json.loads(json_str)

            self.bias_AAs_np = np.zeros(len(alphabet))
            for n, AA in enumerate(alphabet):
                if AA in list(bias_AA_dict.keys()):
                    self.bias_AAs_np[n] = bias_AA_dict[AA]
        else:
            # Not using AA bias
            self.bias_AAs_np = np.zeros(len(alphabet))

        # Load target-binder FastRelax mover
        xml_binder = os.path.join(script_dir, 'RosettaFastRelaxUtil.xml')
        objs_binder = protocols.rosetta_scripts.XmlObjects.create_from_file(xml_binder)
        self.FastRelax_binder = objs_binder.get_mover('FastRelax')

        # Load monomer FastRelax mover (new)
        xml_monomer = os.path.join(script_dir, 'RosettaFastRelaxUtilMonomer.xml')
        objs_monomer = protocols.rosetta_scripts.XmlObjects.create_from_file(xml_monomer)
        self.FastRelax_monomer = objs_monomer.get_mover('FastRelaxMonomer')

        self.relax_max_cycles = args.relax_max_cycles

    def relax_pose(self, sample_feats):
        '''
        Run FastRelax on the current pose using appropriate protocol
        '''
        relaxT0 = int(time.time())

        if len(sample_feats.chains) == 1:
            print('Running monomer FastRelax')
            self.FastRelax_monomer.apply(sample_feats.pose)
        else:
            print('Running binder FastRelax')
            self.FastRelax_binder.apply(sample_feats.pose)

        print(f"Completed one cycle of FastRelax in {int(time.time()) - relaxT0} seconds")

    def sequence_optimize(self, sample_feats, num_seqs=None):
        if num_seqs is None:  # Backwards compatibility
            num_seqs = self.seqs_per_struct
        
        mpnn_t0 = time.time()

        # Once we have figured out pose I/O without Rosetta this will be easy to swap in
        pdbfile = 'temp.pdb'
        sample_feats.pose.dump_pdb(pdbfile)

        feature_dict = mpnn_util.generate_seqopt_features(pdbfile, sample_feats.chains)

        os.remove(pdbfile)

        arg_dict = mpnn_util.set_default_args(num_seqs, omit_AAs=self.omit_AAs)
        arg_dict['temperature'] = self.temperature

        if len(sample_feats.chains) == 1:
            # For single chain design: mask the chain to design it
            masked_chains = sample_feats.chains
            visible_chains = []
        else:
            # For multi-chain design: mask binder(s), keep target visible
            masked_chains = sample_feats.chains[:-1]
            visible_chains = [sample_feats.chains[-1]]

        fixed_positions_dict = {pdbfile[:-len('.pdb')]: sample_feats.fixed_res}

        if self.debug:
            print(f'Fixed positions dict: {fixed_positions_dict}')

        sequences = mpnn_util.generate_sequences(
            self.mpnn_model,
            self.device,
            feature_dict,
            arg_dict,
            masked_chains,
            visible_chains,
            bias_AAs_np=self.bias_AAs_np,
            fixed_positions_dict=fixed_positions_dict
        )
        print( f"ProteinMPNN generated {len(sequences)} sequences in {int( time.time() - mpnn_t0 )} seconds" ) 
        for i, (seq, score) in enumerate(sequences):
            print(f"Seq {i+1}: {seq} (Score: {score:.2f})")
        
        return sequences

    def proteinmpnn(self, sample_feats):
        '''
        Run ProteinMPNN sequence optimization on the pose, this does not use FastRelax
        '''
        seqs_scores = self.sequence_optimize(sample_feats)

        # Iterate though each seq score pair and thread the sequence onto the pose
        # Then write each pose to a pdb file
        prefix = f"{sample_feats.tag}_seq"
        for idx, (seq, score) in enumerate(seqs_scores): 
            sample_feats.thread_mpnn_seq(seq)
            
            print(f"Final Sequence {idx} for {sample_feats.tag}: {seq} (Score: {score:.2f})")
            
            outtag = f"{prefix}_{idx}"
            if args.relax_output:
                # Final relaxation and output
                print(f"Relaxing output for Seq {seq_idx}")
                self.relax_pose(sample_feats)
            self.struct_manager.dump_pose(sample_feats.pose, outtag, seq, score)

    def proteinmpnn_fastrelax(self, sample_feats):
        '''
        Run ProteinMPNN plus FastRelax on the pose being designed
        '''
        from pyrosetta.rosetta.core.scoring import CA_rmsd
        
        original_pose = sample_feats.pose.clone()
        
        # Generate initial sequences from ORIGINAL structure
        seqs_scores = self.sequence_optimize(sample_feats)
        
        for seq_idx, (initial_seq, score) in enumerate(seqs_scores):
            print(f"Performing optimisation of Seq {seq_idx}: {initial_seq} (Score: {score:.2f})")
            trajectory_pose = original_pose.clone()
            sample_feats.pose = trajectory_pose
            sample_feats.thread_mpnn_seq(initial_seq)
            
            previous_best_score = None
            previous_pose = trajectory_pose.clone()
            convergence_counter = 0
            last_best_seq = initial_seq  # Track last cycle's best sequence
            
            for cycle in range(args.relax_max_cycles):
                # Relax current structure
                self.relax_pose(sample_feats)
                
                # Generate new sequences from relaxed structure
                new_seqs = self.sequence_optimize(sample_feats, num_seqs=self.relax_seqs_per_cycle)
                
                # Select best sequence (lowest score)
                sorted_seqs = sorted(new_seqs, key=lambda x: x[1])
                best_seq, best_score = sorted_seqs[0]
                print(f"Cycle {cycle} best sequence: {best_seq} (Score: {best_score:.2f})")

                # Track structural changes
                current_pose = sample_feats.pose.clone()
                rmsd = CA_rmsd(current_pose, previous_pose)
                print(f"Cycle {cycle} Ca-RMSD: {rmsd:.2f}Å")
                
                # Update tracking variables
                last_best_seq = best_seq
                sample_feats.thread_mpnn_seq(best_seq)
                previous_pose = current_pose.clone()

                # Convergence check parameters
                RMSD_THRESHOLD = args.relax_convergence_rmsd # Å - structural convergence
                SCORE_DELTA_THRESHOLD = args.relax_convergence_score  # Score improvement threshold
                CONVERGENCE_CYCLES = args.relax_convergence_max_cycles # Require X consecutive cycles

                # Convergence check
                if previous_best_score is not None:
                    score_delta = previous_best_score - best_score
                    
                    # Dual criteria check
                    if (rmsd <= RMSD_THRESHOLD and 
                        abs(score_delta) < SCORE_DELTA_THRESHOLD):
                        convergence_counter += 1
                    else:
                        convergence_counter = 0  # Reset counter if either condition fails
                    
                    # Early termination conditions
                    if convergence_counter >= CONVERGENCE_CYCLES:
                        print(f"Convergence at cycle {cycle}: "
                            f"RMSD ≤{RMSD_THRESHOLD}Å ({rmsd:.2f}Å) "
                            f"and score Δ<{SCORE_DELTA_THRESHOLD} ({score_delta:.2f}) for {CONVERGENCE_CYCLES} cycles")
                        break
                previous_best_score = best_score

                # Output intermediates
                if args.output_intermediates:
                    tag = f"{sample_feats.tag}_seq_{seq_idx}_cycle{cycle}"
                    self.struct_manager.dump_pose(current_pose, tag, best_seq, best_score)

            # Final sequence is from last completed cycle
            print(f"Final Sequence {seq_idx} for {sample_feats.tag}: {last_best_seq} (Score: {best_score:.2f})")
            
            if args.relax_output:
                # Final relaxation and output
                print(f"Relaxing output for Seq {seq_idx}")
                self.relax_pose(sample_feats)
            final_tag = f"{sample_feats.tag}_seq_{seq_idx}"
            self.struct_manager.dump_pose(sample_feats.pose, final_tag, last_best_seq, best_score)

    def run_model(self, tag, args):
        t0 = time.time()

        print(f"Attempting pose: {tag}")
        
        # Load the pose 
        pose = self.struct_manager.load_pose(tag)

        # Initialize the features
        sample_feats = sample_features(pose, tag)

        # Parse the fixed residues from the pose remarks
        sample_feats.parse_fixed_res()

        # Now determine which type of run we are doing and execute it
        if args.relax_max_cycles > 0:
            self.proteinmpnn_fastrelax(sample_feats)

        else:
            self.proteinmpnn(sample_feats)

        seconds = int(time.time() - t0)

        print( f"Struct: {pdb} reported success in {seconds} seconds\n" )

class StructManager():
    '''
    This class handles all of the input and output for the ProteinMPNN model. It deals with silent files vs. pdbs,
    checkpointing, and writing of outputs

    Note: This class could be moved to a separate file
    '''

    def __init__(self, args):
        self.args = args

        self.silent = False
        if not args.silent == '':
            self.silent = True

            self.struct_iterator = silent_tools.get_silent_index(args.silent)['tags']

            self.sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
            self.sfd_in.read_file(args.silent)

            self.sfd_out = core.io.silent.SilentFileData(args.outsilent, False, False, "binary", core.io.silent.SilentFileOptions())

            self.outsilent = args.outsilent

        self.pdb = False
        if not args.pdbdir == '':
            self.pdb = True

            self.pdbdir    = args.pdbdir
            self.outpdbdir = args.outpdbdir

            self.struct_iterator = glob.glob(os.path.join(args.pdbdir, '*.pdb'))

            # Parse the runlist and determine which structures to process
            if args.runlist != '':
                with open(args.runlist, 'r') as f:
                    self.runlist = set([line.strip() for line in f])

                    # Filter the struct iterator to only include those in the runlist
                    self.struct_iterator = [struct for struct in self.struct_iterator if os.path.basename(struct).split('.')[0] in self.runlist]

                    print(f'After filtering by runlist, {len(self.struct_iterator)} structures remain')

        # Assert that either silent or pdb is true, but not both
        assert(self.silent ^ self.pdb), f'Both silent and pdb are set to {args.silent} and {args.pdb} respectively. Only one of these may be active at a time'

        # Setup checkpointing
        self.chkfn = args.checkpoint_name
        self.finished_structs = set()

        if os.path.isfile(self.chkfn):
            with open(self.chkfn, 'r') as f:
                for line in f:
                    self.finished_structs.add(line.strip())

    def record_checkpoint(self, tag):
        '''
        Record the fact that this tag has been processed.
        Write this tag to the list of finished structs
        '''
        with open(self.chkfn, 'a') as f:
            f.write(f'{tag}\n')

    def iterate(self):
        '''
        Iterate over the silent file or pdb directory and run the model on each structure
        '''

        # Iterate over the structs and for each, check that the struct has not already been processed
        for struct in self.struct_iterator:
            tag = os.path.basename(struct).split('.')[0]
            if tag in self.finished_structs:
                print(f'{tag} has already been processed. Skipping')
                continue

            yield struct

    def dump_pose(self, pose, tag, sequence=None, score=None):
        '''
        Dump this pose and create JSON metadata if sequence/score provided
        '''
        json_data = None
        if sequence is not None and score is not None:
            json_data = {
                "design": tag,
                "sequence": sequence,
                "score": f"{score:.2f}"
            }

        if self.pdb:
            if not os.path.exists(self.outpdbdir):
                os.makedirs(self.outpdbdir)
            
            pdbfile = os.path.join(self.outpdbdir, tag + '.pdb')
            pose.dump_pdb(pdbfile)
            
            if json_data:  # Write JSON to same directory as PDBs
                json_path = os.path.join(self.outpdbdir, f"mpnn_{tag}.json")
                with open(json_path, 'w') as f:
                    f.write(json.dumps(json_data) + '\n')

        if self.silent:
            struct = self.sfd_out.create_SilentStructOP()
            struct.fill_struct(pose, tag)
            self.sfd_out.add_structure(struct)
            self.sfd_out.write_silent_struct(struct, self.outsilent)
            
            if json_data:  # Write JSON to silent file directory
                silent_dir = os.path.dirname(self.outsilent)

                json_path = os.path.join(silent_dir, f"mpnn_{tag}.json")
                with open(json_path, 'w') as f:
                    f.write(json.dumps(json_data) + '\n')

    def load_pose(self, tag):
        '''
        Load a pose from either a silent file or a pdb file depending on the input arguments
        '''

        if not self.pdb and not self.silent:
            raise Exception('Neither pdb nor silent is set to True. Cannot load pose')

        if self.pdb:
            pose = pose_from_pdb(tag)
        
        if self.silent:
            pose = Pose()
            self.sfd_in.get_structure(tag).fill_pose(pose)
        
        return pose


####################
####### Main #######
####################

struct_manager     = StructManager(args)
proteinmpnn_runner = ProteinMPNN_runner(args, struct_manager)

for pdb in struct_manager.iterate():

    if args.debug: proteinmpnn_runner.run_model(pdb, args)

    else: # When not in debug mode the script will continue to run even when some poses fail
        t0 = time.time()

        try: proteinmpnn_runner.run_model(pdb, args)

        except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

        except:
            seconds = int(time.time() - t0)
            print( "Struct with tag %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )

    # We are done with one pdb, record that we finished
    struct_manager.record_checkpoint(pdb)