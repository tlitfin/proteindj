class Utils {
    // Shared residue-spec grammar for hotspot_residues/bg_not_binding_residues/flexible_residues
    // etc: comma-separated tokens, each a chain-qualified single residue ('A56'), a chain-qualified
    // range ('A115-120'), or a bare chain ID meaning the whole chain ('B'). A chain identifier is
    // always required, for consistency across RFdiffusion, BindCraft, and BoltzGen.
    static final String RESIDUE_SPEC_REGEX = /^([A-Za-z]+(\d+(-\d+)?)?)(,[A-Za-z]+(\d+(-\d+)?)?)*$/

    // motifscaff_spec grammar (rfd_motifscaff/boltzgen_motifscaff): an ordered comma-separated list of
    // chain-A 'keep' tokens ('A<start>-<end>' or 'A<n>') and bare-digit 'insert' tokens ('<n>' or
    // '<min>-<max>'), e.g. '7-10,A1-60,5,A70-100,10'. Residues not covered by a keep token are deleted.
    static final String MOTIFSCAFF_SPEC_REGEX = /^((A\d+(-\d+)?)|(\d+(-\d+)?))(,((A\d+(-\d+)?)|(\d+(-\d+)?)))*$/

    // motifscaff_inpaint_seq grammar (rfd_motifscaff/boltzgen_motifscaff): plain chain-A residue/range
    // tokens only (no insert tokens), e.g. 'A10-50,A60'.
    static final String DESIGN_CHAIN_SPEC_REGEX = /^(A(\d+(-\d+)?)?)(,A(\d+(-\d+)?)?)*$/

    // rfd_partialdiff_spec grammar (rfd_partialdiff): an ordered comma-separated list of chain-A 'keep'
    // tokens ('A<start>-<end>' or 'A<n>') and bare-digit 'diffuse' tokens ('<n>' only - a fixed count,
    // no ranges, since partial diffusion cannot change chain A's length), e.g. 'A1-19,59'.
    static final String PARTIALDIFF_SPEC_REGEX = /^((A\d+(-\d+)?)|(\d+))(,((A\d+(-\d+)?)|(\d+)))*$/

    // Method to rebatch channels of tuples of PDBs and JSON files
    static def rebatchTuples(input_channel, batch_size = 50) {
        return input_channel 
            .transpose()
            .buffer(size: batch_size, remainder: true)
            .map { pairs -> 
                def first_elements = pairs.collect { it[0] }
                def second_elements = pairs.collect { it[1] }
                return [first_elements, second_elements]
            }
    }

    // Method to rebatch channels of PDBs by number of GPUs
    static def rebatchGPU(input_channel, gpus) {
        return input_channel
            .collect()
            .flatMap { all_pdbs ->
                def total_size = all_pdbs.size()
                def nbatches = Math.min(gpus, total_size)
                def bsize = (total_size / nbatches).doubleValue()
                def idx = 0
                all_pdbs.collect { pdb ->
                    def batch_id = Math.floor(idx++ / bsize).intValue()
                    [batch_id, pdb]
                }
            }
            .groupTuple()
    }
    
    // Method to rebatch channels of PDBs by number of GPUs, sorted by residue count
    static def rebatchGPUByNumRes(input_channel, gpus) {
        return input_channel
            .collect()
            .flatMap { all_pdbs ->
                // Sort PDB files by number of residues
                def sorted_pdbs = all_pdbs.sort { pdb ->
                    return countResidues(pdb)
                }
                
                def total_size = sorted_pdbs.size()
                def nbatches = Math.min(gpus, total_size)
                def bsize = (total_size / nbatches).doubleValue()
                def idx = 0
                sorted_pdbs.collect { pdb ->
                    def batch_id = Math.floor(idx++ / bsize).intValue()
                    [batch_id, pdb]
                }
            }
            .groupTuple()
    }

    // Helper function to count unique residues in a PDB file
    static def countResidues(pdb_file) {
        def residue_set = new HashSet()
        
        pdb_file.eachLine { line ->
            // Only process ATOM and HETATM lines
            if (line.startsWith("ATOM  ") || line.startsWith("HETATM")) {
                // Extract chain ID (column 22) and residue number (columns 23-26)
                if (line.length() >= 26) {
                    def chain_id = line.substring(21, 22)
                    def res_num = line.substring(22, 26).trim()
                    def res_name = line.substring(17, 20).trim()
                    
                    // Create unique identifier: chain + residue_number + residue_name
                    def residue_key = "${chain_id}_${res_num}_${res_name}"
                    residue_set.add(residue_key)
                }
            }
        }
        
        return residue_set.size()
    }
    
    /**
     * Format filter parameters for command-line arguments
     * @param params The Nextflow params object
     * @param paramPrefix The prefix for parameter names
     * @param paramNames List of parameter names to process
     * @return Formatted parameter string for command line
     */
    static def formatFilterParams(params, paramPrefix, paramNames) {
        return paramNames.collect { name ->
            def paramValue = params["${paramPrefix}_${name}"]
            if (paramValue != null) {
                def cmdParam = name.replaceAll('_', '-')
                return "--${paramPrefix}-${cmdParam} ${paramValue}"
            } else {
                return ""
            }
        }.findAll { it != "" }.join(' ')
    }

    /**
     * Count PDB files from a channel
     * @param channel The input channel containing files
     * @return A value channel with the count of PDB files
     */
    static def countPdbFiles(channel) {
        return channel
            .flatten()
            .collect()
            .map { files -> 
                files.findAll { file -> 
                    file.toString().endsWith('.pdb') 
                }.size() 
            }
            .ifEmpty(0)
    }

    // Extract distinct protein chain IDs from a PDB's ATOM records (excludes HETATM/ligands/waters)
    static Set<String> getPdbChainIds(pdbFile) {
        def chainIds = [] as Set
        pdbFile.eachLine { line ->
            if (line.startsWith("ATOM  ") && line.length() >= 22) {
                chainIds << line.substring(21, 22)
            }
        }
        return chainIds
    }

    // Extract the sorted, distinct residue numbers present for a given chain in a PDB's ATOM
    // records. Used to expand a bare chain ID hotspot token (e.g. 'B') into individual residues.
    static List<Integer> getPdbChainResidueNumbers(pdbFile, String chainId) {
        def resNums = [] as Set
        pdbFile.eachLine { line ->
            if (line.startsWith("ATOM  ") && line.length() >= 26 && line.substring(21, 22) == chainId) {
                resNums << line.substring(22, 26).trim().toInteger()
            }
        }
        return resNums.sort()
    }

    // Validate design length (required, one or two comma-separated integers, min<=max)
    static void validateDesignLength(design_length) {
        if (!design_length) {
            throw new IllegalArgumentException("Please provide a value for design_length, e.g. '65', '65-150'.")
        }
        // Nextflow's CLI parser coerces bare-numeric params (e.g. --design_length 65) to Integer,
        // so coerce to String before splitting to support both String and Integer param values.
        def designLengthVals = design_length.toString().split('-')
        if (designLengthVals.size() > 2 || !designLengthVals.every { it.isInteger() }) {
            throw new IllegalArgumentException("design_length parameter must contain one or two integers (dash-separated) e.g. '65' '65-150'.")
        }
        if (designLengthVals.size() == 2) {
            def minLength = Integer.parseInt(designLengthVals[0])
            def maxLength = Integer.parseInt(designLengthVals[1])
            if (minLength > maxLength || minLength < 1) {
                throw new IllegalArgumentException("design_length values must be valid: min ≤ max and min ≥ 1.")
            }
        }
    }

    // Format design_length as an RFdiffusion contig length range 'N-M'. A single-value design_length
    // (e.g. '65') is expanded to 'N-N' (e.g. '65-65') rather than passed through bare: RFdiffusion's
    // Hydra CLI parses an unquoted single-integer contig element (e.g. contigmap.contigs=[65]) as an
    // int rather than a string, which crashes contigs.py (expects str.strip()/split()).
    static String designLengthContigRange(design_length) {
        def str = design_length.toString()
        return str.contains('-') ? str : "${str}-${str}"
    }
    
}
