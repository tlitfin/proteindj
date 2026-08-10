class RFDEngine extends DesignEngine {

    RFDEngine(Map params, boolean isBinderMode) {
        super(params, isBinderMode)
    }

    // 'denovo' | 'foldcond' | 'motifscaff' | 'partialdiff'
    private String variant() {
        return params.design_mode - 'rfd_'
    }

    boolean requiresInputPdb() {
        return variant() in ['motifscaff', 'partialdiff'] || isBinderMode
    }

    // RFdiffusion requires xyz coordinates even with no real target - true only for the
    // monomer denovo/foldcond variants, which don't otherwise require an input_pdb.
    boolean needsPlaceholderPdb() {
        return variant() in ['denovo', 'foldcond'] && !isBinderMode
    }

    void validateParams() {
        switch (params.design_mode) {
            case 'rfd_partialdiff':
                if (!params.rfd_partialdiff_timesteps) {
                    throw new IllegalArgumentException("rfd_partialdiff_timesteps is required when mode is 'rfd_partialdiff'")
                }
                if (params.rfd_partialdiff_fixed_seq && !params.rfd_partialdiff_fixed_seq.matches(Utils.RESIDUE_SPEC_REGEX)) {
                    throw new IllegalArgumentException("rfd_partialdiff_fixed_seq format invalid. Acceptable: 'A10-13,A16' (chain identifier required, must reference the design chain only).")
                }
                if (params.rfd_partialdiff_spec && !params.rfd_partialdiff_spec.matches(Utils.PARTIALDIFF_SPEC_REGEX)) {
                    throw new IllegalArgumentException("rfd_partialdiff_spec format invalid. Must be an ordered list of chain-A keep tokens ('A<start>-<end>') and/or bare fixed-count diffuse tokens ('<n>', no ranges since length cannot change), e.g. 'A1-19,59'.")
                }
                break
            case 'rfd_foldcond':
                if (!params.rfd_foldcond_scaffold_dir) {
                    throw new IllegalArgumentException("Please provide path to directory containing scaffold files for fold conditioning (rfd_foldcond_scaffold_dir)")
                }
                def scaffoldsDir = new File(params.rfd_foldcond_scaffold_dir.toString())
                if (!scaffoldsDir.exists() || !scaffoldsDir.isDirectory()) {
                    throw new IllegalArgumentException("rfd_foldcond_scaffold_dir does not exist or is not a directory")
                }
                def hasSsFile = scaffoldsDir.listFiles()?.any { it.name.endsWith('_ss.pt') }
                def hasAdjFile = scaffoldsDir.listFiles()?.any { it.name.endsWith('_adj.pt') }
                if (!hasSsFile || !hasAdjFile) {
                    throw new IllegalArgumentException("rfd_foldcond_scaffold_dir does not contain required _ss.pt and _adj.pt files")
                }
                break
            case 'rfd_motifscaff':
                if (!(params.motifscaff_spec || params.motifscaff_inpaint_seq || params.flexible_residues)) {
                    throw new IllegalArgumentException("rfd_motifscaff mode requires at least one of motifscaff_spec/motifscaff_inpaint_seq/flexible_residues.")
                }
                if (params.motifscaff_spec && !params.motifscaff_spec.matches(Utils.MOTIFSCAFF_SPEC_REGEX)) {
                    throw new IllegalArgumentException("motifscaff_spec format invalid. Must be an ordered list of chain-A keep tokens ('A<start>-<end>') and/or bare insert-count tokens ('<n>' or '<min>-<max>'), e.g. '7-10,A1-60,5,A70-100,10'.")
                }
                if (params.motifscaff_inpaint_seq && !params.motifscaff_inpaint_seq.matches(Utils.DESIGN_CHAIN_SPEC_REGEX)) {
                    throw new IllegalArgumentException("motifscaff_inpaint_seq format invalid. Must reference chain A only, e.g. 'A10-50,A60'.")
                }
                break
        }

        if (params.flexible_residues) {
            if (!params.flexible_residues.matches(Utils.RESIDUE_SPEC_REGEX)) {
                throw new IllegalArgumentException("flexible_residues format invalid. Acceptable: 'A10-13,A16,B'.")
            }
            if (needsPlaceholderPdb()) {
                throw new IllegalArgumentException("flexible_residues requires a real input structure - not supported for monomer 'rfd_denovo'/'rfd_foldcond' (no target chain to mark flexible).")
            }
        }

        if (params.design_mode == 'rfd_denovo') {
            Utils.validateDesignLength(params.design_length)
        }
    }

    // Contig-related logging happens inline alongside channel setup in main.nf, since it depends
    // on which contigs channel branch is taken (user-provided / auto-generated / foldcond).
    List<String> startupMessages() {
        return []
    }

    List<String> collectInputFiles() {
        def inputs = []
        if (requiresInputPdb() && params.input_pdb) {
            inputs << params.input_pdb
        }
        if (params.design_mode == 'rfd_foldcond' && params.rfd_foldcond_scaffold_dir) {
            inputs << params.rfd_foldcond_scaffold_dir
        }
        return inputs
    }
}
