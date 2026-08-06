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
                if (!params.rfd_partial_diffusion_timesteps) {
                    throw new IllegalArgumentException("rfd_partial_diffusion_timesteps is required when mode is 'rfd_partialdiff'")
                }
                break
            case 'rfd_foldcond':
                if (!params.rfd_scaffold_dir) {
                    throw new IllegalArgumentException("Please provide path to directory containing scaffold files for fold conditioning (rfd_scaffold_dir)")
                }
                def scaffoldsDir = new File(params.rfd_scaffold_dir.toString())
                if (!scaffoldsDir.exists() || !scaffoldsDir.isDirectory()) {
                    throw new IllegalArgumentException("rfd_scaffold_dir does not exist or is not a directory")
                }
                def hasSsFile = scaffoldsDir.listFiles()?.any { it.name.endsWith('_ss.pt') }
                def hasAdjFile = scaffoldsDir.listFiles()?.any { it.name.endsWith('_adj.pt') }
                if (!hasSsFile || !hasAdjFile) {
                    throw new IllegalArgumentException("rfd_scaffold_dir does not contain required _ss.pt and _adj.pt files")
                }
                break
            case 'rfd_motifscaff':
                if (!params.rfd_contigs) {
                    throw new IllegalArgumentException("rfd_contigs is required for rfd_motifscaff mode.")
                }
                break
        }

        // For rfd_denovo, skip design_length validation when rfd_contigs is provided (it contains the design length)
        if (params.design_mode == 'rfd_denovo' && !params.rfd_contigs) {
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
        if (params.design_mode == 'rfd_foldcond' && params.rfd_scaffold_dir) {
            inputs << params.rfd_scaffold_dir
        }
        return inputs
    }

    // Human-readable description of what an RFdiffusion contig string will do
    static String describeContigs(String contigs) {
        def cleaned = contigs.replaceAll(/[\[\]]/, '').trim()
        def chainParts = cleaned.split(/\s+/).findAll { it }
        def description = ["The contigs for RFdiffusion ${contigs} include ${chainParts.size()} chain${chainParts.size() > 1 ? 's' : ''}. RFdiffusion will:"]

        chainParts.each { part ->
            def segments = part.split('/')
            def chainId = null

            // Determine chain ID from first segment with chain designation
            segments.find { seg ->
                def matcher = seg =~ /^([A-Za-z])?(\d+)-(\d+)$/
                if (matcher.matches() && matcher.group(1)) {
                    chainId = matcher.group(1)
                    return true
                }
                return false
            }

            segments.each { seg ->
                switch (seg) {
                    case '0':
                        description << "* Insert a chainbreak ${chainId ? "after chain ${chainId}" : ""}"
                        break
                    default:
                        def matcher = seg =~ /^([A-Za-z])?(\d+)-(\d+)$/
                        if (matcher.matches()) {
                            def (segChain, start, end) = [matcher.group(1), matcher.group(2), matcher.group(3)]
                            if (segChain) {
                                description << "* Keep residues ${start}-${end} of chain ${segChain}"
                            } else {
                                // No chain ID - this is a partial diffusion or motifscaffolding mode
                                if (start == end) {
                                    description << "* Diffuse ${start} residues${chainId ? " for chain ${chainId}" : " for a new chain"}"
                                } else {
                                    description << "* Diffuse ${start}-${end} residues${chainId ? " for chain ${chainId}" : " for a new chain"}"
                                }
                            }
                        }
                }
            }
        }

        return description.join('\n')
    }
}
