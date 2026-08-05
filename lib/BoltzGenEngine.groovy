class BoltzGenEngine extends DesignEngine {

    // bg_redesign_spec grammar: an ordered comma-separated list of chain-A 'keep' tokens
    // ('A<start>-<end>' or 'A<n>') and bare-digit 'insert' tokens ('<n>' or '<min>-<max>'),
    // e.g. '7-10,A1-60,5,A70-100,10'. Residues not covered by a keep token are deleted.
    private static final String REDESIGN_SPEC_REGEX = /^((A\d+(-\d+)?)|(\d+(-\d+)?))(,((A\d+(-\d+)?)|(\d+(-\d+)?)))*$/

    // bg_redesign_inpaint_seq grammar: plain chain-A residue/range tokens only (no insert tokens).
    private static final String CHAIN_A_SPEC_REGEX = /^(A(\d+(-\d+)?)?)(,A(\d+(-\d+)?)?)*$/

    BoltzGenEngine(Map params, boolean isBinderMode) {
        super(params, isBinderMode)
    }

    boolean requiresInputPdb() {
        return params.design_mode == 'boltzgen_redesign' || isBinderMode
    }

    void validateParams() {
        if (params.design_mode == 'boltzgen_denovo') {
            Utils.validateDesignLength(params.design_length)
        }

        if (params.hotspot_residues && !params.hotspot_residues.matches(Utils.RESIDUE_SPEC_REGEX)) {
            throw new IllegalArgumentException("hotspot_residues format invalid. Acceptable: 'A56,A115-120,B' (chain identifier is required).")
        }
        if (params.bg_not_binding_residues && !params.bg_not_binding_residues.matches(Utils.RESIDUE_SPEC_REGEX)) {
            throw new IllegalArgumentException("bg_not_binding_residues format invalid. Acceptable: 'A200,A210-215,B'.")
        }
        if (params.design_mode == 'boltzgen_denovo' && !params.input_pdb && (params.hotspot_residues || params.bg_not_binding_residues)) {
            throw new IllegalArgumentException("hotspot_residues/bg_not_binding_residues require a target - please provide input_pdb, or leave them null for monomer design.")
        }
        if (params.bg_flexible_residues && !params.bg_flexible_residues.matches(Utils.RESIDUE_SPEC_REGEX)) {
            throw new IllegalArgumentException("bg_flexible_residues format invalid. Acceptable: 'A10-13,A16,B'.")
        }
        if (params.bg_redesign_spec) {
            if (params.design_mode == 'boltzgen_denovo') {
                throw new IllegalArgumentException("bg_redesign_spec only applies to boltzgen_redesign mode.")
            }
            if (!params.bg_redesign_spec.matches(REDESIGN_SPEC_REGEX)) {
                throw new IllegalArgumentException("bg_redesign_spec format invalid. Must be an ordered list of chain-A keep tokens ('A<start>-<end>') and/or bare insert-count tokens ('<n>' or '<min>-<max>'), e.g. '7-10,A1-60,5,A70-100,10'.")
            }
        }
        if (params.bg_redesign_inpaint_seq) {
            if (params.design_mode == 'boltzgen_denovo') {
                throw new IllegalArgumentException("bg_redesign_inpaint_seq only applies to boltzgen_redesign mode.")
            }
            if (!params.bg_redesign_inpaint_seq.matches(CHAIN_A_SPEC_REGEX)) {
                throw new IllegalArgumentException("bg_redesign_inpaint_seq format invalid. Must reference chain A only, e.g. 'A10-50,A60'.")
            }
        }
        if (params.design_mode == 'boltzgen_redesign' && !params.bg_redesign_spec && !params.bg_redesign_inpaint_seq && !params.bg_flexible_residues) {
            throw new IllegalArgumentException("boltzgen_redesign mode requires at least one of bg_redesign_spec, bg_redesign_inpaint_seq, or bg_flexible_residues to be set - otherwise chain A would be left completely unchanged (wasted computation).")
        }
    }

    List<String> startupMessages() {
        def messages = []
        if (params.design_mode == 'boltzgen_denovo') {
            messages << (isBinderMode ? "Using BoltzGen to diffuse binders with the following design parameters:" : "Using BoltzGen to diffuse monomers with the following design parameters:")
            messages << "* Design length = ${params.design_length}"
        } else {
            messages << (isBinderMode ? "Using BoltzGen to redesign an existing binder with the following design parameters:" : "Using BoltzGen to redesign an existing monomer with the following design parameters:")
        }
        if (params.hotspot_residues) {
            messages << "* Target hotspots = ${params.hotspot_residues}"
        }
        if (params.bg_not_binding_residues) {
            messages << "* Target anti-hotspots = ${params.bg_not_binding_residues}"
        }
        if (params.design_mode == 'boltzgen_redesign' && params.bg_redesign_spec) {
            messages << "* Redesign spec = ${params.bg_redesign_spec}"
        }
        if (params.design_mode == 'boltzgen_redesign' && params.bg_redesign_inpaint_seq) {
            messages << "* Redesign inpaint seq = ${params.bg_redesign_inpaint_seq}"
        }
        if (params.bg_flexible_residues) {
            messages << "* Flexible residues = ${params.bg_flexible_residues}"
        }
        return messages
    }

    List<String> collectInputFiles() {
        return params.input_pdb ? [params.input_pdb] : []
    }
}
