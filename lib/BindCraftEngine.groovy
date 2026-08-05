class BindCraftEngine extends DesignEngine {

    BindCraftEngine(Map params, boolean isBinderMode) {
        super(params, isBinderMode)
    }

    void validateParams() {
        Utils.validateDesignLength(params.design_length)

        if (params.bc_chains && !params.bc_chains.matches(/^([A-Za-z]+)(,[A-Za-z]+)*$/)) {
            throw new IllegalArgumentException("bc_chains parameter must be a comma-separated list of chain identifiers, e.g. 'A,B' or 'A'.")
        }

        def hotspot = params.hotspot_residues
        if (hotspot && !hotspot.matches(/^([A-Za-z]*[0-9]+([\-][0-9]+)?)(,[A-Za-z]*[0-9]+([\-][0-9]+)?)*$|^([A-Za-z]+,?)*$/)) {
            throw new IllegalArgumentException("hotspot_residues format invalid. Acceptable: '1,2-10', 'A10,A12,B2-10', or chains 'A,B'.")
        }

        if (params.bc_advanced_json && !new File(params.bc_advanced_json.toString()).exists()) {
            throw new FileNotFoundException("Advanced settings JSON file not found at path: ${params.bc_advanced_json}. Please ensure the file exists and the path is correct.")
        }
    }

    List<String> startupMessages() {
        def messages = ["Using BindCraft to hallucinate binders with the following design parameters:"]
        messages << "* Design length = ${params.design_length}"
        if (params.bc_chains) {
            messages << "* Target chains = ${params.bc_chains}"
        }
        if (params.hotspot_residues) {
            messages << "* Target hotspots = ${params.hotspot_residues}"
        }
        return messages
    }

    List<String> collectInputFiles() {
        def inputs = [params.input_pdb]
        if (params.bc_advanced_json) {
            inputs << params.bc_advanced_json
        }
        return inputs
    }

    // Relative path (under lib/bindcraft/settings_advanced/) of the advanced settings JSON
    // implied by bc_design_protocol/bc_template_protocol, when bc_advanced_json isn't overridden.
    String advancedSettingsRelativePath() {
        def designProtocolTag
        switch (params.bc_design_protocol) {
            case "default":
                designProtocolTag = "default_4stage_multimer"
                break
            case "betasheet":
                designProtocolTag = "betasheet_4stage_multimer"
                break
            case "peptide":
                designProtocolTag = "peptide_3stage_multimer"
                break
            default:
                throw new IllegalArgumentException("Unsupported BindCraft design protocol: ${params.bc_design_protocol}")
        }

        def templateProtocolTag
        switch (params.bc_template_protocol) {
            case "default":
                templateProtocolTag = ""
                break
            case "flexible":
                templateProtocolTag = "_flexible"
                break
            default:
                throw new IllegalArgumentException("Unsupported BindCraft template protocol: ${params.bc_template_protocol}")
        }

        return "${designProtocolTag}${templateProtocolTag}.json"
    }
}
