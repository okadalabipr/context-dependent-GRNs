# This script generates executable mathematical models using BioMathForge (biomathforge).
# For installation and detailed usage, please refer to: https://github.com/okada-lab/BioMathForge

# Note: Efforts have been made to ensure reproducibility as much as possible, 
# but since web search results depend on the date of retrieval, full reproducibility is not guaranteed.
# The complete set of results is provided in `data/fig7/models`.

import json
from dotenv import load_dotenv
import pandas as pd

from biomathforge import (
        generate_formatted_reactions,
        run_pathway_analysis,
        integrate_reactions,
        run_enhance_feedback_crosstalk,
        finalize_reactions
)

#### Step 1. Generate Formatted Reactions
# Convert raw equations to a standardized format using the `generate_formatted_reactions` function.
biomodels_reactions = pd.read_csv("../data/fig7/ranked_biomodels_reactions_weight10.csv")
validated_reactions = generate_formatted_reactions(biomodels_reactions)
with open("../data/fig7/models/step1_formatted_reactions.txt", "w") as f:
    for reaction in validated_reactions:
        f.write(f"{reaction}\n")


#### Step 2. Analyze Pathways with Web Search
# Use web-based research to identify key signaling pathways and expected readouts under experimental conditions.
report = run_pathway_analysis(
    reactions_path="../data/fig7/models/step1_formatted_reactions.txt",
    condition_path="../data/fig7/models/experimental_condition.txt"
)

equations = [eq.strip() for eq in open("../data/fig7/models/step1_formatted_reactions.txt") if eq.strip()]
experimental_condition = open("../data/fig7/models/experimental_condition.txt").read().strip()
output_data = {
    "Reaction Equations": equations,
    "Experimental Condition": experimental_condition,
    "Main Signaling Pathway": report.main_signaling_pathway,
    "Expected Readouts": report.expected_readouts
}

with open("../data/fig7/models/step2_pathway_analysis_result.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)


#### Step 3. Integrate Reactions
# Integrate the equations into a complete network using biological constraints and inferred readouts.
report = json.loads(open("../data/fig7/models/step2_pathway_analysis_result.json").read()) # as dict
integrated_equations, source_nodes, sink_nodes = integrate_reactions(equations, report)

with open("../data/fig7/models/step3_integrated_reactions.txt", "w") as f:
    for eq in integrated_equations:
        f.write(f"{eq}\n")

terminal_nodes = {"source": ", ".join(source_nodes), "sink": ", ".join(sink_nodes)}
with open("../data/fig7/models/step3_terminal_nodes.json", "w") as f:
    json.dump(terminal_nodes, f, indent=4, ensure_ascii=False)


#### Step 4. Enhance Feedback and Crosstalk
# Add plausible feedback loops and crosstalk reactions to improve biological realism.
report, enhancement_summary, added_reactions = run_enhance_feedback_crosstalk(
    reactions_path="../data/fig7/models/step3_integrated_reactions.txt",
    terminal_nodes_path="../data/fig7/models/step3_terminal_nodes.json",
    reactions_overviews_path="../data/fig7/models/step2_pathway_analysis_result.json"
)
with open("../data/fig7/models/step4_enhanced_reactions.txt", "w") as f:
        f.write(report)


#### Step 5. Finalize the Model
# Clean up, deduplicate, and finalize the set of reactions for downstream use (e.g., simulation or export).
equations = [eq.strip() for eq in open("../data/fig7/models/step4_enhanced_reactions.txt") if eq.strip()]
finalized_equations = finalize_reactions(equations)
with open("../data/fig7/models/step5_breast_cancer_reactions_finalized.txt", "w") as f:
    for eq in finalized_equations:
        f.write(f"{eq}\n")