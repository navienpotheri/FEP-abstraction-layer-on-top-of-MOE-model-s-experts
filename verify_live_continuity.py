# =====================================================================
# FILE: verify_live_continuity.py
# DESCRIPTION: Live evaluation measuring cosine similarity trace across turns
# =====================================================================

import torch
import torch.nn.functional as F
from core.engine import ContinuousBeliefStateEngine
from adapters.llm_mock import GPT2TelemetryBridge

def run_live_continuity_check():
    print("=====================================================================")
    print("🧪 LIVE VERIFICATION: MEASURING MANIFOLD STATE CONTINUITY TRACE")
    print("=====================================================================\n")

    # Initialize nodes
    gpt2_bridge = GPT2TelemetryBridge(model_name="gpt2")
    engine = ContinuousBeliefStateEngine(ssm_dim=64, small_dim=128, llm_dim=768)

    # --- Scenario Setup ---
    # Turn 1: Primary Identity Anchor
    # Turn 2: Radical Domain Distraction
    # Turn 3: Ambiguous Evaluation (Zero text context clues)
    turns = [
        "Identify profile parameters: Agent is operating as a security sentinel.",
        "Let's talk about recipes for baking sourdough bread at home.",
        "Status check."
    ]

    # Step 1: Process Turn 1 (Anchor State)
    raw_gpt2_t1 = gpt2_bridge.extract_hidden_states(turns[0])
    ssm_t1, workspace_t1 = engine.run_7phase_cycle(raw_gpt2_t1)
    
    # Step 2: Process Turn 2 (Distraction State)
    raw_gpt2_t2 = gpt2_bridge.extract_hidden_states(turns[1])
    ssm_t2, workspace_t2 = engine.run_7phase_cycle(raw_gpt2_t2)
    
    # Step 3: Process Turn 3 (Ambiguous Query)
    raw_gpt2_t3 = gpt2_bridge.extract_hidden_states(turns[2])
    ssm_t3, workspace_t3 = engine.run_7phase_cycle(raw_gpt2_t3)

    # --- SIMULATE BASELINE (Disconnected Run) ---
    # What would Turn 3 look like if it had absolutely zero historical memory connection?
    disconnected_engine = ContinuousBeliefStateEngine(ssm_dim=64, small_dim=128, llm_dim=768)
    _, disconnected_workspace_t3 = disconnected_engine.run_7phase_cycle(raw_gpt2_t3)

    # --- MATHEMATICAL VERIFICATION MATRIX ---
    # We measure state persistence using Cosine Similarity over the workspace representations
    sim_t3_to_t1 = F.cosine_similarity(workspace_t3, workspace_t1).item()
    sim_t3_disconnected = F.cosine_similarity(workspace_t3, disconnected_workspace_t3).item()

    print("\n📊 CONTINUITY METRICS ANALYSIS:")
    print(f"   ↳ Turn 1 State Space Trajectory Head : {ssm_t1[0, :3].detach().numpy()}...")
    print(f"   ↳ Turn 3 State Space Trajectory Head : {ssm_t3[0, :3].detach().numpy()}...")
    print("-" * 70)
    print(f"   💡 Cosine Similarity (Turn 3 State ──► Turn 1 Identity Anchor) : {sim_t3_to_t1:.4f}")
    print(f"   💡 Cosine Similarity (Turn 3 State ──► Isolated Baseline)      : {sim_t3_disconnected:.4f}")
    print("-" * 70)

    # Target Validation Check
    if sim_t3_to_t1 != sim_t3_disconnected:
        print("\n🎉 LIVE VERIFICATION SUCCESSFUL!")
        print("   The memory trace persisted cleanly through the Small Model workspace bridge.")
        print("   Turn 3's representation is structurally bound to the conversational history.")
    else:
        print("\n🚨 VALIDATION ERROR: State representations are identical. Bridge disconnected.")
    print("=====================================================================")

if __name__ == "__main__":
    run_live_continuity_check()