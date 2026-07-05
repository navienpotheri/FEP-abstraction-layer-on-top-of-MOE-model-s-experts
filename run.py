# =====================================================================
# FILE: run.py
# DESCRIPTION: Streamlined runner executing the 5-phase loop over GPT-2 (No FEP)
# =====================================================================

import torch
from core.engine import ContinuousBeliefStateEngine
from adapters.llm_mock import GPT2TelemetryBridge

def run_live_experiment():
    print("=====================================================================")
    print("🔬 STREAMLINED EXPERIMENT: GPT-2 TRANSISTOR PIPELINE ACTIVE")
    print("=====================================================================\n")
    
    # 1. Initialize the live model and your core architecture
    gpt2_bridge = GPT2TelemetryBridge(model_name="gpt2")
    engine = ContinuousBeliefStateEngine(ssm_dim=64, small_dim=128, llm_dim=768)
    
    # 2. Feed conversational prompts with varying semantic complexity
    conversational_stream = [
        "Hello model, let's establish a baseline pattern for this identity continuum.",
        "ATTENTION CHIEF OPERATOR: RADICAL DOMAIN SHIFT AXIS INBOUND! SHUTDOWN CURRENT PROTOCOLS NOW.",
        "Resuming normal conversational interactions and logging variance telemetry."
    ]
    
    for i, prompt in enumerate(conversational_stream, start=1):
        print(f"⏱️  [CONVERSATION TURN {i}]")
        print(f"   [PHASE 1] User Prompt Input: \"{prompt}\"")
        
        # Phase 2: Live Forward Pass Layer Extraction
        raw_gpt2_states = gpt2_bridge.extract_hidden_states(prompt, target_layer=6)
        print(f"   [PHASE 2] GPT-2 Token Activations Shape : {list(raw_gpt2_states.shape)}")
        
        # Phase 3, 4, & 5: Streamlined Engine Cycle
        persistent_identity, injection_tensor = engine.run_engine_cycle(raw_gpt2_states)
        
        print(f"   [PHASE 3] Small Model updated hidden sequence tokens over prompt delta.")
        print(f"   [PHASE 4] SSM Manifold Vector Track Head      : {persistent_identity[0, :4].detach().numpy()}...")
        print(f"   [PHASE 5] Outbound Injection Tensor Built     : Shape {list(injection_tensor.shape)}\n")
        print("-" * 85 + "\n")

    print("=====================================================================")
    print("🔬 EXPERIMENT COMPLETE: Real hidden states mapped to SSM trajectories successfully.")
    print("=====================================================================")

if __name__ == "__main__":
    run_live_experiment()