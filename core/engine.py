# =====================================================================
# FILE: core/engine.py
# DESCRIPTION: 7-Phase Dual-Model State Space Bridge Engine
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallModelWorkspace(nn.Module):
    """
    Phase 3, 6 & 7: Actively coordinates token features from the frozen LLM
    and long-term memory tracks from the SSM to establish state continuity.
    """
    def __init__(self, input_dim=128, hidden_dim=128, ssm_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_llm = nn.Linear(input_dim, hidden_dim)
        self.W_ssm = nn.Linear(ssm_dim, hidden_dim)
        self.W_recurrence = nn.Linear(hidden_dim, hidden_dim)
        
        self.cell_state = torch.zeros(1, hidden_dim)

    def forward(self, projected_tokens, ssm_distribution=None):
        # Reset local turn state
        self.cell_state = torch.zeros(1, self.hidden_dim)
        
        # Phase 5 & 6: If SSM feedback is provided, project and absorb it
        ssm_injection = self.W_ssm(ssm_distribution) if ssm_distribution is not None else torch.zeros(1, self.hidden_dim)
        
        # Recurrent token integration sequence loop
        for t in range(projected_tokens.size(1)):
            token_t = projected_tokens[:, t, :]
            # Phase 6: Absorb both vectors simultaneously into a unified hidden space
            combined_gate = self.W_llm(token_t) + self.W_recurrence(self.cell_state) + ssm_injection
            self.cell_state = torch.tanh(combined_gate)
            
        return self.cell_state


class ContinuousBeliefStateEngine(nn.Module):
    """
    Phase 4 & 5: State Persistence substrate tracking identity trajectories.
    """
    def __init__(self, ssm_dim=64, small_dim=128, llm_dim=768):
        super().__init__()
        self.ssm_dim = ssm_dim
        self.small_dim = small_dim
        
        # Projections & Workspace Modules
        self.llm_to_small_proj = nn.Linear(llm_dim, small_dim)
        self.workspace = SmallModelWorkspace(input_dim=small_dim, hidden_dim=small_dim, ssm_dim=ssm_dim)
        
        # Phase 4 Matrix: Drives transitions based on extracted hidden distributions
        self.ssm_transition = nn.Sequential(
            nn.Linear(small_dim + ssm_dim, 128),
            nn.Tanh(),
            nn.Linear(128, ssm_dim)
        )
        
        # Long-term trajectory vector tracking (s_t) on a stable unit hypersphere manifold
        self.s_t = F.normalize(torch.randn(1, ssm_dim), p=2, dim=-1)

    def run_7phase_cycle(self, raw_llm_hidden_states):
        # --- Phase 3: Small Model Workspace ---
        # Map raw activations from Phase 2 into the lower-dimensional workspace
        projected_tokens = self.llm_to_small_proj(raw_llm_hidden_states)
        
        # Base forward pass to extract intermediate tracking distribution
        base_workspace_state = self.workspace(projected_tokens, ssm_distribution=None)
        
        # --- Phase 4: SSM State Vector Update ---
        # Update long-term distribution coordinates using the extracted workspace features
        combined_context = torch.cat([self.s_t, base_workspace_state], dim=-1)
        trajectory_velocity = self.ssm_transition(combined_context)
        self.s_t = F.normalize(self.s_t + trajectory_velocity, p=2, dim=-1)
        
        # --- Phase 5 & 6: Feedback & Absorption ---
        # Send updated distribution back to the workspace; absorb SSM + LLM vectors
        conditioned_workspace_state = self.workspace(projected_tokens, ssm_distribution=self.s_t)
        
        # --- Phase 7: State Continuity Delivery ---
        # The conditioned workspace state is ready to hook into downstream heads or context generation
        return self.s_t, conditioned_workspace_state