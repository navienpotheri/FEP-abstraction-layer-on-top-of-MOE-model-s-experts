# =====================================================================
# FILE: adapters/llm_mock.py (GPT-2 Telemetry Bridge)
# DESCRIPTION: Hooks into live GPT-2 hidden layers at inference time
# =====================================================================

import torch
from transformers import GPT2Model, GPT2Tokenizer

class GPT2TelemetryBridge:
    def __init__(self, model_name="gpt2"):
        print(f"📡 Loading live {model_name} weights and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        
        # Freeze GPT-2 weights completely—no training overhead
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        print("📡 GPT-2 frozen forward pass layers online.")

    def extract_hidden_states(self, user_prompt, target_layer=6):
        """
        Phase 2 Pass: Tokenizes the input, runs a forward pass, and extracts
        the hidden states from a specific mid-transformer layer.
        GPT-2 hidden states shape: [batch_size, sequence_length, 768]
        """
        # Tokenize user prompt inputs
        inputs = self.tokenizer(user_prompt, return_tensors="pt")
        
        # Run forward pass with hidden state tracking enabled
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Extract target layer hidden states (GPT-2 has layers 0 to 12)
        # outputs.hidden_states[0] is the embedding layer; 1-12 are transformer blocks.
        raw_hidden_states = outputs.hidden_states[target_layer]
        
        return raw_hidden_states