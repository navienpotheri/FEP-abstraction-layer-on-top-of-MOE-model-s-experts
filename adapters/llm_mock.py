import torch
from transformers import GPT2Model, GPT2Tokenizer

class GPT2TelemetryBridge:
    def __init__(self, model_name="gpt2"):
        print(f"📡 Loading live {model_name} weights and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        
        # Freeze weights for clean forward pass telemetry
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        print("📡 GPT-2 frozen forward pass layers online.")

    def extract_hidden_states(self, user_prompt, target_layer=6):
        inputs = self.tokenizer(user_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[target_layer]