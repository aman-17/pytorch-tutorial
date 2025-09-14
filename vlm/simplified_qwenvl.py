import torch 
import torch.nn as nn
from vit import VisualAdapter, ViT

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()

    def forward(self, x, attn_mask=None):
        normed_x = self.norm1(x)
        attn_output, _ = self.self_attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)
        x = x + attn_output
        ff = self.linear2(self.activation(self.linear1(self.norm2(x))))
        x = x + ff
        return x

class SimplifiedQwenLLM(nn.Module):
    def __init__(self, vocab_size=151936, hidden_size=4096, num_layers=32, num_heads=32, intermediate_size=11008, max_position_embeddings=32768):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs_embeds, attention_mask=None):
        seq_length = inputs_embeds.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.size(0), -1)
        position_embeds = self.position_embeddings(position_ids)
        x = inputs_embeds + position_embeds
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
    
class QwenVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = ViT(output_dim=1408)
        self.adapter = VisualAdapter()
        self.llm = SimplifiedQwenLLM()

    def forward(self, images, text_embeds, attention_mask=None):
        # In real usage, text_embeds come from embed_tokens on input_ids
        # Fusion: Adapt visual features and insert/concatenate into text sequence
        visual_features = self.vision_encoder(images)
        adapted_visual = self.adapter(visual_features)
        # Simplified fusion: concatenate visual to text embeds (in practice, replace special tokens)
        combined_embeds = torch.cat((text_embeds, adapted_visual), dim=1)
        if attention_mask is not None:
            visual_mask = torch.ones((adapted_visual.size(0), adapted_visual.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat((attention_mask, visual_mask), dim=1)
        logits = self.llm(combined_embeds, attention_mask)
        return logits