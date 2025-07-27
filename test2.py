import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

class DeepMLPCombiner(nn.Module):
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int, num_layers: int = 3, num_attn_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.text_proj = nn.Linear(clip_feature_dim, projection_dim)
        self.image_proj = nn.Linear(clip_feature_dim, projection_dim)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=projection_dim * 2, num_heads=num_heads, batch_first=True)
            for _ in range(num_attn_layers)
        ])
        layers = []
        input_dim = projection_dim * 2
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def combine_features(self, image_features, text_features):
        img_proj = F.relu(self.image_proj(image_features))
        txt_proj = F.relu(self.text_proj(text_features))
        fused = torch.cat([img_proj, txt_proj], dim=-1)  # [B, 2*proj_dim]
        # Add sequence dimension for attention: [B, 1, 2*proj_dim]
        fused = fused.unsqueeze(1)
        for attn in self.attn_layers:
            fused, _ = attn(fused, fused, fused)
        fused = fused.squeeze(1)  # [B, 2*proj_dim]
        x = self.mlp(fused)
        out = self.output_layer(x)
        return F.normalize(out, dim=-1)

    def forward(self, image_features, text_features, target_features):
        pred = self.combine_features(image_features, text_features)
        target_features = F.normalize(target_features, dim=-1)
        logits = self.logit_scale.exp() * pred @ target_features.T
        return logits

