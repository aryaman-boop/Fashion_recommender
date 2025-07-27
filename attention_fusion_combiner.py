import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionCombiner(nn.Module):
    def __init__(self, feature_dim, projection_dim, hidden_dim):
        super(AttentionFusionCombiner, self).__init__()
        # Linear projections for image and text features
        self.image_proj = nn.Linear(feature_dim, hidden_dim)
        self.text_proj = nn.Linear(feature_dim, hidden_dim)
        # Attention weights
        self.attn = nn.Linear(hidden_dim * 2, 1)
        # Output projection - should output feature_dim to match CLIP features
        self.output_proj = nn.Linear(hidden_dim * 3, feature_dim)

    def forward(self, image_features, text_features):
        # Project features
        img_proj = F.relu(self.image_proj(image_features))
        txt_proj = F.relu(self.text_proj(text_features))
        # Concatenate and compute attention
        concat = torch.cat([img_proj, txt_proj], dim=-1)
        attn_weights = torch.sigmoid(self.attn(concat))
        # Weighted sum
        fused = attn_weights * img_proj + (1 - attn_weights) * txt_proj
        # Concatenate fused with both for richer representation
        combined = torch.cat([fused, concat], dim=-1)
        # Final projection - output feature_dim dimensions
        out = self.output_proj(combined)
        out = F.normalize(out, dim=-1)
        return out

    def combine_features(self, image_features, text_features):
        return self.forward(image_features, text_features) 