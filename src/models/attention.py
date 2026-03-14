"""
Attention mechanisms for multi-modal fusion in AQI prediction.
Implements various attention strategies for combining satellite and street view features.
"""

import torch
import torch.nn as nn


class SigmoidGatedFusion(nn.Module):
    """Independent Sigmoid-based gating for each input and then concatenates."""
    def __init__(self, in_dim1, in_dim2):
        super(SigmoidGatedFusion, self).__init__()
        self.attn1 = nn.Linear(in_dim1, 1)  # Produces weighted features
        self.attn2 = nn.Linear(in_dim2, 1)

    def forward(self, x1, x2):
        weight1 = torch.sigmoid(self.attn1(x1))  
        weight2 = torch.sigmoid(self.attn2(x2))  
        fused = torch.cat((weight1 * x1, weight2 * x2), dim=1)  # Concatenate weighted features
        return fused


class SoftmaxGatedFusion(nn.Module):
    """Softmax-based gating where attention weights sum to 1."""
    def __init__(self, in_dim1, in_dim2):
        super(SoftmaxGatedFusion, self).__init__()
        self.gate = nn.Linear(in_dim1 + in_dim2, 2)  # Two attention scores

    def forward(self, x1, x2):
        combined_features = torch.cat((x1, x2), dim=1)  
        weights = torch.softmax(self.gate(combined_features), dim=1)  # Softmax over two inputs
        weighted_x1 = weights[:, 0:1] * x1
        weighted_x2 = weights[:, 1:2] * x2
        fused = torch.cat((weighted_x1, weighted_x2), dim=1)  # Concatenate weighted features
        return fused


class CrossAttention(nn.Module):
    """Multihead Cross Attention treating one input as query and other as key/value."""
    def __init__(self, in_dim1, in_dim2, shared_dim=256, num_heads=4):
        super(CrossAttention, self).__init__()
        
        # Project both inputs to a shared dimension
        self.query_proj = nn.Linear(in_dim1, shared_dim)
        self.key_proj = nn.Linear(in_dim2, shared_dim)
        self.value_proj = nn.Linear(in_dim2, shared_dim)

        # Multihead attention with consistent dimensions
        self.attn = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=num_heads, batch_first=True)
        
        # Add projection layer to __init__ instead of forward method
        self.projection = nn.Linear(shared_dim + in_dim1 + in_dim2, 1024)

    def forward(self, x1, x2):
        # Project to shared dimension
        q = self.query_proj(x1).unsqueeze(1)  # (batch_size, 1, shared_dim)
        k = self.key_proj(x2).unsqueeze(1)    # (batch_size, 1, shared_dim)
        v = self.value_proj(x2).unsqueeze(1)  # (batch_size, 1, shared_dim)
    
        # Attention
        attn_output, _ = self.attn(q, k, v)
        attn_output = attn_output.squeeze(1)
    
        # Ensure consistent concatenation
        fused = torch.cat((attn_output, x1, x2), dim=1)
    
        # Use the projection layer from __init__
        fused = self.projection(fused)
    
        return fused


def get_attention_module(attention_type, in_dim1, in_dim2, num_heads=4):
    """
    Factory method to return different attention mechanisms for concatenated features.

    Args:
    - attention_type (str): Type of attention ("sigmoid_gated", "softmax_gated", "cross")
    - in_dim1 (int): Feature dimension of input 1
    - in_dim2 (int): Feature dimension of input 2
    - num_heads (int): Number of heads (only for cross attention)

    Returns:
    - nn.Module: The selected attention mechanism
    """
    if attention_type == "sigmoid_gated":
        return SigmoidGatedFusion(in_dim1, in_dim2)
    elif attention_type == "softmax_gated":
        return SoftmaxGatedFusion(in_dim1, in_dim2)
    elif attention_type == "cross":
        return CrossAttention(in_dim1, in_dim2, num_heads=num_heads)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
