"""
Contrastive Learning Module
Implements contrastive loss to separate normal and anomalous patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for anomaly detection
    Enforces separation between normal and anomalous patterns
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        """
        Args:
            temperature: Temperature parameter for softmax
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            anchor: Anchor embeddings (batch, d_model)
            positive: Positive (normal) embeddings (batch, d_model)
            negative: Negative (anomalous) embeddings (batch, d_model)
            
        Returns:
            loss: Contrastive loss value
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute similarities
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)
        
        # Contrastive loss: maximize pos_sim, minimize neg_sim
        # Using margin-based contrastive loss
        pos_loss = torch.mean((1 - pos_sim) ** 2)
        neg_loss = torch.mean(torch.clamp(neg_sim - self.margin, min=0) ** 2)
        
        loss = pos_loss + neg_loss
        
        return loss


class ContrastiveModule(nn.Module):
    """
    Contrastive learning module for anomaly detection
    """
    
    def __init__(self, d_model: int, projection_dim: int = 64):
        """
        Args:
            d_model: Input embedding dimension
            projection_dim: Projection dimension for contrastive learning
        """
        super().__init__()
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
        
        self.contrastive_loss = ContrastiveLoss()
    
    def forward(self, 
                normal_emb: torch.Tensor, 
                anomalous_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contrastive embeddings and loss
        
        Args:
            normal_emb: Normal sequence embeddings (batch, seq_len, d_model)
            anomalous_emb: Anomalous sequence embeddings (batch, seq_len, d_model)
            
        Returns:
            normal_proj: Projected normal embeddings
            anomalous_proj: Projected anomalous embeddings
        """
        # Average pooling over sequence dimension
        normal_pooled = torch.mean(normal_emb, dim=1)  # (batch, d_model)
        anomalous_pooled = torch.mean(anomalous_emb, dim=1)  # (batch, d_model)
        
        # Project to contrastive space
        normal_proj = self.projection(normal_pooled)
        anomalous_proj = self.projection(anomalous_pooled)
        
        return normal_proj, anomalous_proj
    
    def compute_loss(self, 
                     anchor: torch.Tensor,
                     positive: torch.Tensor,
                     negative: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            anchor: Anchor embeddings (batch, projection_dim)
            positive: Positive embeddings (batch, projection_dim)
            negative: Negative embeddings (batch, projection_dim)
        """
        return self.contrastive_loss(anchor, positive, negative)

