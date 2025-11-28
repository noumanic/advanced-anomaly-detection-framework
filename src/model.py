"""
Main Model Architecture
Integrates Transformer, Contrastive Learning, and GAN components
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

from .transformer_encoder import TransformerEncoder
from .contrastive_learning import ContrastiveModule
from .gan_module import GANModule


class AnomalyDetectionModel(nn.Module):
    """
    Complete anomaly detection model combining:
    - Transformer encoder for feature extraction
    - Contrastive learning for representation quality
    - GAN for regularization
    """
    
    def __init__(self,
                 input_dim: int = 7,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 seq_len: int = 100,
                 use_gan: bool = True,
                 use_contrastive: bool = True):
        """
        Args:
            input_dim: Number of input features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            seq_len: Sequence length
            use_gan: Whether to use GAN regularization
            use_contrastive: Whether to use contrastive learning
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.use_gan = use_gan
        self.use_contrastive = use_contrastive
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=seq_len * 2
        )
        
        # Contrastive learning module
        if use_contrastive:
            self.contrastive = ContrastiveModule(d_model=d_model)
        
        # GAN module
        if use_gan:
            self.gan = GANModule(
                seq_len=seq_len,
                feature_dim=input_dim,
                latent_dim=100,
                hidden_dim=256
            )
        
        # Anomaly scoring head
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary containing:
            - encoded: Transformer encodings
            - reconstructed: Reconstructed sequences
            - anomaly_scores: Anomaly scores
            - contrastive_emb: Contrastive embeddings (if enabled)
        """
        # Transformer encoding
        encoded, reconstructed = self.transformer(x, mask)
        
        # Anomaly scoring (using mean pooled encoding)
        pooled = torch.mean(encoded, dim=1)  # (batch, d_model)
        anomaly_scores = self.anomaly_scorer(pooled).squeeze(-1)  # (batch,)
        
        outputs = {
            'encoded': encoded,
            'reconstructed': reconstructed,
            'anomaly_scores': anomaly_scores
        }
        
        # Contrastive embeddings (if enabled)
        if self.use_contrastive and self.training:
            # For contrastive learning, we'll use this in the training loop
            outputs['pooled_emb'] = pooled
        
        return outputs
    
    def compute_reconstruction_loss(self, 
                                   reconstructed: torch.Tensor, 
                                   original: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE)
        
        Args:
            reconstructed: Reconstructed sequences
            original: Original sequences
            
        Returns:
            Reconstruction loss
        """
        return nn.functional.mse_loss(reconstructed, original)
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for input sequences
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            
        Returns:
            Anomaly scores (batch,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['anomaly_scores']

