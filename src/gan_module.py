"""
GAN Module for Regularization
Implements Generator and Discriminator for GAN-based regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Generator(nn.Module):
    """
    Generator network for GAN
    Learns to generate realistic normal time-series patterns
    """
    
    def __init__(self, 
                 latent_dim: int = 100,
                 seq_len: int = 100,
                 output_dim: int = 7,
                 hidden_dim: int = 256):
        """
        Args:
            latent_dim: Dimension of latent noise vector
            seq_len: Sequence length
            output_dim: Number of output features
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # Initial projection
        self.fc1 = nn.Linear(latent_dim, hidden_dim * seq_len // 4)
        
        # Convolutional layers for sequence generation
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim // 4, output_dim, kernel_size=3, stride=1, padding=1)
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(output_dim, hidden_dim // 2, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate time-series from noise
        
        Args:
            z: Noise tensor (batch, latent_dim)
            
        Returns:
            Generated sequence (batch, seq_len, output_dim)
        """
        batch_size = z.size(0)
        
        # Initial projection
        x = self.fc1(z)
        x = x.view(batch_size, -1, self.seq_len // 4)
        
        # Convolutional generation
        x = self.conv_layers(x)
        
        # Transpose for LSTM (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM refinement
        lstm_out, _ = self.lstm(x)
        x = self.fc_out(lstm_out)
        
        # No Tanh - let the data distribution determine the range
        # StandardScaler normalizes to mean=0, std=1, so we keep raw output
        return x


class Discriminator(nn.Module):
    """
    Discriminator network for GAN
    Distinguishes between real and generated time-series
    """
    
    def __init__(self,
                 seq_len: int = 100,
                 input_dim: int = 7,
                 hidden_dim: int = 128):
        """
        Args:
            seq_len: Sequence length
            input_dim: Number of input features
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
        )
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(hidden_dim * 4, hidden_dim, num_layers=2, 
                           batch_first=True, bidirectional=True)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify sequence as real or fake
        
        Args:
            x: Input sequence (batch, seq_len, input_dim)
            
        Returns:
            Probability of being real (batch, 1)
        """
        # Transpose for conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional features
        x = self.conv_layers(x)
        
        # Transpose back: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Classification
        x = self.fc(x)
        
        return x


class GANModule(nn.Module):
    """
    GAN module for regularization
    """
    
    def __init__(self,
                 seq_len: int = 100,
                 feature_dim: int = 7,
                 latent_dim: int = 100,
                 hidden_dim: int = 256):
        """
        Args:
            seq_len: Sequence length
            feature_dim: Number of features
            latent_dim: Generator latent dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.generator = Generator(latent_dim, seq_len, feature_dim, hidden_dim)
        self.discriminator = Discriminator(seq_len, feature_dim, hidden_dim // 2)
    
    def generate(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate synthetic sequences
        
        Args:
            batch_size: Batch size
            device: Device to generate on
            
        Returns:
            Generated sequences (batch, seq_len, feature_dim)
        """
        z = torch.randn(batch_size, self.generator.latent_dim, device=device)
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake
        
        Args:
            x: Input sequences (batch, seq_len, feature_dim)
            
        Returns:
            Real/fake probabilities (batch, 1)
        """
        return self.discriminator(x)

