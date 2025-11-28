"""
Geometric Masking Module
Implements various geometric masking techniques for data augmentation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class GeometricMasking:
    """
    Applies geometric masking to time-series sequences
    Expands training data and improves model robustness
    """
    
    def __init__(self, 
                 mask_ratio: float = 0.15,
                 mask_type: str = 'random',
                 min_mask_length: int = 1,
                 max_mask_length: int = 10):
        """
        Args:
            mask_ratio: Proportion of features to mask
            mask_type: 'random', 'block', 'geometric', or 'feature'
            min_mask_length: Minimum length of masked block
            max_mask_length: Maximum length of masked block
        """
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.min_mask_length = min_mask_length
        self.max_mask_length = max_mask_length
    
    def random_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random masking: randomly mask features across time and dimensions
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            masked_x: Masked tensor
            mask: Binary mask (1 = keep, 0 = mask)
        """
        batch_size, seq_len, features = x.shape
        
        # Calculate number of elements to mask
        total_elements = batch_size * seq_len * features
        num_mask = int(total_elements * self.mask_ratio)
        
        # Create mask without in-place operations
        # Create a boolean tensor to identify which positions to mask
        indices = torch.randperm(total_elements, device=x.device)[:num_mask]
        # Create mask using zeros and scatter (non-in-place)
        mask_flat = torch.zeros(total_elements, device=x.device, dtype=x.dtype)
        mask_flat = mask_flat.scatter(0, indices, 1.0)  # scatter creates new tensor
        # Invert: 1 where we want to keep, 0 where we want to mask
        mask_flat = 1.0 - mask_flat
        mask = mask_flat.reshape(batch_size, seq_len, features)
        
        # Apply mask (set masked values to 0)
        masked_x = x * mask
        
        return masked_x, mask
    
    def block_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Block masking: mask contiguous blocks in time dimension
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            masked_x: Masked tensor
            mask: Binary mask
        """
        batch_size, seq_len, features = x.shape
        mask = torch.ones(batch_size, seq_len, features, device=x.device, dtype=x.dtype)
        
        for b in range(batch_size):
            num_blocks = int(seq_len * self.mask_ratio / self.max_mask_length) + 1
            
            for _ in range(num_blocks):
                block_length = np.random.randint(self.min_mask_length, 
                                                min(self.max_mask_length + 1, seq_len))
                start_idx = np.random.randint(0, max(1, seq_len - block_length))
                end_idx = min(start_idx + block_length, seq_len)
                
                # Randomly mask some features in this block
                feature_mask = torch.rand(features, device=x.device) < self.mask_ratio
                # Use clone and assignment to avoid in-place issues
                mask_b_slice = mask[b, start_idx:end_idx, :].clone()
                mask_b_slice[:, feature_mask] = 0
                mask[b, start_idx:end_idx, :] = mask_b_slice
        
        masked_x = x * mask
        return masked_x, mask
    
    def geometric_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Geometric masking: mask geometric patterns (rectangles, lines)
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            masked_x: Masked tensor
            mask: Binary mask
        """
        batch_size, seq_len, features = x.shape
        mask = torch.ones(batch_size, seq_len, features, device=x.device, dtype=x.dtype)
        
        for b in range(batch_size):
            # Create rectangular mask regions
            num_regions = int(seq_len * features * self.mask_ratio / 20) + 1
            
            for _ in range(num_regions):
                # Random rectangle
                t_start = np.random.randint(0, seq_len)
                t_end = min(t_start + np.random.randint(1, seq_len // 4), seq_len)
                f_start = np.random.randint(0, features)
                f_end = min(f_start + np.random.randint(1, features // 2), features)
                
                # Use clone to avoid in-place operations
                mask_b_slice = mask[b, t_start:t_end, f_start:f_end].clone()
                mask_b_slice[:] = 0
                mask[b, t_start:t_end, f_start:f_end] = mask_b_slice
        
        masked_x = x * mask
        return masked_x, mask
    
    def feature_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feature masking: mask entire feature dimensions
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            masked_x: Masked tensor
            mask: Binary mask
        """
        batch_size, seq_len, features = x.shape
        mask = torch.ones(batch_size, seq_len, features, device=x.device, dtype=x.dtype)
        
        # Select features to mask
        num_features_to_mask = max(1, int(features * self.mask_ratio))
        features_to_mask = np.random.choice(features, num_features_to_mask, replace=False)
        
        # Use clone to avoid in-place operations
        mask_clone = mask.clone()
        mask_clone[:, :, features_to_mask] = 0
        mask = mask_clone
        masked_x = x * mask
        
        return masked_x, mask
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking based on mask_type
        
        Args:
            x: Input tensor (batch, seq_len, features)
            
        Returns:
            masked_x: Masked tensor
            mask: Binary mask
        """
        if self.mask_type == 'random':
            return self.random_mask(x)
        elif self.mask_type == 'block':
            return self.block_mask(x)
        elif self.mask_type == 'geometric':
            return self.geometric_mask(x)
        elif self.mask_type == 'feature':
            return self.feature_mask(x)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")
