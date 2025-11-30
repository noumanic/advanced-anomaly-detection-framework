"""
Training Module
Handles model training with all loss components
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import os

from .model import AnomalyDetectionModel
from .geometric_masking import GeometricMasking


class Trainer:
    """
    Trainer for anomaly detection model
    """
    
    def __init__(self,
                 model: AnomalyDetectionModel,
                 device: torch.device,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-5,
                 gan_weight: float = 0.1,
                 contrastive_weight: float = 0.1,
                 reconstruction_weight: float = 1.0):
        """
        Args:
            model: Anomaly detection model
            device: Training device
            lr: Learning rate
            weight_decay: Weight decay
            gan_weight: Weight for GAN loss
            contrastive_weight: Weight for contrastive loss
            reconstruction_weight: Weight for reconstruction loss
        """
        self.model = model.to(device)
        self.device = device
        
        # Enable cuDNN benchmarking for faster training (if using GPU)
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Faster, but less reproducible
        
        # Optimizers
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
            # GAN optimizers (if using GAN)
        # Use lower learning rates for GAN to prevent instability
        if model.use_gan:
            self.gen_optimizer = optim.Adam(
                model.gan.generator.parameters(),
                lr=lr * 0.1,  # Much lower LR for generator
                betas=(0.5, 0.999)
            )
            self.disc_optimizer = optim.Adam(
                model.gan.discriminator.parameters(),
                lr=lr * 0.2,  # Lower LR for discriminator
                betas=(0.5, 0.999)
            )
        
        # Loss weights (adjusted for better balance)
        # GAN loss is raw BCE, so needs much smaller weight
        self.gan_weight = gan_weight * 0.1  # Scale down GAN weight
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        
        # Geometric masking
        self.geometric_masking = GeometricMasking(
            mask_ratio=0.15,
            mask_type='random'
        )
        
        # Contrastive module reference
        self.contrastive_module = model.contrastive if model.use_contrastive else None
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_gan_loss = 0.0
        total_contrastive_loss = 0.0
        total_disc_loss = 0.0
        
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            batch_size = x.size(0)
            
            # Apply geometric masking
            x_masked, mask = self.geometric_masking(x)
            
            # Forward pass
            outputs = self.model(x_masked, mask=None)
            
            reconstructed = outputs['reconstructed']
            encoded = outputs['encoded']
            
            # Reconstruction loss
            recon_loss = self.model.compute_reconstruction_loss(reconstructed, x)
            
            # Main model loss (reconstruction + contrastive)
            main_loss = self.reconstruction_weight * recon_loss
            
            # Contrastive learning loss
            contrastive_loss = None
            gen_loss = None
            
            if self.model.use_contrastive:
                # Improved contrastive sampling strategy
                # Positive: augmented version with stronger perturbation
                # Use geometric masking as positive augmentation
                x_positive, _ = self.geometric_masking(x.clone())
                outputs_positive = self.model(x_positive)
                
                # Negative: create more distinct negative samples
                # Option 1: Time-shifted (creates temporal anomalies)
                shift = self.model.seq_len // 4
                x_negative = torch.roll(x, shifts=shift, dims=1)
                # Option 2: Feature-shuffled (creates feature anomalies)
                perm = torch.randperm(x.size(2), device=self.device)
                x_negative = x[:, :, perm]
                # Option 3: Add noise to create anomalies
                x_negative = x_negative + torch.randn_like(x_negative) * 0.3
                
                outputs_negative = self.model(x_negative)
                
                # Get contrastive embeddings
                # Anchor: current encoded sequence (normal)
                # Positive: augmented normal (should be similar)
                # Negative: anomalous (should be different)
                anchor_emb, _ = self.model.contrastive(encoded, encoded)  # (batch, proj_dim)
                positive_emb, _ = self.model.contrastive(
                    encoded, outputs_positive['encoded']
                )  # (batch, proj_dim)
                _, negative_emb = self.model.contrastive(
                    encoded, outputs_negative['encoded']
                )  # (batch, proj_dim)
                
                # Compute contrastive loss
                contrastive_loss = self.model.contrastive.compute_loss(
                    anchor_emb, positive_emb, negative_emb
                )
                
                main_loss = main_loss + self.contrastive_weight * contrastive_loss
                total_contrastive_loss += contrastive_loss.item()
            
            # GAN training (separate from main model)
            # Train discriminator less frequently to balance with generator
            if self.model.use_gan:
                # Train discriminator (every batch)
                self.disc_optimizer.zero_grad()
                
                # Real data - use label smoothing for stability
                real_pred = self.model.gan.discriminate(x.detach())
                real_target = torch.ones_like(real_pred, device=self.device) * 0.9  # Label smoothing
                real_loss = self.bce_loss(real_pred, real_target)
                
                # Generated data
                fake_data = self.model.gan.generate(batch_size, self.device)
                fake_pred = self.model.gan.discriminate(fake_data.detach())
                fake_target = torch.ones_like(fake_pred, device=self.device) * 0.1  # Label smoothing
                fake_loss = self.bce_loss(fake_pred, fake_target)
                
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                # Clip discriminator gradients
                torch.nn.utils.clip_grad_norm_(self.model.gan.discriminator.parameters(), max_norm=1.0)
                self.disc_optimizer.step()
                
                total_disc_loss += disc_loss.item()
                
                # Train generator (every batch, but with lower weight)
                # Only train generator if discriminator is not too strong
                if disc_loss.item() < 1.5:  # Only train gen if disc is not dominating
                    self.gen_optimizer.zero_grad()
                    
                    # Generate new fake data for generator training
                    fake_data_gen = self.model.gan.generate(batch_size, self.device)
                    fake_pred_gen = self.model.gan.discriminate(fake_data_gen)
                    gen_target = torch.ones_like(fake_pred_gen, device=self.device) * 0.9  # Label smoothing
                    gen_loss = self.bce_loss(fake_pred_gen, gen_target)
                    
                    gen_loss.backward()
                    # Clip generator gradients
                    torch.nn.utils.clip_grad_norm_(self.model.gan.generator.parameters(), max_norm=1.0)
                    self.gen_optimizer.step()
                    
                    total_gan_loss += gen_loss.item()
                    gen_loss_value = gen_loss.item()
                else:
                    # Discriminator too strong, skip generator update
                    gen_loss_value = 0.0
                    total_gan_loss += 0.0
                    gen_loss = torch.tensor(0.0, device=self.device)
            
            # Backward pass for main model (reconstruction + contrastive)
            self.optimizer.zero_grad()
            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Total loss for logging (includes all components)
            # Don't add GAN loss to main loss - it's trained separately
            total_loss_batch = main_loss.item()
            # GAN loss is tracked separately, not added to main loss
            
            # Accumulate losses
            total_loss += total_loss_batch
            total_recon_loss += recon_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'recon': total_recon_loss / num_batches,
                'gan': total_gan_loss / num_batches if self.model.use_gan else 0,
                'contrastive': total_contrastive_loss / num_batches if self.model.use_contrastive else 0
            })
        
        return {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'gan_loss': total_gan_loss / num_batches if self.model.use_gan else 0.0,
            'contrastive_loss': total_contrastive_loss / num_batches if self.model.use_contrastive else 0.0,
            'discriminator_loss': total_disc_loss / num_batches if self.model.use_gan else 0.0
        }
    
    def train(self, 
              train_loader: DataLoader,
              num_epochs: int,
              save_dir: Optional[str] = None,
              save_every: int = 10) -> Dict[str, list]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history
        """
        history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'gan_loss': [],
            'contrastive_loss': [],
            'discriminator_loss': []
        }
        
        for epoch in range(1, num_epochs + 1):
            losses = self.train_epoch(train_loader, epoch)
            
            for key in history:
                history[key].append(losses.get(key, 0.0))
            
            # Save checkpoint
            if save_dir and epoch % save_every == 0:
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': history
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        return history
    
    def save_model(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'seq_len': self.model.seq_len,
                'use_gan': self.model.use_gan,
                'use_contrastive': self.model.use_contrastive
            }
        }, path)
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

