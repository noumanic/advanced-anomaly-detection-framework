"""
Main Training Script
Trains the Advanced Anomaly Detection Framework
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from datetime import datetime

from src.data_preprocessing import DataPreprocessor
from src.model import AnomalyDetectionModel
from src.training import Trainer
from src.evaluation import Evaluator


def create_anomaly_labels(data: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """
    Create synthetic anomaly labels for evaluation
    Uses statistical methods to identify potential anomalies
    
    Args:
        data: Input sequences (N, seq_len, features)
        contamination: Expected proportion of anomalies
        
    Returns:
        Binary labels (1 = anomaly, 0 = normal)
    """
    # Compute reconstruction error using simple baseline
    mean = np.mean(data, axis=(0, 1), keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)
    
    # Z-score based anomaly detection
    z_scores = np.abs((data - mean) / (std + 1e-8))
    max_z_scores = np.max(z_scores, axis=(1, 2))
    
    # Threshold based on contamination rate
    threshold = np.percentile(max_z_scores, (1 - contamination) * 100)
    labels = (max_z_scores >= threshold).astype(int)
    
    return labels


def main():
    parser = argparse.ArgumentParser(description='Train Advanced Anomaly Detection Framework')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='dataset/LCL-June2015v2_94.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('--customer_id', type=str, default=None,
                       help='Specific customer ID to use (None for all)')
    parser.add_argument('--window_size', type=int, default=100,
                       help='Sequence window size')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sliding window')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/test split ratio')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=128,
                       help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--use_gan', action='store_true', default=True,
                       help='Use GAN regularization')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                       help='Use contrastive learning')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--gan_weight', type=float, default=0.01,
                       help='Weight for GAN loss (reduced for stability)')
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                       help='Weight for contrastive loss')
    parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and results')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use: auto (detect), cuda (force GPU), cpu (force CPU)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (if multiple GPUs available)')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Device selection with diagnostics
    print("="*60)
    print("Device Configuration")
    print("="*60)
    
    if args.device == 'cpu':
        device = torch.device('cpu')
        print("⚠️  Forcing CPU usage (--device cpu)")
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ ERROR: CUDA requested but not available!")
            print("   PyTorch was likely installed without CUDA support.")
            print("   Please install PyTorch with CUDA from: https://pytorch.org/get-started/locally/")
            print("   Falling back to CPU...")
            device = torch.device('cpu')
        else:
            if args.gpu_id >= torch.cuda.device_count():
                print(f"⚠️  GPU {args.gpu_id} not available. Using GPU 0 instead.")
                args.gpu_id = 0
            device = torch.device(f'cuda:{args.gpu_id}')
            print(f"✅ Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.2f} GB")
    else:  # auto
        if torch.cuda.is_available():
            if args.gpu_id >= torch.cuda.device_count():
                args.gpu_id = 0
            device = torch.device(f'cuda:{args.gpu_id}')
            print(f"✅ Auto-detected GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device('cpu')
            print("⚠️  No GPU detected. Using CPU.")
            print("   To use GPU, install PyTorch with CUDA: https://pytorch.org/get-started/locally/")
    
    print(f"Final device: {device}")
    print("="*60)
    
    # Data preprocessing
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor(
        window_size=args.window_size,
        stride=args.stride,
        normalization='standard',
        handle_missing='forward_fill'
    )
    
    X_train, X_test, train_indices, test_indices = preprocessor.prepare_data(
        args.data_path,
        customer_id=args.customer_id,
        train_split=args.train_split
    )
    
    print(f"Training sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")
    print(f"Feature dimension: {X_train.shape[2]}")
    
    # Create anomaly labels for evaluation (synthetic)
    # In real scenario, these would come from ground truth
    print("Creating anomaly labels for evaluation...")
    train_labels = create_anomaly_labels(X_train, contamination=0.1)
    test_labels = create_anomaly_labels(X_test, contamination=0.1)
    
    print(f"Training anomalies: {train_labels.sum()} / {len(train_labels)} ({train_labels.mean()*100:.2f}%)")
    print(f"Test anomalies: {test_labels.sum()} / {len(test_labels)} ({test_labels.mean()*100:.2f}%)")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(train_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(test_labels)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False  # Faster GPU transfer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False  # Faster GPU transfer
    )
    
    # Create model
    print("Creating model...")
    model = AnomalyDetectionModel(
        input_dim=X_train.shape[2],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seq_len=args.window_size,
        use_gan=args.use_gan,
        use_contrastive=args.use_contrastive
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Move model to device before creating trainer
    model = model.to(device)
    if device.type == 'cuda':
        # Print GPU memory info
        print(f"\nGPU Memory Info:")
        print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gan_weight=args.gan_weight,
        contrastive_weight=args.contrastive_weight,
        reconstruction_weight=args.reconstruction_weight
    )
    
    # Create evaluator once (reused for all evaluations)
    evaluator = Evaluator(model, device)
    
    # Function to evaluate and save results
    def evaluate_and_save(epoch: int, is_final: bool = False):
        """Evaluate model and save results"""
        epoch_dir = output_dir if is_final else os.path.join(output_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        if is_final:
            print("Final Evaluation")
        else:
            print(f"Evaluation at Epoch {epoch}")
        print("="*60)
        metrics = evaluator.evaluate(test_loader, labels=test_labels)
        
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Optimal Threshold: {metrics['threshold']:.4f}")
        print("="*60)
        
        # Generate comprehensive evaluation visualizations
        print("\nGenerating evaluation visualizations...")
        scores, _ = evaluator.compute_anomaly_scores(test_loader)
        predictions = metrics['predictions']
        
        # Generate all evaluation plots
        evaluator.plot_comprehensive_evaluation(
            scores=scores,
            labels=test_labels,
            predictions=predictions,
            threshold=metrics['threshold'],
            save_dir=epoch_dir
        )
        
        # Save metrics
        import json
        metrics_to_save = {k: v for k, v in metrics.items() 
                          if k not in ['anomaly_scores', 'predictions']}
        metrics_to_save['epoch'] = epoch
        metrics_path = os.path.join(epoch_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
        
        return metrics
    
    # Train with periodic evaluation
    print("Starting training...")
    print(f"Results will be saved every {args.save_every} epochs\n")
    
    history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'gan_loss': [],
        'contrastive_loss': [],
        'discriminator_loss': []
    }
    
    for epoch in range(1, args.num_epochs + 1):
        # Train one epoch
        losses = trainer.train_epoch(train_loader, epoch)
        
        # Update history
        for key in history:
            history[key].append(losses.get(key, 0.0))
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'history': history
            }, checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
            
            # Evaluate and save results
            evaluate_and_save(epoch, is_final=False)
            
            # Save training history up to this point
            history_path = os.path.join(output_dir, f'epoch_{epoch}', 'training_history.json')
            with open(history_path, 'w') as f:
                import json
                json.dump(history, f, indent=2)
    
    # Save final model
    model_path = os.path.join(output_dir, 'final_model.pt')
    trainer.save_model(model_path)
    print(f"\nSaved final model to {model_path}")
    
    # Final evaluation
    final_metrics = evaluate_and_save(args.num_epochs, is_final=True)
    
    # Save final training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        import json
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print("="*60)
    print(f"All outputs saved to: {output_dir}")
    print(f"\nResults saved at epochs: {', '.join([str(e) for e in range(args.save_every, args.num_epochs + 1, args.save_every)] + [str(args.num_epochs)])}")
    print("="*60)


if __name__ == '__main__':
    main()

