"""
Evaluation Module
Computes evaluation metrics for anomaly detection
"""

import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import os


class Evaluator:
    """
    Evaluator for anomaly detection model
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Args:
            model: Trained anomaly detection model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def compute_anomaly_scores(self, 
                               data_loader: torch.utils.data.DataLoader,
                               use_reconstruction_error: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for test data
        
        Args:
            data_loader: Data loader for test data
            use_reconstruction_error: Whether to use reconstruction error as score
            
        Returns:
            scores: Anomaly scores
            labels: True labels (if available)
        """
        all_scores = []
        all_labels = []
        all_recon_errors = []
        
        with torch.no_grad():
            for x, labels in data_loader:
                x = x.to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                
                # Get anomaly scores
                scores = outputs['anomaly_scores'].cpu().numpy()
                all_scores.append(scores)
                
                # Compute reconstruction error
                if use_reconstruction_error:
                    reconstructed = outputs['reconstructed']
                    recon_error = torch.mean((reconstructed - x) ** 2, dim=(1, 2)).cpu().numpy()
                    all_recon_errors.append(recon_error)
                
                # Store labels if available
                if labels is not None:
                    all_labels.append(labels.numpy())
        
        scores = np.concatenate(all_scores)
        recon_errors = np.concatenate(all_recon_errors) if all_recon_errors else None
        
        # Combine scores
        if use_reconstruction_error and recon_errors is not None:
            # Normalize and combine
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            recon_errors = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)
            combined_scores = 0.6 * scores + 0.4 * recon_errors
        else:
            combined_scores = scores
        
        labels = np.concatenate(all_labels) if all_labels else None
        
        return combined_scores, labels
    
    def find_optimal_threshold(self, 
                              scores: np.ndarray,
                              labels: np.ndarray,
                              metric: str = 'f1') -> float:
        """
        Find optimal threshold for anomaly detection
        
        Args:
            scores: Anomaly scores
            labels: True labels
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold
        """
        thresholds = np.linspace(scores.min(), scores.max(), 1000)
        best_threshold = thresholds[0]
        best_score = 0.0
        
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(labels, predictions, zero_division=0)
            elif metric == 'precision':
                score = precision_score(labels, predictions, zero_division=0)
            elif metric == 'recall':
                score = recall_score(labels, predictions, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def evaluate(self, 
                 data_loader: torch.utils.data.DataLoader,
                 labels: Optional[np.ndarray] = None,
                 threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            data_loader: Test data loader
            labels: True labels (if available)
            threshold: Anomaly threshold (if None, will find optimal)
            
        Returns:
            Dictionary of metrics
        """
        scores, true_labels = self.compute_anomaly_scores(data_loader)
        
        # Use provided labels or true_labels
        if labels is not None:
            true_labels = labels
        
        if true_labels is None:
            # No labels available, return scores only
            return {
                'anomaly_scores': scores,
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std())
            }
        
        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self.find_optimal_threshold(scores, true_labels, metric='f1')
        
        # Compute predictions
        predictions = (scores >= threshold).astype(int)
        
        # Compute metrics
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(true_labels, scores)
        except ValueError:
            roc_auc = 0.0
        
        # PR-AUC
        try:
            pr_auc = average_precision_score(true_labels, scores)
        except ValueError:
            pr_auc = 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'threshold': float(threshold),
            'anomaly_scores': scores,
            'predictions': predictions
        }
    
    def plot_results(self,
                    scores: np.ndarray,
                    labels: Optional[np.ndarray] = None,
                    threshold: Optional[float] = None,
                    save_path: Optional[str] = None):
        """
        Plot evaluation results
        
        Args:
            scores: Anomaly scores
            labels: True labels
            threshold: Anomaly threshold
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Anomaly scores over time
        axes[0, 0].plot(scores, alpha=0.7, label='Anomaly Score')
        if threshold is not None:
            axes[0, 0].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
        if labels is not None:
            anomaly_indices = np.where(labels == 1)[0]
            axes[0, 0].scatter(anomaly_indices, scores[anomaly_indices], 
                             color='red', s=10, alpha=0.5, label='True Anomalies')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Anomaly Score')
        axes[0, 0].set_title('Anomaly Scores Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Score distribution
        axes[0, 1].hist(scores, bins=50, alpha=0.7, edgecolor='black')
        if threshold is not None:
            axes[0, 1].axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
        axes[0, 1].set_xlabel('Anomaly Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        if labels is not None:
            # Plot 3: ROC Curve
            fpr, tpr, _ = roc_curve(labels, scores)
            axes[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(labels, scores):.3f})')
            axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(labels, scores)
            pr_auc = average_precision_score(labels, scores)
            axes[1, 1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
            axes[1, 1].set_xlabel('Recall')
            axes[1, 1].set_ylabel('Precision')
            axes[1, 1].set_title('Precision-Recall Curve')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self,
                             predictions: np.ndarray,
                             labels: np.ndarray,
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            predictions: Predicted labels
            labels: True labels
            save_path: Path to save plot
        """
        cm = confusion_matrix(labels, predictions)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')
        
        # Labels and title
        classes = ['Normal', 'Anomaly']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add metrics text
        metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
        ax.text(1.5, -0.3, metrics_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return cm
    
    def plot_comprehensive_evaluation(self,
                                     scores: np.ndarray,
                                     labels: np.ndarray,
                                     predictions: np.ndarray,
                                     threshold: float,
                                     save_dir: Optional[str] = None):
        """
        Generate comprehensive evaluation visualizations
        
        Args:
            scores: Anomaly scores
            labels: True labels
            predictions: Predicted labels
            threshold: Anomaly threshold
            save_dir: Directory to save all plots
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. Main evaluation plots (ROC, PR, scores, distribution)
        main_plot_path = os.path.join(save_dir, 'evaluation_plots.png') if save_dir else None
        self.plot_results(scores, labels, threshold, main_plot_path)
        
        # 2. Confusion matrix
        cm_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        cm = self.plot_confusion_matrix(predictions, labels, cm_path)
        
        # 3. Classification report
        if save_dir:
            report_path = os.path.join(save_dir, 'classification_report.txt')
            report = classification_report(labels, predictions, 
                                         target_names=['Normal', 'Anomaly'])
            with open(report_path, 'w') as f:
                f.write("Classification Report\n")
                f.write("="*60 + "\n\n")
                f.write(report)
                f.write("\n\nConfusion Matrix:\n")
                f.write("-"*60 + "\n")
                f.write(f"                Predicted\n")
                f.write(f"              Normal  Anomaly\n")
                f.write(f"True Normal    {cm[0,0]:6d}  {cm[0,1]:6d}\n")
                f.write(f"True Anomaly   {cm[1,0]:6d}  {cm[1,1]:6d}\n")
            print(f"Saved classification report to {report_path}")
        
        # 4. Additional detailed plots
        self._plot_detailed_analysis(scores, labels, predictions, threshold, save_dir)
    
    def _plot_detailed_analysis(self,
                                scores: np.ndarray,
                                labels: np.ndarray,
                                predictions: np.ndarray,
                                threshold: float,
                                save_dir: Optional[str] = None):
        """
        Generate additional detailed analysis plots
        
        Args:
            scores: Anomaly scores
            labels: True labels
            predictions: Predicted labels
            threshold: Anomaly threshold
            save_dir: Directory to save plots
        """
        if save_dir is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Anomaly Detection Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Score distribution by class
        ax = axes[0, 0]
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='green', edgecolor='black')
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', edgecolor='black')
        ax.axvline(x=threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Precision-Recall at different thresholds
        ax = axes[0, 1]
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        precisions = []
        recalls = []
        for t in thresholds:
            preds = (scores >= t).astype(int)
            if len(np.unique(preds)) > 1:  # Both classes present
                prec = precision_score(labels, preds, zero_division=0)
                rec = recall_score(labels, preds, zero_division=0)
            else:
                prec = 0
                rec = 0
            precisions.append(prec)
            recalls.append(rec)
        ax.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal Threshold')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Error analysis
        ax = axes[1, 0]
        false_positives = (predictions == 1) & (labels == 0)
        false_negatives = (predictions == 0) & (labels == 1)
        true_positives = (predictions == 1) & (labels == 1)
        true_negatives = (predictions == 0) & (labels == 0)
        
        error_counts = {
            'True Positive': true_positives.sum(),
            'True Negative': true_negatives.sum(),
            'False Positive': false_positives.sum(),
            'False Negative': false_negatives.sum()
        }
        colors = ['green', 'blue', 'orange', 'red']
        ax.bar(error_counts.keys(), error_counts.values(), color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title('Classification Results Breakdown')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (key, value) in enumerate(error_counts.items()):
            ax.text(i, value, str(value), ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Score vs Index with predictions
        ax = axes[1, 1]
        sample_size = min(5000, len(scores))  # Sample for visibility
        indices = np.random.choice(len(scores), sample_size, replace=False)
        indices = np.sort(indices)
        
        ax.scatter(indices, scores[indices], 
                  c=labels[indices], cmap='RdYlGn', 
                  alpha=0.6, s=10, label='True Labels')
        ax.axhline(y=threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold')
        
        # Highlight misclassifications
        misclassified = (predictions[indices] != labels[indices])
        if misclassified.any():
            ax.scatter(indices[misclassified], scores[indices][misclassified],
                      color='red', marker='x', s=50, linewidths=2,
                      label='Misclassified', zorder=5)
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Scores with Predictions (Sampled)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        detailed_path = os.path.join(save_dir, 'detailed_analysis.png')
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        print(f"Saved detailed analysis to {detailed_path}")
        plt.close()

