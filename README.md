# Advanced Anomaly Detection Framework

A comprehensive anomaly detection framework that combines **geometric masking**, **Transformer-based feature extraction**, **contrastive learning**, and **GAN-based regularization** to detect anomalies in multivariate time-series data, even when training data contains contaminated or mixed anomalous values.

## Overview

This framework addresses the challenging problem of anomaly detection when training data contains contaminated or mixed anomalous values, which can distort the model's understanding of "normal" behavior. The solution integrates multiple advanced techniques:

- **Geometric Masking**: Expands training data and improves robustness through data augmentation
- **Transformer Encoder**: Captures long-range dependencies and extracts rich features
- **Contrastive Learning**: Enforces separation between normal and anomalous patterns
- **GAN Regularization**: Learns realistic normal patterns and reduces overfitting

## Dataset Used

### Dataset Information
- **Dataset Name**: LCL-June2015v2_94 (UK Domestic Energy Consumption)
- **Source**: London Energy Consumption Dataset
- **Number of Features**: 7 (multivariate features derived from energy consumption)
- **Number of Samples**: 1,000,000+ time-series points
- **Number of Customers**: 37 unique customer IDs
- **Time Range**: 2012-05-18 to 2014-02-28
- **Temporal Resolution**: Half-hourly (48 measurements per day)

### Feature Engineering
The framework creates multivariate features from the raw energy consumption data:
1. **Energy Consumption**: Raw KWH/hh values
2. **Hour**: Normalized hour of day (0-23)
3. **Day of Week**: Normalized day of week (0-6)
4. **Day of Month**: Normalized day of month (1-31)
5. **Month**: Normalized month (1-12)
6. **Rolling Mean**: 24-hour rolling mean of energy consumption
7. **Rolling Std**: 24-hour rolling standard deviation

### Train/Test Split
- **Training Set**: 80% of sequences
- **Test Set**: 20% of sequences
- **Sequence Length**: 100 time steps (default)
- **Stride**: 1 (sliding window)

## Preprocessing Steps

### 1. Data Loading
- Load CSV data with proper datetime parsing
- Filter by customer ID (optional)
- Sort chronologically

### 2. Feature Creation
- Extract temporal features (hour, day, month)
- Compute rolling statistics (mean, std) over 24-hour windows
- Handle missing values using forward-fill strategy

### 3. Normalization
- **Method**: StandardScaler (zero mean, unit variance)
- Applied per feature dimension
- Fitted on training data, applied to test data

### 4. Sequence Windowing
- Create sliding windows of fixed length (default: 100 time steps)
- Stride parameter controls overlap (default: 1)
- Results in sequences of shape (N, window_size, features)

### 5. Missing Value Handling
- **Strategy**: Forward-fill (propagate last valid value forward)
- Alternative strategies available: backward-fill, interpolation, zero-fill

## Model Architecture

### Overall Architecture

```
Input Sequence (B, T, F)
    ↓
Geometric Masking (Data Augmentation)
    ↓
Transformer Encoder
    ├── Input Projection (F → d_model)
    ├── Positional Encoding
    ├── Multi-Head Self-Attention (×N layers)
    ├── Feedforward Networks
    └── Output Projection (d_model → F)
    ↓
    ├── Encoded Representations (B, T, d_model)
    ├── Reconstructed Sequence (B, T, F)
    └── Anomaly Scores (B,)
    ↓
Contrastive Learning Module (Training)
    └── Projection Head → Contrastive Loss
    ↓
GAN Module (Training)
    ├── Generator → Synthetic Normal Patterns
    └── Discriminator → Real vs Fake Classification
```

### 1. Transformer Encoder Block

**Components**:
- **Input Projection**: Linear layer mapping input features to model dimension
- **Positional Encoding**: Sinusoidal positional encodings for temporal information
- **Multi-Head Self-Attention**: Captures dependencies across time and features
  - Number of heads: 8 (default)
  - Model dimension: 128 (default)
- **Feedforward Network**: Two-layer MLP with GELU activation
  - Hidden dimension: 512 (default)
- **Layer Normalization**: Applied before each sub-layer
- **Residual Connections**: Around attention and feedforward layers
- **Output Projection**: Maps back to input feature dimension for reconstruction

**Key Parameters**:
- `d_model`: 128 (model dimension)
- `nhead`: 8 (attention heads)
- `num_layers`: 4 (transformer blocks)
- `dim_feedforward`: 512
- `dropout`: 0.1

### 2. Contrastive Learning Module

**Purpose**: Enforce separation between normal and anomalous patterns in latent space

**Architecture**:
- **Projection Head**: 2-layer MLP
  - Input: d_model (128)
  - Hidden: d_model (128)
  - Output: projection_dim (64)
- **Contrastive Loss**: Improved margin-based contrastive loss
  - Maximizes similarity between normal samples (positive pairs)
  - Minimizes similarity between normal and anomalous samples (negative pairs)
  - Temperature: 0.07
  - Margin: 0.5 (reduced for better gradient flow)
  - Diversity regularization: 0.01 × |pos_sim| (prevents collapse)

**Training Strategy**:
- **Anchor**: Original sequence embeddings
- **Positive**: Geometric masking augmented normal sequences (stronger augmentation)
- **Negative**: Combined strategy:
  - Time-shifted sequences (temporal anomalies)
  - Feature-shuffled sequences (feature anomalies)
  - Noise-injected sequences (0.3× noise scale)
- **Loss Components**:
  - Positive loss: Pulls similar samples together `(1 - sim(anchor, positive))²`
  - Negative loss: Pushes dissimilar samples apart `max(0, margin - sim(anchor, negative))²`
  - Diversity regularization: Prevents embedding collapse

### 3. GAN Module

**Generator**:
- **Input**: Random noise vector (latent_dim=100)
- **Architecture**:
  - Initial fully connected layer
  - Transposed 1D convolutions (upsampling)
  - Batch normalization and ReLU activations
  - Bidirectional LSTM for temporal refinement
  - Output projection to feature dimension
  - **No Tanh activation**: Outputs raw values matching StandardScaler distribution (mean=0, std=1)
- **Output**: Synthetic time-series sequences (B, T, F) matching normalized data distribution

**Discriminator**:
- **Input**: Time-series sequences (B, T, F)
- **Architecture**:
  - 1D convolutions (downsampling)
  - Batch normalization and LeakyReLU
  - Bidirectional LSTM
  - Classification head (real vs fake)
- **Output**: Probability of being real (B, 1)

**Purpose**: Learn realistic normal patterns and provide adversarial regularization

### 4. Anomaly Scoring Head

- **Input**: Mean-pooled transformer encodings (B, d_model)
- **Architecture**: 2-layer MLP with Sigmoid output
- **Output**: Anomaly scores in [0, 1] range

## Training Procedure

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 32-48 | Number of sequences per batch (48 recommended for 8GB GPU) |
| Learning Rate | 1e-4 | AdamW optimizer learning rate (main model) |
| GAN Generator LR | 1e-5 | Generator learning rate (0.1× main LR) |
| GAN Discriminator LR | 2e-5 | Discriminator learning rate (0.2× main LR) |
| Weight Decay | 1e-5 | L2 regularization |
| Epochs | 50 | Number of training epochs |
| Window Size | 100 | Sequence length |
| Stride | 1 | Sliding window stride |
| Gradient Clipping | 1.0 | Max gradient norm for all components |

### Loss Functions

The total loss combines multiple components:

```
Total Loss = λ_recon × L_reconstruction 
           + λ_contrastive × L_contrastive 
           + λ_gan × L_gan
```

**Loss Weights** (default):
- `λ_recon`: 1.0 (reconstruction loss)
- `λ_contrastive`: 0.5 (contrastive loss) - increased for better contribution
- `λ_gan`: 0.01 (GAN generator loss) - reduced for stability

**Note**: GAN loss is trained separately and not added to main loss. It's tracked independently for monitoring.

**Individual Losses**:

1. **Reconstruction Loss**: Mean Squared Error (MSE)
   ```
   L_reconstruction = MSE(reconstructed, original)
   ```

2. **Contrastive Loss**: Improved margin-based contrastive loss with diversity regularization
   ```
   L_contrastive = (1 - sim(anchor, positive))² 
                 + max(0, margin - sim(anchor, negative))²
                 + 0.01 × |sim(anchor, positive)|
   ```
   - Margin: 0.5 (reduced from 1.0 for better gradients)
   - Diversity term prevents embedding collapse

3. **GAN Losses** (trained separately from main model):
   - **Discriminator**: Binary Cross-Entropy with label smoothing
     - Real targets: 0.9 (instead of 1.0)
     - Fake targets: 0.1 (instead of 0.0)
     - Loss: `(BCE(real_pred, 0.9) + BCE(fake_pred, 0.1)) / 2`
   - **Generator**: Binary Cross-Entropy with label smoothing
     - Target: 0.9 (fool discriminator)
     - Loss: `BCE(fake_pred, 0.9)`
   - **Conditional Training**: Generator only updates when discriminator loss < 1.5 (prevents discriminator dominance)

### Optimization Algorithm

- **Main Model Optimizer**: AdamW
  - Learning Rate: 1e-4
  - Beta Parameters: (0.9, 0.999)
  - Weight Decay: 1e-5
  
- **GAN Optimizers**: Adam (separate optimizers)
  - Generator Learning Rate: 1e-5 (0.1× main LR)
  - Discriminator Learning Rate: 2e-5 (0.2× main LR)
  - Beta Parameters: (0.5, 0.999) for both
  
- **Gradient Clipping**: Max norm = 1.0
  - Applied to: Main model, Generator, Discriminator
  - Prevents exploding gradients and stabilizes training

### Training Pipeline

1. **Data Augmentation**: Apply geometric masking to input sequences (15% mask ratio)

2. **Forward Pass**: 
   - Transformer encoding and reconstruction
   - Contrastive embedding computation (with improved sampling)
   - GAN generation (if enabled)

3. **Loss Computation**:
   - Reconstruction loss (MSE)
   - Contrastive loss (if enabled) - with diversity regularization
   - GAN losses (if enabled) - trained separately

4. **Backward Pass** (separate passes for stability):
   - **Discriminator Update**:
     - Compute discriminator loss (real + fake)
     - Backward pass with gradient clipping
     - Update discriminator optimizer
   - **Generator Update** (conditional):
     - Only if discriminator loss < 1.5 (prevents discriminator dominance)
     - Compute generator loss
     - Backward pass with gradient clipping
     - Update generator optimizer
   - **Main Model Update**:
     - Compute main loss (reconstruction + contrastive)
     - Backward pass with gradient clipping
     - Update main model optimizer

5. **Gradient Clipping**: Applied to all components (max_norm=1.0)

6. **Loss Tracking**: All losses tracked separately for monitoring

### Geometric Masking During Training

Applied to each batch with:
- **Mask Ratio**: 15% of features
- **Mask Types**: Random, Block, Geometric, Feature
- **Purpose**: Data augmentation and robustness

## Evaluation Metrics

### Metrics Computed

1. **Precision**: `TP / (TP + FP)`
   - Proportion of predicted anomalies that are actually anomalies

2. **Recall**: `TP / (TP + FN)`
   - Proportion of actual anomalies that are detected

3. **F1 Score**: `2 × (Precision × Recall) / (Precision + Recall)`
   - Harmonic mean of precision and recall

4. **ROC-AUC**: Area Under ROC Curve
   - Measures ability to distinguish between normal and anomalous samples
   - Range: [0, 1], higher is better

5. **PR-AUC**: Area Under Precision-Recall Curve
   - Better metric for imbalanced datasets
   - Range: [0, 1], higher is better

### Thresholding Strategy

**Optimal Threshold Selection**:
- Grid search over score range (1000 thresholds)
- Optimize F1 score by default
- Alternative: optimize precision or recall

**Anomaly Score Computation**:
- Primary: Transformer-based anomaly scores (sigmoid output)
- Secondary: Reconstruction error (MSE per sequence)
- Combined: Weighted average (60% scores + 40% reconstruction error)

### Evaluation Visualizations

The framework generates comprehensive plots:

1. **Main Evaluation Plots** (`evaluation_plots.png`):
   - Anomaly Scores Over Time: Time-series plot with threshold line and true anomalies
   - Score Distribution: Histogram of anomaly scores
   - ROC Curve: Receiver Operating Characteristic curve with AUC
   - Precision-Recall Curve: PR curve with AUC

2. **Confusion Matrix** (`confusion_matrix.png`):
   - Visual confusion matrix with color-coded cells
   - Metrics overlay (Accuracy, Precision, Recall, F1-Score)

3. **Classification Report** (`classification_report.txt`):
   - Detailed per-class metrics
   - Confusion matrix in text format

4. **Detailed Analysis** (`detailed_analysis.png`):
   - Score Distribution by Class: Separate histograms for normal vs anomaly
   - Precision-Recall vs Threshold: How metrics change with threshold
   - Classification Breakdown: Bar chart of TP, TN, FP, FN
   - Anomaly Scores with Misclassifications: Highlighted errors

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib for visualization
- tqdm for progress bars

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd advanced-anomaly-detection-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure dataset is in `dataset/` directory:
```
dataset/
  └── LCL-June2015v2_94.csv
```

## Usage

### Basic Training

Train the model with default parameters:
```bash
python train.py
```

### Custom Training

Train with custom parameters:
```bash
python train.py \
    --data_path dataset/LCL-June2015v2_94.csv \
    --window_size 100 \
    --batch_size 48 \
    --num_epochs 50 \
    --d_model 128 \
    --nhead 8 \
    --num_layers 4 \
    --lr 1e-4 \
    --gan_weight 0.01 \
    --contrastive_weight 0.5 \
    --output_dir outputs
```

**Recommended for 8GB GPU**:
```bash
python train.py --batch_size 48 --num_epochs 50
```

### Training on Specific Customer

Train on a specific customer's data:
```bash
python train.py --customer_id MAC004020
```

### Command-Line Arguments

**Data Arguments**:
- `--data_path`: Path to dataset CSV
- `--customer_id`: Specific customer ID (None for all)
- `--window_size`: Sequence window size (default: 100)
- `--stride`: Sliding window stride (default: 1)
- `--train_split`: Train/test split ratio (default: 0.8)

**Model Arguments**:
- `--d_model`: Transformer model dimension (default: 128)
- `--nhead`: Number of attention heads (default: 8)
- `--num_layers`: Number of transformer layers (default: 4)
- `--dim_feedforward`: Feedforward dimension (default: 512)
- `--dropout`: Dropout rate (default: 0.1)
- `--use_gan`: Enable GAN regularization (default: True)
- `--use_contrastive`: Enable contrastive learning (default: True)

**Training Arguments**:
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-5)
- `--gan_weight`: GAN loss weight (default: 0.01, reduced for stability)
- `--contrastive_weight`: Contrastive loss weight (default: 0.5, increased for better contribution)
- `--reconstruction_weight`: Reconstruction loss weight (default: 1.0)

**Output Arguments**:
- `--output_dir`: Output directory (default: outputs)
- `--save_every`: Save checkpoint every N epochs (default: 10)

**Device Arguments**:
- `--device`: Device to use - 'auto' (detect), 'cuda' (force GPU), 'cpu' (force CPU) (default: auto)
- `--gpu_id`: GPU ID to use if multiple GPUs available (default: 0)

## Results

### Expected Performance

The framework is designed to achieve:
- **High Recall**: Detect most anomalies (minimize false negatives)
- **Reasonable Precision**: Balance false positives
- **Strong ROC-AUC**: Good separation between normal and anomalous patterns
- **Robust Performance**: Works well even with contaminated training data

### Training Stability

The updated training procedure ensures:
- **Stable GAN Training**: GAN loss should stabilize around 0.5-1.5 (not explode)
- **Active Contrastive Learning**: Contrastive loss should show meaningful values (0.1-2.0 range)
- **Decreasing Total Loss**: Main loss should decrease over time
- **Consistent Training Speed**: No sudden slowdowns due to gradient issues
- **Balanced Components**: All loss components contribute meaningfully

### Output Files

After training, the following files are generated in `outputs/run_<timestamp>/`:

1. **`final_model.pt`**: Trained model checkpoint
2. **`checkpoints/checkpoint_epoch_*.pt`**: Periodic checkpoints (every N epochs)
3. **`metrics.json`**: Evaluation metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
4. **`training_history.json`**: Training loss history (all components tracked separately)
5. **`evaluation_plots.png`**: Main evaluation visualizations (ROC, PR, scores, distribution)
6. **`confusion_matrix.png`**: Confusion matrix with metrics overlay
7. **`classification_report.txt`**: Detailed classification report
8. **`detailed_analysis.png`**: Additional analysis plots (score distributions, threshold analysis, error breakdown)

### Interpreting Results

- **Anomaly Scores**: Higher scores indicate higher likelihood of anomaly
- **Threshold**: Optimal threshold balances precision and recall
- **ROC-AUC > 0.8**: Good discrimination ability
- **PR-AUC**: More informative for imbalanced datasets

## Project Structure

```
advanced-anomaly-detection-framework/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── geometric_masking.py        # Geometric masking techniques
│   ├── transformer_encoder.py      # Transformer architecture
│   ├── contrastive_learning.py     # Contrastive learning module
│   ├── gan_module.py               # GAN generator and discriminator
│   ├── model.py                    # Main model integration
│   ├── training.py                 # Training pipeline
│   └── evaluation.py               # Evaluation metrics and plots
├── dataset/
│   └── LCL-June2015v2_94.csv      # Dataset file
├── train.py                        # Main training script
├── requirements.txt                # Python dependencies
├── PROJECT_REQUIREMENTS.md         # Project requirements
└── README.md                       # This file
```

## Key Features

### 1. Geometric Masking
- **Random Masking**: Randomly mask features across time and dimensions
- **Block Masking**: Mask contiguous blocks in time dimension
- **Geometric Masking**: Mask geometric patterns (rectangles, lines)
- **Feature Masking**: Mask entire feature dimensions

### 2. Transformer Encoder
- Multi-head self-attention for long-range dependencies
- Positional encoding for temporal information
- Layer normalization and residual connections
- Bidirectional feature extraction

### 3. Contrastive Learning
- Enforces tight clustering of normal samples
- Pushes anomalous samples away in latent space
- Improves representation quality
- Improved margin-based contrastive loss (margin=0.5)
- Enhanced positive/negative sampling strategy
- Diversity regularization prevents embedding collapse

### 4. GAN Regularization
- Generator learns realistic normal patterns
- Discriminator provides adversarial pressure
- Reduces overfitting
- Improves generalization
- **Stability Features**:
  - Label smoothing (0.9/0.1) prevents overconfidence
  - Lower learning rates (0.1×/0.2×) for GAN components
  - Conditional generator training (only when discriminator balanced)
  - Gradient clipping for both generator and discriminator
  - Generator output matches data distribution (no Tanh)

## Reproducibility

The framework includes:
- **Fixed Random Seeds**: For reproducibility (can be added)
- **Checkpointing**: Save and resume training
- **Comprehensive Logging**: All metrics and losses logged
- **Visualization**: Automatic plot generation

## Future Improvements

Potential enhancements:
1. **Adaptive Thresholding**: Dynamic threshold selection
2. **Multi-Scale Attention**: Capture patterns at different time scales
3. **Uncertainty Quantification**: Confidence intervals for predictions
4. **Online Learning**: Update model with new data
5. **Interpretability**: Attention visualization and feature importance

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{anomaly_detection_framework,
  title={Advanced Anomaly Detection Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/noumanic/advanced-anomaly-detection-framework}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact [noumanhafeez.nh11@gmail.com].

---

**Note**: This framework is designed for research and educational purposes. For production use, additional validation, testing, and optimization may be required.