# Advanced Anomaly Detection Framework  
### Project Requirements & Task Description

Anomaly detection becomes challenging when the training data contains *contaminated or mixed anomalous values*, as this can distort the model’s understanding of what "normal" behavior actually looks like.  
The proposed framework mitigates this issue by **combining geometric masking, Transformer-based feature extraction, contrastive learning, and GAN-based regularization** to enhance generalization and performance.

---

## Key Components to Implement

### 1. **Geometric Masking**
Apply geometric masking techniques to:
- Expand effective training data  
- Improve model robustness  
- Simulate partial missing patterns  
- Help the model focus on structure instead of noise  

---

### 2. **Transformer Encoder**
Use a Transformer architecture for:
- Feature extraction  
- Sequence encoding  
- Reconstruction of multivariate time-series  
- Capturing long-range dependencies between features  

---

### 3. **Contrastive Learning**
Implement contrastive loss to:
- Enforce separation between *normal* and *anomalous* patterns  
- Improve latent representation quality  
- Make normal samples cluster tightly while anomalies stay distant  

---

### 4. **GAN-Based Regularization**
Integrate a GAN module to:
- Learn realistic "normal" time-series patterns  
- Reduce overfitting  
- Improve generalization when training data is contaminated  
- Provide adversarial pressure to improve reconstruction quality  

Together, these techniques aim to **reduce overfitting** and **improve representation quality** through strong regularization and self-supervised learning.

---

## Dataset Requirements

You must demonstrate the framework on a publicly available **multivariate time-series dataset**.

Suggested datasets include:
- NASA Datasets (SMAP, MSL)
- eBay Anomaly Dataset
- Nanyang Technological University (NTU) datasets

A convenient dataset collection:
https://github.com/elisejiuqizhang/TS-AD-Datasets

---

## Project Deliverables

You must submit the following through GitHub:

### 1. **Working Implementation**
A fully functioning codebase that integrates:
- Geometric masking  
- Transformer encoder  
- Contrastive loss  
- GAN regularization  
- Anomaly scoring & detection  

### 2. **Comprehensive README**
Your README must clearly explain:

#### Dataset Used
- Dataset name  
- Number of features  
- Train/test split  

#### Preprocessing Steps
- Normalization  
- Sequence windowing  
- Masking  
- Handling missing values  

#### Model Architecture
- Transformer block details  
- Contrastive module  
- GAN generator & discriminator  
- Loss functions  
- Training pipeline diagram  

#### Training Procedure
- Hyperparameters  
- Batch size  
- Optimization algorithm  
- Epochs  

#### Evaluation Metrics
- Precision, Recall, F1  
- ROC-AUC  
- PR-AUC  
- Thresholding strategy  

#### Results
- Anomaly detection plots  
- Reconstruction error  
- Latent space visualization (optional)  

---

## Grading Criteria

The following aspects will be evaluated:

- **Correctness** and complete functionality of the implementation  
- **Effectiveness** of anomaly detection results  
- **Clarity** and organization of the documentation  
- **Reproducibility** through clean code structure  
- **Completeness** of evaluation and experiments  

---

## End of Document
This file summarizes the entire project specification and required components for the Advanced Anomaly Detection Framework.