# Dimensionality Reduction Techniques: PCA, LDA, and Kernel PCA

This project explores standard linear and non-linear dimensionality reduction techniques applied to classification problems. It includes both manual "from-scratch" implementations and comparisons with `scikit-learn` equivalents.

## Overview

The codebase demonstrates how dimensionality reduction can simplify datasets while retaining essential information for classification. It covers:
1.  **Principal Component Analysis (PCA):** Unsupervised linear transformation that maximizes variance.
2.  **Linear Discriminant Analysis (LDA):** Supervised linear transformation that maximizes class separability.
3.  **Kernel PCA (KPCA):** Non-linear transformation using the RBF kernel to separate complex data structures.

## Dependencies

* Python 3.x
* numpy
* pandas
* matplotlib
* scikit-learn
* scipy

## Datasets Used

* **UCI Wine Dataset:** A 13-feature dataset used for benchmarking PCA and LDA.
* **Synthetic Datasets (`make_moons`, `make_circles`):** Non-linearly separable datasets used to demonstrate the capabilities of Kernel PCA.

## Project Structure & Key Results

### Part 1: PCA on Wine Dataset
* **Implementation:** Manual eigendecomposition of the covariance matrix and `sklearn.decomposition.PCA`.
* **Goal:** Reduce the 13-feature Wine dataset to 2 principal components.
* **Result:** PCA captured the majority of the variance, but as an unsupervised technique, it slightly reduced classification accuracy compared to the full dataset in this specific run.
    * *PCA Test Accuracy:* 92.6%

### Part 2: LDA on Wine Dataset
* **Implementation:** Manual calculation of within-class ($S_W$) and between-class ($S_B$) scatter matrices, solving the generalized eigenvalue problem, and `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.
* **Goal:** Find the feature subspace that optimizes class separability.
* **Result:** LDA outperformed PCA on this dataset because it utilizes class labels during transformation.
    * *LDA Test Accuracy:* 100%

#### Performance Comparison (Wine Dataset)
| Method | Test Accuracy | Computation Time (approx) |
| :--- | :--- | :--- |
| Original Data (13 features) | 1.000 | 0.015s |
| PCA (2 components) | 0.926 | 0.011s |
| LDA (2 components) | 1.000 | 0.013s |

### Part 3: RBF Kernel PCA
* **Implementation:** Manual implementation using `scipy` for pairwise distances and kernel matrix centering; verified against `sklearn.decomposition.KernelPCA`.
* **Goal:** Separate non-linear data (half-moons and concentric circles) that standard linear PCA cannot handle.
* **Result:**
    * Successfully "unrolled" the `make_moons` dataset using an RBF kernel ($\gamma=15$).
    * Successfully separated the `make_circles` dataset into distinct clusters suitable for linear classification.
    * Demonstrated the impact of the hyperparameter $\gamma$ (gamma) on the resulting subspace.

## Usage
The code is structured as a sequential script or notebook. Ensure all dependencies are installed and run the sections sequentially to load data, perform transformations, and visualize decision regions.
