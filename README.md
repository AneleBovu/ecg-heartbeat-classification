# ECG Heartbeat Classification: Decision Tree vs Random Forest with PCA
## Project Overview
A complete machine learning pipeline classifying ECG heartbeats as **Normal** or **Abnormal**
using a curated dataset of 4,999 instances, each represented as a variable-length
time-series amplitude sequence.

## Pipeline Summary (10 Tasks)
| Task | Description |
|------|-------------|
| 1 | Exploratory Data Analysis — class distribution, waveform morphology |
| 2 | Feature Engineering → 12 morphological/statistical features + StandardScaler |
| 3 | Pairwise Correlation Analysis — multicollinearity detection |
| 4 | Principal Component Analysis — 3 PCs retaining 75.6% variance |
| 5 | PC–Target Correlations & Feature Selection |
| 6 | Dataset Partition (60% train / 40% test) |
| 7 | Decision Tree — depth tuning, bias-variance analysis |
| 8 | Random Forest — n_estimators tuning, misclassification analysis |
| 9 | F1 & ROC-AUC Comparison + Condorcet Jury Theorem |
| 10 | 10-Fold Cross-Validation + Paired t-test (p=0.0026) |

## Key Results
| Model | Subset-B F1 | CV Mean F1 |
|-------|-------------|------------|
| Decision Tree | 0.9413 | 0.9389 |
| **Random Forest** | **0.9571** | **0.9504** |

RF outperforms DT with statistical significance (p < 0.01).

## Libraries Used
- `scikit-learn` — classifiers, PCA, cross-validation, metrics
- `pandas`, `NumPy` — data wrangling and numerical computation
- `matplotlib`, `seaborn` — visualisation
- `scipy` — feature engineering (skewness, kurtosis) and statistical testing

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook ECGHeartbeats.ipynb
```

## File Structure
```
ecg-heartbeat-classification/
├── ECGHeartbeats.ipynb   # Full analysis notebook
├── ecg_curated.csv       # Dataset (4,999 ECG instances)
└── README.md
```
