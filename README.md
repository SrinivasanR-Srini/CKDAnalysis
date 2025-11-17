# CKD Analysis - Machine Learning Pipeline

A comprehensive machine learning pipeline for Chronic Kidney Disease (CKD) analysis using the UCI CKD dataset.

## Overview

This project implements a two-stage machine learning pipeline:

1. **Solution 1**: Creatinine Prediction (Regression)
2. **Solution 2**: CKD Level Classification (Multiclass Classification)

## Dataset

- **Source**: [UCI Machine Learning Repository - Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- **Features Used**:
  - Age
  - Albumin (urine)
  - RBC (Red Blood Cells - nominal)
  - Pus Cell (nominal)
  - Bacteria (nominal)
  - Urine pH

## Solution 1: Creatinine Prediction

### Objective
Predict serum creatinine levels from the given features.

### Models
- **Perceptron** (MLPRegressor with single neuron)
- **XGBoost** (Ensemble)
- **CatBoost** (Ensemble)

### Features
- Exploratory Data Analysis (EDA)
- Feature preprocessing and encoding
- Model comparison and evaluation
- Feature importance analysis

### Output
Predicted creatinine values are saved for use in Solution 2.

## Solution 2: CKD Level Classification

### Objective
Classify CKD stages based on eGFR calculations using predicted creatinine.

### Process
1. Calculate eGFR using **CKD-EPI equation**
2. Apply **KDIGO classification** for CKD stages:
   - **G1**: eGFR ≥ 90 (Normal or high)
   - **G2**: eGFR 60-89 (Mildly decreased)
   - **G3a**: eGFR 45-59 (Mildly to moderately decreased)
   - **G3b**: eGFR 30-44 (Moderately to severely decreased)
   - **G4**: eGFR 15-29 (Severely decreased)
   - **G5**: eGFR < 15 (Kidney failure)

### Models
- **Multi-Layer Perceptron (MLP)**
- **Support Vector Machine (SVM)**

### Features
- eGFR calculation using CKD-EPI equation
- KDIGO-based multiclass classification
- Model comparison with cross-validation
- Confusion matrices and classification reports
- Feature importance analysis

## Installation

### Requirements
- Python 3.8+
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SrinivasanR-Srini/CKDAnalysis.git
cd CKDAnalysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebooks

1. **Solution 1 - Creatinine Prediction**:
```bash
jupyter notebook solution1_creatinine_prediction.ipynb
```
Run all cells to:
- Load and explore the dataset
- Preprocess features
- Train regression models (Perceptron, XGBoost, CatBoost)
- Evaluate and compare models
- Save predictions for Solution 2

2. **Solution 2 - CKD Classification**:
```bash
jupyter notebook solution2_ckd_classification.ipynb
```
Run all cells to:
- Load predicted creatinine from Solution 1
- Calculate eGFR using CKD-EPI equation
- Apply KDIGO classification
- Train classification models (MLP, SVM)
- Evaluate and compare models

### Expected Workflow

1. Run `solution1_creatinine_prediction.ipynb` first
2. This will create `solution1_output.pkl` with predictions
3. Run `solution2_ckd_classification.ipynb` to complete the pipeline

## Project Structure

```
CKDAnalysis/
├── README.md
├── requirements.txt
├── solution1_creatinine_prediction.ipynb
├── solution2_ckd_classification.ipynb
└── solution1_output.pkl (generated after running Solution 1)
```

## Key Features

### Data Preprocessing
- Missing value imputation
- Categorical feature encoding
- Feature scaling
- Train/test split with stratification

### Exploratory Data Analysis
- Distribution analysis
- Correlation analysis
- Feature vs target visualization
- Statistical summaries

### Model Evaluation Metrics

#### Regression (Solution 1)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

#### Classification (Solution 2)
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-Score (macro)
- Confusion Matrix
- Classification Report

### Visualizations
- Distribution plots
- Correlation heatmaps
- Scatter plots
- Box plots
- Confusion matrices
- Feature importance plots
- Model comparison charts

## Technical Details

### CKD-EPI Equation
The eGFR is calculated using the CKD-EPI (Chronic Kidney Disease Epidemiology Collaboration) equation:
- Incorporates: serum creatinine, age, sex
- More accurate than older equations (e.g., MDRD)
- Widely accepted in clinical practice

### KDIGO Guidelines
Classification follows the Kidney Disease: Improving Global Outcomes (KDIGO) guidelines for CKD staging based on eGFR levels.

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Visualization
- **scikit-learn**: Machine learning models and preprocessing
- **scipy**: Scientific computing
- **xgboost**: Gradient boosting
- **catboost**: Gradient boosting
- **jupyter**: Interactive notebooks

## Authors

Created as part of the CKD Analysis project.

## License

This project is for educational and research purposes.

## Acknowledgments

- UCI Machine Learning Repository for the CKD dataset
- KDIGO for CKD staging guidelines
- CKD-EPI collaboration for the eGFR equation
