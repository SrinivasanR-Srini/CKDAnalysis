# CKD Analysis - Machine Learning Pipeline

A comprehensive machine learning pipeline for Chronic Kidney Disease (CKD) analysis using UCI CKD datasets.

## Overview

This project implements comprehensive machine learning analyses for CKD:

1. **Solution 1**: Creatinine Prediction (Regression) - Uses UCI dataset ID 336
2. **Solution 2**: CKD Level Classification (Multiclass Classification) - Uses UCI dataset ID 336
3. **CKD Risk Factor Analysis**: Comprehensive risk factor prediction - Uses UCI dataset ID 857

## Datasets

### Original Analysis (Solutions 1 & 2)
- **Source**: [UCI Machine Learning Repository - Chronic Kidney Disease Dataset (ID: 336)](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- **Features Used**:
  - Age
  - Albumin (urine)
  - RBC (Red Blood Cells - nominal)
  - Pus Cell (nominal)
  - Bacteria (nominal)
  - Urine pH

### New Analysis: CKD Risk Factor Prediction
- **Source**: [UCI Machine Learning Repository - Risk Factor Prediction of CKD (ID: 857)](https://archive.ics.uci.edu/dataset/857/risk+factor+prediction+of+chronic+kidney+disease)
- **Features Used**:
  - age: Patient age
  - al: Albumin levels
  - urinestate: Derived feature (1 if any of rbc, pc, pcc, ba equals 1, else 0)
- **Analysis includes**:
  - **Problem 1**: Multi-class classification for CKD stage (s1/s2/s3/s4/s5)
  - **Problem 2**: Binary classification for CKD diagnosis (ckd vs notckd)

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

## CKD Risk Factor Analysis

### Objective
Comprehensive analysis for CKD risk factor prediction using multiple classification approaches.

### Features
- **Automated data loading** from UCI repository with fallback to synthetic data
- **Feature engineering**: Creation of 'urinestate' feature
- **Comprehensive EDA** with multiple visualizations
- **Two classification problems**:
  - Multi-class stage prediction (s1-s5)
  - Binary CKD diagnosis (ckd vs notckd)

### Models Implemented
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- Logistic Regression

### Analysis Pipeline
1. Data import and preprocessing
2. Missing value handling
3. Feature engineering (urinestate creation)
4. Exploratory Data Analysis
5. Model training with hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
6. Comprehensive evaluation metrics
7. Visualization generation

### Evaluation Metrics
- **Multi-class (Stage)**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrices
  - Feature importance
  
- **Binary (Class)**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC scores
  - ROC curves
  - Precision-Recall curves
  - Confusion matrices
  - Feature importance

### Visualizations Generated
- EDA overview (feature distributions, correlations, target distributions)
- Confusion matrices for all models (both problems)
- Model comparison charts
- ROC curves (binary classification)
- Precision-Recall curves (binary classification)

All visualizations are automatically saved to the `visualizations/` directory.

## Installation

### Requirements
- Python 3.8+
- Jupyter Notebook (for notebooks)

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

### Running the Original Notebooks

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

### Running the CKD Risk Factor Analysis

Execute the Python script:
```bash
python3 ckd_risk_analysis.py
```

This will:
- Attempt to load data from UCI repository (dataset ID: 857)
- Generate synthetic data if UCI access fails
- Perform comprehensive preprocessing and EDA
- Train and evaluate 4 models on 2 classification problems
- Generate visualizations in the `visualizations/` directory
- Print detailed analysis results and recommendations

### Expected Workflow

**For Original Analysis:**
1. Run `solution1_creatinine_prediction.ipynb` first
2. This will create `solution1_output.pkl` with predictions
3. Run `solution2_ckd_classification.ipynb` to complete the pipeline

**For Risk Factor Analysis:**
1. Simply run `python3 ckd_risk_analysis.py`
2. Check the `visualizations/` directory for generated plots
3. Review the console output for detailed metrics and insights

## Project Structure

```
CKDAnalysis/
├── README.md
├── requirements.txt
├── solution1_creatinine_prediction.ipynb
├── solution2_ckd_classification.ipynb
├── ckd_risk_analysis.py (NEW)
├── visualizations/ (generated - NEW)
│   ├── eda_overview.png
│   ├── problem1_confusion_matrices.png
│   ├── problem2_confusion_matrices.png
│   └── problem2_roc_pr_curves.png
└── solution1_output.pkl (generated after running Solution 1)
```

## Key Features

### Data Preprocessing
- Missing value imputation
- Categorical feature encoding
- Feature scaling
- Train/test split with stratification
- Feature engineering (urinestate creation)

### Exploratory Data Analysis
- Distribution analysis
- Correlation analysis
- Feature vs target visualization
- Statistical summaries
- Comprehensive visualization suite

### Model Evaluation Metrics

#### Regression (Solution 1)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

#### Classification (Solution 2, Risk Factor Analysis)
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-Score (macro)
- ROC-AUC (binary classification)
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
- ROC curves (binary classification)
- Precision-Recall curves (binary classification)

## Technical Details

### CKD-EPI Equation
The eGFR is calculated using the CKD-EPI (Chronic Kidney Disease Epidemiology Collaboration) equation:
- Incorporates: serum creatinine, age, sex
- More accurate than older equations (e.g., MDRD)
- Widely accepted in clinical practice

### KDIGO Guidelines
Classification follows the Kidney Disease: Improving Global Outcomes (KDIGO) guidelines for CKD staging based on eGFR levels.

### Hyperparameter Tuning
- GridSearchCV for exhaustive search
- RandomizedSearchCV for efficient exploration
- Cross-validation (5-fold) for model selection
- Stratified splits to preserve class distributions

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Visualization
- **scikit-learn**: Machine learning models and preprocessing
- **scipy**: Scientific computing
- **xgboost**: Gradient boosting
- **catboost**: Gradient boosting
- **jupyter**: Interactive notebooks
- **ucimlrepo**: UCI ML Repository data access

## Authors

Created as part of the CKD Analysis project.

## License

This project is for educational and research purposes.

## Acknowledgments

- UCI Machine Learning Repository for the CKD datasets
- KDIGO for CKD staging guidelines
- CKD-EPI collaboration for the eGFR equation

