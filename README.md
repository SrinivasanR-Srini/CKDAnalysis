# CKDAnalysis
For Chronic Kidney Disease Analysis

## Overview
This project implements a comprehensive two-part machine learning pipeline for analyzing and predicting Chronic Kidney Disease (CKD) stages using the UCI CKD dataset.

## Pipeline Structure

### Part 1: Regression - Predict Serum Creatinine
- **Objective**: Predict serum creatinine levels
- **Features**: age, albumin, red_blood_cells, pus_cell, bacteria, urine_ph (specific gravity)
- **Algorithms**: 
  - Perceptron-based Regressor (MLPRegressor)
  - XGBoost Regressor
  - CatBoost Regressor

### Part 2: Classification - Predict CKD Stage
- **Objective**: Predict CKD stage based on KDIGO guidelines
- **Features**: predicted creatinine (from Part 1) and age
- **Method**: Calculate eGFR using CKD-EPI equation, classify using KDIGO stages
- **Algorithms**:
  - Multi-Layer Perceptron (MLP) Classifier
  - Support Vector Machine (SVM) Classifier

## Dataset
- **Source**: [UCI Machine Learning Repository - Chronic Kidney Disease Dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
- **Format**: ARFF file
- **Size**: 400 instances with 25 attributes

## CKD Staging (KDIGO Guidelines)
Based on estimated Glomerular Filtration Rate (eGFR):
- **Stage 1 (Normal/High)**: eGFR ≥ 90 mL/min/1.73m²
- **Stage 2 (Mild)**: eGFR 60-89 mL/min/1.73m²
- **Stage 3a (Mild to Moderate)**: eGFR 45-59 mL/min/1.73m²
- **Stage 3b (Moderate to Severe)**: eGFR 30-44 mL/min/1.73m²
- **Stage 4 (Severe)**: eGFR 15-29 mL/min/1.73m²
- **Stage 5 (Kidney Failure)**: eGFR < 15 mL/min/1.73m²

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/SrinivasanR-Srini/CKDAnalysis.git
cd CKDAnalysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook
```bash
jupyter notebook ckd_ml_pipeline.ipynb
```

### Notebook Contents
1. **Data Loading**: Downloads and loads the CKD dataset from UCI repository
2. **Exploratory Data Analysis (EDA)**: Visualizations and statistical analysis
3. **Data Preprocessing**: Handles missing values, encodes categorical variables
4. **Part 1 - Regression**: Trains and evaluates models to predict creatinine
5. **Part 2 - Classification**: Calculates eGFR and classifies CKD stages
6. **Model Comparison**: Compares performance of different algorithms
7. **Pipeline Demonstration**: End-to-end prediction example

## Key Features
- ✅ Complete data preprocessing pipeline
- ✅ Comprehensive EDA with visualizations
- ✅ Multiple regression algorithms for creatinine prediction
- ✅ eGFR calculation using CKD-EPI equation
- ✅ KDIGO-based CKD stage classification
- ✅ Multiple classification algorithms
- ✅ Performance metrics and model comparison
- ✅ End-to-end prediction demonstration

## Results
The notebook provides:
- Model performance metrics (RMSE, MAE, R² for regression; Accuracy, Precision, Recall for classification)
- Confusion matrices for classification models
- Comparative analysis of different algorithms
- Visualizations of predictions vs actual values

## Dependencies
See `requirements.txt` for a complete list of dependencies.

## License
This project is for educational and research purposes.

## Acknowledgments
- UCI Machine Learning Repository for the CKD dataset
- KDIGO guidelines for CKD classification standards
