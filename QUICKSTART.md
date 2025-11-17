# CKD Analysis - Quick Start Guide

## Overview
This guide will help you get started with the CKD (Chronic Kidney Disease) Analysis ML Pipeline.

## Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/SrinivasanR-Srini/CKDAnalysis.git
cd CKDAnalysis
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualization
- scikit-learn - ML models and preprocessing
- xgboost, catboost - Gradient boosting models
- scipy - ARFF file reading
- jupyter, notebook - Notebook environment

### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook ckd_ml_pipeline.ipynb
```

## What's Inside the Notebook

### 1. Data Loading (Cells 1-3)
- Automatically downloads the CKD dataset from UCI repository
- Loads and decodes the ARFF format data
- Displays basic dataset information

### 2. Exploratory Data Analysis (Cells 4-7)
- Statistical summaries
- Missing value analysis
- Distribution visualizations
- Correlation analysis

### 3. Data Preprocessing (Cells 8-10)
- Handles missing values
- Converts data types
- Encodes categorical variables
- Prepares data for modeling

### 4. Part 1 - Regression (Cells 11-18)
**Goal**: Predict serum creatinine levels

**Features used**:
- age
- albumin (al)
- red blood cells (rbc)
- pus cell (pc)
- bacteria (ba)
- specific gravity (sg)

**Models**:
1. **Perceptron-based Regressor** (MLPRegressor)
   - Single layer neural network
   - Good for linear relationships

2. **XGBoost Regressor**
   - Gradient boosting decision trees
   - Handles non-linear relationships well
   - Usually the best performer

3. **CatBoost Regressor**
   - Alternative gradient boosting
   - Good for categorical features

**Output**: Predicted creatinine values with performance metrics (RMSE, MAE, R²)

### 5. Part 2 - Classification (Cells 19-30)
**Goal**: Classify CKD stage based on KDIGO guidelines

**Process**:
1. Use predicted creatinine from Part 1
2. Calculate eGFR using CKD-EPI equation:
   - eGFR = 175 × (Creatinine)^(-1.154) × (Age)^(-0.203)
3. Classify into 6 CKD stages based on eGFR:
   - Stage 1: eGFR ≥ 90 (Normal/High)
   - Stage 2: eGFR 60-89 (Mild)
   - Stage 3a: eGFR 45-59 (Mild-Moderate)
   - Stage 3b: eGFR 30-44 (Moderate-Severe)
   - Stage 4: eGFR 15-29 (Severe)
   - Stage 5: eGFR < 15 (Kidney Failure)

**Models**:
1. **MLP Classifier**
   - Multi-layer neural network
   - Two hidden layers (100, 50 neurons)

2. **SVM Classifier**
   - Support Vector Machine with RBF kernel
   - Good for complex decision boundaries

**Output**: CKD stage predictions with accuracy and confusion matrices

### 6. Summary and Pipeline Demo (Cells 31-35)
- Model comparison tables
- Complete end-to-end prediction example
- Performance insights

## Running the Pipeline

### Option 1: Run All Cells
In Jupyter Notebook:
1. Click "Kernel" → "Restart & Run All"
2. Wait for all cells to execute (takes 2-5 minutes)
3. Review outputs and visualizations

### Option 2: Run Step by Step
1. Execute cells sequentially using Shift+Enter
2. Review each output before proceeding
3. Understand each stage of the pipeline

## Expected Results

### Part 1 - Regression Performance
Typical results (may vary based on data):
- XGBoost: R² ≈ 0.70-0.85
- CatBoost: R² ≈ 0.70-0.80
- Perceptron: R² ≈ 0.50-0.65

### Part 2 - Classification Performance
Typical results:
- MLP: Accuracy ≈ 0.85-0.95
- SVM: Accuracy ≈ 0.85-0.95

## Customization

### Using Your Own Data
To use custom data, modify the data loading section (Cell 3):
```python
# Instead of downloading, load your own CSV
df = pd.read_csv('your_data.csv')
```

Ensure your data has columns: age, al, rbc, pc, ba, sg, sc

### Adjusting Model Parameters
For better performance, tune hyperparameters:

**XGBoost**:
```python
xgb_reg = XGBRegressor(
    n_estimators=200,  # More trees
    learning_rate=0.05,  # Slower learning
    max_depth=7  # Deeper trees
)
```

**MLP Classifier**:
```python
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),  # More/deeper layers
    max_iter=2000  # More iterations
)
```

## Troubleshooting

### Dataset Download Fails
If automatic download fails:
1. Manually download from: https://archive.ics.uci.edu/ml/machine-learning-databases/00336/chronic_kidney_disease.arff
2. Place in the same directory as the notebook
3. Run the notebook

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Kernel Crashes
- Reduce model complexity (fewer estimators, smaller networks)
- Restart kernel and clear outputs before re-running

## Next Steps

1. **Experiment with Features**: Try different feature combinations
2. **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters
3. **Cross-Validation**: Implement k-fold cross-validation for robustness
4. **Feature Engineering**: Create new features from existing ones
5. **Model Ensemble**: Combine multiple models for better predictions

## Support

For issues or questions:
- Check the notebook comments and markdown cells
- Review the README.md for detailed documentation
- Examine the code in each cell for implementation details

## References

- KDIGO Guidelines: https://kdigo.org/guidelines/
- UCI CKD Dataset: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- CatBoost Documentation: https://catboost.ai/
