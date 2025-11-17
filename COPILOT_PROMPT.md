# GitHub Copilot Prompt for CKD Analysis ML Pipeline

## Role
You are an expert Machine Learning Engineer specializing in healthcare analytics and biomedical data science. You have extensive experience with:
- Medical data preprocessing and feature engineering
- Clinical calculations (eGFR, KDIGO classifications)
- Regression and classification models
- Ensemble methods and neural networks
- Python scientific computing stack

## Task
Create a comprehensive two-stage machine learning pipeline for Chronic Kidney Disease (CKD) analysis using Python Jupyter notebooks.

## Context
Chronic Kidney Disease is a serious health condition that requires accurate prediction and classification for early intervention. This project will:
1. First predict serum creatinine levels (a key kidney function marker)
2. Then use those predictions to classify CKD severity stages

## Dataset
- **Source**: UCI Machine Learning Repository - Chronic Kidney Disease Dataset
- **URL**: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- **Format**: ARFF file format
- **Import Method**: Use Python with `scipy.io.arff` and `requests` libraries

## Features to Use
### Required Features
1. **Age** (numeric) - Patient age in years
2. **Albumin** (numeric) - Urine albumin levels (al)
3. **RBC** (nominal/categorical) - Red Blood Cells in urine (rbc)
4. **Pus Cell** (nominal/categorical) - Pus cells in urine (pc)
5. **Bacteria** (nominal/categorical) - Bacteria presence in urine (ba)
6. **Urine pH** (numeric) - Urine pH level (ph)

### Additional Feature
- **Serum Creatinine** (sc) - Used as target in Solution 1, predictor in Solution 2

## Solution 1: Creatinine Prediction (Regression)

### Objective
Predict serum creatinine values from the selected features.

### Requirements

#### 1. Data Loading and Exploration
- Download dataset from UCI repository programmatically
- Parse ARFF format using scipy
- Display dataset shape, columns, and basic statistics
- Handle byte string decoding from ARFF format

#### 2. Data Preprocessing
- Identify and handle missing values (replace '?', '\t?', etc. with NaN)
- Impute numeric features using median strategy
- Impute categorical features using mode strategy
- Convert data types appropriately (numeric vs categorical)

#### 3. Exploratory Data Analysis (EDA)
Create comprehensive visualizations:
- Target variable (serum creatinine) distribution with histogram, box plot, and Q-Q plot
- Numeric feature distributions (age, albumin, pH)
- Categorical feature distributions (RBC, Pus Cell, Bacteria)
- Correlation heatmap for numeric features
- Scatter plots: numeric features vs target
- Box plots: categorical features vs target

#### 4. Feature Engineering
- Encode categorical features using LabelEncoder
- Create feature matrix (X) and target vector (y)
- Split data into train/test sets (80/20 split)
- Apply StandardScaler for feature scaling

#### 5. Model Training - Three Models Required

**Model 1: Perceptron-based**
- Use MLPRegressor with single hidden layer (perceptron architecture)
- Linear activation function for regression
- SGD solver
- Include early stopping

**Model 2: XGBoost (Ensemble)**
- XGBRegressor with appropriate hyperparameters
- n_estimators=100, learning_rate=0.1, max_depth=5
- Calculate feature importance

**Model 3: CatBoost (Ensemble)**
- CatBoostRegressor with similar hyperparameters
- iterations=100, learning_rate=0.1, depth=5
- Suppress verbose output

#### 6. Model Evaluation
For each model, calculate:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

For both training and test sets.

#### 7. Model Comparison
- Create comparison table with all metrics
- Visualize RMSE and R² scores across models
- Create scatter plots: actual vs predicted for each model
- Identify best performing model

#### 8. Save Output for Solution 2
Save to pickle file (`solution1_output.pkl`):
- Test set features (X_test)
- Actual creatinine values (y_test)
- Predicted creatinine values from best model
- Best model object
- Scaler object
- Label encoders
- Original dataframe
- Test set indices

## Solution 2: CKD Level Classification (Multiclass)

### Objective
Classify CKD stages using eGFR calculations based on predicted creatinine values.

### Requirements

#### 1. Load Data from Solution 1
- Load pickle file from Solution 1
- Extract predicted creatinine values
- Extract necessary metadata (age, sex)

#### 2. eGFR Calculation - CKD-EPI Equation

Implement the CKD-EPI (Chronic Kidney Disease Epidemiology Collaboration) equation:

**Formula varies by sex and creatinine level:**

For **Females**:
- If Scr ≤ 0.7: eGFR = 144 × (Scr/0.7)^(-0.329) × 0.993^Age
- If Scr > 0.7: eGFR = 144 × (Scr/0.7)^(-1.209) × 0.993^Age

For **Males**:
- If Scr ≤ 0.9: eGFR = 141 × (Scr/0.9)^(-0.411) × 0.993^Age
- If Scr > 0.9: eGFR = 141 × (Scr/0.9)^(-1.209) × 0.993^Age

**Inputs:**
- Serum creatinine (mg/dL)
- Age (years)
- Sex (male/female)

**Output:**
- eGFR in mL/min/1.73m²

Create function `calculate_egfr_ckd_epi(creatinine, age, sex)` to implement this.

#### 3. KDIGO Classification

Apply KDIGO (Kidney Disease: Improving Global Outcomes) guidelines for CKD staging:

| Stage | eGFR Range | Description |
|-------|-----------|-------------|
| **G1** | ≥ 90 | Normal or high kidney function |
| **G2** | 60-89 | Mildly decreased kidney function |
| **G3a** | 45-59 | Mildly to moderately decreased |
| **G3b** | 30-44 | Moderately to severely decreased |
| **G4** | 15-29 | Severely decreased kidney function |
| **G5** | < 15 | Kidney failure |

Create function `classify_ckd_stage(egfr)` to implement this.

#### 4. Data Preparation for Classification
- Calculate eGFR for all samples (both predicted and actual creatinine)
- Classify all samples into CKD stages
- Display stage distributions
- Visualize:
  - eGFR distributions (actual vs predicted)
  - eGFR scatter plot (actual vs predicted)
  - CKD stage distributions (bar plot comparison)
  - eGFR vs Age scatter plot

#### 5. Feature Engineering for Classification
- Use original features plus creatinine
- Encode all categorical features (including sex)
- Create feature matrix with: age, albumin, pH, serum creatinine, RBC, Pus Cell, Bacteria, Sex
- Encode target CKD stages using LabelEncoder
- Split data with stratification to preserve class distribution
- Apply StandardScaler

#### 6. Model Training - Two Classification Models

**Model 1: Multi-Layer Perceptron (MLP)**
- MLPClassifier with multiple hidden layers: (100, 50, 25)
- ReLU activation function
- Adam solver
- Early stopping with 10% validation split
- max_iter=500

**Model 2: Support Vector Machine (SVM)**
- SVC with RBF kernel
- C=10, gamma='scale'
- Enable probability estimates (probability=True)

#### 7. Model Evaluation

For each model, calculate:
- Accuracy (train and test)
- Precision (macro average)
- Recall (macro average)
- F1-Score (macro average)
- Confusion Matrix
- Classification Report (per-class metrics)

#### 8. Cross-Validation
- Perform 5-fold cross-validation
- Calculate mean and standard deviation for:
  - Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1-Score (macro)

#### 9. Feature Importance
- Use permutation importance for both models
- Visualize feature importance with horizontal bar plots
- Identify most important features for CKD classification

#### 10. Visualizations
Create comprehensive visualizations:
- Confusion matrices (heatmaps) for both models
- Model comparison bar charts (Accuracy, F1-Score)
- Feature importance plots
- eGFR distribution plots
- CKD stage distribution comparisons

#### 11. Final Summary
Print comprehensive summary including:
- Dataset statistics
- Features used
- eGFR calculation method
- Number of classes and class names
- Train/test split sizes
- Model performance comparison table
- Best performing model identification
- Complete pipeline summary

## Guidelines

### Code Style
- Use clear, descriptive variable names
- Add comments for complex calculations (especially eGFR)
- Use markdown cells to organize notebook into clear sections
- Include section headers and explanations

### Best Practices
- Set random seed (42) for reproducibility
- Suppress warnings for cleaner output
- Use appropriate visualization color schemes
- Label all plots with clear titles and axis labels
- Use tight_layout() for better plot spacing

### Error Handling
- Handle missing values appropriately
- Check for byte string encoding issues in ARFF data
- Validate data types after conversion
- Handle division by zero in calculations

### Output Format
- Both solutions must be Jupyter notebooks (.ipynb format)
- Notebooks should be runnable end-to-end
- File names:
  - `solution1_creatinine_prediction.ipynb`
  - `solution2_ckd_classification.ipynb`

## Required Libraries

Create a `requirements.txt` with:
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
scipy>=1.10.0
xgboost>=1.7.0
catboost>=1.2.0
jupyter>=1.0.0
notebook>=6.5.0
requests>=2.28.0
```

## Deliverables

1. **solution1_creatinine_prediction.ipynb** - Complete regression pipeline
2. **solution2_ckd_classification.ipynb** - Complete classification pipeline
3. **requirements.txt** - All dependencies
4. **README.md** - Comprehensive documentation
5. **solution1_output.pkl** - Intermediate results (auto-generated)

## Success Criteria

✅ Both notebooks run end-to-end without errors  
✅ All required models implemented (Perceptron, XGBoost, CatBoost, MLP, SVM)  
✅ eGFR calculated correctly using CKD-EPI equation  
✅ KDIGO classification implemented correctly  
✅ Comprehensive EDA with visualizations  
✅ Proper preprocessing and feature engineering  
✅ Model evaluation with multiple metrics  
✅ Clear documentation and code comments  
✅ Results saved and passed between solutions  

## Expected Outcomes

- Accurate creatinine prediction models (R² > 0.7 expected)
- Reliable CKD stage classification (Accuracy > 0.8 expected)
- Clear model comparisons showing best performers
- Actionable insights from feature importance
- Production-ready, well-documented code

## Medical Context Notes

- Serum creatinine is a key biomarker for kidney function
- Higher creatinine indicates worse kidney function
- eGFR decreases as kidney disease progresses
- Early CKD detection (stages G1-G2) enables intervention
- Advanced stages (G4-G5) require specialized treatment

## Implementation Notes

- Solution 1 focuses on **regression** (predicting continuous values)
- Solution 2 focuses on **multiclass classification** (6 CKD stages)
- Pipeline demonstrates realistic medical ML workflow
- Results from Solution 1 directly feed into Solution 2
- Both solutions should maintain high code quality and documentation standards
