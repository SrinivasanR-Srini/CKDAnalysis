"""
Comprehensive CKD Risk Factor Prediction Analysis
Dataset: UCI Chronic Kidney Disease Risk Factor Prediction (ID: 857)

This script performs:
1. Data loading and preprocessing
2. Feature engineering (creating 'urinestate' column)
3. Problem 1: Multi-class classification for CKD stage (s1-s5)
4. Problem 2: Binary classification for CKD diagnosis (ckd vs notckd)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, 
                             roc_curve, precision_recall_curve, auc)
from sklearn.inspection import permutation_importance
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("CKD RISK FACTOR PREDICTION ANALYSIS")
print("=" * 80)


def load_data():
    """Load the UCI CKD Risk Factor dataset or generate synthetic data"""
    print("\n1. DATA LOADING")
    print("-" * 80)
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset
        ckd_risk = fetch_ucirepo(id=857)
        
        # Data (as pandas dataframes)
        X_data = ckd_risk.data.features
        y_data = ckd_risk.data.targets
        
        # Combine features and target into a single dataframe
        df = pd.concat([X_data, y_data], axis=1)
        
        print("Dataset loaded successfully from UCI repository!")
        print(f"Dataset Shape: {df.shape}")
        
    except Exception as e:
        print(f"Could not load from UCI repository: {e}")
        print("\nGenerating synthetic data based on dataset structure...")
        
        # Generate synthetic data that matches the expected structure
        np.random.seed(42)
        n_samples = 200
        
        # Generate features
        data = {
            'age': np.random.randint(20, 80, n_samples),
            'bp': np.random.randint(60, 140, n_samples),
            'sg': np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_samples),
            'al': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
            'su': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
            'rbc': np.random.choice([0, 1], n_samples),
            'pc': np.random.choice([0, 1], n_samples),
            'pcc': np.random.choice([0, 1], n_samples),
            'ba': np.random.choice([0, 1], n_samples),
            'bgr': np.random.randint(70, 200, n_samples),
            'bu': np.random.randint(10, 150, n_samples),
            'sc': np.random.uniform(0.5, 8.0, n_samples),
            'sod': np.random.randint(120, 160, n_samples),
            'pot': np.random.uniform(2.5, 6.0, n_samples),
            'hemo': np.random.uniform(8.0, 18.0, n_samples),
            'pcv': np.random.randint(25, 55, n_samples),
            'wbcc': np.random.randint(4000, 15000, n_samples),
            'rbcc': np.random.uniform(2.5, 6.5, n_samples),
            'htn': np.random.choice([0, 1], n_samples),
            'dm': np.random.choice([0, 1], n_samples),
            'cad': np.random.choice([0, 1], n_samples),
            'appet': np.random.choice([0, 1], n_samples),
            'pe': np.random.choice([0, 1], n_samples),
            'ane': np.random.choice([0, 1], n_samples),
        }
        
        # Generate stage (s1-s5) based on some logic
        stage_probs = []
        for i in range(n_samples):
            # Simple heuristic: worse indicators -> higher stage
            score = (data['al'][i] / 5 + data['sc'][i] / 8 +
                     (1 if data['htn'][i] == 1 else 0) +
                     (1 if data['dm'][i] == 1 else 0)) / 4
            
            if score < 0.2:
                stage_probs.append(np.random.choice(['s1', 's2'], p=[0.7, 0.3]))
            elif score < 0.4:
                stage_probs.append(np.random.choice(['s2', 's3'], p=[0.5, 0.5]))
            elif score < 0.6:
                stage_probs.append(np.random.choice(['s3', 's4'], p=[0.5, 0.5]))
            elif score < 0.8:
                stage_probs.append(np.random.choice(['s4', 's5'], p=[0.5, 0.5]))
            else:
                stage_probs.append('s5')
        
        data['stage'] = stage_probs
        
        # Generate class (ckd/notckd) based on stage
        data['class'] = ['notckd' if s in ['s1', 's2'] and np.random.random() > 0.3
                         else 'ckd' for s in data['stage']]
        
        df = pd.DataFrame(data)
        print("Synthetic dataset generated successfully!")
        print(f"Dataset Shape: {df.shape}")
    
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


def preprocess_data(df):
    """Preprocess the data and create urinestate feature"""
    print("\n2. DATA PREPROCESSING")
    print("-" * 80)
    
    # Check for missing values
    print("Missing values per column:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
        
        # Handle missing values
        print("\nHandling missing values...")
        
        # For numerical columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns: fill with mode
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print(f"Remaining missing values: {df.isnull().sum().sum()}")
    else:
        print("No missing values found!")
    
    # Create 'urinestate' feature
    # urinestate = 1 if ANY of (rbc, pc, pcc, ba) equals 1, else 0
    df['urinestate'] = ((df['rbc'] == 1) | (df['pc'] == 1) |
                         (df['pcc'] == 1) | (df['ba'] == 1)).astype(int)
    
    print("\n'urinestate' feature created successfully!")
    print(f"Urinestate distribution:")
    print(df['urinestate'].value_counts())
    print(f"Percentage with urinestate=1: {df['urinestate'].mean()*100:.2f}%")
    
    return df


def exploratory_analysis(df):
    """Perform exploratory data analysis"""
    print("\n3. EXPLORATORY DATA ANALYSIS")
    print("-" * 80)
    
    # Target variable distributions
    print("\nTarget Variable Distributions:")
    print("\n1. CKD Stage Distribution:")
    print(df['stage'].value_counts().sort_index())
    
    print("\n2. CKD Class Distribution:")
    print(df['class'].value_counts())
    
    # Feature distributions
    print("\nSelected Feature Statistics:")
    print(df[['age', 'al', 'urinestate']].describe())
    
    # Create visualization directory
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Create visualizations
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # Row 1: Feature distributions
    axes[0, 0].hist(df['age'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Age Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    al_counts = df['al'].value_counts().sort_index()
    axes[0, 1].bar(al_counts.index, al_counts.values, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Albumin (al) Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Albumin Level')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    urinestate_counts = df['urinestate'].value_counts().sort_index()
    axes[0, 2].bar(urinestate_counts.index, urinestate_counts.values, edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Urinestate Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Urinestate (0/1)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Target variable distributions
    stage_counts = df['stage'].value_counts().sort_index()
    axes[1, 0].bar(range(len(stage_counts)), stage_counts.values,
                   tick_label=stage_counts.index, edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('CKD Stage Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Stage')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    class_counts = df['class'].value_counts()
    axes[1, 1].bar(class_counts.index, class_counts.values, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('CKD Class Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Box plot for age by class
    df.boxplot(column='age', by='class', ax=axes[1, 2])
    axes[1, 2].set_title('Age Distribution by CKD Class', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Age')
    plt.sca(axes[1, 2])
    plt.xticks(rotation=0)
    
    # Row 3: Relationships
    scatter = axes[2, 0].scatter(df['age'], df['al'], alpha=0.5, c=df['urinestate'], cmap='coolwarm')
    axes[2, 0].set_title('Age vs Albumin (colored by urinestate)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Age')
    axes[2, 0].set_ylabel('Albumin Level')
    axes[2, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2, 0])
    
    # Correlation heatmap
    selected_features = ['age', 'al', 'urinestate', 'rbc', 'pc', 'pcc', 'ba']
    corr_matrix = df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=axes[2, 1], cbar_kws={'shrink': 0.8})
    axes[2, 1].set_title('Correlation Heatmap', fontsize=12, fontweight='bold')
    
    # Feature importance overview
    feature_data = df[['age', 'al', 'urinestate']].copy()
    axes[2, 2].boxplot([feature_data['age']/feature_data['age'].max(),
                         feature_data['al']/feature_data['al'].max(),
                         feature_data['urinestate']])
    axes[2, 2].set_xticklabels(['Age\n(normalized)', 'Albumin\n(normalized)', 'Urinestate'])
    axes[2, 2].set_title('Feature Value Distributions (Normalized)', fontsize=12, fontweight='bold')
    axes[2, 2].set_ylabel('Normalized Value')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/eda_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nEDA visualizations saved to 'visualizations/eda_overview.png'")


def problem1_stage_classification(df):
    """Problem 1: Multi-class Classification for CKD Stage"""
    print("\n4. PROBLEM 1: MULTI-CLASS CLASSIFICATION FOR CKD STAGE")
    print("=" * 80)
    
    # Select features and target
    feature_cols = ['age', 'al', 'urinestate']
    X_stage = df[feature_cols].copy()
    y_stage = df['stage'].copy()
    
    print(f"\nFeatures shape: {X_stage.shape}")
    print(f"Target shape: {y_stage.shape}")
    print(f"Target classes: {sorted(y_stage.unique())}")
    print(f"\nClass distribution:")
    print(y_stage.value_counts().sort_index())
    
    # Encode target labels
    le_stage = LabelEncoder()
    y_stage_encoded = le_stage.fit_transform(y_stage)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_stage, y_stage_encoded, test_size=0.2, random_state=42, stratify=y_stage_encoded
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\nTraining models with hyperparameter tuning...")
    print("-" * 80)
    
    models = {}
    results = {}
    
    # 1. Random Forest
    print("\n1. Random Forest Classifier...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = RandomizedSearchCV(rf, rf_params, n_iter=15, cv=5, random_state=42, n_jobs=-1)
    rf_grid.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf_grid.best_estimator_
    print(f"   Best params: {rf_grid.best_params_}")
    print(f"   Best CV score: {rf_grid.best_score_:.4f}")
    
    # 2. Gradient Boosting
    print("\n2. Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(random_state=42)
    gb_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb_grid = RandomizedSearchCV(gb, gb_params, n_iter=15, cv=5, random_state=42, n_jobs=-1)
    gb_grid.fit(X_train_scaled, y_train)
    models['Gradient Boosting'] = gb_grid.best_estimator_
    print(f"   Best params: {gb_grid.best_params_}")
    print(f"   Best CV score: {gb_grid.best_score_:.4f}")
    
    # 3. SVM
    print("\n3. SVM Classifier...")
    svm = SVC(random_state=42, probability=True)
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1)
    svm_grid.fit(X_train_scaled, y_train)
    models['SVM'] = svm_grid.best_estimator_
    print(f"   Best params: {svm_grid.best_params_}")
    print(f"   Best CV score: {svm_grid.best_score_:.4f}")
    
    # 4. Logistic Regression
    print("\n4. Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
    lr_params = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'saga']
    }
    lr_grid = GridSearchCV(lr, lr_params, cv=5, n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr_grid.best_estimator_
    print(f"   Best params: {lr_grid.best_params_}")
    print(f"   Best CV score: {lr_grid.best_score_:.4f}")
    
    # Evaluate models
    print("\n" + "=" * 80)
    print("MODEL EVALUATION - STAGE CLASSIFICATION")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 80)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        results[name] = {
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Predictions': y_test_pred
        }
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1-Score:       {f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred,
                                   target_names=le_stage.classes_,
                                   zero_division=0))
    
    # Results comparison
    results_df = pd.DataFrame(results).T
    results_df_display = results_df.drop('Predictions', axis=1)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - STAGE CLASSIFICATION")
    print("=" * 80)
    print(results_df_display.to_string())
    
    best_model = results_df['Test Accuracy'].idxmax()
    print(f"\n>>> Best Model: {best_model} (Test Accuracy: {results_df.loc[best_model, 'Test Accuracy']:.4f})")
    
    # Visualizations
    visualize_problem1_results(models, results, le_stage, y_test)
    
    return models, results, le_stage, X_test_scaled, y_test, feature_cols


def visualize_problem1_results(models, results, le_stage, y_test):
    """Create visualizations for Problem 1"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Confusion matrices
    for idx, (name, model) in enumerate(models.items()):
        row = idx // 3
        col = idx % 3
        
        y_pred = results[name]['Predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le_stage.classes_,
                    yticklabels=le_stage.classes_,
                    ax=axes[row, col], cbar_kws={'shrink': 0.8})
        axes[row, col].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    # Model comparison
    results_df = pd.DataFrame(results).T.drop('Predictions', axis=1)
    axes[1, 2].bar(range(len(models)), results_df['Test Accuracy'],
                   tick_label=list(models.keys()), alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Model Comparison - Test Accuracy', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(results_df['Test Accuracy']):
        axes[1, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/problem1_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nProblem 1 visualizations saved to 'visualizations/problem1_confusion_matrices.png'")
    
    # Feature importance
    visualize_feature_importance(models, 'problem1', y_test)


def visualize_feature_importance(models, problem_name, y_test, X_test_scaled=None, feature_cols=None):
    """Visualize feature importance for models"""
    # This is a placeholder - actual implementation would need X_test_scaled and feature_cols
    print(f"Feature importance analysis for {problem_name} would be performed here")


def problem2_class_classification(df):
    """Problem 2: Binary Classification for CKD Diagnosis"""
    print("\n5. PROBLEM 2: BINARY CLASSIFICATION FOR CKD DIAGNOSIS")
    print("=" * 80)
    
    # Select features and target
    feature_cols = ['age', 'al', 'urinestate']
    X_class = df[feature_cols].copy()
    y_class = df['class'].copy()
    
    print(f"\nFeatures shape: {X_class.shape}")
    print(f"Target shape: {y_class.shape}")
    print(f"Target classes: {sorted(y_class.unique())}")
    print(f"\nClass distribution:")
    print(y_class.value_counts())
    
    # Encode target labels
    le_class = LabelEncoder()
    y_class_encoded = le_class.fit_transform(y_class)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_class, y_class_encoded, test_size=0.2, random_state=42, stratify=y_class_encoded
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\nTraining models with hyperparameter tuning...")
    print("-" * 80)
    
    models = {}
    results = {}
    
    # 1. Random Forest
    print("\n1. Random Forest Classifier...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    rf_grid = RandomizedSearchCV(rf, rf_params, n_iter=15, cv=5, random_state=42, n_jobs=-1)
    rf_grid.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf_grid.best_estimator_
    print(f"   Best params: {rf_grid.best_params_}")
    print(f"   Best CV score: {rf_grid.best_score_:.4f}")
    
    # 2. Gradient Boosting
    print("\n2. Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(random_state=42)
    gb_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb_grid = RandomizedSearchCV(gb, gb_params, n_iter=15, cv=5, random_state=42, n_jobs=-1)
    gb_grid.fit(X_train_scaled, y_train)
    models['Gradient Boosting'] = gb_grid.best_estimator_
    print(f"   Best params: {gb_grid.best_params_}")
    print(f"   Best CV score: {gb_grid.best_score_:.4f}")
    
    # 3. SVM
    print("\n3. SVM Classifier...")
    svm = SVC(random_state=42, probability=True)
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1)
    svm_grid.fit(X_train_scaled, y_train)
    models['SVM'] = svm_grid.best_estimator_
    print(f"   Best params: {svm_grid.best_params_}")
    print(f"   Best CV score: {svm_grid.best_score_:.4f}")
    
    # 4. Logistic Regression
    print("\n4. Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_params = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    lr_grid = GridSearchCV(lr, lr_params, cv=5, n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr_grid.best_estimator_
    print(f"   Best params: {lr_grid.best_params_}")
    print(f"   Best CV score: {lr_grid.best_score_:.4f}")
    
    # Evaluate models
    print("\n" + "=" * 80)
    print("MODEL EVALUATION - CLASS CLASSIFICATION")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 80)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        
        results[name] = {
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Predictions': y_test_pred,
            'Probabilities': y_test_proba
        }
        
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1-Score:       {f1:.4f}")
        print(f"ROC-AUC:        {roc_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred,
                                   target_names=le_class.classes_,
                                   zero_division=0))
    
    # Results comparison
    results_df = pd.DataFrame(results).T
    results_df_display = results_df.drop(['Predictions', 'Probabilities'], axis=1)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON - CLASS CLASSIFICATION")
    print("=" * 80)
    print(results_df_display.to_string())
    
    best_model = results_df['Test Accuracy'].idxmax()
    print(f"\n>>> Best Model: {best_model} (Test Accuracy: {results_df.loc[best_model, 'Test Accuracy']:.4f})")
    
    # Visualizations
    visualize_problem2_results(models, results, le_class, y_test)
    
    return models, results, le_class


def visualize_problem2_results(models, results, le_class, y_test):
    """Create visualizations for Problem 2"""
    # Confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    for idx, (name, model) in enumerate(models.items()):
        row = idx // 3
        col = idx % 3
        
        y_pred = results[name]['Predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=le_class.classes_,
                    yticklabels=le_class.classes_,
                    ax=axes[row, col], cbar_kws={'shrink': 0.8})
        axes[row, col].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    # Model comparison
    results_df = pd.DataFrame(results).T.drop(['Predictions', 'Probabilities'], axis=1)
    axes[1, 2].bar(range(len(models)), results_df['Test Accuracy'],
                   tick_label=list(models.keys()), alpha=0.7, edgecolor='black', color='steelblue')
    axes[1, 2].set_title('Model Comparison - Test Accuracy', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(results_df['Test Accuracy']):
        axes[1, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/problem2_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nProblem 2 visualizations saved to 'visualizations/problem2_confusion_matrices.png'")
    
    # ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curves
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['Probabilities'])
        roc_auc = result['ROC-AUC']
        axes[0].plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    axes[0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0].set_title('ROC Curves - Binary Classification', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curves
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['Probabilities'])
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, linewidth=2, label=f'{name} (AUC = {pr_auc:.3f})')
    
    axes[1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1].set_title('Precision-Recall Curves - Binary Classification', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/problem2_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("ROC and PR curves saved to 'visualizations/problem2_roc_pr_curves.png'")


def print_final_summary(df, results_stage, results_class):
    """Print final summary of the analysis"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE CKD RISK FACTOR PREDICTION ANALYSIS - FINAL SUMMARY")
    print("=" * 80)
    
    print("\n1. DATASET INFORMATION")
    print("-" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Features used: age, al, urinestate")
    print(f"Target variables: stage (multi-class), class (binary)")
    
    print("\n2. PROBLEM 1: STAGE CLASSIFICATION (Multi-class)")
    print("-" * 80)
    # Create DataFrame and ensure numeric types
    results_df_stage = pd.DataFrame(results_stage).T
    results_df_stage = results_df_stage.drop('Predictions', axis=1)
    # Convert to numeric
    for col in results_df_stage.columns:
        results_df_stage[col] = pd.to_numeric(results_df_stage[col], errors='coerce')
    
    print("Top 3 models:")
    top_3_stage = results_df_stage.nlargest(3, 'Test Accuracy')[['Test Accuracy', 'F1-Score']]
    for idx, (model_name, row) in enumerate(top_3_stage.iterrows(), 1):
        print(f"{idx}. {model_name}: Test Acc = {row['Test Accuracy']:.4f}, F1 = {row['F1-Score']:.4f}")
    
    print("\n3. PROBLEM 2: CLASS CLASSIFICATION (Binary)")
    print("-" * 80)
    # Create DataFrame and ensure numeric types
    results_df_class = pd.DataFrame(results_class).T
    results_df_class = results_df_class.drop(['Predictions', 'Probabilities'], axis=1)
    # Convert to numeric
    for col in results_df_class.columns:
        results_df_class[col] = pd.to_numeric(results_df_class[col], errors='coerce')
    
    print("Top 3 models:")
    top_3_class = results_df_class.nlargest(3, 'Test Accuracy')[['Test Accuracy', 'F1-Score', 'ROC-AUC']]
    for idx, (model_name, row) in enumerate(top_3_class.iterrows(), 1):
        print(f"{idx}. {model_name}: Test Acc = {row['Test Accuracy']:.4f}, F1 = {row['F1-Score']:.4f}, ROC-AUC = {row['ROC-AUC']:.4f}")
    
    print("\n4. KEY INSIGHTS")
    print("-" * 80)
    print("• Feature 'urinestate' was derived from rbc, pc, pcc, and ba")
    print("• Three features (age, al, urinestate) were used for both classification problems")
    print("• Multiple models were trained with hyperparameter tuning")
    print("• Comprehensive evaluation metrics were calculated for both problems")
    print("• Visualizations saved in 'visualizations/' directory")
    
    print("\n5. RECOMMENDATIONS")
    print("-" * 80)
    print(f"• For stage prediction: Use {results_df_stage['Test Accuracy'].idxmax()}")
    print(f"• For CKD diagnosis: Use {results_df_class['Test Accuracy'].idxmax()}")
    print("• Consider feature engineering to improve model performance")
    print("• Collect more data if possible to improve generalization")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def main():
    """Main function to run the complete analysis"""
    # Load data
    df = load_data()
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Exploratory analysis
    exploratory_analysis(df)
    
    # Problem 1: Stage classification
    models_stage, results_stage, le_stage, X_test_scaled, y_test, feature_cols = problem1_stage_classification(df)
    
    # Problem 2: Class classification
    models_class, results_class, le_class = problem2_class_classification(df)
    
    # Print final summary
    print_final_summary(df, results_stage, results_class)
    
    print("\nAll visualizations have been saved to the 'visualizations/' directory")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
