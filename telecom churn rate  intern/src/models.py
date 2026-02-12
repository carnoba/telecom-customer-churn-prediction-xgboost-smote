from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import pandas as pd
import numpy as np

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance the training data.
    """
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Post-SMOTE class distribution: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled

def train_models(X_train, y_train):
    """
    Train and optimize Logistic Regression, Random Forest, and XGBoost.
    """
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # 3. XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train, y_train)
    
    return {
        'Logistic Regression': lr,
        'Random Forest': rf,
        'XGBoost': xgb
    }

def evaluate_models(models, X_test, y_test):
    """
    Evaluate each model and return metrics.
    """
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        results[name] = {
            'auc': auc,
            'cm': cm,
            'fpr': fpr,
            'tpr': tpr,
            'prob': y_prob
        }
        print(f"{name} AUC: {auc:.4f}")
        
    return results
