import joblib
import numpy as np
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.data import get_preprocessed_data

def train_and_evaluate_model():
    X_train, X_test, y_train, y_test, scaler, feature_names = get_preprocessed_data()
    
    # Production tuned hyperparameters
    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=42
    )
    
    # 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # Train primary production estimator
    model.fit(X_train, y_train)
    
    # Holdout validation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Game-theoretic TreeSHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    artifacts = {
        'model': model,
        'scaler': scaler,
        'explainer': explainer,
        'feature_names': feature_names,
        'cv_auc_mean': float(np.mean(cv_scores)),
        'test_auc': float(test_auc)
    }
    joblib.dump(artifacts, 'model_artifacts.pkl')
    print(f"Model fitted successfully. 5-Fold CV AUC: {np.mean(cv_scores):.4f} | Test AUC: {test_auc:.4f}")
    return artifacts

def load_artifacts():
    try:
        return joblib.load('model_artifacts.pkl')
    except Exception:
        return train_and_evaluate_model()

if __name__ == "__main__":
    train_and_evaluate_model()
