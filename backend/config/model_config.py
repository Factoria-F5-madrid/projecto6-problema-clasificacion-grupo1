"""
Configuration for ML models and hyperparameters
"""

from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier  # Temporarily disabled due to OpenMP issue

class ModelConfig:
    """Configuration class for ML models and their hyperparameters"""
    
    # Base models for classification
    BASE_MODELS = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'svm': SVC(random_state=42),
        'naive_bayes': GaussianNB(),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        # 'xgboost': XGBClassifier(random_state=42, eval_metric='logloss')  # Temporarily disabled
    }
    
    # Hyperparameter grids for optimization
    HYPERPARAMETER_GRIDS = {
        'logistic_regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        },
        'naive_bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'decision_tree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        },
        # 'xgboost': {
        #     'n_estimators': [50, 100, 200],
        #     'learning_rate': [0.01, 0.1, 0.2],
        #     'max_depth': [3, 5, 7],
        #     'subsample': [0.8, 0.9, 1.0]
        # }
    }
    
    # Cross-validation settings
    CV_SETTINGS = {
        'k_fold': 5,
        'stratified': True,
        'shuffle': True,
        'random_state': 42
    }
    
    # Overfitting control settings
    OVERFITTING_THRESHOLD = 0.05  # 5% maximum difference
    
    # Evaluation metrics
    EVALUATION_METRICS = [
        'accuracy',
        'precision_macro',
        'recall_macro', 
        'f1_macro',
        'roc_auc_ovr'
    ]
    
    @classmethod
    def get_model(cls, model_name: str):
        """Get a model instance by name"""
        if model_name not in cls.BASE_MODELS:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(cls.BASE_MODELS.keys())}")
        return cls.BASE_MODELS[model_name]
    
    @classmethod
    def get_hyperparameters(cls, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter grid for a model"""
        if model_name not in cls.HYPERPARAMETER_GRIDS:
            raise ValueError(f"Hyperparameters for '{model_name}' not found. Available models: {list(cls.HYPERPARAMETER_GRIDS.keys())}")
        return cls.HYPERPARAMETER_GRIDS[model_name]
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names"""
        return list(cls.BASE_MODELS.keys())
