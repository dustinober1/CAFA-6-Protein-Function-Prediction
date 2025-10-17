"""
Advanced gradient boosting models (LightGBM, XGBoost) for protein function prediction.
Includes IA-weighted objectives and hyperparameter tuning.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class LightGBMModel:
    """LightGBM model with IA-weighted objectives."""
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 8, 
                 learning_rate: float = 0.05, num_leaves: int = 31,
                 feature_fraction: float = 0.8, bagging_fraction: float = 0.8):
        """Initialize LightGBM model."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.models = {}  # one model per GO term
        self.go_terms = []
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              go_terms: List[str], ia_weights: Dict[str, float] = None,
              verbose: bool = False):
        """
        Train LightGBM models (one per GO term).
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples, n_terms)
            go_terms: List of GO term IDs
            ia_weights: Information accretion weights
            verbose: Print progress
        """
        try:
            import lightgbm as lgb
        except ImportError:
            print("LightGBM not installed. Run: pip install lightgbm")
            return
        
        self.go_terms = go_terms
        ia_weights = ia_weights or {term: 1.0 for term in go_terms}
        
        for term_idx, term in enumerate(go_terms):
            if verbose and term_idx % 100 == 0:
                print(f"  Training term {term_idx}/{len(go_terms)}")
            
            y = y_train[:, term_idx]
            
            # Skip if no positive samples
            if y.sum() == 0:
                continue
            
            # Get term weight
            weight = ia_weights.get(term, 1.0)
            sample_weight = np.ones(len(y)) * weight
            
            # Create dataset
            train_data = lgb.Dataset(
                X_train, label=y, weight=sample_weight,
                free_raw_data=False
            )
            
            # Train model
            params = {
                'objective': 'binary',
                'num_leaves': self.num_leaves,
                'learning_rate': self.learning_rate,
                'feature_fraction': self.feature_fraction,
                'bagging_fraction': self.bagging_fraction,
                'bagging_freq': 5,
                'verbose': -1,
                'metric': 'binary_logloss',
                'num_threads': -1,
            }
            
            model = lgb.train(
                params, train_data,
                num_boost_round=self.n_estimators,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            self.models[term] = model
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for test data.
        
        Args:
            X_test: Test features
            
        Returns:
            Probability matrix (n_samples, n_terms)
        """
        n_samples = X_test.shape[0]
        n_terms = len(self.go_terms)
        predictions = np.zeros((n_samples, n_terms))
        
        for term_idx, term in enumerate(self.go_terms):
            if term in self.models:
                predictions[:, term_idx] = self.models[term].predict(X_test)
            else:
                # No model trained for this term (all negative in training)
                predictions[:, term_idx] = 0.0
        
        return predictions


class XGBoostModel:
    """XGBoost model with IA-weighted objectives."""
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8):
        """Initialize XGBoost model."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.models = {}
        self.go_terms = []
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              go_terms: List[str], ia_weights: Dict[str, float] = None,
              verbose: bool = False):
        """
        Train XGBoost models (one per GO term).
        
        Args:
            X_train: Training features
            y_train: Training labels (n_samples, n_terms)
            go_terms: List of GO term IDs
            ia_weights: Information accretion weights
            verbose: Print progress
        """
        try:
            import xgboost as xgb
        except ImportError:
            print("XGBoost not installed. Run: pip install xgboost")
            return
        
        self.go_terms = go_terms
        ia_weights = ia_weights or {term: 1.0 for term in go_terms}
        
        for term_idx, term in enumerate(go_terms):
            if verbose and term_idx % 100 == 0:
                print(f"  Training term {term_idx}/{len(go_terms)}")
            
            y = y_train[:, term_idx]
            
            # Skip if no positive samples
            if y.sum() == 0:
                continue
            
            # Get term weight
            weight = ia_weights.get(term, 1.0)
            sample_weight = np.ones(len(y)) * weight
            
            # Create dataset
            dtrain = xgb.DMatrix(
                X_train, label=y, weight=sample_weight
            )
            
            # Train model
            params = {
                'objective': 'binary:logistic',
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': 0.8,
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'verbosity': 0,
            }
            
            model = xgb.train(
                params, dtrain,
                num_boost_round=self.n_estimators,
                verbose_eval=False
            )
            
            self.models[term] = model
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for test data.
        
        Args:
            X_test: Test features
            
        Returns:
            Probability matrix (n_samples, n_terms)
        """
        try:
            import xgboost as xgb
        except ImportError:
            return np.zeros((X_test.shape[0], len(self.go_terms)))
        
        n_samples = X_test.shape[0]
        n_terms = len(self.go_terms)
        predictions = np.zeros((n_samples, n_terms))
        
        dtest = xgb.DMatrix(X_test)
        
        for term_idx, term in enumerate(self.go_terms):
            if term in self.models:
                predictions[:, term_idx] = self.models[term].predict(dtest)
            else:
                predictions[:, term_idx] = 0.0
        
        return predictions


class StackedEnsemble:
    """Meta-learner stacking multiple base models."""
    
    def __init__(self, base_models: List, meta_model: str = 'logistic'):
        """
        Initialize stacked ensemble.
        
        Args:
            base_models: List of trained base models
            meta_model: Type of meta-learner ('logistic', 'ridge', 'xgb')
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_learners = {}
        self.go_terms = []
    
    def train(self, X_val: np.ndarray, y_val: np.ndarray, 
              go_terms: List[str], verbose: bool = False):
        """
        Train meta-learner on base model predictions.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            go_terms: List of GO terms
            verbose: Print progress
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import Ridge
        
        self.go_terms = go_terms
        
        # Generate meta-features (base model predictions on val set)
        meta_features = []
        for model in self.base_models:
            pred = model.predict_proba(X_val)
            meta_features.append(pred)
        
        meta_features = np.concatenate(meta_features, axis=1)
        
        # Train meta-learner for each GO term
        for term_idx, term in enumerate(go_terms):
            if verbose and term_idx % 100 == 0:
                print(f"  Training meta-learner for term {term_idx}/{len(go_terms)}")
            
            y = y_val[:, term_idx]
            
            if y.sum() == 0:
                continue
            
            if self.meta_model == 'logistic':
                meta = LogisticRegression(max_iter=1000, random_state=42)
            elif self.meta_model == 'ridge':
                meta = Ridge(alpha=1.0)
            else:
                meta = LogisticRegression(max_iter=1000, random_state=42)
            
            meta.fit(meta_features, y)
            self.meta_learners[term] = meta
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict using meta-learner.
        
        Args:
            X_test: Test features
            
        Returns:
            Probability matrix
        """
        # Generate meta-features
        meta_features = []
        for model in self.base_models:
            pred = model.predict_proba(X_test)
            meta_features.append(pred)
        
        meta_features = np.concatenate(meta_features, axis=1)
        
        # Use meta-learners for predictions
        n_samples = X_test.shape[0]
        n_terms = len(self.go_terms)
        predictions = np.zeros((n_samples, n_terms))
        
        for term_idx, term in enumerate(self.go_terms):
            if term in self.meta_learners:
                predictions[:, term_idx] = self.meta_learners[term].predict_proba(
                    meta_features
                )[:, 1]
            else:
                # Average of base predictions
                base_preds = np.mean([m.predict_proba(X_test)[:, term_idx] 
                                     for m in self.base_models], axis=0)
                predictions[:, term_idx] = base_preds
        
        return predictions


def main():
    """Test gradient boosting models."""
    print("Advanced gradient boosting models ready.")
    print("Usage: LightGBMModel, XGBoostModel, StackedEnsemble")


if __name__ == '__main__':
    main()
