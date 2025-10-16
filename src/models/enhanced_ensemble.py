"""
Enhanced ensemble methods for combining multiple models.

Implements multiple ensemble strategies including stacking, weighted averaging,
and voting with soft/hard decision rules.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class WeightedEnsemble:
    """Weighted averaging ensemble."""
    
    def __init__(self, weights: Optional[np.ndarray] = None):
        """
        Initialize weighted ensemble.
        
        Args:
            weights: Weights for each model (auto-computed from CV performance if None)
        """
        self.weights = weights
        self.models = []
    
    def add_model(self, model, weight: float = 1.0):
        """Add model with weight."""
        self.models.append((model, weight))
    
    def predict(self, X: np.ndarray, normalize_weights: bool = True) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Input features
            normalize_weights: Whether to normalize weights to sum to 1
            
        Returns:
            Ensemble predictions (n_samples, n_outputs)
        """
        predictions = []
        weights = []
        
        for model, weight in self.models:
            pred = model.predict(X)
            
            # Handle different output formats
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            
            predictions.append(pred)
            weights.append(weight)
        
        # Stack predictions
        stacked = np.stack(predictions, axis=0)  # (n_models, n_samples, n_outputs)
        weights = np.array(weights).reshape(-1, 1, 1)
        
        if normalize_weights:
            weights = weights / weights.sum()
        
        # Weighted average
        ensemble_pred = (stacked * weights).sum(axis=0)
        
        return ensemble_pred
    
    def predict_proba(self, X: np.ndarray, normalize_weights: bool = True) -> np.ndarray:
        """Generate ensemble probabilities."""
        return self.predict(X, normalize_weights=normalize_weights)


class VotingEnsemble:
    """Hard voting ensemble with majority or plurality rule."""
    
    def __init__(self, voting_rule: str = 'majority'):
        """
        Initialize voting ensemble.
        
        Args:
            voting_rule: 'majority' (>50%) or 'plurality' (most votes)
        """
        self.voting_rule = voting_rule
        self.models = []
    
    def add_model(self, model):
        """Add model to ensemble."""
        self.models.append(model)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Generate ensemble predictions via voting.
        
        Args:
            X: Input features
            threshold: Probability threshold for binary prediction
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            
            # Convert to binary if needed
            if pred.ndim == 1:
                pred = (pred > threshold).astype(int)
            else:
                pred = (pred > threshold).astype(int)
            
            predictions.append(pred)
        
        # Stack predictions (n_models, n_samples, ...)
        stacked = np.stack(predictions, axis=0)
        
        # Apply voting rule
        if self.voting_rule == 'majority':
            # Majority vote: > 50%
            votes_sum = stacked.sum(axis=0)
            ensemble_pred = (votes_sum > len(self.models) / 2).astype(int)
        else:  # plurality
            # Plurality: most votes
            votes_sum = stacked.sum(axis=0)
            ensemble_pred = (votes_sum >= (len(self.models) + 1) / 2).astype(int)
        
        return ensemble_pred


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, base_models: List, meta_model, cv_folds: int = 5):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_model: Meta-learner model
            cv_folds: Number of CV folds for generating meta-features
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.trained_base_models = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit stacking ensemble.
        
        Args:
            X: Training features
            y: Training targets
        """
        from sklearn.model_selection import KFold
        
        logger.info("Fitting stacking ensemble...")
        
        # Generate meta-features using cross-validation
        meta_X = np.zeros((X.shape[0], len(self.base_models)))
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for base_idx, base_model in enumerate(self.base_models):
            logger.info(f"  Training base model {base_idx+1}/{len(self.base_models)}")
            
            meta_X_fold = np.zeros(X.shape[0])
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                # Train base model on fold
                model_clone = self._clone_model(base_model)
                model_clone.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                if hasattr(model_clone, 'predict_proba'):
                    preds = model_clone.predict_proba(X_val_fold)[:, 1]
                else:
                    preds = model_clone.predict(X_val_fold)
                
                meta_X_fold[val_idx] = preds
            
            meta_X[:, base_idx] = meta_X_fold
        
        # Train meta-model
        logger.info("Training meta-model...")
        self.meta_model.fit(meta_X, y)
        
        # Train base models on full training set
        for base_idx, base_model in enumerate(self.base_models):
            model_clone = self._clone_model(base_model)
            model_clone.fit(X, y)
            self.trained_base_models.append(model_clone)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate stacking predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        meta_X = np.zeros((X.shape[0], len(self.trained_base_models)))
        
        for base_idx, base_model in enumerate(self.trained_base_models):
            if hasattr(base_model, 'predict_proba'):
                preds = base_model.predict_proba(X)[:, 1]
            else:
                preds = base_model.predict(X)
            
            meta_X[:, base_idx] = preds
        
        return self.meta_model.predict(meta_X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate stacking probabilities."""
        meta_X = np.zeros((X.shape[0], len(self.trained_base_models)))
        
        for base_idx, base_model in enumerate(self.trained_base_models):
            if hasattr(base_model, 'predict_proba'):
                preds = base_model.predict_proba(X)[:, 1]
            else:
                preds = base_model.predict(X)
            
            meta_X[:, base_idx] = preds
        
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_X)
        else:
            return self.meta_model.predict(meta_X)
    
    @staticmethod
    def _clone_model(model):
        """Clone a model for retraining."""
        import copy
        return copy.deepcopy(model)


class BlendingEnsemble:
    """Blending ensemble with separate holdout set."""
    
    def __init__(self, base_models: List, test_size: float = 0.2):
        """
        Initialize blending ensemble.
        
        Args:
            base_models: List of base models
            test_size: Fraction of data to use for blending
        """
        self.base_models = base_models
        self.test_size = test_size
        self.trained_base_models = []
        self.blend_weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit blending ensemble.
        
        Args:
            X: Training features
            y: Training targets
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("Fitting blending ensemble...")
        
        # Split into train and blend sets
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        
        # Generate blend predictions
        blend_X = np.zeros((X_blend.shape[0], len(self.base_models)))
        
        for base_idx, base_model in enumerate(self.base_models):
            logger.info(f"  Training base model {base_idx+1}/{len(self.base_models)}")
            
            model_clone = self._clone_model(base_model)
            model_clone.fit(X_train, y_train)
            self.trained_base_models.append(model_clone)
            
            # Get blend predictions
            if hasattr(model_clone, 'predict_proba'):
                preds = model_clone.predict_proba(X_blend)[:, 1]
            else:
                preds = model_clone.predict(X_blend)
            
            blend_X[:, base_idx] = preds
        
        # Optimize blend weights on blend set
        self.blend_weights = self._optimize_blend_weights(blend_X, y_blend)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate blending predictions."""
        base_X = np.zeros((X.shape[0], len(self.trained_base_models)))
        
        for base_idx, base_model in enumerate(self.trained_base_models):
            if hasattr(base_model, 'predict_proba'):
                preds = base_model.predict_proba(X)[:, 1]
            else:
                preds = base_model.predict(X)
            
            base_X[:, base_idx] = preds
        
        if self.blend_weights is None:
            # Equal weights
            return base_X.mean(axis=1)
        
        return base_X @ self.blend_weights / self.blend_weights.sum()
    
    def _optimize_blend_weights(self, blend_X: np.ndarray, 
                               y_blend: np.ndarray) -> np.ndarray:
        """Optimize blend weights on blend set."""
        best_weights = np.ones(blend_X.shape[1])
        best_score = 0.0
        
        # Grid search over weight combinations
        for w1 in np.linspace(0, 1, 11):
            for w2 in np.linspace(0, 1, 11):
                weights = np.array([w1, w2])
                weights = weights / weights.sum()
                
                blend_pred = blend_X @ weights
                blend_pred_binary = (blend_pred > 0.5).astype(int)
                
                score = f1_score(y_blend, blend_pred_binary, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_weights = weights
        
        logger.info(f"Blend weights optimized: {best_weights}")
        return best_weights
    
    @staticmethod
    def _clone_model(model):
        """Clone a model."""
        import copy
        return copy.deepcopy(model)


class EnsembleOptimizer:
    """Optimize ensemble combination strategy."""
    
    @staticmethod
    def find_best_ensemble_strategy(models: List, 
                                   X_val: np.ndarray, 
                                   y_val: np.ndarray,
                                   metric_fn: Callable = None) -> Tuple[str, Dict]:
        """
        Compare different ensemble strategies.
        
        Args:
            models: List of trained models
            X_val: Validation features
            y_val: Validation targets
            metric_fn: Metric function for evaluation
            
        Returns:
            (best_strategy, strategy_results)
        """
        if metric_fn is None:
            metric_fn = lambda y_true, y_pred: f1_score(y_true, y_pred > 0.5, zero_division=0)
        
        results = {}
        
        # Strategy 1: Simple averaging
        logger.info("Evaluating averaging ensemble...")
        ensemble_avg = WeightedEnsemble(weights=np.ones(len(models)) / len(models))
        for model in models:
            ensemble_avg.add_model(model, weight=1.0/len(models))
        
        pred_avg = ensemble_avg.predict(X_val)
        score_avg = metric_fn(y_val, pred_avg)
        results['averaging'] = {
            'score': score_avg,
            'ensemble': ensemble_avg,
        }
        logger.info(f"  Averaging score: {score_avg:.4f}")
        
        # Strategy 2: Weighted average (by individual model performance)
        logger.info("Evaluating weighted ensemble...")
        weights = []
        for model in models:
            pred = model.predict(X_val)
            score = metric_fn(y_val, pred)
            weights.append(score)
        
        weights = np.array(weights) / sum(weights)
        ensemble_weighted = WeightedEnsemble()
        for i, model in enumerate(models):
            ensemble_weighted.add_model(model, weight=weights[i])
        
        pred_weighted = ensemble_weighted.predict(X_val)
        score_weighted = metric_fn(y_val, pred_weighted)
        results['weighted'] = {
            'score': score_weighted,
            'ensemble': ensemble_weighted,
            'weights': weights,
        }
        logger.info(f"  Weighted score: {score_weighted:.4f}")
        
        # Find best
        best_strategy = max(results.keys(), key=lambda k: results[k]['score'])
        logger.info(f"Best ensemble strategy: {best_strategy}")
        
        return best_strategy, results
