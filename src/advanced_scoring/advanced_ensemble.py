"""
Advanced ensemble methods for improved CAFA-6 scoring.
Implements sophisticated ensemble strategies including meta-learning, 
adaptive weighting, and hierarchical ensembling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')


class AdaptiveWeightedEnsemble:
    """Adaptive weighted ensemble that learns optimal weights per GO term."""
    
    def __init__(self, n_terms: int, learning_rate: float = 0.01):
        """
        Initialize adaptive ensemble.
        
        Args:
            n_terms: Number of GO terms
            learning_rate: Learning rate for weight updates
        """
        self.n_terms = n_terms
        self.learning_rate = learning_rate
        self.weights = None
        self.base_models = []
    
    def add_model(self, model, name: str):
        """Add a base model to the ensemble."""
        self.base_models.append((model, name))
    
    def fit_adaptive_weights(self, X_val: np.ndarray, y_val: np.ndarray,
                           ia_weights: Dict[str, float] = None):
        """
        Learn optimal weights for each GO term using gradient descent.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            ia_weights: Information accretion weights
        """
        n_models = len(self.base_models)
        self.weights = np.ones((n_terms, n_models)) / n_models
        
        # Get predictions from all models
        model_predictions = []
        for model, _ in self.base_models:
            pred = model.predict_proba(X_val)
            model_predictions.append(pred)
        
        model_predictions = np.stack(model_predictions, axis=2)  # (n_samples, n_terms, n_models)
        
        # Gradient descent to optimize weights
        for epoch in range(100):
            # Current ensemble predictions
            ensemble_pred = np.sum(model_predictions * self.weights[None, :, :], axis=2)
            
            # Calculate gradients
            for term_idx in range(self.n_terms):
                y_t = y_val[:, term_idx]
                y_p = ensemble_pred[:, term_idx]
                
                # Skip if no positive samples
                if y_t.sum() == 0:
                    continue
                
                # Calculate weighted F1 gradient
                weight = ia_weights.get(f"GO:{term_idx:07d}", 1.0) if ia_weights else 1.0
                
                for model_idx in range(n_models):
                    model_pred = model_predictions[:, term_idx, model_idx]
                    
                    # Simple gradient based on F1 improvement
                    current_weight = self.weights[term_idx, model_idx]
                    
                    # Try small weight change
                    new_weight = current_weight + 0.01
                    new_weight = np.clip(new_weight, 0.01, 1.0)
                    
                    # Renormalize weights
                    temp_weights = self.weights[term_idx].copy()
                    temp_weights[model_idx] = new_weight
                    temp_weights = temp_weights / temp_weights.sum()
                    
                    # Calculate new ensemble prediction
                    new_ensemble = np.sum(model_predictions[:, term_idx, :] * temp_weights[None, :], axis=1)
                    
                    # Calculate F1 scores
                    current_f1 = f1_score(y_t, (y_p > 0.5).astype(int), zero_division=0)
                    new_f1 = f1_score(y_t, (new_ensemble > 0.5).astype(int), zero_division=0)
                    
                    # Update weight if improvement
                    if new_f1 > current_f1:
                        gradient = (new_f1 - current_f1) * weight
                        self.weights[term_idx, model_idx] += self.learning_rate * gradient
            
            # Normalize weights
            self.weights = self.weights / self.weights.sum(axis=1, keepdims=True)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions with learned weights."""
        model_predictions = []
        for model, _ in self.base_models:
            pred = model.predict_proba(X)
            model_predictions.append(pred)
        
        model_predictions = np.stack(model_predictions, axis=2)
        
        # Apply learned weights
        ensemble_pred = np.sum(model_predictions * self.weights[None, :, :], axis=2)
        
        return ensemble_pred


class HierarchicalEnsemble:
    """Ensemble that respects GO hierarchy in model combination."""
    
    def __init__(self, go_hierarchy, ia_weights: Dict[str, float] = None):
        """
        Initialize hierarchical ensemble.
        
        Args:
            go_hierarchy: GOHierarchy object
            ia_weights: Information accretion weights
        """
        self.go_hierarchy = go_hierarchy
        self.ia_weights = ia_weights or {}
        self.level_models = {}  # level -> list of models
        self.ancestor_weights = {}
    
    def add_level_models(self, level: int, models: List):
        """Add models for specific GO hierarchy level."""
        self.level_models[level] = models
    
    def predict_hierarchical(self, X: np.ndarray, go_terms: List[str]) -> np.ndarray:
        """
        Generate predictions that respect GO hierarchy.
        
        Args:
            X: Input features
            go_terms: List of GO terms
            
        Returns:
            Hierarchical predictions
        """
        n_samples = X.shape[0]
        n_terms = len(go_terms)
        predictions = np.zeros((n_samples, n_terms))
        
        # Group terms by hierarchy level
        term_levels = {}
        for i, term in enumerate(go_terms):
            if term in self.go_hierarchy.terms:
                level = self.go_hierarchy.get_depth(term)
                term_levels[i] = level
            else:
                term_levels[i] = 0
        
        # Predict level by level, respecting parent-child relationships
        max_level = max(term_levels.values()) if term_levels else 0
        
        for level in range(max_level + 1):
            level_terms = [i for i, l in term_levels.items() if l == level]
            
            if not level_terms:
                continue
            
            # Get models for this level
            level_preds = []
            if level in self.level_models:
                for model in self.level_models[level]:
                    pred = model.predict_proba(X)
                    level_preds.append(pred)
            
            if level_preds:
                # Average predictions for this level
                level_pred = np.mean(level_preds, axis=0)
                
                # Apply hierarchy constraints
                for term_idx in level_terms:
                    term = go_terms[term_idx]
                    
                    # Get parent terms
                    if term in self.go_hierarchy.terms:
                        parents = self.go_hierarchy.get_parents(term)
                        parent_indices = [go_terms.index(p) for p in parents if p in go_terms]
                        
                        if parent_indices:
                            # Parent score should be >= child score
                            max_parent_score = np.max(predictions[:, parent_indices], axis=1)
                            level_pred[:, term_idx] = np.maximum(
                                level_pred[:, term_idx], max_parent_score * 0.9
                            )
                
                    predictions[:, term_idx] = level_pred[:, term_idx]
        
        return predictions


class MetaLearningEnsemble:
    """Meta-learning ensemble that learns to combine base models."""
    
    def __init__(self, meta_model_type: str = 'gradient_boosting'):
        """
        Initialize meta-learning ensemble.
        
        Args:
            meta_model_type: Type of meta-model ('logistic', 'ridge', 'gradient_boosting')
        """
        self.meta_model_type = meta_model_type
        self.base_models = []
        self.meta_models = {}  # One per GO term
        self.fitted = False
    
    def add_base_model(self, model, name: str):
        """Add a base model."""
        self.base_models.append((model, name))
    
    def fit_meta_models(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       go_terms: List[str]):
        """
        Train meta-models on validation predictions.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            go_terms: List of GO terms
        """
        print("Training meta-models...")
        
        # Get predictions from base models on validation set
        val_predictions = []
        for model, _ in self.base_models:
            pred = model.predict_proba(X_val)
            val_predictions.append(pred)
        
        val_predictions = np.stack(val_predictions, axis=2)  # (n_val, n_terms, n_models)
        
        # Train meta-model for each GO term
        for term_idx, term in enumerate(go_terms):
            y_t = y_val[:, term_idx]
            
            # Skip if no positive samples
            if y_t.sum() == 0:
                continue
            
            # Prepare meta-features
            meta_X = val_predictions[:, term_idx, :]  # (n_val, n_models)
            
            # Select meta-model
            if self.meta_model_type == 'logistic':
                meta_model = LogisticRegression(random_state=42, max_iter=1000)
            elif self.meta_model_type == 'ridge':
                meta_model = Ridge(alpha=1.0)
            else:  # gradient_boosting
                meta_model = GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, random_state=42
                )
            
            # Train meta-model
            try:
                if self.meta_model_type == 'logistic':
                    meta_model.fit(meta_X, y_t)
                else:
                    meta_model.fit(meta_X, y_t)
                
                self.meta_models[term_idx] = meta_model
            except:
                # Fallback to simple averaging
                pass
        
        self.fitted = True
        print(f"Trained {len(self.meta_models)} meta-models")
    
    def predict_proba(self, X: np.ndarray, go_terms: List[str]) -> np.ndarray:
        """Generate predictions using meta-models."""
        if not self.fitted:
            # Fallback to simple averaging
            predictions = []
            for model, _ in self.base_models:
                pred = model.predict_proba(X)
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        
        # Get base model predictions
        base_predictions = []
        for model, _ in self.base_models:
            pred = model.predict_proba(X)
            base_predictions.append(pred)
        
        base_predictions = np.stack(base_predictions, axis=2)  # (n_samples, n_terms, n_models)
        
        # Apply meta-models
        n_samples, n_terms, n_models = base_predictions.shape
        final_predictions = np.zeros((n_samples, n_terms))
        
        for term_idx in range(n_terms):
            if term_idx in self.meta_models:
                meta_X = base_predictions[:, term_idx, :]
                meta_model = self.meta_models[term_idx]
                
                if self.meta_model_type == 'logistic':
                    pred = meta_model.predict_proba(meta_X)[:, 1]
                else:
                    pred = meta_model.predict(meta_X)
                
                final_predictions[:, term_idx] = pred
            else:
                # Fallback to averaging
                final_predictions[:, term_idx] = np.mean(base_predictions[:, term_idx, :], axis=1)
        
        return final_predictions


class ConfidenceWeightedEnsemble:
    """Ensemble that weights models based on prediction confidence."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize confidence-weighted ensemble.
        
        Args:
            confidence_threshold: Threshold for high confidence predictions
        """
        self.confidence_threshold = confidence_threshold
        self.base_models = []
        self.model_confidences = {}
    
    def add_model(self, model, name: str):
        """Add a base model."""
        self.base_models.append((model, name))
    
    def calculate_model_confidence(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calculate confidence scores for each model."""
        for model, name in self.base_models:
            pred_proba = model.predict_proba(X_val)
            
            # Calculate confidence as average max probability per sample
            confidences = []
            for i in range(pred_proba.shape[0]):
                max_probs = np.max(pred_proba[i], axis=0) if pred_proba.ndim > 1 else pred_proba[i]
                confidences.append(max_probs)
            
            self.model_confidences[name] = np.mean(confidences)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate confidence-weighted predictions."""
        predictions = []
        weights = []
        
        for model, name in self.base_models:
            pred = model.predict_proba(X)
            predictions.append(pred)
            
            # Use confidence as weight
            weight = self.model_confidences.get(name, 1.0)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred


class AdvancedEnsembleOptimizer:
    """Optimize ensemble strategy using multiple criteria."""
    
    @staticmethod
    def find_optimal_strategy(models: List, X_val: np.ndarray, y_val: np.ndarray,
                            go_terms: List[str], ia_weights: Dict[str, float] = None,
                            go_hierarchy=None) -> Tuple[str, object, Dict]:
        """
        Find optimal ensemble strategy using multiple approaches.
        
        Args:
            models: List of trained models
            X_val: Validation features
            y_val: Validation targets
            go_terms: List of GO terms
            ia_weights: Information accretion weights
            go_hierarchy: GO hierarchy object
            
        Returns:
            (best_strategy, best_ensemble, results)
        """
        if ia_weights is None:
            ia_weights = {}
        
        results = {}
        
        # Strategy 1: Simple averaging
        print("Evaluating simple averaging...")
        avg_pred = np.mean([m.predict_proba(X_val) for m in models], axis=0)
        avg_f1 = AdvancedEnsembleOptimizer._calculate_weighted_f1(
            y_val, avg_pred, go_terms, ia_weights
        )
        results['averaging'] = {'score': avg_f1, 'predictions': avg_pred}
        
        # Strategy 2: Adaptive weighted ensemble
        print("Evaluating adaptive weighted ensemble...")
        adaptive = AdaptiveWeightedEnsemble(len(go_terms))
        for i, model in enumerate(models):
            adaptive.add_model(model, f'model_{i}')
        adaptive.fit_adaptive_weights(X_val, y_val, ia_weights)
        adaptive_pred = adaptive.predict_proba(X_val)
        adaptive_f1 = AdvancedEnsembleOptimizer._calculate_weighted_f1(
            y_val, adaptive_pred, go_terms, ia_weights
        )
        results['adaptive'] = {'score': adaptive_f1, 'ensemble': adaptive, 'predictions': adaptive_pred}
        
        # Strategy 3: Meta-learning ensemble
        print("Evaluating meta-learning ensemble...")
        meta_ensemble = MetaLearningEnsemble('gradient_boosting')
        for i, model in enumerate(models):
            meta_ensemble.add_base_model(model, f'model_{i}')
        
        # Use part of validation for meta-training
        split_idx = len(X_val) // 2
        meta_ensemble.fit_meta_models(
            X_val[:split_idx], y_val[:split_idx],
            X_val[split_idx:], y_val[split_idx:],
            go_terms
        )
        meta_pred = meta_ensemble.predict_proba(X_val[split_idx:], go_terms)
        meta_f1 = AdvancedEnsembleOptimizer._calculate_weighted_f1(
            y_val[split_idx:], meta_pred, go_terms, ia_weights
        )
        results['meta_learning'] = {'score': meta_f1, 'ensemble': meta_ensemble, 'predictions': meta_pred}
        
        # Strategy 4: Hierarchical ensemble (if hierarchy available)
        if go_hierarchy is not None:
            print("Evaluating hierarchical ensemble...")
            hierarchical = HierarchicalEnsemble(go_hierarchy, ia_weights)
            # Add all models to level 0
            hierarchical.add_level_models(0, models)
            hier_pred = hierarchical.predict_hierarchical(X_val, go_terms)
            hier_f1 = AdvancedEnsembleOptimizer._calculate_weighted_f1(
                y_val, hier_pred, go_terms, ia_weights
            )
            results['hierarchical'] = {'score': hier_f1, 'ensemble': hierarchical, 'predictions': hier_pred}
        
        # Find best strategy
        best_strategy = max(results.keys(), key=lambda k: results[k]['score'])
        print(f"Best ensemble strategy: {best_strategy} (score: {results[best_strategy]['score']:.4f})")
        
        return best_strategy, results[best_strategy].get('ensemble'), results
    
    @staticmethod
    def _calculate_weighted_f1(y_true: np.ndarray, y_pred_proba: np.ndarray,
                             go_terms: List[str], ia_weights: Dict[str, float]) -> float:
        """Calculate IA-weighted F1 score."""
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        weighted_f1s = []
        for i, term in enumerate(go_terms):
            y_t = y_true[:, i]
            y_p = y_pred_binary[:, i]
            
            if y_t.sum() == 0:
                continue
            
            f1 = f1_score(y_t, y_p, zero_division=0)
            weight = ia_weights.get(term, 1.0)
            weighted_f1s.append(f1 * weight)
        
        return np.mean(weighted_f1s) if weighted_f1s else 0.0


def main():
    """Test advanced ensemble methods."""
    print("Advanced ensemble methods ready for use!")
    print("Available classes:")
    print("- AdaptiveWeightedEnsemble")
    print("- HierarchicalEnsemble") 
    print("- MetaLearningEnsemble")
    print("- ConfidenceWeightedEnsemble")
    print("- AdvancedEnsembleOptimizer")


if __name__ == '__main__':
    main()
