"""
Advanced threshold optimization and calibration for CAFA-6 scoring.
Implements sophisticated threshold selection, probability calibration,
and hierarchy-aware optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')


class HierarchyAwareThresholdOptimizer:
    """Optimize thresholds respecting GO hierarchy constraints."""
    
    def __init__(self, go_hierarchy, ia_weights: Dict[str, float] = None):
        """
        Initialize hierarchy-aware optimizer.
        
        Args:
            go_hierarchy: GOHierarchy object
            ia_weights: Information accretion weights
        """
        self.go_hierarchy = go_hierarchy
        self.ia_weights = ia_weights or {}
        self.thresholds = {}
        self.hierarchy_constraints = {}
    
    def optimize_with_hierarchy(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               go_terms: List[str], optimization_method: str = 'differential_evolution') -> Dict[str, float]:
        """
        Optimize thresholds with GO hierarchy constraints.
        
        Args:
            y_true: True labels (n_samples, n_terms)
            y_pred_proba: Predicted probabilities (n_samples, n_terms)
            go_terms: List of GO term IDs
            optimization_method: 'differential_evolution' or 'gradient_descent'
            
        Returns:
            Dictionary of term -> optimal threshold
        """
        n_terms = len(go_terms)
        
        # Build hierarchy constraints
        self._build_hierarchy_constraints(go_terms)
        
        # Objective function
        def objective_function(thresholds):
            """Objective to maximize IA-weighted F1."""
            thresholds = np.clip(thresholds, 0.01, 0.99)
            
            # Apply hierarchy constraints
            constrained_thresholds = self._apply_hierarchy_constraints(thresholds, go_terms)
            
            # Generate predictions
            y_pred = (y_pred_proba >= constrained_thresholds).astype(int)
            
            # Calculate IA-weighted F1
            weighted_f1 = self._calculate_ia_weighted_f1(y_true, y_pred, go_terms)
            
            return -weighted_f1  # Minimize negative F1
        
        # Initial thresholds
        initial_thresholds = np.full(n_terms, 0.5)
        
        # Bounds for each threshold
        bounds = [(0.01, 0.99) for _ in range(n_terms)]
        
        # Optimize
        if optimization_method == 'differential_evolution':
            result = differential_evolution(
                objective_function, bounds, maxiter=50, popsize=15, seed=42
            )
        else:
            result = minimize(
                objective_function, initial_thresholds, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 100}
            )
        
        # Apply hierarchy constraints to final thresholds
        final_thresholds = self._apply_hierarchy_constraints(result.x, go_terms)
        
        # Create threshold dictionary
        self.thresholds = {term: final_thresholds[i] for i, term in enumerate(go_terms)}
        
        return self.thresholds
    
    def _build_hierarchy_constraints(self, go_terms: List[str]):
        """Build parent-child relationship constraints."""
        self.hierarchy_constraints = {}
        
        for i, term in enumerate(go_terms):
            if term in self.go_hierarchy.terms:
                parents = self.go_hierarchy.get_parents(term)
                children = self.go_hierarchy.children.get(term, set())
                
                # Find indices of parents and children in go_terms
                parent_indices = [go_terms.index(p) for p in parents if p in go_terms]
                child_indices = [go_terms.index(c) for c in children if c in go_terms]
                
                self.hierarchy_constraints[i] = {
                    'parents': parent_indices,
                    'children': child_indices
                }
    
    def _apply_hierarchy_constraints(self, thresholds: np.ndarray, go_terms: List[str]) -> np.ndarray:
        """Apply hierarchy constraints to thresholds."""
        constrained = thresholds.copy()
        
        # Parent threshold should be <= child threshold (easier to predict parents)
        for term_idx, constraints in self.hierarchy_constraints.items():
            parent_indices = constraints['parents']
            child_indices = constraints['children']
            
            # Parent threshold <= child threshold
            for parent_idx in parent_indices:
                constrained[parent_idx] = min(constrained[parent_idx], constrained[term_idx])
            
            # Child threshold >= parent threshold
            for child_idx in child_indices:
                constrained[child_idx] = max(constrained[child_idx], constrained[term_idx])
        
        return np.clip(constrained, 0.01, 0.99)
    
    def _calculate_ia_weighted_f1(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 go_terms: List[str]) -> float:
        """Calculate IA-weighted F1 score."""
        from sklearn.metrics import f1_score
        
        weighted_f1s = []
        for i, term in enumerate(go_terms):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            
            if y_t.sum() == 0:
                continue
            
            f1 = f1_score(y_t, y_p, zero_division=0)
            weight = self.ia_weights.get(term, 1.0)
            weighted_f1s.append(f1 * weight)
        
        return np.mean(weighted_f1s) if weighted_f1s else 0.0


class BayesianThresholdOptimizer:
    """Bayesian optimization for threshold selection."""
    
    def __init__(self, ia_weights: Dict[str, float] = None):
        """
        Initialize Bayesian optimizer.
        
        Args:
            ia_weights: Information accretion weights
        """
        self.ia_weights = ia_weights or {}
        self.threshold_distributions = {}
    
    def optimize_bayesian(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                         go_terms: List[str], n_iterations: int = 50) -> Dict[str, float]:
        """
        Bayesian optimization of thresholds.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            go_terms: List of GO terms
            n_iterations: Number of optimization iterations
            
        Returns:
            Dictionary of term -> optimal threshold
        """
        thresholds = {}
        
        for term_idx, term in enumerate(go_terms):
            y_t = y_true[:, term_idx]
            y_p = y_pred_proba[:, term_idx]
            
            if y_t.sum() == 0:
                thresholds[term] = 0.5
                continue
            
            # Bayesian optimization using Beta distribution prior
            best_threshold = self._optimize_single_term_bayesian(
                y_t, y_p, self.ia_weights.get(term, 1.0), n_iterations
            )
            
            thresholds[term] = best_threshold
        
        return thresholds
    
    def _optimize_single_term_bayesian(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                     weight: float, n_iterations: int) -> float:
        """Bayesian optimization for a single GO term."""
        # Initialize with Beta prior (favoring thresholds around 0.5)
        alpha, beta_param = 2, 2
        
        best_threshold = 0.5
        best_score = 0
        
        for iteration in range(n_iterations):
            # Sample threshold from current posterior
            threshold = np.random.beta(alpha, beta_param)
            
            # Calculate score
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = self._calculate_weighted_f1_single(y_true, y_pred, weight)
            
            # Update best
            if f1 > best_score:
                best_score = f1
                best_threshold = threshold
            
            # Update posterior (simple update rule)
            if f1 > best_score * 0.9:  # Good threshold
                alpha += 1
            else:  # Poor threshold
                beta_param += 1
        
        return best_threshold
    
    def _calculate_weighted_f1_single(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    weight: float) -> float:
        """Calculate weighted F1 for single term."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, zero_division=0) * weight


class AdvancedCalibration:
    """Advanced probability calibration methods."""
    
    def __init__(self):
        """Initialize calibrator."""
        self.calibrators = {}
        self.calibration_method = None
    
    def fit_calibrators(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       go_terms: List[str], method: str = 'isotonic'):
        """
        Fit calibration models for each GO term.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            go_terms: List of GO terms
            method: Calibration method ('isotonic', 'platt', 'beta')
        """
        self.calibration_method = method
        
        for term_idx, term in enumerate(go_terms):
            y_t = y_true[:, term_idx]
            y_p = y_pred_proba[:, term_idx]
            
            if y_t.sum() == 0:
                continue
            
            if method == 'isotonic':
                calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                calibrator.fit(y_p, y_t)
                self.calibrators[term] = calibrator
            
            elif method == 'platt':
                # Platt scaling (sigmoid calibration)
                A, B = self._fit_platt_scaling(y_t, y_p)
                self.calibrators[term] = {'A': A, 'B': B}
            
            elif method == 'beta':
                # Beta calibration
                alpha, beta_param = self._fit_beta_calibration(y_t, y_p)
                self.calibrators[term] = {'alpha': alpha, 'beta': beta_param}
    
    def calibrate_predictions(self, y_pred_proba: np.ndarray, go_terms: List[str]) -> np.ndarray:
        """
        Calibrate probability predictions.
        
        Args:
            y_pred_proba: Predicted probabilities
            go_terms: List of GO terms
            
        Returns:
            Calibrated predictions
        """
        calibrated = y_pred_proba.copy()
        
        for term_idx, term in enumerate(go_terms):
            if term not in self.calibrators:
                continue
            
            y_p = y_pred_proba[:, term_idx]
            
            if self.calibration_method == 'isotonic':
                calibrated[:, term_idx] = self.calibrators[term].transform(y_p)
            
            elif self.calibration_method == 'platt':
                A = self.calibrators[term]['A']
                B = self.calibrators[term]['B']
                calibrated[:, term_idx] = 1.0 / (1.0 + np.exp(-(A * y_p + B)))
            
            elif self.calibration_method == 'beta':
                alpha = self.calibrators[term]['alpha']
                beta_param = self.calibrators[term]['beta']
                calibrated[:, term_idx] = beta.cdf(y_p, alpha, beta_param)
        
        return np.clip(calibrated, 0.001, 0.999)
    
    def _fit_platt_scaling(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """Fit Platt scaling parameters."""
        from scipy.optimize import minimize
        
        def log_loss(params):
            A, B = params
            p = 1.0 / (1.0 + np.exp(-(A * y_pred_proba + B)))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            return loss
        
        try:
            result = minimize(log_loss, x0=[1, 0], method='L-BFGS-B')
            return result.x[0], result.x[1]
        except:
            return 1.0, 0.0
    
    def _fit_beta_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """Fit Beta calibration parameters."""
        from scipy.stats import beta
        from scipy.optimize import minimize
        
        def neg_log_likelihood(params):
            alpha, beta_param = params
            # Ensure positive parameters
            alpha = max(alpha, 0.1)
            beta_param = max(beta_param, 0.1)
            
            # Calculate log likelihood
            ll = np.sum(beta.logpdf(y_pred_proba[y_true == 1], alpha, beta_param))
            ll += np.sum(beta.logpdf(1 - y_pred_proba[y_true == 0], alpha, beta_param))
            
            return -ll
        
        try:
            result = minimize(neg_log_likelihood, x0=[2, 2], method='L-BFGS-B')
            return max(result.x[0], 0.1), max(result.x[1], 0.1)
        except:
            return 2.0, 2.0


class AdaptiveThresholdSelector:
    """Adaptive threshold selection based on prediction confidence."""
    
    def __init__(self, base_threshold: float = 0.5):
        """
        Initialize adaptive selector.
        
        Args:
            base_threshold: Base threshold to adapt from
        """
        self.base_threshold = base_threshold
        self.confidence_thresholds = {}
    
    def fit_confidence_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                go_terms: List[str]):
        """
        Learn confidence-based threshold adjustments.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            go_terms: List of GO terms
        """
        for term_idx, term in enumerate(go_terms):
            y_t = y_true[:, term_idx]
            y_p = y_pred_proba[:, term_idx]
            
            if y_t.sum() == 0:
                self.confidence_thresholds[term] = self.base_threshold
                continue
            
            # Find optimal threshold for different confidence levels
            confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
            thresholds = []
            
            for confidence in confidence_levels:
                # Select predictions with confidence >= level
                high_conf_mask = (y_p >= confidence) | (y_p <= 1 - confidence)
                
                if high_conf_mask.sum() > 10:  # Need enough samples
                    y_t_high = y_t[high_conf_mask]
                    y_p_high = y_p[high_conf_mask]
                    
                    # Find optimal threshold for this confidence level
                    best_thresh = self._find_optimal_threshold(y_t_high, y_p_high)
                    thresholds.append(best_thresh)
                else:
                    thresholds.append(self.base_threshold)
            
            self.confidence_thresholds[term] = {
                'levels': confidence_levels,
                'thresholds': thresholds
            }
    
    def predict_adaptive(self, y_pred_proba: np.ndarray, go_terms: List[str]) -> np.ndarray:
        """
        Generate predictions with adaptive thresholds.
        
        Args:
            y_pred_proba: Predicted probabilities
            go_terms: List of GO terms
            
        Returns:
            Binary predictions with adaptive thresholds
        """
        y_pred = np.zeros_like(y_pred_proba)
        
        for term_idx, term in enumerate(go_terms):
            if term not in self.confidence_thresholds:
                y_pred[:, term_idx] = (y_pred_proba[:, term_idx] >= self.base_threshold).astype(int)
                continue
            
            thresholds_info = self.confidence_thresholds[term]
            levels = thresholds_info['levels']
            thresholds = thresholds_info['thresholds']
            
            # Apply adaptive threshold based on confidence
            for i, prob in enumerate(y_pred_proba[:, term_idx]):
                # Find appropriate confidence level
                confidence = max(prob, 1 - prob)
                
                # Find threshold for this confidence level
                threshold = self.base_threshold
                for level, thresh in zip(levels, thresholds):
                    if confidence >= level:
                        threshold = thresh
                        break
                
                y_pred[i, term_idx] = int(prob >= threshold)
        
        return y_pred
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Find optimal threshold for single term."""
        from sklearn.metrics import f1_score
        
        best_threshold = 0.5
        best_f1 = 0
        
        thresholds = np.linspace(0.1, 0.9, 17)
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        return best_threshold


class EnsembleThresholdOptimizer:
    """Combine multiple threshold optimization strategies."""
    
    def __init__(self, go_hierarchy=None, ia_weights: Dict[str, float] = None):
        """
        Initialize ensemble optimizer.
        
        Args:
            go_hierarchy: GOHierarchy object
            ia_weights: Information accretion weights
        """
        self.go_hierarchy = go_hierarchy
        self.ia_weights = ia_weights or {}
        self.optimizers = {}
        self.best_strategy = None
    
    def optimize_ensemble(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                         go_terms: List[str]) -> Tuple[Dict[str, float], str]:
        """
        Find best threshold optimization strategy.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            go_terms: List of GO terms
            
        Returns:
            (best_thresholds, best_strategy_name)
        """
        strategies = {}
        
        # Strategy 1: Standard per-term optimization
        print("Evaluating standard optimization...")
        standard_opt = ThresholdOptimizer(self.ia_weights)
        standard_thresholds = standard_opt.optimize_thresholds(y_true, y_pred_proba, go_terms)
        standard_score = self._evaluate_thresholds(y_true, y_pred_proba, go_terms, standard_thresholds)
        strategies['standard'] = {'thresholds': standard_thresholds, 'score': standard_score}
        
        # Strategy 2: Hierarchy-aware optimization
        if self.go_hierarchy is not None:
            print("Evaluating hierarchy-aware optimization...")
            hier_opt = HierarchyAwareThresholdOptimizer(self.go_hierarchy, self.ia_weights)
            hier_thresholds = hier_opt.optimize_with_hierarchy(y_true, y_pred_proba, go_terms)
            hier_score = self._evaluate_thresholds(y_true, y_pred_proba, go_terms, hier_thresholds)
            strategies['hierarchy'] = {'thresholds': hier_thresholds, 'score': hier_score}
        
        # Strategy 3: Bayesian optimization
        print("Evaluating Bayesian optimization...")
        bayes_opt = BayesianThresholdOptimizer(self.ia_weights)
        bayes_thresholds = bayes_opt.optimize_bayesian(y_true, y_pred_proba, go_terms)
        bayes_score = self._evaluate_thresholds(y_true, y_pred_proba, go_terms, bayes_thresholds)
        strategies['bayesian'] = {'thresholds': bayes_thresholds, 'score': bayes_score}
        
        # Strategy 4: Adaptive thresholds
        print("Evaluating adaptive thresholds...")
        adaptive_opt = AdaptiveThresholdSelector()
        adaptive_opt.fit_confidence_thresholds(y_true, y_pred_proba, go_terms)
        adaptive_pred = adaptive_opt.predict_adaptive(y_pred_proba, go_terms)
        adaptive_score = self._calculate_weighted_f1(y_true, adaptive_pred, go_terms)
        
        # Convert adaptive to thresholds for comparison
        adaptive_thresholds = {term: 0.5 for term in go_terms}  # Simplified
        strategies['adaptive'] = {'thresholds': adaptive_thresholds, 'score': adaptive_score}
        
        # Find best strategy
        best_strategy_name = max(strategies.keys(), key=lambda k: strategies[k]['score'])
        best_thresholds = strategies[best_strategy_name]['thresholds']
        
        print(f"Best threshold strategy: {best_strategy_name} (score: {strategies[best_strategy_name]['score']:.4f})")
        
        self.best_strategy = best_strategy_name
        self.optimizers = strategies
        
        return best_thresholds, best_strategy_name
    
    def _evaluate_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                           go_terms: List[str], thresholds: Dict[str, float]) -> float:
        """Evaluate threshold performance."""
        y_pred = np.zeros_like(y_pred_proba)
        
        for i, term in enumerate(go_terms):
            threshold = thresholds.get(term, 0.5)
            y_pred[:, i] = (y_pred_proba[:, i] >= threshold).astype(int)
        
        return self._calculate_weighted_f1(y_true, y_pred, go_terms)
    
    def _calculate_weighted_f1(self, y_true: np.ndarray, y_pred: np.ndarray,
                             go_terms: List[str]) -> float:
        """Calculate IA-weighted F1 score."""
        from sklearn.metrics import f1_score
        
        weighted_f1s = []
        for i, term in enumerate(go_terms):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            
            if y_t.sum() == 0:
                continue
            
            f1 = f1_score(y_t, y_p, zero_division=0)
            weight = self.ia_weights.get(term, 1.0)
            weighted_f1s.append(f1 * weight)
        
        return np.mean(weighted_f1s) if weighted_f1s else 0.0


# Import the original ThresholdOptimizer for reference
from src.evaluation.threshold_optimizer import ThresholdOptimizer


def main():
    """Test advanced threshold optimization methods."""
    print("Advanced threshold optimization methods ready!")
    print("Available classes:")
    print("- HierarchyAwareThresholdOptimizer")
    print("- BayesianThresholdOptimizer")
    print("- AdvancedCalibration")
    print("- AdaptiveThresholdSelector")
    print("- EnsembleThresholdOptimizer")


if __name__ == '__main__':
    main()
