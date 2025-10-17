"""
Threshold optimization and GO hierarchy constraint enforcement.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings('ignore')


class ThresholdOptimizer:
    """Optimize per-term prediction thresholds to maximize IA-weighted F1."""
    
    def __init__(self, ia_weights: Dict[str, float]):
        """
        Initialize optimizer.
        
        Args:
            ia_weights: Dictionary of GO term -> information accretion weight
        """
        self.ia_weights = ia_weights
        self.thresholds = {}
    
    def optimize_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                           go_terms: List[str], grid_search: bool = True,
                           n_thresholds: int = 50) -> Dict[str, float]:
        """
        Optimize thresholds per GO term to maximize IA-weighted F1.
        
        Args:
            y_true: True binary labels (n_samples, n_terms)
            y_pred_proba: Predicted probabilities (n_samples, n_terms)
            go_terms: List of GO term IDs
            grid_search: Whether to use grid search (vs global threshold)
            n_thresholds: Number of thresholds to try
            
        Returns:
            Dictionary of term -> optimal threshold
        """
        self.thresholds = {}
        best_f1 = 0
        
        # Try different threshold ranges
        for term_idx, term in enumerate(go_terms):
            y_t = y_true[:, term_idx]
            y_p = y_pred_proba[:, term_idx]
            
            # Skip if no positive samples
            if y_t.sum() == 0:
                self.thresholds[term] = 0.5
                continue
            
            weight = self.ia_weights.get(term, 1.0)
            best_thresh = 0.5
            best_f1_term = 0
            
            # Grid search over thresholds
            thresholds_to_try = np.linspace(0.01, 0.99, n_thresholds)
            
            for thresh in thresholds_to_try:
                y_pred = (y_p >= thresh).astype(int)
                
                # Calculate weighted precision, recall, F1
                tp = np.sum((y_t == 1) & (y_pred == 1))
                fp = np.sum((y_t == 0) & (y_pred == 1))
                fn = np.sum((y_t == 1) & (y_pred == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                # Weight by IA
                weighted_f1 = f1 * weight
                
                if weighted_f1 > best_f1_term:
                    best_f1_term = weighted_f1
                    best_thresh = thresh
            
            self.thresholds[term] = best_thresh
        
        return self.thresholds
    
    def apply_thresholds(self, y_pred_proba: np.ndarray, 
                        go_terms: List[str]) -> np.ndarray:
        """
        Apply optimized thresholds to predictions.
        
        Args:
            y_pred_proba: Probability predictions (n_samples, n_terms)
            go_terms: List of GO terms
            
        Returns:
            Binary predictions with optimized thresholds
        """
        y_pred_binary = np.zeros_like(y_pred_proba)
        
        for term_idx, term in enumerate(go_terms):
            threshold = self.thresholds.get(term, 0.5)
            y_pred_binary[:, term_idx] = (y_pred_proba[:, term_idx] >= threshold).astype(int)
        
        return y_pred_binary


class GOHierarchyEnforcer:
    """Enforce GO hierarchy constraints: parent score >= max(children scores)."""
    
    def __init__(self, go_hierarchy):
        """
        Initialize enforcer.
        
        Args:
            go_hierarchy: GOHierarchy object with parent/child relationships
        """
        self.go_hierarchy = go_hierarchy
    
    def enforce_constraints(self, predictions: Dict[str, Dict[str, float]],
                           go_terms: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Enforce hierarchy constraints on predictions.
        
        Args:
            predictions: Dict of protein_id -> {go_term: score}
            go_terms: List of all GO terms
            
        Returns:
            Modified predictions with enforced constraints
        """
        go_set = set(go_terms)
        enforced = {}
        
        for protein_id, term_scores in predictions.items():
            enforced_scores = term_scores.copy()
            
            # For each term, ensure parent score >= max(children)
            for term in go_terms:
                if term not in self.go_hierarchy.terms:
                    continue
                
                children = self.go_hierarchy.children.get(term, set())
                valid_children = [c for c in children if c in go_set]
                
                if valid_children:
                    max_child_score = max(
                        enforced_scores.get(child, 0) for child in valid_children
                    )
                    current_score = enforced_scores.get(term, 0)
                    
                    if max_child_score > current_score:
                        enforced_scores[term] = max_child_score
            
            # For each term with score, propagate to all ancestors
            for term, score in list(term_scores.items()):
                if term not in self.go_hierarchy.terms:
                    continue
                
                ancestors = self.go_hierarchy.get_ancestors(term, include_self=False)
                valid_ancestors = [a for a in ancestors if a in go_set]
                
                for ancestor in valid_ancestors:
                    ancestor_current = enforced_scores.get(ancestor, 0)
                    # Propagate with slightly reduced score to avoid flooding
                    enforced_scores[ancestor] = max(ancestor_current, score * 0.95)
            
            enforced[protein_id] = enforced_scores
        
        return enforced
    
    def propagate_to_root(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Ensure every predicted term also has all ancestors predicted.
        This is a simpler version that just propagates scores up.
        
        Args:
            predictions: Dict of protein_id -> {go_term: score}
            
        Returns:
            Modified predictions with full ancestry
        """
        propagated = {}
        
        for protein_id, term_scores in predictions.items():
            prop_scores = term_scores.copy()
            
            for term, score in term_scores.items():
                if term not in self.go_hierarchy.terms:
                    continue
                
                ancestors = self.go_hierarchy.get_ancestors(term, include_self=False)
                for ancestor in ancestors:
                    # Use max aggregation for ancestor scores
                    ancestor_score = prop_scores.get(ancestor, 0)
                    prop_scores[ancestor] = max(ancestor_score, score * 0.9)
            
            propagated[protein_id] = prop_scores
        
        return propagated


class PredictionCalibrator:
    """Calibrate probability predictions to improve reliability."""
    
    @staticmethod
    def platt_scaling(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """
        Fit Platt scaling (sigmoid calibration).
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (A, B) parameters for sigmoid: 1 / (1 + exp(A*x + B))
        """
        from scipy.optimize import fmin_bfgs
        
        def log_loss(params):
            A, B = params
            p = 1.0 / (1.0 + np.exp(-(A * y_pred_proba + B)))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            return loss
        
        try:
            A, B = fmin_bfgs(log_loss, x0=[1, 0], disp=False)
        except:
            A, B = 1, 0
        
        return A, B
    
    @staticmethod
    def isotonic_calibration(y_true: np.ndarray, y_pred_proba: np.ndarray):
        """
        Fit isotonic regression for calibration.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Isotonic regression model
        """
        from sklearn.isotonic import IsotonicRegression
        
        iso_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso_model.fit(y_pred_proba, y_true)
        
        return iso_model


class ConfidenceBinner:
    """Bin confidence scores to improve calibration."""
    
    @staticmethod
    def bin_predictions(predictions: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """
        Bin confidence scores into discrete levels.
        
        Args:
            predictions: Probability predictions
            n_bins: Number of bins
            
        Returns:
            Binned predictions
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        binned = np.digitize(predictions, bin_edges) / n_bins
        
        return np.clip(binned, 0.01, 0.99)


def main():
    """Test threshold optimization and constraint enforcement."""
    print("Threshold optimization and GO hierarchy tools ready.")
    print("Usage: ThresholdOptimizer, GOHierarchyEnforcer, PredictionCalibrator")


if __name__ == '__main__':
    main()
