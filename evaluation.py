"""
Comprehensive evaluation and submission pipeline.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict


class SubmissionGenerator:
    """Generate submission files in the required format."""
    
    def __init__(self, ia_weights: Dict[str, float]):
        """
        Initialize generator.
        
        Args:
            ia_weights: Dictionary of GO term -> information accretion weight
        """
        self.ia_weights = ia_weights
    
    def create_submission(self, predictions: Dict[str, Dict[str, float]], 
                         output_path: str, max_predictions_per_protein: int = 1500):
        """
        Create submission file in the required format.
        
        Args:
            predictions: Dict of protein_id -> {go_term: probability}
            output_path: Path to save submission file
            max_predictions_per_protein: Max predictions per protein
        """
        lines = []
        
        for protein_id, go_predictions in predictions.items():
            # Extract just the protein ID if it contains taxon info (format: ID\t9606 from sequences)
            prot_id = protein_id.split('\t')[0] if '\t' in protein_id else protein_id
            
            # Sort by probability (descending)
            sorted_preds = sorted(go_predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Limit predictions
            sorted_preds = sorted_preds[:max_predictions_per_protein]
            
            # Add to submission (exclude 0 scores)
            for go_term, score in sorted_preds:
                if score > 0:
                    # Format score with up to 3 significant figures
                    score_str = f"{score:.3f}".rstrip('0').rstrip('.')
                    if not score_str:
                        score_str = "0.001"
                    lines.append(f"{prot_id}\t{go_term}\t{score_str}")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Submission file created: {output_path}")
        print(f"  Total predictions: {len(lines)}")
    
    @staticmethod
    def calculate_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                             ia_weights: Dict[str, float], 
                             go_terms: List[str]) -> float:
        """
        Calculate weighted F1 score using information accretion weights.
        
        Args:
            y_true: True labels (n_samples, n_terms)
            y_pred: Predicted labels (n_samples, n_terms)
            ia_weights: IA weights for each term
            go_terms: List of GO terms
            
        Returns:
            Weighted F1 score
        """
        from sklearn.metrics import precision_recall_curve
        
        weighted_precisions = []
        weighted_recalls = []
        
        for i, term in enumerate(go_terms):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            
            # Skip if no positive samples
            if y_t.sum() == 0:
                continue
            
            # Calculate precision and recall
            tp = np.sum((y_t == 1) & (y_p == 1))
            fp = np.sum((y_t == 0) & (y_p == 1))
            fn = np.sum((y_t == 1) & (y_p == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            weight = ia_weights.get(term, 0)
            weighted_precisions.append(precision * weight)
            weighted_recalls.append(recall * weight)
        
        if not weighted_precisions:
            return 0
        
        avg_precision = np.sum(weighted_precisions) / (np.sum(weighted_precisions) + 1e-8)
        avg_recall = np.sum(weighted_recalls) / (np.sum(weighted_recalls) + 1e-8)
        
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-8)
        return f1


class ModelEvaluator:
    """Evaluate and compare multiple models."""
    
    def __init__(self, ia_weights: Dict[str, float]):
        self.ia_weights = ia_weights
        self.results = {}
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, 
                      y_pred: np.ndarray, go_terms: List[str]) -> Dict:
        """
        Evaluate a model.
        
        Args:
            model_name: Name of model
            y_true: True labels
            y_pred: Predicted labels
            go_terms: List of GO terms
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics = {
            'model': model_name,
            'precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def print_comparison(self):
        """Print comparison of all models."""
        if not self.results:
            print("No results to compare")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        df = pd.DataFrame(self.results).T
        print(df.to_string())
        print("="*80)
        
        # Find best model for each metric
        print("\nBest Models:")
        for metric in ['f1_micro', 'f1_macro', 'f1_weighted']:
            best_model = df[metric].idxmax()
            best_score = df[metric].max()
            print(f"  Best {metric}: {best_model} ({best_score:.4f})")


class EnsemblePredictor:
    """Ensemble predictions from multiple models."""
    
    @staticmethod
    def average_ensemble(predictions_list: List[np.ndarray], 
                        weights: List[float] = None) -> np.ndarray:
        """
        Average probability predictions from multiple models.
        
        Args:
            predictions_list: List of prediction arrays (probability format)
            weights: Optional weights for each model
            
        Returns:
            Averaged predictions
        """
        if weights is None:
            weights = [1.0] * len(predictions_list)
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        ensemble = np.zeros_like(predictions_list[0], dtype=float)
        
        for pred, weight in zip(predictions_list, weights):
            ensemble += pred * weight
        
        return ensemble
    
    @staticmethod
    def voting_ensemble(predictions_list: List[np.ndarray], 
                       threshold: float = 0.5) -> np.ndarray:
        """
        Majority voting ensemble of multiple models.
        
        Args:
            predictions_list: List of binary prediction arrays
            threshold: Threshold for binary predictions
            
        Returns:
            Ensemble predictions
        """
        votes = np.zeros_like(predictions_list[0], dtype=float)
        
        for pred in predictions_list:
            votes += pred
        
        # Majority vote
        return (votes > len(predictions_list) / 2).astype(int)


class AnalysisUtils:
    """Utilities for analysis."""
    
    @staticmethod
    def analyze_predictions_distribution(predictions: np.ndarray, 
                                        go_terms: List[str]) -> Dict:
        """
        Analyze distribution of predictions.
        
        Args:
            predictions: Prediction array (n_samples, n_terms)
            go_terms: List of GO terms
            
        Returns:
            Dictionary with statistics
        """
        n_samples, n_terms = predictions.shape
        
        # Overall statistics
        total_predictions = (predictions > 0).sum()
        predictions_per_protein = (predictions > 0).sum(axis=1)
        predictions_per_term = (predictions > 0).sum(axis=0)
        
        stats = {
            'n_proteins': n_samples,
            'n_terms': n_terms,
            'total_predictions': total_predictions,
            'avg_per_protein': predictions_per_protein.mean(),
            'max_per_protein': predictions_per_protein.max(),
            'min_per_protein': predictions_per_protein.min(),
            'top_terms': [
                (go_terms[i], count) 
                for i, count in sorted(
                    enumerate(predictions_per_term),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ],
        }
        
        return stats
    
    @staticmethod
    def print_analysis(stats: Dict):
        """Print analysis statistics."""
        print("\nPrediction Analysis:")
        print(f"  Proteins: {stats['n_proteins']}")
        print(f"  GO Terms: {stats['n_terms']}")
        print(f"  Total Predictions: {stats['total_predictions']}")
        print(f"  Avg per Protein: {stats['avg_per_protein']:.2f}")
        print(f"  Max per Protein: {stats['max_per_protein']}")
        print(f"  Min per Protein: {stats['min_per_protein']}")
        print("\n  Top 10 Predicted Terms:")
        for term, count in stats['top_terms']:
            print(f"    {term}: {count}")


def main():
    """Test evaluation pipeline."""
    print("Testing evaluation pipeline...")
    
    # Create mock data
    n_proteins = 100
    n_terms = 50
    
    y_true = np.random.binomial(1, 0.1, size=(n_proteins, n_terms))
    y_pred_1 = np.random.binomial(1, 0.15, size=(n_proteins, n_terms))
    y_pred_2 = np.random.binomial(1, 0.12, size=(n_proteins, n_terms))
    
    go_terms = [f"GO:{i:07d}" for i in range(n_terms)]
    
    # Create weights
    ia_weights = {term: np.random.uniform(0, 1) for term in go_terms}
    
    # Evaluate models
    evaluator = ModelEvaluator(ia_weights)
    
    evaluator.evaluate_model("Model 1", y_true, y_pred_1, go_terms)
    evaluator.evaluate_model("Model 2", y_true, y_pred_2, go_terms)
    
    evaluator.print_comparison()
    
    # Test ensemble
    print("\nTesting ensemble...")
    ensemble_proba = EnsemblePredictor.average_ensemble([y_pred_1, y_pred_2])
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    evaluator.evaluate_model("Ensemble", y_true, ensemble_pred, go_terms)
    
    # Analyze
    stats = AnalysisUtils.analyze_predictions_distribution(ensemble_pred, go_terms)
    AnalysisUtils.print_analysis(stats)
    
    # Test submission generation
    print("\nTesting submission generation...")
    
    # Create sample predictions
    predictions = {}
    for i in range(n_proteins):
        protein_id = f"P{i:05d}"
        go_preds = {}
        for j in range(n_terms):
            if ensemble_pred[i, j] == 1:
                go_preds[go_terms[j]] = float(ensemble_proba[i, j])
        predictions[protein_id] = go_preds
    
    gen = SubmissionGenerator(ia_weights)
    output_file = '/tmp/test_submission.tsv'
    gen.create_submission(predictions, output_file)
    
    # Read and show sample
    with open(output_file, 'r') as f:
        lines = f.readlines()[:5]
    
    print("Sample submission lines:")
    for line in lines:
        print(f"  {line.strip()}")
    
    print("\nEvaluation pipeline test completed successfully!")


if __name__ == '__main__':
    main()
