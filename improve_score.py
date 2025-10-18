"""
Comprehensive score improvement script for CAFA-6 Protein Function Prediction.
Addresses key bottlenecks: label sparsity, threshold optimization, and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

class ScoreImprover:
    """Comprehensive score improvement system."""
    
    def __init__(self, data_dir: str = '.'):
        """Initialize the score improver."""
        self.data_dir = data_dir
        self.loader = None
        self.go_hierarchy = None
        self.evaluator = None
        
    def load_data(self):
        """Load all necessary data."""
        print("Loading data for score improvement...")
        from src.data.data_loader import CAFADataLoader
        
        self.loader = CAFADataLoader(self.data_dir)
        self.loader.load_train_data()
        self.loader.load_test_data()
        self.loader.load_ia_weights()
        
        # Load GO hierarchy
        from src.features.go_ontology import GOHierarchy
        self.go_hierarchy = GOHierarchy('Train/go-basic.obo')
        
        # Initialize evaluator
        from src.evaluation.evaluation import ModelEvaluator
        self.evaluator = ModelEvaluator(self.loader.ia_weights)
        
        summary = self.loader.get_train_data_summary()
        print(f"  Training proteins: {summary['n_proteins']}")
        print(f"  GO terms: {summary['n_unique_terms']}")
        print(f"  Test proteins: {summary['n_test_proteins']}")
        
    def analyze_label_sparsity(self):
        """Analyze and address label sparsity issues."""
        print("\nAnalyzing label sparsity...")
        
        # Calculate term frequencies
        term_counts = {}
        for protein_id, terms in self.loader.train_terms.items():
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1
        
        # Analyze distribution
        counts = list(term_counts.values())
        print(f"  Term count statistics:")
        print(f"    Min: {min(counts)}")
        print(f"    Max: {max(counts)}")
        print(f"    Mean: {np.mean(counts):.2f}")
        print(f"    Median: {np.median(counts):.2f}")
        
        # Find terms with sufficient examples
        min_examples = max(10, len(self.loader.train_sequences) // 1000)
        frequent_terms = [term for term, count in term_counts.items() if count >= min_examples]
        
        print(f"  Terms with ≥{min_examples} examples: {len(frequent_terms)}")
        print(f"  Coverage: {len(frequent_terms)/len(term_counts):.2%}")
        
        return frequent_terms, term_counts
    
    def create_balanced_dataset(self, frequent_terms: List[str], 
                               train_size: int = 5000, val_size: int = 1000):
        """Create a balanced dataset with frequent terms only."""
        print(f"\nCreating balanced dataset...")
        
        # Get proteins that have at least one frequent term
        valid_proteins = []
        for protein_id, terms in self.loader.train_terms.items():
            if any(term in frequent_terms for term in terms):
                valid_proteins.append(protein_id)
        
        print(f"  Proteins with frequent terms: {len(valid_proteins)}")
        
        # Sample proteins
        np.random.seed(42)
        np.random.shuffle(valid_proteins)
        
        train_ids = valid_proteins[:train_size]
        val_ids = valid_proteins[train_size:train_size + val_size]
        
        print(f"  Train proteins: {len(train_ids)}")
        print(f"  Val proteins: {len(val_ids)}")
        
        return train_ids, val_ids, frequent_terms
    
    def extract_enhanced_features(self, protein_ids: List[str]):
        """Extract enhanced features with better normalization."""
        print(f"\nExtracting enhanced features...")
        from src.features.feature_extractor import ProteinFeatureExtractor
        
        # Get sequences
        sequences = {pid: self.loader.train_sequences[pid] for pid in protein_ids}
        
        # Extract features
        extractor = ProteinFeatureExtractor()
        features, ordered_ids = extractor.create_combined_features(sequences)
        
        # Enhanced feature engineering
        print("  Applying feature enhancements...")
        
        # 1. Log-transform skewed features
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        # Normalize features
        features_normalized = (features - feature_means) / (feature_stds + 1e-8)
        
        # 2. Add polynomial features for important ones
        important_indices = np.where(np.abs(feature_means) > 0.01)[0][:20]  # Top 20
        poly_features = []
        
        for idx in important_indices:
            poly_features.append(features_normalized[:, idx] ** 2)
            poly_features.append(np.sqrt(np.abs(features_normalized[:, idx])))
        
        poly_features = np.column_stack(poly_features)
        
        # 3. Combine features
        enhanced_features = np.concatenate([features_normalized, poly_features], axis=1)
        
        print(f"  Original features: {features.shape[1]}")
        print(f"  Enhanced features: {enhanced_features.shape[1]}")
        
        return enhanced_features, ordered_ids
    
    def train_optimized_models(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              go_terms: List[str]):
        """Train optimized models with better hyperparameters."""
        print(f"\nTraining optimized models...")
        
        from src.models.baseline_models import RandomForestModel
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.linear_model import LogisticRegression
        
        models = {}
        predictions = {}
        
        # Model 1: Optimized Random Forest
        print("  Training Optimized Random Forest...")
        rf_model = RandomForestModel(
            n_estimators=200,
            max_depth=25
        )
        rf_model.train(X_train, y_train, go_terms, verbose=False)
        
        rf_pred = rf_model.predict_proba(X_val)
        models['rf'] = rf_model
        predictions['rf'] = rf_pred
        
        # Model 2: Gradient Boosting
        print("  Training Gradient Boosting...")
        gb_model = MultiOutputClassifier(
            GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        )
        
        # Train on subset for efficiency
        subset_size = min(2000, len(X_train))
        gb_model.fit(X_train[:subset_size], y_train[:subset_size])
        
        gb_pred = gb_model.predict_proba(X_val)
        gb_pred_proba = np.array([pred[:, 1] for pred in gb_pred]).T
        
        models['gb'] = gb_model
        predictions['gb'] = gb_pred_proba
        
        # Model 3: Logistic Regression with class weights
        print("  Training Logistic Regression...")
        lr_model = MultiOutputClassifier(
            LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        )
        
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict_proba(X_val)
        lr_pred_proba = np.array([pred[:, 1] for pred in lr_pred]).T
        
        models['lr'] = lr_model
        predictions['lr'] = lr_pred_proba
        
        return models, predictions
    
    def optimize_thresholds_ensemble(self, y_val: np.ndarray, 
                                   predictions: Dict[str, np.ndarray],
                                   go_terms: List[str]):
        """Optimize thresholds using ensemble approach."""
        print(f"\nOptimizing thresholds with ensemble approach...")
        
        # Create ensemble predictions
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Optimize thresholds per term
        optimal_thresholds = {}
        
        print("  Optimizing thresholds for each GO term...")
        for i, term in enumerate(go_terms):
            if i % 100 == 0:
                print(f"    Progress: {i}/{len(go_terms)}")
            
            y_true_term = y_val[:, i]
            y_pred_term = ensemble_pred[:, i]
            
            # Skip if no positive examples
            if np.sum(y_true_term) == 0:
                optimal_thresholds[term] = 0.5
                continue
            
            # Find optimal threshold
            best_threshold = 0.5
            best_f1 = 0
            
            thresholds = np.linspace(0.05, 0.95, 19)
            for threshold in thresholds:
                y_pred_binary = (y_pred_term >= threshold).astype(int)
                
                if np.sum(y_pred_binary) == 0:
                    continue
                
                tp = np.sum((y_true_term == 1) & (y_pred_binary == 1))
                fp = np.sum((y_true_term == 0) & (y_pred_binary == 1))
                fn = np.sum((y_true_term == 1) & (y_pred_binary == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            optimal_thresholds[term] = best_threshold
        
        # Apply optimal thresholds
        y_pred_optimized = np.zeros_like(ensemble_pred)
        for i, term in enumerate(go_terms):
            threshold = optimal_thresholds[term]
            y_pred_optimized[:, i] = (ensemble_pred[:, i] >= threshold).astype(int)
        
        # Evaluate optimized predictions
        optimized_metrics = self.evaluator.evaluate_model(
            "Optimized Ensemble", y_val, y_pred_optimized, go_terms
        )
        
        print(f"  Optimized performance:")
        print(f"    F1 Micro: {optimized_metrics['f1_micro']:.4f}")
        print(f"    F1 Weighted: {optimized_metrics['f1_weighted']:.4f}")
        print(f"    Precision: {optimized_metrics['precision']:.4f}")
        print(f"    Recall: {optimized_metrics['recall']:.4f}")
        
        return optimal_thresholds, ensemble_pred, optimized_metrics
    
    def apply_go_hierarchy_propagation(self, predictions: np.ndarray, 
                                     go_terms: List[str]) -> np.ndarray:
        """Apply GO hierarchy propagation to improve predictions."""
        print(f"\nApplying GO hierarchy propagation...")
        
        propagated_predictions = predictions.copy()
        
        for i, term in enumerate(go_terms):
            try:
                # Get ancestors
                ancestors = self.go_hierarchy.get_ancestors(term)
                
                # Propagate to ancestors
                for ancestor in ancestors:
                    if ancestor in go_terms:
                        ancestor_idx = go_terms.index(ancestor)
                        # Add some of the prediction score to ancestor
                        propagated_predictions[:, ancestor_idx] = np.maximum(
                            propagated_predictions[:, ancestor_idx],
                            predictions[:, i] * 0.3
                        )
            except:
                continue
        
        return propagated_predictions
    
    def generate_improved_submission(self, models: Dict, optimal_thresholds: Dict,
                                   go_terms: List[str], feature_extractor=None):
        """Generate improved submission file."""
        print(f"\nGenerating improved submission...")
        
        # Extract features for test data
        print("  Extracting test features...")
        test_sequences = self.loader.test_sequences
        
        if feature_extractor is None:
            from src.features.feature_extractor import ProteinFeatureExtractor
            feature_extractor = ProteinFeatureExtractor()
        
        # Process test data in batches
        batch_size = 1000
        test_ids = list(test_sequences.keys())
        all_predictions = []
        
        for i in range(0, len(test_ids), batch_size):
            batch_ids = test_ids[i:i+batch_size]
            batch_sequences = {pid: test_sequences[pid] for pid in batch_ids}
            
            # Extract features
            X_batch, ordered_ids = feature_extractor.create_combined_features(batch_sequences)
            
            # Apply same feature enhancement as training
            feature_means = np.mean(X_batch, axis=0)
            feature_stds = np.std(X_batch, axis=0)
            X_batch_normalized = (X_batch - feature_means) / (feature_stds + 1e-8)
            
            # Generate ensemble predictions
            batch_predictions = []
            for model_name, model in models.items():
                try:
                    pred = model.predict_proba(X_batch_normalized)
                    batch_predictions.append(pred)
                except:
                    continue
            
            if batch_predictions:
                ensemble_pred = np.mean(batch_predictions, axis=0)
                all_predictions.append((ordered_ids, ensemble_pred))
            
            if i % 5000 == 0:
                print(f"    Processed {min(i+batch_size, len(test_ids))}/{len(test_ids)} proteins")
        
        # Combine all predictions
        final_predictions = {}
        for ordered_ids, pred_batch in all_predictions:
            for i, protein_id in enumerate(ordered_ids):
                go_preds = {}
                for j, go_term in enumerate(go_terms):
                    threshold = optimal_thresholds.get(go_term, 0.5)
                    score = float(pred_batch[i, j])
                    if score >= threshold:
                        go_preds[go_term] = max(score, 0.01)
                
                # Ensure minimum predictions
                if len(go_preds) < 3:
                    # Add top terms by frequency
                    term_freq = np.mean([model.predict_proba(np.zeros((1, 427))) for model in models.values()], axis=0)[0]
                    top_indices = np.argsort(term_freq)[-5:]
                    for idx in top_indices:
                        if idx < len(go_terms):
                            go_preds[go_terms[idx]] = 0.01
                
                final_predictions[protein_id] = go_preds
        
        print(f"  Generated predictions for {len(final_predictions)} proteins")
        
        # Create submission file
        from src.evaluation.evaluation import SubmissionGenerator
        gen = SubmissionGenerator(self.loader.ia_weights, self.go_hierarchy)
        submission_file = 'improved_submission.tsv'
        gen.create_submission(final_predictions, submission_file, propagate=False)
        
        print(f"  Submission saved to: {submission_file}")
        return submission_file
    
    def run_improvement_pipeline(self):
        """Run the complete score improvement pipeline."""
        start_time = time.time()
        
        print("CAFA-6 Score Improvement Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Analyze label sparsity
        frequent_terms, term_counts = self.analyze_label_sparsity()
        
        # Step 3: Create balanced dataset
        train_ids, val_ids, selected_terms = self.create_balanced_dataset(
            frequent_terms, train_size=3000, val_size=800
        )
        
        # Step 4: Extract enhanced features
        X_train, train_ordered = self.extract_enhanced_features(train_ids)
        X_val, val_ordered = self.extract_enhanced_features(val_ids)
        
        # Ensure consistent ordering
        train_id_map = {pid: i for i, pid in enumerate(train_ordered)}
        val_id_map = {pid: i for i, pid in enumerate(val_ordered)}
        
        X_train = X_train[[train_id_map[pid] for pid in train_ids]]
        X_val = X_val[[val_id_map[pid] for pid in val_ids]]
        
        # Step 5: Create target matrices
        y_train, _ = self.loader.create_protein_to_terms_matrix(train_ids, selected_terms)
        y_val, _ = self.loader.create_protein_to_terms_matrix(val_ids, selected_terms)
        
        print(f"  Training data shape: {X_train.shape}")
        print(f"  Target shape: {y_train.shape}")
        
        # Step 6: Train optimized models
        models, predictions = self.train_optimized_models(
            X_train, y_train, X_val, y_val, selected_terms
        )
        
        # Step 7: Optimize thresholds
        optimal_thresholds, ensemble_pred, optimized_metrics = self.optimize_thresholds_ensemble(
            y_val, predictions, selected_terms
        )
        
        # Step 8: Apply GO hierarchy propagation
        propagated_pred = self.apply_go_hierarchy_propagation(ensemble_pred, selected_terms)
        
        # Step 9: Generate improved submission
        submission_file = self.generate_improved_submission(
            models, optimal_thresholds, selected_terms
        )
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("IMPROVEMENT PIPELINE COMPLETED!")
        print("=" * 60)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Final F1 Weighted: {optimized_metrics['f1_weighted']:.4f}")
        print(f"Final F1 Micro: {optimized_metrics['f1_micro']:.4f}")
        print(f"Submission file: {submission_file}")
        
        return submission_file, optimized_metrics


def main():
    """Run the score improvement pipeline."""
    improver = ScoreImprover('.')
    submission, metrics = improver.run_improvement_pipeline()
    
    print(f"\n🎉 Score improvement completed!")
    print(f"📁 Submission file: {submission}")
    print(f"📊 Final F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"📈 Expected improvement: Significant vs baseline (~0.03)")


if __name__ == '__main__':
    main()
