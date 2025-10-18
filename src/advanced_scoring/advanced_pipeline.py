"""
Advanced CAFA-6 scoring pipeline integrating all advanced techniques.
Combines advanced ensembling, threshold optimization, and feature engineering
for maximum competition performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import advanced components
from .advanced_ensemble import (
    AdaptiveWeightedEnsemble, HierarchicalEnsemble, MetaLearningEnsemble,
    AdvancedEnsembleOptimizer
)
from .advanced_thresholds import (
    HierarchyAwareThresholdOptimizer, BayesianThresholdOptimizer,
    AdvancedCalibration, EnsembleThresholdOptimizer
)
from .advanced_features import AdvancedFeatureFusion

# Import existing components
from ..data.data_loader import CAFADataLoader
from ..features.go_ontology import GOHierarchy
from ..evaluation.evaluation import ModelEvaluator, SubmissionGenerator
from ..models.baseline_models import RandomForestModel
from ..models.neural_models import NeuralNetworkModel, DeepNeuralNetwork, ProteinDataset
from ..models.enhanced_ensemble import WeightedEnsemble, StackingEnsemble


class AdvancedScoringPipeline:
    """Advanced pipeline for CAFA-6 with state-of-the-art scoring techniques."""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """
        Initialize advanced pipeline.
        
        Args:
            data_dir: Path to data directory
            output_dir: Path to output directory
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Core components
        self.loader = None
        self.go_hierarchy = None
        self.evaluator = None
        
        # Advanced components
        self.feature_fusion = None
        self.ensemble_optimizer = None
        self.threshold_optimizer = None
        self.calibrator = None
        
        # Models and predictions
        self.models = {}
        self.predictions = {}
        self.best_ensemble = None
        self.best_thresholds = None
        
        # Performance tracking
        self.performance_log = []
    
    def load_data(self):
        """Load and prepare data."""
        print("Loading data...")
        self.loader = CAFADataLoader(str(self.data_dir))
        self.loader.load_train_data()
        self.loader.load_test_data()
        self.loader.load_ia_weights()
        
        # Load GO hierarchy
        print("Loading GO hierarchy...")
        go_obo_path = self.data_dir / 'Train' / 'go-basic.obo'
        self.go_hierarchy = GOHierarchy(str(go_obo_path))
        
        self.evaluator = ModelEvaluator(self.loader.ia_weights)
        
        summary = self.loader.get_train_data_summary()
        print("Data Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    def prepare_advanced_data(self, train_size: int = 3000, val_size: int = 800,
                             n_terms: int = 1000, feature_types: List[str] = None):
        """
        Prepare data with advanced feature extraction.
        
        Args:
            train_size: Number of training proteins
            val_size: Number of validation proteins
            n_terms: Number of GO terms to use
            feature_types: List of advanced feature types
        """
        if feature_types is None:
            feature_types = ['multiscale', 'graph', 'evolutionary']
        
        print(f"\nPreparing advanced data (train={train_size}, val={val_size}, terms={n_terms})...")
        
        # Get sample proteins
        sample_ids = list(self.loader.train_sequences.keys())[:(train_size + val_size)]
        self.train_ids = sample_ids[:train_size]
        self.val_ids = sample_ids[train_size:]
        
        # Prepare sequences
        train_seqs = {pid: self.loader.train_sequences[pid] for pid in self.train_ids}
        val_seqs = {pid: self.loader.train_sequences[pid] for pid in self.val_ids}
        
        print(f"  Train proteins: {len(train_seqs)}")
        print(f"  Val proteins: {len(val_seqs)}")
        
        # Extract advanced features
        print("  Extracting advanced features...")
        self.feature_fusion = AdvancedFeatureFusion(feature_types)
        
        X_train, train_ids_ordered, feature_names = self.feature_fusion.extract_all_features(train_seqs)
        X_val, val_ids_ordered, _ = self.feature_fusion.extract_all_features(val_seqs)
        
        # Ensure consistent ordering
        train_id_map = {pid: i for i, pid in enumerate(train_ids_ordered)}
        val_id_map = {pid: i for i, pid in enumerate(val_ids_ordered)}
        
        X_train = X_train[[train_id_map[pid] for pid in self.train_ids]]
        X_val = X_val[[val_id_map[pid] for pid in self.val_ids]]
        
        print(f"  Advanced feature shape: {X_train.shape}")
        
        # Create target matrices
        all_go_terms = sorted(self.loader.go_terms)
        y_train_full, _ = self.loader.create_protein_to_terms_matrix(self.train_ids, all_go_terms)
        y_val_full, _ = self.loader.create_protein_to_terms_matrix(self.val_ids, all_go_terms)
        
        # Select top terms by frequency
        term_frequencies = y_train_full.sum(axis=0)
        top_indices = np.argsort(term_frequencies)[-n_terms:]
        
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train_full[:, top_indices]
        self.y_val = y_val_full[:, top_indices]
        self.go_terms = [all_go_terms[i] for i in top_indices]
        self.feature_names = feature_names
        
        print(f"  Using {len(self.go_terms)} GO terms")
        print(f"  Target shape: {self.y_train.shape}")
    
    def train_advanced_models(self):
        """Train advanced models with enhanced techniques."""
        print("\n" + "="*80)
        print("TRAINING ADVANCED MODELS")
        print("="*80)
        
        # Model 1: Enhanced Random Forest
        print("\nTraining Enhanced Random Forest...")
        rf_model = RandomForestModel(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced'
        )
        rf_model.train(self.X_train, self.y_train, self.go_terms, verbose=True)
        
        rf_pred = rf_model.predict_proba(self.X_val)
        rf_metrics = self.evaluator.evaluate_model(
            "Enhanced Random Forest", self.y_val, 
            (rf_pred > 0.5).astype(int), self.go_terms
        )
        self.models['enhanced_rf'] = rf_model
        self.predictions['enhanced_rf'] = rf_pred
        
        # Model 2: Deep Neural Network
        print("\nTraining Deep Neural Network...")
        import torch
        from torch.utils.data import DataLoader
        
        device = 'cpu'
        train_dataset = ProteinDataset(self.X_train, self.y_train, self.train_ids)
        val_dataset = ProteinDataset(self.X_val, self.y_val, self.val_ids)
        
        network = DeepNeuralNetwork(
            input_dim=self.X_train.shape[1],
            output_dim=len(self.go_terms),
            hidden_dims=[512, 256, 128],
            dropout_rate=0.4,
            batch_norm=True
        )
        
        dnn_model = NeuralNetworkModel(network, name="Deep Neural Network", device=device)
        dnn_model.train(
            train_dataset, val_dataset, self.go_terms,
            epochs=15, batch_size=64, learning_rate=0.001,
            verbose=True
        )
        
        dnn_pred = dnn_model.predict_proba(self.X_val)
        dnn_metrics = self.evaluator.evaluate_model(
            "Deep Neural Network", self.y_val,
            (dnn_pred > 0.5).astype(int), self.go_terms
        )
        self.models['deep_nn'] = dnn_model
        self.predictions['deep_nn'] = dnn_pred
        
        # Model 3: Gradient Boosting (if available)
        try:
            print("\nTraining Gradient Boosting...")
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.multioutput import MultiOutputClassifier
            
            gb_model = MultiOutputClassifier(
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            )
            
            # Train on subset for efficiency
            subset_size = min(1000, len(self.X_train))
            gb_model.fit(self.X_train[:subset_size], self.y_train[:subset_size])
            
            gb_pred = gb_model.predict_proba(self.X_val)
            # Extract positive class probabilities
            gb_pred_proba = np.array([pred[:, 1] for pred in gb_pred]).T
            
            gb_metrics = self.evaluator.evaluate_model(
                "Gradient Boosting", self.y_val,
                (gb_pred_proba > 0.5).astype(int), self.go_terms
            )
            self.models['gradient_boosting'] = gb_model
            self.predictions['gradient_boosting'] = gb_pred_proba
            
        except Exception as e:
            print(f"Gradient Boosting failed: {e}")
        
        print(f"\nIndividual model performance:")
        self.evaluator.print_comparison()
    
    def create_advanced_ensemble(self):
        """Create advanced ensemble with multiple strategies."""
        print("\n" + "="*80)
        print("CREATING ADVANCED ENSEMBLE")
        print("="*80)
        
        # Prepare model list
        model_list = list(self.models.values())
        
        # Find optimal ensemble strategy
        print("Optimizing ensemble strategy...")
        best_strategy, best_ensemble, results = AdvancedEnsembleOptimizer.find_optimal_strategy(
            model_list, self.X_val, self.y_val, self.go_terms,
            self.loader.ia_weights, self.go_hierarchy
        )
        
        self.best_ensemble = best_ensemble
        self.ensemble_results = results
        
        # Generate ensemble predictions
        if hasattr(best_ensemble, 'predict_proba'):
            ensemble_pred = best_ensemble.predict_proba(self.X_val)
        elif hasattr(best_ensemble, 'predict'):
            ensemble_pred = best_ensemble.predict(self.X_val)
        else:
            # Fallback to averaging
            ensemble_pred = np.mean([self.predictions[name] for name in self.predictions.keys()], axis=0)
        
        # Evaluate ensemble
        ensemble_metrics = self.evaluator.evaluate_model(
            f"Advanced Ensemble ({best_strategy})", self.y_val,
            (ensemble_pred > 0.5).astype(int), self.go_terms
        )
        
        self.predictions['advanced_ensemble'] = ensemble_pred
        
        print(f"Advanced ensemble performance:")
        print(f"  Strategy: {best_strategy}")
        print(f"  F1 Score: {ensemble_metrics['f1_micro']:.4f}")
        print(f"  Precision: {ensemble_metrics['precision']:.4f}")
        print(f"  Recall: {ensemble_metrics['recall']:.4f}")
    
    def optimize_advanced_thresholds(self):
        """Optimize thresholds with advanced techniques."""
        print("\n" + "="*80)
        print("OPTIMIZING ADVANCED THRESHOLDS")
        print("="*80)
        
        # Get best predictions (ensemble)
        y_pred_proba = self.predictions['advanced_ensemble']
        
        # Initialize ensemble threshold optimizer
        self.threshold_optimizer = EnsembleThresholdOptimizer(
            self.go_hierarchy, self.loader.ia_weights
        )
        
        # Find optimal threshold strategy
        best_thresholds, best_strategy = self.threshold_optimizer.optimize_ensemble(
            self.y_val, y_pred_proba, self.go_terms
        )
        
        self.best_thresholds = best_thresholds
        self.best_threshold_strategy = best_strategy
        
        # Apply optimal thresholds
        y_pred_optimized = np.zeros_like(y_pred_proba)
        for i, term in enumerate(self.go_terms):
            threshold = best_thresholds.get(term, 0.5)
            y_pred_optimized[:, i] = (y_pred_proba[:, i] >= threshold).astype(int)
        
        # Evaluate optimized predictions
        optimized_metrics = self.evaluator.evaluate_model(
            f"Optimized Thresholds ({best_strategy})", self.y_val,
            y_pred_optimized, self.go_terms
        )
        
        print(f"Optimized threshold performance:")
        print(f"  Strategy: {best_strategy}")
        print(f"  F1 Score: {optimized_metrics['f1_micro']:.4f}")
        print(f"  Precision: {optimized_metrics['precision']:.4f}")
        print(f"  Recall: {optimized_metrics['recall']:.4f}")
        
        # Store final predictions
        self.final_predictions = y_pred_optimized
        self.final_predictions_proba = y_pred_proba
    
    def calibrate_predictions(self):
        """Calibrate probability predictions."""
        print("\n" + "="*80)
        print("CALIBRATING PREDICTIONS")
        print("="*80)
        
        # Initialize calibrator
        self.calibrator = AdvancedCalibration()
        
        # Fit calibrators
        calibration_methods = ['isotonic', 'platt']
        best_method = None
        best_score = 0
        
        for method in calibration_methods:
            print(f"Trying {method} calibration...")
            
            # Fit calibrator
            self.calibrator.fit_calibrators(
                self.y_val, self.final_predictions_proba, 
                self.go_terms, method=method
            )
            
            # Calibrate predictions
            calibrated_proba = self.calibrator.calibrate_predictions(
                self.final_predictions_proba, self.go_terms
            )
            
            # Apply thresholds
            calibrated_pred = np.zeros_like(calibrated_proba)
            for i, term in enumerate(self.go_terms):
                threshold = self.best_thresholds.get(term, 0.5)
                calibrated_pred[:, i] = (calibrated_proba[:, i] >= threshold).astype(int)
            
            # Evaluate
            calibrated_metrics = self.evaluator.evaluate_model(
                f"Calibrated ({method})", self.y_val, calibrated_pred, self.go_terms
            )
            
            if calibrated_metrics['f1_micro'] > best_score:
                best_score = calibrated_metrics['f1_micro']
                best_method = method
                self.best_calibrated_proba = calibrated_proba
                self.best_calibrated_pred = calibrated_pred
        
        print(f"Best calibration method: {best_method} (F1: {best_score:.4f})")
        
        # Update final predictions if calibration improved
        if best_score > optimized_metrics['f1_micro']:
            self.final_predictions = self.best_calibrated_pred
            self.final_predictions_proba = self.best_calibrated_proba
            print("Calibration improved performance, updating final predictions")
    
    def generate_test_predictions(self):
        """Generate predictions on test set with full pipeline."""
        print("\n" + "="*80)
        print("GENERATING TEST PREDICTIONS")
        print("="*80)
        
        # Extract features for test data
        print("Extracting features for test data...")
        test_seqs = self.loader.test_sequences
        
        X_test, test_ids_ordered, _ = self.feature_fusion.extract_all_features(test_seqs)
        
        print(f"Test feature shape: {X_test.shape}")
        
        # Generate predictions from all models
        test_predictions = []
        model_names = []
        
        for model_name, model in self.models.items():
            print(f"Generating predictions with {model_name}...")
            
            try:
                pred = model.predict_proba(X_test)
                test_predictions.append(pred)
                model_names.append(model_name)
            except Exception as e:
                print(f"  Failed: {e}")
        
        if not test_predictions:
            print("No successful predictions, using fallback strategy")
            # Fallback: use term frequencies
            term_freq = np.mean(self.y_train, axis=0)
            test_pred_proba = np.tile(term_freq, (len(test_ids_ordered), 1))
        else:
            # Use ensemble strategy
            if self.best_ensemble and hasattr(self.best_ensemble, 'predict_proba'):
                # Try to use the trained ensemble
                try:
                    if hasattr(self.best_ensemble, 'base_models'):
                        # Re-create ensemble with test models
                        # This is simplified - in practice you'd need to handle model recreation
                        test_pred_proba = np.mean(test_predictions, axis=0)
                    else:
                        test_pred_proba = np.mean(test_predictions, axis=0)
                except:
                    test_pred_proba = np.mean(test_predictions, axis=0)
            else:
                test_pred_proba = np.mean(test_predictions, axis=0)
        
        # Apply calibration if available
        if hasattr(self, 'calibrator') and self.calibrator.calibrators:
            print("Applying calibration to test predictions...")
            test_pred_proba = self.calibrator.calibrate_predictions(
                test_pred_proba, self.go_terms
            )
        
        # Apply optimized thresholds
        test_pred_binary = np.zeros_like(test_pred_proba)
        for i, term in enumerate(self.go_terms):
            threshold = self.best_thresholds.get(term, 0.5)
            test_pred_binary[:, i] = (test_pred_proba[:, i] >= threshold).astype(int)
        
        # Convert to submission format
        predictions = {}
        for i, protein_id in enumerate(test_ids_ordered):
            go_preds = {}
            for j, go_term in enumerate(self.go_terms):
                if test_pred_binary[i, j] == 1:
                    score = float(test_pred_proba[i, j])
                    if score > 0.001:
                        go_preds[go_term] = score
            
            # Ensure minimum predictions
            if len(go_preds) < 5:
                # Add top terms by frequency
                term_freq = np.mean(self.y_train, axis=0)
                top_terms = np.argsort(term_freq)[-10:]
                for j in top_terms:
                    go_preds[self.go_terms[j]] = max(0.01, float(test_pred_proba[i, j]))
            
            predictions[protein_id] = go_preds
        
        print(f"Generated predictions for {len(predictions)} proteins")
        return predictions
    
    def create_submission(self, predictions: Dict, output_file: str = None):
        """Create optimized submission file."""
        if output_file is None:
            output_file = str(self.output_dir / 'advanced_submission.tsv')
        
        print(f"\nCreating advanced submission file: {output_file}")
        
        # Use advanced submission generator
        gen = SubmissionGenerator(self.loader.ia_weights, self.go_hierarchy)
        gen.create_submission(predictions, output_file, propagate=True)
        
        return output_file
    
    def save_pipeline(self, output_file: str = None):
        """Save the complete pipeline for future use."""
        if output_file is None:
            output_file = str(self.output_dir / 'advanced_pipeline.pkl')
        
        pipeline_data = {
            'models': self.models,
            'best_ensemble': self.best_ensemble,
            'best_thresholds': self.best_thresholds,
            'best_threshold_strategy': getattr(self, 'best_threshold_strategy', None),
            'calibrator': getattr(self, 'calibrator', None),
            'feature_fusion': self.feature_fusion,
            'go_terms': self.go_terms,
            'feature_names': getattr(self, 'feature_names', []),
            'performance_log': self.performance_log
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Pipeline saved to: {output_file}")
    
    def run_full_advanced_pipeline(self, train_size: int = 3000, val_size: int = 800,
                                  n_terms: int = 1000, feature_types: List[str] = None):
        """Run the complete advanced pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Prepare advanced features
            self.prepare_advanced_data(train_size, val_size, n_terms, feature_types)
            
            # Step 3: Train advanced models
            self.train_advanced_models()
            
            # Step 4: Create advanced ensemble
            self.create_advanced_ensemble()
            
            # Step 5: Optimize thresholds
            self.optimize_advanced_thresholds()
            
            # Step 6: Calibrate predictions
            self.calibrate_predictions()
            
            # Step 7: Generate test predictions
            test_predictions = self.generate_test_predictions()
            
            # Step 8: Create submission
            submission_file = self.create_submission(test_predictions)
            
            # Step 9: Save pipeline
            self.save_pipeline()
            
            total_time = time.time() - start_time
            
            print("\n" + "="*80)
            print("ADVANCED PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Submission file: {submission_file}")
            
            # Print final performance summary
            print("\nFinal Performance Summary:")
            self.evaluator.print_comparison()
            
            return submission_file
            
        except Exception as e:
            print(f"\nPipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Run the advanced pipeline."""
    print("Advanced CAFA-6 Scoring Pipeline")
    print("="*50)
    
    # Example usage
    data_dir = '/path/to/cafa-6-data'
    output_dir = './advanced_results'
    
    pipeline = AdvancedScoringPipeline(data_dir, output_dir)
    
    # Run with advanced features
    submission = pipeline.run_full_advanced_pipeline(
        train_size=2000,
        val_size=500,
        n_terms=500,
        feature_types=['multiscale', 'graph', 'evolutionary']
    )
    
    if submission:
        print(f"Advanced submission created: {submission}")


if __name__ == '__main__':
    main()
