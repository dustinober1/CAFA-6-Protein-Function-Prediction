"""
Advanced winning pipeline integrating all techniques for CAFA-6 protein function prediction.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data.data_loader import CAFADataLoader
from features.advanced_features import AdvancedFeatureExtractor
from features.go_ontology import GOHierarchy
from models.baseline_models import RandomForestModel
from models.neural_models import NeuralNetworkModel, DeepNeuralNetwork, ProteinDataset
from models.gradient_boosting import LightGBMModel, XGBoostModel, StackedEnsemble
from evaluation.evaluation import ModelEvaluator, EnsemblePredictor, SubmissionGenerator
from evaluation.threshold_optimizer import ThresholdOptimizer, GOHierarchyEnforcer


class AdvancedWinningPipeline:
    """Advanced pipeline with all winning techniques."""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """Initialize pipeline."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.loader = None
        self.evaluator = None
        self.go_hierarchy = None
        self.models = {}
        self.predictions = {}
        self.threshold_optimizer = None
        self.go_enforcer = None
    
    def load_data(self):
        """Load all data."""
        print("Loading data...")
        self.loader = CAFADataLoader(str(self.data_dir))
        self.loader.load_train_data()
        self.loader.load_test_data()
        self.loader.load_ia_weights()
        
        print("Loading GO hierarchy...")
        go_obo_path = self.data_dir / 'Train' / 'go-basic.obo'
        self.go_hierarchy = GOHierarchy(str(go_obo_path))
        
        self.evaluator = ModelEvaluator(self.loader.ia_weights)
        self.threshold_optimizer = ThresholdOptimizer(self.loader.ia_weights)
        self.go_enforcer = GOHierarchyEnforcer(self.go_hierarchy)
        
        summary = self.loader.get_train_data_summary()
        print(f"Data summary: {summary}")
    
    def prepare_data(self, train_size: int = 3000, val_size: int = 1000, 
                    n_terms: int = 800):
        """Prepare training/validation data with advanced features."""
        print(f"\nPreparing data (train={train_size}, val={val_size}, terms={n_terms})...")
        
        # Select top GO terms by frequency
        from collections import defaultdict
        term_freq = defaultdict(int)
        for pid, terms in self.loader.train_terms.items():
            for term in terms:
                term_freq[term] += 1
        
        # Get top n_terms
        top_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:n_terms]
        self.go_terms = [t[0] for t in top_terms]
        
        # Split data
        protein_ids = list(self.loader.train_sequences.keys())
        np.random.seed(42)
        np.random.shuffle(protein_ids)
        
        split_idx = train_size
        self.train_ids = protein_ids[:split_idx]
        self.val_ids = protein_ids[split_idx:split_idx + val_size]
        
        print(f"  Train proteins: {len(self.train_ids)}")
        print(f"  Val proteins: {len(self.val_ids)}")
        print(f"  Using {len(self.go_terms)} GO terms")
        
        # Extract advanced features
        print("Extracting advanced features...")
        adv_extractor = AdvancedFeatureExtractor(self.loader.train_taxonomy)
        
        train_seqs = {pid: self.loader.train_sequences[pid] for pid in self.train_ids}
        val_seqs = {pid: self.loader.train_sequences[pid] for pid in self.val_ids}
        
        self.X_train, _, feature_names = adv_extractor.extract_batch(train_seqs)
        self.X_val, _, _ = adv_extractor.extract_batch(val_seqs)
        
        print(f"  Feature shape: {self.X_train.shape}")
        
        # Create label matrices
        self.y_train, _ = self.loader.create_protein_to_terms_matrix(
            self.train_ids, self.go_terms
        )
        self.y_val, _ = self.loader.create_protein_to_terms_matrix(
            self.val_ids, self.go_terms
        )
    
    def train_all_models(self):
        """Train all model types."""
        print("\n" + "="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        
        # Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestModel(n_estimators=50, max_depth=15)
        rf_model.train(self.X_train, self.y_train, self.go_terms, verbose=True)
        self.models['RF'] = rf_model
        self.predictions['RF'] = rf_model.predict_proba(self.X_val)
        
        # LightGBM
        print("\nTraining LightGBM...")
        try:
            lgb_model = LightGBMModel(n_estimators=150, max_depth=8, learning_rate=0.05)
            lgb_model.train(self.X_train, self.y_train, self.go_terms, 
                          ia_weights=self.loader.ia_weights, verbose=True)
            self.models['LGB'] = lgb_model
            self.predictions['LGB'] = lgb_model.predict_proba(self.X_val)
            print("  LightGBM trained successfully")
        except Exception as e:
            print(f"  LightGBM failed: {e}")
        
        # XGBoost
        print("\nTraining XGBoost...")
        try:
            xgb_model = XGBoostModel(n_estimators=150, max_depth=6, learning_rate=0.05)
            xgb_model.train(self.X_train, self.y_train, self.go_terms,
                          ia_weights=self.loader.ia_weights, verbose=True)
            self.models['XGB'] = xgb_model
            self.predictions['XGB'] = xgb_model.predict_proba(self.X_val)
            print("  XGBoost trained successfully")
        except Exception as e:
            print(f"  XGBoost failed: {e}")
        
        # Neural Network
        print("\nTraining Neural Network...")
        try:
            nn_model = NeuralNetworkModel(
                DeepNeuralNetwork(
                    input_dim=self.X_train.shape[1],
                    output_dim=len(self.go_terms),
                    hidden_dims=[512, 256, 128],
                    dropout_rate=0.3
                ),
                name="Neural Network",
                device='cpu'
            )
            
            train_dataset = ProteinDataset(self.X_train, self.y_train, self.train_ids)
            val_dataset = ProteinDataset(self.X_val, self.y_val, self.val_ids)
            
            nn_model.train(
                train_dataset, val_dataset, self.go_terms,
                epochs=20, batch_size=64, learning_rate=0.001, verbose=True
            )
            
            self.models['NN'] = nn_model
            self.predictions['NN'] = nn_model.predict_proba(self.X_val)
            print("  Neural Network trained successfully")
        except Exception as e:
            print(f"  Neural Network failed: {e}")
    
    def optimize_ensemble_and_thresholds(self):
        """Create ensemble and optimize thresholds."""
        print("\n" + "="*80)
        print("OPTIMIZING ENSEMBLE AND THRESHOLDS")
        print("="*80)
        
        # Create ensemble
        print("Averaging predictions from all models...")
        pred_arrays = list(self.predictions.values())
        self.ensemble_proba = EnsemblePredictor.average_ensemble(pred_arrays)
        
        # Optimize thresholds
        print("Optimizing per-term thresholds...")
        self.threshold_optimizer.optimize_thresholds(
            self.y_val, self.ensemble_proba, self.go_terms,
            grid_search=True, n_thresholds=50
        )
        
        # Evaluate with optimized thresholds
        print("Evaluating with optimized thresholds...")
        ensemble_binary = self.threshold_optimizer.apply_thresholds(
            self.ensemble_proba, self.go_terms
        )
        
        metrics = self.evaluator.evaluate_model(
            "Ensemble (Optimized)", self.y_val, ensemble_binary, self.go_terms
        )
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
    
    def generate_test_predictions(self):
        """Generate final test predictions."""
        print("\n" + "="*80)
        print("GENERATING TEST PREDICTIONS")
        print("="*80)
        
        # Extract features for test data
        print("Extracting features for ALL test data...")
        adv_extractor = AdvancedFeatureExtractor(self.loader.train_taxonomy)
        X_test, test_ids, _ = adv_extractor.extract_batch(self.loader.test_sequences)
        
        print(f"  Test shape: {X_test.shape}")
        
        # Get predictions from best available model
        print("Making predictions...")
        test_preds_list = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict_proba(X_test)
                test_preds_list.append(pred)
                print(f"  {model_name} predictions: shape {pred.shape}")
            except Exception as e:
                print(f"  {model_name} failed: {e}")
        
        if test_preds_list:
            test_pred_proba = EnsemblePredictor.average_ensemble(test_preds_list)
        else:
            # Fallback: frequency-based
            test_pred_proba = np.tile(
                np.mean(self.y_train, axis=0),
                (len(test_ids), 1)
            )
        
        # Apply optimized thresholds
        print("Applying optimized thresholds...")
        test_pred_binary = self.threshold_optimizer.apply_thresholds(
            test_pred_proba, self.go_terms
        )
        
        # Convert to submission format
        print("Converting to submission format...")
        predictions = {}
        for i, protein_id in enumerate(test_ids):
            go_preds = {}
            for j, go_term in enumerate(self.go_terms):
                if test_pred_binary[i, j] == 1:
                    # Use probability from proba matrix for scoring
                    score = float(test_pred_proba[i, j])
                    if score > 0.01:
                        go_preds[go_term] = score
            
            # Ensure every protein has predictions
            if not go_preds:
                top_indices = np.argsort(test_pred_proba[i])[-10:]
                for idx in top_indices:
                    go_preds[self.go_terms[idx]] = float(test_pred_proba[i, idx])
            
            predictions[protein_id] = go_preds
        
        print(f"Generated predictions for {len(predictions)} proteins")
        return predictions
    
    def create_submission(self, predictions: Dict):
        """Create final submission with GO hierarchy enforcement."""
        print("\n" + "="*80)
        print("CREATING SUBMISSION")
        print("="*80)
        
        output_file = str(self.output_dir / 'submission.tsv')
        
        # Enforce GO hierarchy constraints
        print("Enforcing GO hierarchy constraints...")
        predictions = self.go_enforcer.propagate_to_root(predictions)
        
        # Create submission
        print(f"Writing submission to {output_file}...")
        gen = SubmissionGenerator(self.loader.ia_weights, self.go_hierarchy)
        gen.create_submission(predictions, output_file, propagate=False, 
                            max_predictions_per_protein=1500)
        
        return output_file
    
    def run_full_pipeline(self, train_size: int = 3000, val_size: int = 1000,
                         n_terms: int = 800):
        """Run complete advanced pipeline."""
        self.load_data()
        self.prepare_data(train_size, val_size, n_terms)
        self.train_all_models()
        self.optimize_ensemble_and_thresholds()
        
        test_predictions = self.generate_test_predictions()
        submission_file = self.create_submission(test_predictions)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED!")
        print(f"Submission: {submission_file}")
        print("="*80)


def main():
    """Run advanced pipeline."""
    data_dir = '/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/cafa-6-protein-function-prediction'
    output_dir = '/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/results'
    
    pipeline = AdvancedWinningPipeline(data_dir, output_dir)
    pipeline.run_full_pipeline(train_size=3000, val_size=1000, n_terms=800)


if __name__ == '__main__':
    main()
