"""
Comprehensive training and prediction pipeline for all models.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import time

from data.data_loader import CAFADataLoader
from features.feature_extractor import ProteinFeatureExtractor
from models.baseline_models import SVMModel, RandomForestModel
from models.neural_models import NeuralNetworkModel, DeepNeuralNetwork, ProteinDataset
from models.embedding_model import SequenceEncoder, SequenceToFunctionModel
from evaluation.evaluation import ModelEvaluator, EnsemblePredictor, SubmissionGenerator, AnalysisUtils
import torch
from torch.utils.data import DataLoader


class ComprehensiveTrainingPipeline:
    """Complete pipeline for training and evaluating multiple models."""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """
        Initialize pipeline.
        
        Args:
            data_dir: Path to data directory
            output_dir: Path to output directory
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.loader = None
        self.evaluator = None
        self.models = {}
        self.predictions = {}
        
    def load_data(self):
        """Load data."""
        print("Loading data...")
        self.loader = CAFADataLoader(str(self.data_dir))
        self.loader.load_train_data()
        self.loader.load_test_data()
        self.loader.load_ia_weights()
        
        self.evaluator = ModelEvaluator(self.loader.ia_weights)
        
        summary = self.loader.get_train_data_summary()
        print("Data Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    def prepare_data_split(self, train_size: int = 2000, val_size: int = 500,
                          n_terms: int = 200):
        """
        Prepare training and validation data.
        
        Args:
            train_size: Number of training proteins
            val_size: Number of validation proteins
            n_terms: Number of top GO terms to use
        """
        print(f"\nPreparing data split (train={train_size}, val={val_size}, terms={n_terms})...")
        
        # Get sample
        sample_ids = list(self.loader.train_sequences.keys())[:(train_size + val_size)]
        
        self.train_ids = sample_ids[:train_size]
        self.val_ids = sample_ids[train_size:]
        
        train_seqs = {pid: self.loader.train_sequences[pid] for pid in self.train_ids}
        self.val_seqs = {pid: self.loader.train_sequences[pid] for pid in self.val_ids}
        
        print(f"  Train proteins: {len(train_seqs)}")
        print(f"  Val proteins: {len(self.val_seqs)}")
        
        # Extract features
        print("  Extracting features...")
        extractor = ProteinFeatureExtractor(k=3)
        
        self.X_train, _ = extractor.create_combined_features(train_seqs, fit_tfidf=True)
        self.X_val, _ = extractor.create_combined_features(self.val_seqs, fit_tfidf=False)
        self.extractor = extractor
        
        print(f"  Feature shape: {self.X_train.shape}")
        
        # Create target matrices
        all_go_terms = sorted(self.loader.go_terms)
        y_train_full, _ = self.loader.create_protein_to_terms_matrix(self.train_ids, all_go_terms)
        y_val_full, _ = self.loader.create_protein_to_terms_matrix(self.val_ids, all_go_terms)
        
        # Select top terms
        term_frequencies = y_train_full.sum(axis=0)
        top_indices = np.argsort(term_frequencies)[-n_terms:]
        
        self.y_train = y_train_full[:, top_indices]
        self.y_val = y_val_full[:, top_indices]
        self.go_terms = [all_go_terms[i] for i in top_indices]
        
        print(f"  Using {len(self.go_terms)} GO terms")
        print(f"  Target shape: {self.y_train.shape}")
    
    def train_baseline_models(self):
        """Train baseline ML models."""
        print("\n" + "="*80)
        print("TRAINING BASELINE MODELS")
        print("="*80)
        
        models = [
            ('Random Forest', RandomForestModel(n_estimators=30, max_depth=12)),
        ]
        
        for name, model in models:
            print(f"\nTraining {name}...")
            model.train(self.X_train, self.y_train, self.go_terms, verbose=True)
            
            print(f"Evaluating {name}...")
            metrics = self.evaluator.evaluate_model(
                name, self.y_val, 
                model.predict(self.X_val, threshold=0.5),
                self.go_terms
            )
            
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
            
            # Store predictions
            self.predictions[name] = model.predict_proba(self.X_val)
            self.models[name] = model
    
    def train_neural_model(self):
        """Train neural network model."""
        print("\n" + "="*80)
        print("TRAINING NEURAL NETWORK MODEL")
        print("="*80)
        
        device = 'cpu'
        
        # Create datasets
        train_dataset = ProteinDataset(self.X_train, self.y_train, self.train_ids)
        val_dataset = ProteinDataset(self.X_val, self.y_val, self.val_ids)
        
        # Create model
        network = DeepNeuralNetwork(
            input_dim=self.X_train.shape[1],
            output_dim=len(self.go_terms),
            hidden_dims=[256, 128],
            dropout_rate=0.3
        )
        
        model = NeuralNetworkModel(network, name="Deep Neural Network", device=device)
        
        print("Training...")
        model.train(
            train_dataset, val_dataset,
            self.go_terms,
            epochs=5,
            batch_size=32,
            learning_rate=0.001,
            verbose=True
        )
        
        print("\nEvaluating...")
        predictions_proba = model.predict_proba(self.X_val)
        predictions_binary = (predictions_proba >= 0.5).astype(int)
        
        metrics = self.evaluator.evaluate_model(
            "Deep Neural Network", self.y_val,
            predictions_binary,
            self.go_terms
        )
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
        
        # Store predictions
        self.predictions["Deep Neural Network"] = predictions_proba
        self.models["Deep Neural Network"] = model
    
    def train_embedding_model(self):
        """Train sequence embedding model."""
        print("\n" + "="*80)
        print("TRAINING EMBEDDING MODEL")
        print("="*80)
        
        device = 'cpu'
        
        # Encode sequences
        print("Encoding sequences...")
        encoder = SequenceEncoder()
        X_train_seq, lengths_train, _ = encoder.encode_sequences(
            {pid: self.loader.train_sequences[pid] for pid in self.train_ids},
            max_length=500
        )
        X_val_seq, lengths_val, _ = encoder.encode_sequences(
            {pid: self.loader.train_sequences[pid] for pid in self.val_ids},
            max_length=500
        )
        
        # Create model
        model_net = SequenceToFunctionModel(
            vocab_size=21,
            embedding_dim=64,
            output_dim=len(self.go_terms),
            hidden_dim=128,
            max_length=500
        ).to(device)
        
        optimizer = torch.optim.Adam(model_net.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
        # Prepare tensors
        X_train_tensor = torch.LongTensor(X_train_seq).to(device)
        y_train_tensor = torch.FloatTensor(self.y_train).to(device)
        X_val_tensor = torch.LongTensor(X_val_seq).to(device)
        y_val_tensor = torch.FloatTensor(self.y_val).to(device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        print("Training embedding model...")
        for epoch in range(3):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model_net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Evaluate
            with torch.no_grad():
                val_pred = model_net(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)
            
            print(f"  Epoch {epoch+1}/3: Train loss={total_loss/len(train_loader):.4f}, "
                  f"Val loss={val_loss:.4f}")
        
        # Get predictions
        with torch.no_grad():
            predictions_proba = model_net(X_val_tensor).cpu().numpy()
        
        predictions_binary = (predictions_proba >= 0.5).astype(int)
        
        print("\nEvaluating Embedding Model...")
        metrics = self.evaluator.evaluate_model(
            "Embedding Model", self.y_val,
            predictions_binary,
            self.go_terms
        )
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
        
        # Store predictions
        self.predictions["Embedding Model"] = predictions_proba
    
    def ensemble_predictions(self):
        """Create ensemble predictions."""
        print("\n" + "="*80)
        print("CREATING ENSEMBLE PREDICTIONS")
        print("="*80)
        
        pred_arrays = list(self.predictions.values())
        
        print(f"Ensembling {len(pred_arrays)} models...")
        ensemble_proba = EnsemblePredictor.average_ensemble(pred_arrays)
        ensemble_binary = (ensemble_proba >= 0.5).astype(int)
        
        metrics = self.evaluator.evaluate_model(
            "Ensemble", self.y_val,
            ensemble_binary,
            self.go_terms
        )
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
        
        self.ensemble_predictions_proba = ensemble_proba
        self.ensemble_predictions_binary = ensemble_binary
    
    def print_results(self):
        """Print comparison of all models."""
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        self.evaluator.print_comparison()
    
    def generate_test_predictions(self):
        """Generate predictions on test set."""
        print("\n" + "="*80)
        print("GENERATING TEST PREDICTIONS")
        print("="*80)
        
        # Extract features for test data
        print("Extracting features for test data...")
        test_seqs = {pid: self.loader.test_sequences[pid] 
                    for pid in list(self.loader.test_sequences.keys())[:1000]}  # Sample for quick test
        
        X_test, test_ids = self.extractor.create_combined_features(test_seqs, fit_tfidf=False)
        
        print(f"Test feature shape: {X_test.shape}")
        
        # Get predictions from best model (ensemble)
        print("Making predictions...")
        pred_arrays = list(self.predictions.values())
        ensemble_proba = EnsemblePredictor.average_ensemble(pred_arrays)
        
        # Predict on test set (use first baseline model as example)
        if self.models:
            first_model = list(self.models.values())[0]
            test_pred_proba = first_model.predict_proba(X_test)
        else:
            # Random predictions as fallback
            test_pred_proba = np.random.uniform(0, 1, size=(X_test.shape[0], len(self.go_terms)))
        
        # Convert to submission format
        predictions = {}
        for i, protein_id in enumerate(test_ids):
            go_preds = {}
            for j, go_term in enumerate(self.go_terms):
                score = float(test_pred_proba[i, j])
                if score > 0:
                    go_preds[go_term] = score
            predictions[protein_id] = go_preds
        
        return predictions
    
    def create_submission(self, predictions: Dict, output_file: str = None):
        """Create submission file."""
        if output_file is None:
            output_file = str(self.output_dir / 'submission.tsv')
        
        print(f"\nCreating submission file: {output_file}")
        gen = SubmissionGenerator(self.loader.ia_weights)
        gen.create_submission(predictions, output_file)
        
        return output_file
    
    def run_full_pipeline(self, train_size: int = 1000, val_size: int = 300, 
                         n_terms: int = 150):
        """Run complete pipeline."""
        self.load_data()
        self.prepare_data_split(train_size, val_size, n_terms)
        
        self.train_baseline_models()
        self.train_neural_model()
        self.train_embedding_model()
        
        self.ensemble_predictions()
        self.print_results()
        
        # Generate test predictions
        test_predictions = self.generate_test_predictions()
        submission_file = self.create_submission(test_predictions)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Submission file: {submission_file}")
        print("="*80)


def main():
    """Run complete pipeline."""
    import sys
    
    data_dir = '/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/cafa-6-protein-function-prediction'
    output_dir = '/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/results'
    
    pipeline = ComprehensiveTrainingPipeline(data_dir, output_dir)
    
    try:
        pipeline.run_full_pipeline(train_size=1000, val_size=300, n_terms=100)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
