"""
Baseline models using traditional ML methods.
"""
import numpy as np
import pickle
from typing import Dict, List, Tuple
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import time


class BaselineModel:
    """Base class for baseline models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.go_terms = None
        self.term_to_idx = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              go_terms: List[str], verbose: bool = True):
        """
        Train the model.
        
        Args:
            X_train: Feature matrix (n_samples, n_features)
            y_train: Target matrix (n_samples, n_terms)
            go_terms: List of GO terms
            verbose: Print training info
        """
        self.go_terms = go_terms
        self.term_to_idx = {term: i for i, term in enumerate(go_terms)}
        
        if verbose:
            print(f"Training {self.name}...")
            print(f"  Training set shape: {X_train.shape}")
            print(f"  Target matrix shape: {y_train.shape}")
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  Training completed in {elapsed:.2f} seconds")
    
    def predict(self, X_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Feature matrix
            threshold: Probability threshold
            
        Returns:
            Prediction matrix
        """
        predictions = self.model.predict_proba(X_test)
        return (predictions >= threshold).astype(int)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Feature matrix
            y_test: Target matrix
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test, threshold=0.5)
        
        # Calculate per-term metrics
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(y_test.shape[1]):
            y_true = y_test[:, i]
            y_pred = predictions[:, i]
            
            # Skip if no positive examples
            if y_true.sum() == 0:
                continue
            
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        metrics = {
            'name': self.name,
            'avg_precision': np.mean(precisions) if precisions else 0,
            'avg_recall': np.mean(recalls) if recalls else 0,
            'avg_f1': np.mean(f1_scores) if f1_scores else 0,
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model to file."""
        data = {
            'model': self.model,
            'go_terms': self.go_terms,
            'term_to_idx': self.term_to_idx,
            'name': self.name,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {self.name} to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.go_terms = data['go_terms']
        self.term_to_idx = data['term_to_idx']
        print(f"Loaded {self.name} from {filepath}")


class SVMModel(BaselineModel):
    """SVM-based model using OneVsRest strategy."""
    
    def __init__(self, C: float = 1.0):
        super().__init__('SVM (OneVsRest)')
        self.model = OneVsRestClassifier(
            LinearSVC(C=C, max_iter=1000, dual=False, random_state=42),
            n_jobs=-1
        )


class RandomForestModel(BaselineModel):
    """Random Forest model using OneVsRest strategy."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        super().__init__('Random Forest (OneVsRest)')
        self.model = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            ),
            n_jobs=-1
        )


def main():
    """Test baseline models."""
    from data_loader import CAFADataLoader
    from feature_extractor import ProteinFeatureExtractor
    
    # Load data
    print("Loading data...")
    loader = CAFADataLoader('/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/cafa-6-protein-function-prediction')
    loader.load_train_data()
    
    # Split data
    train_seqs, val_seqs, train_terms, val_terms = loader.split_train_test(test_size=0.2)
    
    print(f"Train set: {len(train_seqs)} proteins")
    print(f"Val set: {len(val_seqs)} proteins")
    
    # Extract features
    print("\nExtracting features...")
    extractor = ProteinFeatureExtractor(k=3)
    
    # Use combined features for better performance
    X_train, train_ids = extractor.create_combined_features(train_seqs, fit_tfidf=True)
    X_val, val_ids = extractor.create_combined_features(val_seqs, fit_tfidf=False)
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Create target matrices
    print("Creating target matrices...")
    all_go_terms = sorted(loader.go_terms)
    
    y_train, _ = loader.create_protein_to_terms_matrix(train_ids, all_go_terms)
    y_val, _ = loader.create_protein_to_terms_matrix(val_ids, all_go_terms)
    
    print(f"Target matrix shape: {y_train.shape}")
    
    # Subset to most common terms for faster testing
    term_frequencies = y_train.sum(axis=0)
    common_term_indices = np.argsort(term_frequencies)[-500:]  # Top 500 terms
    
    y_train_subset = y_train[:, common_term_indices]
    y_val_subset = y_val[:, common_term_indices]
    common_terms = [all_go_terms[i] for i in common_term_indices]
    
    print(f"Using {len(common_terms)} common GO terms for testing")
    print(f"Subset target matrix shape: {y_train_subset.shape}")
    
    # Train and evaluate models
    models = [
        SVMModel(C=1.0),
        RandomForestModel(n_estimators=50, max_depth=15),
    ]
    
    results = []
    
    for model in models:
        print(f"\n{'='*60}")
        model.train(X_train, y_train_subset, common_terms)
        
        print(f"\nEvaluating {model.name}...")
        metrics = model.evaluate(X_val, y_val_subset)
        results.append(metrics)
        
        print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
        print(f"  Avg Recall: {metrics['avg_recall']:.4f}")
        print(f"  Avg F1: {metrics['avg_f1']:.4f}")
        
        # Save model
        model.save_model(f'/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/{model.name.replace(" ", "_").lower()}.pkl')
    
    print(f"\n{'='*60}")
    print("\nResults Summary:")
    for result in results:
        print(f"{result['name']}:")
        print(f"  Avg F1: {result['avg_f1']:.4f}")
    
    print("\nBaseline models test completed successfully!")


if __name__ == '__main__':
    main()
