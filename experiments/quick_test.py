"""
Quick baseline model test with smaller dataset.
"""
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import CAFADataLoader
from src.features.feature_extractor import ProteinFeatureExtractor
from src.models.baseline_models import SVMModel, RandomForestModel


def main():
    """Test baseline models on small dataset."""
    print("Loading data...")
    loader = CAFADataLoader('/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/cafa-6-protein-function-prediction')
    loader.load_train_data()
    
    # Use only a sample for quick testing
    sample_ids = list(loader.train_sequences.keys())[:1000]
    train_seqs = {pid: loader.train_sequences[pid] for pid in sample_ids[:800]}
    val_seqs = {pid: loader.train_sequences[pid] for pid in sample_ids[800:]}
    
    print(f"Train set: {len(train_seqs)} proteins")
    print(f"Val set: {len(val_seqs)} proteins")
    
    # Extract features (using only composition + property for speed)
    print("\nExtracting features...")
    extractor = ProteinFeatureExtractor(k=3)
    
    comp_train = np.array([
        extractor.get_composition_features(seq) 
        for seq in train_seqs.values()
    ])
    comp_val = np.array([
        extractor.get_composition_features(seq) 
        for seq in val_seqs.values()
    ])
    
    prop_train = np.array([
        extractor.get_property_features(seq) 
        for seq in train_seqs.values()
    ])
    prop_val = np.array([
        extractor.get_property_features(seq) 
        for seq in val_seqs.values()
    ])
    
    X_train = np.concatenate([comp_train, prop_train], axis=1)
    X_val = np.concatenate([comp_val, prop_val], axis=1)
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Create target matrices with most common terms
    print("Creating target matrices...")
    train_ids = list(train_seqs.keys())
    val_ids = list(val_seqs.keys())
    
    all_go_terms = sorted(loader.go_terms)
    y_train, _ = loader.create_protein_to_terms_matrix(train_ids, all_go_terms)
    y_val, _ = loader.create_protein_to_terms_matrix(val_ids, all_go_terms)
    
    # Use only top 100 terms for speed
    term_frequencies = y_train.sum(axis=0)
    top_indices = np.argsort(term_frequencies)[-100:]
    
    y_train = y_train[:, top_indices]
    y_val = y_val[:, top_indices]
    top_terms = [all_go_terms[i] for i in top_indices]
    
    print(f"Using {len(top_terms)} GO terms")
    print(f"Target matrix shape: {y_train.shape}")
    
    # Train quick model
    print("\nTraining Random Forest model...")
    model = RandomForestModel(n_estimators=10, max_depth=10)
    model.train(X_train, y_train, top_terms)
    
    print("\nEvaluating...")
    metrics = model.evaluate(X_val, y_val)
    
    print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"  Avg Recall: {metrics['avg_recall']:.4f}")
    print(f"  Avg F1: {metrics['avg_f1']:.4f}")
    
    print("\nQuick test completed successfully!")


if __name__ == '__main__':
    main()
