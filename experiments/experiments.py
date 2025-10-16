"""
Utility script for running specific experiments and analyses.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import CAFADataLoader
from src.features.feature_extractor import ProteinFeatureExtractor
from src.models.baseline_models import RandomForestModel
from src.evaluation.evaluation import ModelEvaluator, AnalysisUtils
import time


def experiment_feature_comparison(data_dir: str, n_proteins: int = 500):
    """Compare different feature types on a small dataset."""
    print("\n" + "="*80)
    print("EXPERIMENT: Feature Type Comparison")
    print("="*80)
    
    # Load data
    loader = CAFADataLoader(data_dir)
    loader.load_train_data()
    
    # Sample
    sample_ids = list(loader.train_sequences.keys())[:n_proteins]
    train_ids = sample_ids[:int(n_proteins*0.8)]
    val_ids = sample_ids[int(n_proteins*0.8):]
    
    train_seqs = {pid: loader.train_sequences[pid] for pid in train_ids}
    val_seqs = {pid: loader.train_sequences[pid] for pid in val_ids}
    
    # Extract different feature types
    extractor = ProteinFeatureExtractor(k=3)
    
    feature_types = {}
    
    # Composition
    X_comp_train, _ = extractor.create_composition_features(train_seqs)
    X_comp_val, _ = extractor.create_composition_features(val_seqs)
    feature_types['Composition (20-dim)'] = (X_comp_train, X_comp_val)
    
    # Property
    import numpy as np
    X_prop_train = np.array([
        extractor.get_property_features(seq) for seq in train_seqs.values()
    ])
    X_prop_val = np.array([
        extractor.get_property_features(seq) for seq in val_seqs.values()
    ])
    feature_types['Property (6-dim)'] = (X_prop_train, X_prop_val)
    
    # Combined
    X_comb_train, _ = extractor.create_combined_features(train_seqs, fit_tfidf=True)
    X_comb_val, _ = extractor.create_combined_features(val_seqs, fit_tfidf=False)
    feature_types['Combined (427-dim)'] = (X_comb_train, X_comb_val)
    
    # Create target matrix
    all_go_terms = sorted(loader.go_terms)
    y_train_full, _ = loader.create_protein_to_terms_matrix(train_ids, all_go_terms)
    y_val_full, _ = loader.create_protein_to_terms_matrix(val_ids, all_go_terms)
    
    # Use top terms
    term_freq = y_train_full.sum(axis=0)
    top_idx = np.argsort(term_freq)[-50:]
    y_train = y_train_full[:, top_idx]
    y_val = y_val_full[:, top_idx]
    top_terms = [all_go_terms[i] for i in top_idx]
    
    # Compare
    evaluator = ModelEvaluator(loader.ia_weights)
    
    print("\nTraining models on each feature type...")
    for feat_name, (X_train, X_val) in feature_types.items():
        model = RandomForestModel(n_estimators=20, max_depth=10)
        model.train(X_train, y_train, top_terms, verbose=False)
        
        metrics = evaluator.evaluate_model(feat_name, y_val, 
                                          model.predict(X_val, threshold=0.5),
                                          top_terms)
        
        print(f"\n{feat_name}:")
        print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
    
    evaluator.print_comparison()


def experiment_model_size_comparison(data_dir: str):
    """Compare model performance vs training set size."""
    print("\n" + "="*80)
    print("EXPERIMENT: Model Size Comparison")
    print("="*80)
    
    loader = CAFADataLoader(data_dir)
    loader.load_train_data()
    
    extractor = ProteinFeatureExtractor(k=3)
    evaluator = ModelEvaluator(loader.ia_weights)
    
    # Different training set sizes
    sizes = [200, 500, 1000, 2000]
    results = []
    
    for size in sizes:
        print(f"\nTraining with {size} proteins...")
        
        sample_ids = list(loader.train_sequences.keys())[:size + 300]
        train_ids = sample_ids[:size]
        val_ids = sample_ids[size:]
        
        train_seqs = {pid: loader.train_sequences[pid] for pid in train_ids}
        val_seqs = {pid: loader.train_sequences[pid] for pid in val_ids}
        
        X_train, _ = extractor.create_combined_features(train_seqs, fit_tfidf=True)
        X_val, _ = extractor.create_combined_features(val_seqs, fit_tfidf=False)
        
        # Target matrix
        all_go_terms = sorted(loader.go_terms)
        y_train_full, _ = loader.create_protein_to_terms_matrix(train_ids, all_go_terms)
        y_val_full, _ = loader.create_protein_to_terms_matrix(val_ids, all_go_terms)
        
        term_freq = y_train_full.sum(axis=0)
        top_idx = np.argsort(term_freq)[-80:]
        y_train = y_train_full[:, top_idx]
        y_val = y_val_full[:, top_idx]
        top_terms = [all_go_terms[i] for i in top_idx]
        
        # Train
        start = time.time()
        model = RandomForestModel(n_estimators=30, max_depth=12)
        model.train(X_train, y_train, top_terms, verbose=False)
        elapsed = time.time() - start
        
        # Evaluate
        metrics = evaluator.evaluate_model(f"Size-{size}", y_val,
                                          model.predict(X_val, threshold=0.5),
                                          top_terms)
        metrics['size'] = size
        metrics['training_time'] = elapsed
        results.append(metrics)
        
        print(f"  Training time: {elapsed:.2f}s")
        print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
    
    # Summary
    print("\n" + "-"*80)
    print("Summary:")
    print("-"*80)
    df = pd.DataFrame(results)
    print(df[['size', 'f1_micro', 'training_time']].to_string(index=False))


def experiment_threshold_analysis(data_dir: str, n_proteins: int = 1000):
    """Analyze effect of prediction threshold."""
    print("\n" + "="*80)
    print("EXPERIMENT: Threshold Analysis")
    print("="*80)
    
    loader = CAFADataLoader(data_dir)
    loader.load_train_data()
    
    # Sample
    sample_ids = list(loader.train_sequences.keys())[:n_proteins]
    train_ids = sample_ids[:int(n_proteins*0.8)]
    val_ids = sample_ids[int(n_proteins*0.8):]
    
    train_seqs = {pid: loader.train_sequences[pid] for pid in train_ids}
    val_seqs = {pid: loader.train_sequences[pid] for pid in val_ids}
    
    # Features
    extractor = ProteinFeatureExtractor(k=3)
    X_train, _ = extractor.create_combined_features(train_seqs, fit_tfidf=True)
    X_val, _ = extractor.create_combined_features(val_seqs, fit_tfidf=False)
    
    # Target
    all_go_terms = sorted(loader.go_terms)
    y_train_full, _ = loader.create_protein_to_terms_matrix(train_ids, all_go_terms)
    y_val_full, _ = loader.create_protein_to_terms_matrix(val_ids, all_go_terms)
    
    term_freq = y_train_full.sum(axis=0)
    top_idx = np.argsort(term_freq)[-100:]
    y_train = y_train_full[:, top_idx]
    y_val = y_val_full[:, top_idx]
    top_terms = [all_go_terms[i] for i in top_idx]
    
    # Train
    model = RandomForestModel(n_estimators=30, max_depth=12)
    model.train(X_train, y_train, top_terms, verbose=False)
    
    # Get probabilities
    proba = model.predict_proba(X_val)
    
    # Test different thresholds
    from sklearn.metrics import f1_score
    
    print("\nTesting different thresholds...")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)
    
    results = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = (proba >= threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score
        prec = precision_score(y_val, predictions, average='micro', zero_division=0)
        rec = recall_score(y_val, predictions, average='micro', zero_division=0)
        f1 = f1_score(y_val, predictions, average='micro', zero_division=0)
        
        print(f"{threshold:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
        results.append({'threshold': threshold, 'f1': f1})
    
    best = max(results, key=lambda x: x['f1'])
    print(f"\nBest threshold: {best['threshold']:.1f} (F1: {best['f1']:.4f})")


def main():
    """Run experiments."""
    parser = argparse.ArgumentParser(description='Run CAFA-6 experiments')
    parser.add_argument('experiment', choices=['features', 'sizes', 'threshold'],
                       help='Experiment to run')
    parser.add_argument('--data-dir', default=
                       '/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/cafa-6-protein-function-prediction',
                       help='Path to data directory')
    parser.add_argument('--n-proteins', type=int, default=500,
                       help='Number of proteins to use')
    
    args = parser.parse_args()
    
    if args.experiment == 'features':
        experiment_feature_comparison(args.data_dir, args.n_proteins)
    elif args.experiment == 'sizes':
        experiment_model_size_comparison(args.data_dir)
    elif args.experiment == 'threshold':
        experiment_threshold_analysis(args.data_dir, args.n_proteins)


if __name__ == '__main__':
    main()
