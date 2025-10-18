"""
Demonstration of advanced CAFA-6 scoring pipeline with real data.
Shows the complete workflow with the downloaded CAFA-6 data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import sys
import os
import time

# Add src to path
sys.path.append('src')

def create_simple_submission():
    """Create a simple baseline submission for demonstration."""
    print("Creating Simple Baseline Submission...")
    print("=" * 60)
    
    try:
        from src.data.data_loader import CAFADataLoader
        from src.features.feature_extractor import ProteinFeatureExtractor
        from src.evaluation.evaluation import SubmissionGenerator
        from src.features.go_ontology import GOHierarchy
        
        # Load data
        print("1. Loading CAFA-6 data...")
        loader = CAFADataLoader('.')
        loader.load_train_data()
        loader.load_test_data()
        loader.load_ia_weights()
        
        print(f"   - Training proteins: {len(loader.train_sequences)}")
        print(f"   - Test proteins: {len(loader.test_sequences)}")
        print(f"   - GO terms: {len(loader.go_terms)}")
        print(f"   - IA weights: {len(loader.ia_weights)}")
        
        # Load GO hierarchy for propagation
        print("2. Loading GO hierarchy...")
        go_hierarchy = GOHierarchy('Train/go-basic.obo')
        print(f"   - GO terms in hierarchy: {len(go_hierarchy.terms)}")
        
        # Get most frequent GO terms for demonstration
        print("3. Selecting top GO terms...")
        term_counts = {}
        for protein_id, terms in loader.train_terms.items():
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1
        
        # Select top 500 terms by frequency
        top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:500]
        selected_terms = [term for term, count in top_terms]
        
        print(f"   - Selected {len(selected_terms)} most frequent GO terms")
        
        # Create baseline predictions using term frequencies
        print("4. Creating baseline predictions...")
        predictions = {}
        
        # Calculate term frequencies
        term_freq = {term: count / len(loader.train_sequences) 
                      for term, count in top_terms}
        
        # For each test protein, predict top terms based on sequence length
        for protein_id, sequence in loader.test_sequences.items():
            go_preds = {}
            
            # Simple heuristic: longer sequences get more predictions
            seq_length = len(sequence)
            n_predictions = min(20 + seq_length // 50, 50)  # 20-50 predictions
            
            # Select terms based on frequency
            sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
            
            for i, (term, freq) in enumerate(sorted_terms[:n_predictions]):
                # Add some variation based on sequence length
                score = freq * (1.0 + 0.1 * (seq_length - 100) / 1000)
                score = max(0.01, min(0.99, score))  # Clamp to [0.01, 0.99]
                go_preds[term] = score
            
            # Add some rare terms with lower scores
            for term, freq in sorted_terms[n_predictions:n_predictions+10]:
                score = freq * 0.1  # Lower score for rare terms
                go_preds[term] = max(0.01, score)
            
            predictions[protein_id] = go_preds
        
        print(f"   - Created predictions for {len(predictions)} proteins")
        
        # Propagate predictions using GO hierarchy
        print("5. Propagating predictions using GO hierarchy...")
        final_predictions = {}
        
        for protein_id, go_preds in predictions.items():
            propagated = {}
            
            for go_term, score in go_preds.items():
                # Add the original prediction
                propagated[go_term] = score
                
                # Add ancestors with reduced scores
                try:
                    ancestors = go_hierarchy.get_ancestors(go_term)
                    for ancestor in ancestors[:3]:  # Limit to top 3 ancestors
                        if ancestor not in propagated:
                            propagated[ancestor] = score * 0.5  # Reduced score for ancestors
                except:
                    pass
            
            final_predictions[protein_id] = propagated
        
        print(f"   - Propagated predictions for {len(final_predictions)} proteins")
        
        # Create submission file
        print("6. Creating submission file...")
        submission_file = 'baseline_submission.tsv'
        
        gen = SubmissionGenerator(loader.ia_weights, go_hierarchy)
        gen.create_submission(final_predictions, submission_file, propagate=False)
        
        print(f"   - Submission saved to: {submission_file}")
        
        # Print submission statistics
        total_predictions = sum(len(preds) for preds in final_predictions.values())
        avg_predictions = total_predictions / len(final_predictions)
        
        print(f"\nSubmission Statistics:")
        print(f"  - Total proteins: {len(final_predictions)}")
        print(f"  - Total predictions: {total_predictions}")
        print(f"  - Average predictions per protein: {avg_predictions:.2f}")
        print(f"  - Unique GO terms: {len(set.union(*[set(preds.keys()) for preds in final_predictions.values()]))}")
        
        # Show sample predictions
        print(f"\nSample Predictions:")
        sample_proteins = list(final_predictions.keys())[:3]
        for protein_id in sample_proteins:
            preds = final_predictions[protein_id]
            top_5 = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  {protein_id}:")
            for term, score in top_5:
                print(f"    {term}: {score:.4f}")
        
        return submission_file
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_feature_extraction():
    """Demonstrate advanced feature extraction."""
    print("\nDemonstrating Advanced Feature Extraction...")
    print("=" * 60)
    
    try:
        from src.data.data_loader import CAFADataLoader
        from src.features.feature_extractor import ProteinFeatureExtractor
        
        # Load sample data
        loader = CAFADataLoader('.')
        loader.load_train_data()
        
        # Get sample proteins
        sample_ids = list(loader.train_sequences.keys())[:10]
        sample_sequences = {pid: loader.train_sequences[pid] for pid in sample_ids}
        
        print(f"1. Extracting features for {len(sample_sequences)} proteins...")
        
        # Extract features
        extractor = ProteinFeatureExtractor()
        features, protein_ids = extractor.create_combined_features(sample_sequences)
        
        print(f"   - Feature matrix shape: {features.shape}")
        print(f"   - Feature types included:")
        print(f"     * Amino acid composition (20 features)")
        print(f"     * Biochemical properties (6 features)")
        print(f"     * Length features (1 feature)")
        print(f"     * Dipeptide composition (400 features)")
        
        # Show sample features
        print(f"\n2. Sample feature vectors:")
        for i, pid in enumerate(protein_ids[:3]):
            feature_vec = features[i]
            print(f"   {pid}:")
            print(f"     Length: {len(feature_vec)}")
            print(f"     Sample values: [{feature_vec[0]:.3f}, {feature_vec[1]:.3f}, ...]")
        
        # Analyze feature statistics
        print(f"\n3. Feature Statistics:")
        print(f"   - Mean values: {np.mean(features, axis=0)[:5]}")
        print(f"   - Std values: {np.std(features, axis=0)[:5]}")
        print(f"   - Min values: {np.min(features, axis=0)[:5]}")
        print(f"   - Max values: {np.max(features, axis=0)[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature extraction demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_evaluation_metrics():
    """Demonstrate evaluation metrics."""
    print("\nDemonstrating Evaluation Metrics...")
    print("=" * 60)
    
    try:
        from src.data.data_loader import CAFADataLoader
        from src.evaluation.evaluation import ModelEvaluator
        
        # Load data
        loader = CAFADataLoader('.')
        loader.load_train_data()
        loader.load_ia_weights()
        
        # Create dummy predictions for evaluation
        sample_ids = list(loader.train_sequences.keys())[:100]
        sample_terms = list(loader.go_terms)[:50]
        
        y_true, _ = loader.create_protein_to_terms_matrix(sample_ids, sample_terms)
        
        # Create different prediction scenarios
        scenarios = {
            'Random': np.random.random(y_true.shape),
            'High Precision': (np.random.random(y_true.shape) > 0.8).astype(float),
            'High Recall': (np.random.random(y_true.shape) > 0.2).astype(float),
            'Balanced': (np.random.random(y_true.shape) > 0.5).astype(float)
        }
        
        evaluator = ModelEvaluator(loader.ia_weights)
        
        print("1. Evaluating different prediction scenarios:")
        for name, y_pred in scenarios.items():
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics = evaluator.evaluate_model(name, y_true, y_pred_binary, sample_terms)
            
            print(f"\n   {name} Predictions:")
            print(f"     - Precision: {metrics['precision']:.4f}")
            print(f"     - Recall: {metrics['recall']:.4f}")
            print(f"     - F1 Micro: {metrics['f1_micro']:.4f}")
            print(f"     - F1 Macro: {metrics['f1_macro']:.4f}")
            print(f"     - F1 Weighted: {metrics['f1_weighted']:.4f}")
        
        # Calculate Fmax (maximum F1 across thresholds) - simplified version
        print(f"\n2. Calculating Fmax for balanced predictions...")
        y_pred_proba = scenarios['Balanced']
        
        # Simple Fmax calculation
        thresholds = np.linspace(0.1, 0.9, 9)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_binary = (y_pred_proba > threshold).astype(int)
            metrics = evaluator.evaluate_model(f"Threshold {threshold:.1f}", y_true, y_pred_binary, sample_terms)
            f1_scores.append(metrics['f1_weighted'])
        
        fmax_score = max(f1_scores)
        print(f"   - Fmax Score: {fmax_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete demonstration."""
    print("Advanced CAFA-6 Scoring Pipeline Demonstration")
    print("=" * 70)
    print("This demo shows the complete workflow with real CAFA-6 data.")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run demonstrations
    results = []
    
    # 1. Create baseline submission
    submission = create_simple_submission()
    results.append(("Baseline Submission", submission is not None))
    
    # 2. Demonstrate feature extraction
    features_ok = demonstrate_feature_extraction()
    results.append(("Feature Extraction", features_ok))
    
    # 3. Demonstrate evaluation metrics
    eval_ok = demonstrate_evaluation_metrics()
    results.append(("Evaluation Metrics", eval_ok))
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for demo_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{demo_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} demonstrations successful")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All demonstrations completed successfully!")
        print("\nThe advanced scoring pipeline is ready for full deployment.")
        print("\nNext steps:")
        print("1. Wait for PyTorch installation to complete")
        print("2. Run the full advanced pipeline:")
        print("   python -m src.advanced_scoring.advanced_pipeline")
        print("3. Submit your predictions to CAFA-6")
        
        if submission:
            print(f"\nBaseline submission created: {submission}")
            print("You can upload this to Kaggle to test the submission format.")
    else:
        print(f"\n⚠️  {total - passed} demonstration(s) failed.")
        print("Please check the implementation and try again.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
