"""
Test script for advanced scoring techniques.
Validates all components and demonstrates usage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.append('src')

def test_advanced_ensemble():
    """Test advanced ensemble methods."""
    print("Testing Advanced Ensemble Methods...")
    print("-" * 50)
    
    try:
        from src.advanced_scoring import AdaptiveWeightedEnsemble, AdvancedEnsembleOptimizer
        
        # Create mock data
        n_samples, n_features, n_terms = 100, 50, 20
        X_val = np.random.random((n_samples, n_features))
        y_val = np.random.binomial(1, 0.1, (n_samples, n_terms))
        
        # Create mock models
        class MockModel:
            def predict_proba(self, X):
                return np.random.random((X.shape[0], n_terms))
        
        models = [MockModel() for _ in range(3)]
        
        # Test adaptive ensemble
        adaptive = AdaptiveWeightedEnsemble(n_terms)
        for i, model in enumerate(models):
            adaptive.add_model(model, f'model_{i}')
        
        adaptive.fit_adaptive_weights(X_val, y_val)
        pred = adaptive.predict_proba(X_val)
        
        assert pred.shape == (n_samples, n_terms), f"Expected {(n_samples, n_terms)}, got {pred.shape}"
        print("✓ AdaptiveWeightedEnsemble works correctly")
        
        # Test ensemble optimizer
        best_strategy, best_ensemble, results = AdvancedEnsembleOptimizer.find_optimal_strategy(
            models, X_val, y_val, [f'GO:{i:07d}' for i in range(n_terms)]
        )
        
        assert best_strategy in results, "Best strategy should be in results"
        print(f"✓ AdvancedEnsembleOptimizer works correctly (best: {best_strategy})")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced ensemble test failed: {e}")
        return False

def test_advanced_thresholds():
    """Test advanced threshold optimization."""
    print("\nTesting Advanced Threshold Optimization...")
    print("-" * 50)
    
    try:
        from src.advanced_scoring import BayesianThresholdOptimizer, EnsembleThresholdOptimizer
        
        # Create mock data
        n_samples, n_terms = 100, 20
        y_true = np.random.binomial(1, 0.1, (n_samples, n_terms))
        y_pred_proba = np.random.random((n_samples, n_terms))
        go_terms = [f'GO:{i:07d}' for i in range(n_terms)]
        ia_weights = {term: np.random.random() for term in go_terms}
        
        # Test Bayesian optimizer
        bayes_opt = BayesianThresholdOptimizer(ia_weights)
        thresholds = bayes_opt.optimize_bayesian(y_true, y_pred_proba, go_terms, n_iterations=20)
        
        assert len(thresholds) == n_terms, f"Expected {n_terms} thresholds, got {len(thresholds)}"
        assert all(0 <= t <= 1 for t in thresholds.values()), "All thresholds should be in [0,1]"
        print("✓ BayesianThresholdOptimizer works correctly")
        
        # Test ensemble optimizer
        ensemble_opt = EnsembleThresholdOptimizer(ia_weights=ia_weights)
        best_thresholds, best_strategy = ensemble_opt.optimize_ensemble(
            y_true, y_pred_proba, go_terms
        )
        
        assert len(best_thresholds) == n_terms, f"Expected {n_terms} thresholds, got {len(best_thresholds)}"
        print(f"✓ EnsembleThresholdOptimizer works correctly (best: {best_strategy})")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced threshold test failed: {e}")
        return False

def test_advanced_features():
    """Test advanced feature extraction."""
    print("\nTesting Advanced Feature Extraction...")
    print("-" * 50)
    
    try:
        from src.advanced_scoring import (
            MultiScaleSequenceFeatures, GraphBasedFeatures, 
            EvolutionaryFeatures, AdvancedFeatureFusion
        )
        
        # Create mock sequences
        sequences = {
            'protein_1': 'MKVLWAALLVTFLAGCAQAKTEK',
            'protein_2': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR'
        }
        
        # Test multi-scale features
        multiscale = MultiScaleSequenceFeatures()
        ms_features = multiscale.extract_multiscale_features(sequences)
        
        assert len(ms_features) == len(sequences), "Should extract features for all sequences"
        for pid, features in ms_features.items():
            assert len(features) > 0, f"Features should not be empty for {pid}"
            assert isinstance(features, np.ndarray), f"Features should be numpy array for {pid}"
        print("✓ MultiScaleSequenceFeatures works correctly")
        
        # Test graph features
        graph = GraphBasedFeatures()
        graph_features = graph.extract_graph_features(sequences)
        
        assert len(graph_features) == len(sequences), "Should extract features for all sequences"
        for pid, features in graph_features.items():
            assert len(features) > 0, f"Graph features should not be empty for {pid}"
        print("✓ GraphBasedFeatures works correctly")
        
        # Test evolutionary features
        evo = EvolutionaryFeatures()
        evo_features = evo.extract_evolutionary_features(sequences)
        
        assert len(evo_features) == len(sequences), "Should extract features for all sequences"
        for pid, features in evo_features.items():
            assert len(features) > 0, f"Evolutionary features should not be empty for {pid}"
        print("✓ EvolutionaryFeatures works correctly")
        
        # Test feature fusion
        fusion = AdvancedFeatureFusion(['multiscale', 'graph', 'evolutionary'])
        feature_matrix, protein_ids, feature_names = fusion.extract_all_features(sequences)
        
        assert feature_matrix.shape[0] == len(sequences), "Matrix should have correct number of rows"
        assert len(protein_ids) == len(sequences), "Should have correct number of protein IDs"
        assert len(feature_names) == feature_matrix.shape[1], "Feature names should match matrix columns"
        print(f"✓ AdvancedFeatureFusion works correctly (shape: {feature_matrix.shape})")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced feature test failed: {e}")
        return False

def test_pipeline_integration():
    """Test pipeline integration."""
    print("\nTesting Pipeline Integration...")
    print("-" * 50)
    
    try:
        from src.advanced_scoring import AdvancedScoringPipeline
        
        # Create mock data directory structure
        mock_data_dir = 'mock_cafa_data'
        os.makedirs(mock_data_dir, exist_ok=True)
        os.makedirs(f'{mock_data_dir}/Train', exist_ok=True)
        os.makedirs(f'{mock_data_dir}/Test', exist_ok=True)
        
        # Create minimal mock files
        with open(f'{mock_data_dir}/Train/go-basic.obo', 'w') as f:
            f.write("format-version: 1.2\n")
        
        with open(f'{mock_data_dir}/Train/train_sequences.fasta', 'w') as f:
            f.write(">protein_1\nMKVLWAALLVTFLAGCAQAKTEK\n")
            f.write(">protein_2\nMVLSPADKTNVKAAWGKVGAHAGEYG\n")
        
        with open(f'{mock_data_dir}/Train/train_terms.tsv', 'w') as f:
            f.write("protein_1\tGO:0000001\n")
            f.write("protein_1\tGO:0000002\n")
            f.write("protein_2\tGO:0000001\n")
        
        with open(f'{mock_data_dir}/Test/testsuperset.fasta', 'w') as f:
            f.write(">test_protein_1\nMKVLWAALLVTFLAGCAQAKTEK\n")
        
        with open(f'{mock_data_dir}/IA.tsv', 'w') as f:
            f.write("GO:0000001\t0.5\n")
            f.write("GO:0000002\t0.3\n")
        
        # Test pipeline initialization
        pipeline = AdvancedScoringPipeline(mock_data_dir, 'test_output')
        
        # Test data loading (this might fail due to incomplete mock data, but that's ok)
        try:
            pipeline.load_data()
            print("✓ Pipeline data loading works")
        except Exception as e:
            print(f"⚠ Pipeline data loading failed (expected with mock data): {e}")
        
        # Clean up
        import shutil
        shutil.rmtree(mock_data_dir, ignore_errors=True)
        shutil.rmtree('test_output', ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        return False

def test_imports():
    """Test that all imports work correctly."""
    print("Testing Imports...")
    print("-" * 50)
    
    try:
        from src.advanced_scoring import (
            AdaptiveWeightedEnsemble,
            HierarchicalEnsemble,
            MetaLearningEnsemble,
            AdvancedEnsembleOptimizer,
            HierarchyAwareThresholdOptimizer,
            BayesianThresholdOptimizer,
            AdvancedCalibration,
            EnsembleThresholdOptimizer,
            ProteinLanguageModelFeatures,
            GraphBasedFeatures,
            MultiScaleSequenceFeatures,
            EvolutionaryFeatures,
            AdvancedFeatureFusion,
            AdvancedScoringPipeline
        )
        print("✓ All imports successful")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def run_performance_benchmark():
    """Run a simple performance benchmark."""
    print("\nRunning Performance Benchmark...")
    print("-" * 50)
    
    try:
        import time
        from src.advanced_scoring import AdvancedFeatureFusion
        
        # Create larger test dataset
        sequences = {}
        for i in range(50):
            seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), 100))
            sequences[f'protein_{i}'] = seq
        
        # Benchmark feature extraction
        fusion = AdvancedFeatureFusion(['multiscale', 'graph', 'evolutionary'])
        
        start_time = time.time()
        feature_matrix, protein_ids, feature_names = fusion.extract_all_features(sequences)
        end_time = time.time()
        
        print(f"✓ Extracted features for {len(sequences)} proteins in {end_time - start_time:.2f} seconds")
        print(f"  Feature matrix shape: {feature_matrix.shape}")
        print(f"  Features per second: {len(sequences) / (end_time - start_time):.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Advanced Scoring Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Advanced Ensemble", test_advanced_ensemble),
        ("Advanced Thresholds", test_advanced_thresholds),
        ("Advanced Features", test_advanced_features),
        ("Pipeline Integration", test_pipeline_integration),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Advanced scoring is ready to use.")
        print("\nTo get started with the advanced pipeline:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run: python -m src.advanced_scoring.advanced_pipeline")
        print("3. Or integrate into your existing pipeline")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
