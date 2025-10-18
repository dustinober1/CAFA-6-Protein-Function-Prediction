"""
Basic functionality test for CAFA-6 pipeline without heavy dependencies.
Tests the core data loading and basic feature extraction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.append('src')

def test_data_loading():
    """Test basic data loading functionality."""
    print("Testing Basic Data Loading...")
    print("-" * 50)
    
    try:
        from src.data.data_loader import CAFADataLoader
        
        # Test with the downloaded data
        loader = CAFADataLoader('.')
        
        # Load training data
        print("Loading training data...")
        loader.load_train_data()
        
        # Load test data
        print("Loading test data...")
        loader.load_test_data()
        
        # Load IA weights
        print("Loading IA weights...")
        loader.load_ia_weights()
        
        # Get summary
        summary = loader.get_train_data_summary()
        print("Data Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Test some basic functionality
        print(f"\nSample proteins: {list(loader.train_sequences.keys())[:3]}")
        print(f"Sample GO terms: {list(loader.go_terms)[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_go_hierarchy():
    """Test GO hierarchy loading."""
    print("\nTesting GO Hierarchy...")
    print("-" * 50)
    
    try:
        from src.features.go_ontology import GOHierarchy
        
        # Load GO hierarchy
        go_hierarchy = GOHierarchy('Train/go-basic.obo')
        
        # Test basic functionality
        print(f"Loaded GO hierarchy with {len(go_hierarchy.terms)} terms")
        
        # Test getting ancestors/descendants
        if go_hierarchy.terms:
            sample_term = list(go_hierarchy.terms.keys())[0]
            ancestors = go_hierarchy.get_ancestors(sample_term)
            descendants = go_hierarchy.get_descendants(sample_term)
            print(f"Sample term {sample_term}: {len(ancestors)} ancestors, {len(descendants)} descendants")
        
        return True
        
    except Exception as e:
        print(f"✗ GO hierarchy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_features():
    """Test basic feature extraction without heavy dependencies."""
    print("\nTesting Basic Feature Extraction...")
    print("-" * 50)
    
    try:
        from src.features.feature_extractor import ProteinFeatureExtractor
        
        # Create sample data
        sequences = {
            'protein_1': 'MKVLWAALLVTFLAGCAQAKTEK',
            'protein_2': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR'
        }
        
        # Extract features
        extractor = ProteinFeatureExtractor()
        features, protein_ids = extractor.create_combined_features(sequences)
        
        print(f"Extracted features for {len(protein_ids)} proteins")
        print(f"Feature matrix shape: {features.shape}")
        for i, pid in enumerate(protein_ids):
            print(f"  {pid}: {len(features[i])} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic feature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_models():
    """Test basic model functionality."""
    print("\nTesting Basic Models...")
    print("-" * 50)
    
    try:
        from src.models.baseline_models import RandomForestModel
        from src.data.data_loader import CAFADataLoader
        
        # Load data
        loader = CAFADataLoader('.')
        loader.load_train_data()
        loader.load_ia_weights()
        
        # Get small sample for testing
        sample_ids = list(loader.train_sequences.keys())[:100]
        sample_sequences = {pid: loader.train_sequences[pid] for pid in sample_ids}
        
        # Extract basic features
        from src.features.feature_extractor import ProteinFeatureExtractor
        extractor = ProteinFeatureExtractor()
        features, _ = extractor.create_combined_features(sample_sequences)
        
        # Create feature matrix
        feature_matrix = features
        
        # Create target matrix (small subset)
        sample_terms = list(loader.go_terms)[:50]
        y_train, _ = loader.create_protein_to_terms_matrix(sample_ids, sample_terms)
        
        # Train basic model
        print("Training Random Forest...")
        rf_model = RandomForestModel(n_estimators=10, max_depth=5)  # Small for testing
        rf_model.train(feature_matrix, y_train, sample_terms, verbose=False)
        
        # Test prediction
        predictions = rf_model.predict_proba(feature_matrix)
        print(f"Prediction shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation():
    """Test evaluation functionality."""
    print("\nTesting Evaluation...")
    print("-" * 50)
    
    try:
        from src.evaluation.evaluation import ModelEvaluator
        from src.data.data_loader import CAFADataLoader
        
        # Load data
        loader = CAFADataLoader('.')
        loader.load_train_data()
        loader.load_ia_weights()
        
        # Create dummy predictions for testing
        sample_ids = list(loader.train_sequences.keys())[:50]
        sample_terms = list(loader.go_terms)[:20]
        y_true, _ = loader.create_protein_to_terms_matrix(sample_ids, sample_terms)
        y_pred = np.random.random((len(sample_ids), len(sample_terms)))
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Evaluate
        evaluator = ModelEvaluator(loader.ia_weights)
        metrics = evaluator.evaluate_model("Test Model", y_true, y_pred_binary, sample_terms)
        
        print("Evaluation metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("Basic CAFA-6 Functionality Test Suite")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("GO Hierarchy", test_go_hierarchy),
        ("Basic Features", test_basic_features),
        ("Basic Models", test_basic_models),
        ("Evaluation", test_evaluation)
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
        print("\n🎉 All basic tests passed! Core functionality is working.")
        print("\nNext steps:")
        print("1. Wait for PyTorch installation to complete")
        print("2. Run advanced scoring tests")
        print("3. Test the complete pipeline")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
