#!/usr/bin/env python3
"""
Verification script to check all modules are properly set up.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def verify_imports():
    """Verify all modules can be imported."""
    print("Verifying module imports...\n")
    
    modules = [
        # Data
        ("data.data_loader", "CAFADataLoader"),
        
        # Features
        ("features.feature_extractor", "ProteinFeatureExtractor"),
        ("features.augmentation", "ProteinAugmenter"),
        ("features.go_ontology", "GOHierarchy"),
        ("features.taxonomy_processor", "TaxonomyProcessor"),
        
        # Models
        ("models.baseline_models", "RandomForestModel"),
        ("models.neural_models", "DeepNeuralNetwork"),
        ("models.embedding_model", "SequenceToFunctionModel"),
        ("models.esm_embedder", "ESM2Embedder"),
        ("models.enhanced_ensemble", "StackingEnsemble"),
        ("models.hpo", "BayesianHyperparameterOptimizer"),
        
        # Evaluation
        ("evaluation.evaluation", "ModelEvaluator"),
        ("evaluation.cross_validation", "StratifiedMultiLabelKFold"),
        
        # Pipeline
        ("train_pipeline", "ComprehensiveTrainingPipeline"),
    ]
    
    failed = []
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except ImportError as e:
            print(f"✗ {module_name}.{class_name} - {e}")
            failed.append(f"{module_name}.{class_name}")
        except AttributeError as e:
            print(f"✗ {module_name}.{class_name} - Class not found")
            failed.append(f"{module_name}.{class_name}")
    
    print("\n" + "="*60)
    
    if failed:
        print(f"❌ {len(failed)} module(s) failed to import:")
        for module in failed:
            print(f"  - {module}")
        return False
    else:
        print(f"✅ All {len(modules)} modules imported successfully!")
        return True

def verify_structure():
    """Verify directory structure."""
    print("\nVerifying directory structure...\n")
    
    dirs = [
        "src/data",
        "src/features",
        "src/models",
        "src/evaluation",
        "scripts",
        "results",
        "cafa-6-protein-function-prediction/Train",
        "cafa-6-protein-function-prediction/Test",
    ]
    
    missing = []
    
    for dir_path in dirs:
        full_path = Path(__file__).parent / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - NOT FOUND")
            missing.append(dir_path)
    
    print("\n" + "="*60)
    
    if missing:
        print(f"❌ {len(missing)} directory(ies) missing:")
        for dir_path in missing:
            print(f"  - {dir_path}")
        return False
    else:
        print(f"✅ All {len(dirs)} directories exist!")
        return True

def verify_files():
    """Verify key files exist."""
    print("\nVerifying key files...\n")
    
    files = [
        "run_pipeline.py",
        "README.md",
        "requirements.txt",
        ".gitignore",
        "IMPLEMENTATION_GUIDE.md",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/__init__.py",
        "src/evaluation/__init__.py",
    ]
    
    missing = []
    
    for file_path in files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - NOT FOUND")
            missing.append(file_path)
    
    print("\n" + "="*60)
    
    if missing:
        print(f"❌ {len(missing)} file(s) missing:")
        for file_path in missing:
            print(f"  - {file_path}")
        return False
    else:
        print(f"✅ All {len(files)} key files exist!")
        return True

def main():
    """Run all verifications."""
    print("\n" + "="*60)
    print("CAFA-6 PROJECT VERIFICATION")
    print("="*60 + "\n")
    
    results = []
    
    # Run verifications
    results.append(("Directory Structure", verify_structure()))
    results.append(("Key Files", verify_files()))
    results.append(("Module Imports", verify_imports()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Project is ready!")
        print("="*60)
        print("\nNext steps:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Download CAFA-6 data from Kaggle")
        print("3. Extract to: cafa-6-protein-function-prediction/")
        print("4. Run pipeline: python run_pipeline.py")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
