"""
Advanced scoring module for CAFA-6 protein function prediction.

This module implements state-of-the-art techniques for improving CAFA-6 scores including:
- Advanced ensemble methods (adaptive weighting, meta-learning, hierarchical ensembling)
- Sophisticated threshold optimization (hierarchy-aware, Bayesian, adaptive)
- Advanced feature engineering (protein language models, graph-based, multi-scale)
- Complete pipeline integration with calibration and optimization

Key Components:
- AdvancedEnsemble: Multiple ensemble strategies with automatic optimization
- AdvancedThresholdOptimizer: Sophisticated threshold selection methods
- AdvancedFeatureFusion: Cutting-edge feature extraction techniques
- AdvancedScoringPipeline: Complete end-to-end pipeline

Usage:
    from src.advanced_scoring import AdvancedScoringPipeline
    
    pipeline = AdvancedScoringPipeline(data_dir, output_dir)
    submission = pipeline.run_full_advanced_pipeline()
"""

from .advanced_ensemble import (
    AdaptiveWeightedEnsemble,
    HierarchicalEnsemble,
    MetaLearningEnsemble,
    ConfidenceWeightedEnsemble,
    AdvancedEnsembleOptimizer
)

from .advanced_thresholds import (
    HierarchyAwareThresholdOptimizer,
    BayesianThresholdOptimizer,
    AdvancedCalibration,
    AdaptiveThresholdSelector,
    EnsembleThresholdOptimizer
)

from .advanced_features import (
    ProteinLanguageModelFeatures,
    GraphBasedFeatures,
    MultiScaleSequenceFeatures,
    EvolutionaryFeatures,
    AdvancedFeatureFusion
)

from .advanced_pipeline import AdvancedScoringPipeline

__all__ = [
    # Ensemble methods
    'AdaptiveWeightedEnsemble',
    'HierarchicalEnsemble', 
    'MetaLearningEnsemble',
    'ConfidenceWeightedEnsemble',
    'AdvancedEnsembleOptimizer',
    
    # Threshold optimization
    'HierarchyAwareThresholdOptimizer',
    'BayesianThresholdOptimizer',
    'AdvancedCalibration',
    'AdaptiveThresholdSelector',
    'EnsembleThresholdOptimizer',
    
    # Feature engineering
    'ProteinLanguageModelFeatures',
    'GraphBasedFeatures',
    'MultiScaleSequenceFeatures',
    'EvolutionaryFeatures',
    'AdvancedFeatureFusion',
    
    # Main pipeline
    'AdvancedScoringPipeline'
]

__version__ = "1.0.0"
__author__ = "Advanced CAFA-6 Team"
__description__ = "Advanced scoring techniques for CAFA-6 protein function prediction"
