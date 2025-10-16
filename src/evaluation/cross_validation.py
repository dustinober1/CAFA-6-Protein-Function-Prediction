"""
Cross-validation utilities for protein function prediction.

Implements stratified k-fold splitting that respects class imbalance and
provides comprehensive evaluation across folds.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Iterator
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MultiLabelBinarizer
import logging

logger = logging.getLogger(__name__)


class StratifiedMultiLabelKFold:
    """Stratified k-fold split for multi-label classification."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, 
                 random_state: int = 42, test_size: Optional[float] = None):
        """
        Initialize stratified multi-label k-fold splitter.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle before splitting
            random_state: Random seed
            test_size: Fraction for test set (if None, use 1/n_splits)
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.test_size = test_size or (1.0 / n_splits)
        
        self.rng = np.random.RandomState(random_state)
    
    def split(self, X: np.ndarray, y: np.ndarray, 
              groups: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for stratified k-fold.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Multi-label targets (n_samples, n_labels) - binary matrix
            groups: Optional group labels (ignored for compatibility)
            
        Yields:
            (train_indices, test_indices) tuples
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        # Shuffle indices if requested
        if self.shuffle:
            self.rng.shuffle(indices)
        
        # Create stratification based on most common labels
        # For each sample, find its label pattern
        label_patterns = {}
        for i in range(n_samples):
            pattern = tuple(np.where(y[i] > 0)[0])
            if pattern not in label_patterns:
                label_patterns[pattern] = []
            label_patterns[pattern].append(i)
        
        # Distribute each pattern across folds
        fold_indices = [[] for _ in range(self.n_splits)]
        
        for pattern, indices_for_pattern in label_patterns.items():
            # Shuffle indices for this pattern
            pattern_indices = np.array(indices_for_pattern)
            if self.shuffle:
                self.rng.shuffle(pattern_indices)
            
            # Distribute across folds
            for fold_idx, sample_idx in enumerate(pattern_indices):
                fold = fold_idx % self.n_splits
                fold_indices[fold].append(sample_idx)
        
        # Yield train/test splits
        for fold in range(self.n_splits):
            test_fold_indices = np.array(fold_indices[fold])
            train_fold_indices = np.concatenate([fold_indices[f] for f in range(self.n_splits) if f != fold])
            
            yield train_fold_indices, test_fold_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits


class CrossValidationEvaluator:
    """Evaluate models using cross-validation."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize cross-validation evaluator.
        
        Args:
            n_splits: Number of folds
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.fold_results = []
    
    def cross_validate(self, model_factory, X: np.ndarray, y: np.ndarray,
                      eval_fn=None, verbose: bool = True) -> Dict:
        """
        Perform cross-validation on a model.
        
        Args:
            model_factory: Callable that returns a model instance
            X: Features (n_samples, n_features)
            y: Targets (n_samples, n_labels)
            eval_fn: Evaluation function(y_true, y_pred) -> dict of metrics
            verbose: Whether to print progress
            
        Returns:
            Dict with aggregated CV results
        """
        self.fold_results = []
        
        splitter = StratifiedMultiLabelKFold(n_splits=self.n_splits, 
                                            random_state=self.random_state)
        
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            if verbose:
                logger.info(f"Fold {fold+1}/{self.n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = model_factory()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            if eval_fn:
                metrics = eval_fn(y_test, y_pred)
            else:
                metrics = self._default_metrics(y_test, y_pred)
            
            metrics['fold'] = fold
            fold_metrics.append(metrics)
            self.fold_results.append(metrics)
            
            if verbose:
                logger.info(f"  F1: {metrics.get('f1', 0):.4f}, "
                           f"Precision: {metrics.get('precision', 0):.4f}, "
                           f"Recall: {metrics.get('recall', 0):.4f}")
        
        # Aggregate results
        results = self._aggregate_results(fold_metrics)
        results['fold_results'] = fold_metrics
        
        return results
    
    @staticmethod
    def _default_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute default evaluation metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Convert to binary predictions (threshold > 0.5)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Compute metrics with care for empty predictions
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
            recall = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    
    @staticmethod
    def _aggregate_results(fold_results: List[Dict]) -> Dict:
        """Aggregate results across folds."""
        df = pd.DataFrame(fold_results)
        
        aggregated = {}
        for col in df.columns:
            if col != 'fold' and df[col].dtype in [np.float64, np.int64]:
                aggregated[f'{col}_mean'] = float(df[col].mean())
                aggregated[f'{col}_std'] = float(df[col].std())
        
        return aggregated


class FoldDataBuilder:
    """Build train/val/test splits for experiments."""
    
    def __init__(self, random_state: int = 42):
        """Initialize fold builder."""
        self.random_state = random_state
    
    def create_stratified_split(self, X: np.ndarray, y: np.ndarray,
                               val_size: float = 0.2, 
                               test_size: float = 0.1) -> Dict:
        """
        Create stratified train/val/test split.
        
        Args:
            X: Features
            y: Targets (multi-label binary matrix)
            val_size: Fraction for validation
            test_size: Fraction for test
            
        Returns:
            Dict with train/val/test indices and splits
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        # Create stratified groups based on label frequency
        # Group samples by number of labels
        label_counts = y.sum(axis=1)
        
        from sklearn.model_selection import train_test_split
        
        # Split test first (stratified by label count)
        train_val_idx, test_idx = train_test_split(
            indices, 
            test_size=test_size,
            random_state=self.random_state,
            stratify=label_counts
        )
        
        # Split train/val (stratified by label count)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size/(1-test_size),
            random_state=self.random_state,
            stratify=label_counts[train_val_idx]
        )
        
        return {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'X_train': X[train_idx],
            'X_val': X[val_idx],
            'X_test': X[test_idx],
            'y_train': y[train_idx],
            'y_val': y[val_idx],
            'y_test': y[test_idx],
        }
    
    def create_k_fold_splits(self, X: np.ndarray, y: np.ndarray,
                            n_splits: int = 5) -> List[Dict]:
        """
        Create k stratified folds.
        
        Args:
            X: Features
            y: Targets (multi-label binary matrix)
            n_splits: Number of folds
            
        Returns:
            List of dicts, each containing train/test split for a fold
        """
        splitter = StratifiedMultiLabelKFold(n_splits=n_splits, 
                                            random_state=self.random_state)
        
        folds = []
        for train_idx, test_idx in splitter.split(X, y):
            folds.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'X_train': X[train_idx],
                'X_test': X[test_idx],
                'y_train': y[train_idx],
                'y_test': y[test_idx],
            })
        
        return folds
