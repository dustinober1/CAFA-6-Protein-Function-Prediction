"""
Data augmentation techniques for protein sequences.

Implements various augmentation strategies to increase training data diversity
and improve model robustness.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ProteinAugmenter:
    """Augment protein sequences for improved training."""
    
    # Standard amino acids
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Biochemically similar amino acids
    SIMILARITY_GROUPS = {
        'A': ['A', 'V', 'I', 'L', 'M'],           # Small hydrophobic
        'V': ['A', 'V', 'I', 'L', 'M'],
        'I': ['A', 'V', 'I', 'L', 'M'],
        'L': ['A', 'V', 'I', 'L', 'M'],
        'M': ['A', 'V', 'I', 'L', 'M'],
        
        'D': ['D', 'E'],                           # Acidic
        'E': ['D', 'E'],
        
        'N': ['N', 'Q', 'S', 'T'],                 # Polar
        'Q': ['N', 'Q', 'S', 'T'],
        'S': ['N', 'Q', 'S', 'T'],
        'T': ['N', 'Q', 'S', 'T'],
        
        'K': ['K', 'R', 'H'],                      # Positive
        'R': ['K', 'R', 'H'],
        'H': ['K', 'R', 'H'],
        
        'F': ['F', 'W', 'Y'],                      # Aromatic
        'W': ['F', 'W', 'Y'],
        'Y': ['F', 'W', 'Y'],
        
        'C': ['C'],                                 # Special
        'G': ['G'],
        'P': ['P'],
    }
    
    def __init__(self, random_state: int = 42):
        """
        Initialize augmenter.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_state)
    
    def substitute_similar(self, sequence: str, rate: float = 0.1) -> str:
        """
        Replace amino acids with biochemically similar ones.
        
        Args:
            sequence: Amino acid sequence
            rate: Substitution rate (0-1)
            
        Returns:
            Augmented sequence
        """
        seq_list = list(sequence)
        n_substitutions = max(1, int(len(sequence) * rate))
        
        # Randomly select positions to substitute
        positions = self.rng.choice(len(sequence), size=n_substitutions, replace=False)
        
        for pos in positions:
            aa = seq_list[pos]
            if aa in self.SIMILARITY_GROUPS:
                similar_aas = self.SIMILARITY_GROUPS[aa]
                seq_list[pos] = self.rng.choice(similar_aas)
        
        return ''.join(seq_list)
    
    def random_substitution(self, sequence: str, rate: float = 0.05) -> str:
        """
        Replace amino acids with random amino acids (mutation-like).
        
        Args:
            sequence: Amino acid sequence
            rate: Substitution rate (0-1)
            
        Returns:
            Augmented sequence
        """
        seq_list = list(sequence)
        n_substitutions = max(1, int(len(sequence) * rate))
        
        positions = self.rng.choice(len(sequence), size=n_substitutions, replace=False)
        
        for pos in positions:
            seq_list[pos] = self.rng.choice(list(self.AMINO_ACIDS))
        
        return ''.join(seq_list)
    
    def deletion(self, sequence: str, rate: float = 0.05) -> str:
        """
        Delete amino acids from sequence.
        
        Args:
            sequence: Amino acid sequence
            rate: Deletion rate (0-1)
            
        Returns:
            Augmented sequence
        """
        n_deletions = max(1, int(len(sequence) * rate))
        positions = sorted(self.rng.choice(len(sequence), size=n_deletions, replace=False), reverse=True)
        
        seq_list = list(sequence)
        for pos in positions:
            del seq_list[pos]
        
        return ''.join(seq_list)
    
    def insertion(self, sequence: str, rate: float = 0.05) -> str:
        """
        Insert random amino acids into sequence.
        
        Args:
            sequence: Amino acid sequence
            rate: Insertion rate (0-1)
            
        Returns:
            Augmented sequence
        """
        n_insertions = max(1, int(len(sequence) * rate))
        
        seq_list = list(sequence)
        
        # Insert at random positions (process in reverse to maintain indices)
        for _ in range(n_insertions):
            pos = self.rng.randint(0, len(seq_list) + 1)
            aa = self.rng.choice(list(self.AMINO_ACIDS))
            seq_list.insert(pos, aa)
        
        return ''.join(seq_list)
    
    def masking(self, sequence: str, rate: float = 0.15, mask_token: str = 'X') -> str:
        """
        Mask (replace with X) random positions.
        
        Args:
            sequence: Amino acid sequence
            rate: Masking rate (0-1)
            mask_token: Token to use for masking
            
        Returns:
            Augmented sequence
        """
        seq_list = list(sequence)
        n_masks = max(1, int(len(sequence) * rate))
        positions = self.rng.choice(len(sequence), size=n_masks, replace=False)
        
        for pos in positions:
            seq_list[pos] = mask_token
        
        return ''.join(seq_list)
    
    def rotation(self, sequence: str, shift: Optional[int] = None) -> str:
        """
        Rotate sequence (circular shift).
        
        Args:
            sequence: Amino acid sequence
            shift: Number of positions to shift. If None, random.
            
        Returns:
            Augmented sequence
        """
        if shift is None:
            shift = self.rng.randint(1, len(sequence))
        
        shift = shift % len(sequence)
        return sequence[-shift:] + sequence[:-shift]
    
    def augment(self, sequence: str, augmentation_type: str, **kwargs) -> str:
        """
        Apply specified augmentation to sequence.
        
        Args:
            sequence: Amino acid sequence
            augmentation_type: One of: similar, random, deletion, insertion, masking, rotation
            **kwargs: Additional arguments for specific augmentation
            
        Returns:
            Augmented sequence
        """
        if augmentation_type == 'similar':
            return self.substitute_similar(sequence, rate=kwargs.get('rate', 0.1))
        elif augmentation_type == 'random':
            return self.random_substitution(sequence, rate=kwargs.get('rate', 0.05))
        elif augmentation_type == 'deletion':
            return self.deletion(sequence, rate=kwargs.get('rate', 0.05))
        elif augmentation_type == 'insertion':
            return self.insertion(sequence, rate=kwargs.get('rate', 0.05))
        elif augmentation_type == 'masking':
            return self.masking(sequence, rate=kwargs.get('rate', 0.15))
        elif augmentation_type == 'rotation':
            return self.rotation(sequence)
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    def augment_batch(self, sequences: List[str], augmentation_type: str,
                     rate: float = 0.1, num_augmentations: int = 1) -> List[str]:
        """
        Create multiple augmented versions of sequences.
        
        Args:
            sequences: List of amino acid sequences
            augmentation_type: Type of augmentation to apply
            rate: Augmentation rate parameter
            num_augmentations: Number of augmented versions per sequence
            
        Returns:
            List of augmented sequences (sequences + augmentations)
        """
        augmented = list(sequences)  # Start with originals
        
        for _ in range(num_augmentations):
            for seq in sequences:
                aug_seq = self.augment(seq, augmentation_type, rate=rate)
                augmented.append(aug_seq)
        
        return augmented


class AugmentationPipeline:
    """Apply multiple augmentations to create diverse training data."""
    
    def __init__(self, augmenter: ProteinAugmenter):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmenter: ProteinAugmenter instance
        """
        self.augmenter = augmenter
    
    def create_augmentation_mix(self, sequences: List[str],
                               augmentation_types: Optional[List[str]] = None,
                               num_samples_per_seq: int = 2) -> Dict[str, List[str]]:
        """
        Create augmented dataset using mix of strategies.
        
        Args:
            sequences: Original sequences
            augmentation_types: Types of augmentation to use
            num_samples_per_seq: Number of augmentations per sequence
            
        Returns:
            Dict with original and augmented sequences
        """
        if augmentation_types is None:
            augmentation_types = ['similar', 'masking', 'deletion']
        
        result = {
            'original': sequences,
            'augmented': []
        }
        
        for aug_type in augmentation_types:
            augmented = self.augmenter.augment_batch(sequences, aug_type, 
                                                     num_augmentations=num_samples_per_seq)
            result['augmented'].extend(augmented)
        
        return result
    
    def create_balanced_augmentation(self, sequences: List[str], 
                                     labels: np.ndarray,
                                     minority_threshold: float = 0.1) -> Tuple[List[str], np.ndarray]:
        """
        Create augmented dataset to balance class distribution.
        
        Args:
            sequences: Original sequences
            labels: Multi-label binary matrix (n_samples, n_labels)
            minority_threshold: Fraction of samples to consider minority (rare labels)
            
        Returns:
            (augmented_sequences, augmented_labels)
        """
        # Find rare labels (minority classes)
        label_frequencies = labels.sum(axis=0) / labels.shape[0]
        minority_labels = np.where(label_frequencies < minority_threshold)[0]
        
        # Find samples with minority labels that could be augmented
        samples_with_minority = []
        for label_idx in minority_labels:
            samples_with_minority.extend(np.where(labels[:, label_idx] > 0)[0])
        
        # Remove duplicates and shuffle
        minority_samples = list(set(samples_with_minority))
        self.augmenter.rng.shuffle(minority_samples)
        
        # Create augmentations
        augmented_sequences = list(sequences)
        augmented_labels = list(labels)
        
        augmentation_types = ['similar', 'masking', 'deletion', 'rotation']
        
        for sample_idx in minority_samples:
            original_seq = sequences[sample_idx]
            original_label = labels[sample_idx].copy()
            
            # Create 2-3 augmentations per minority sample
            n_aug = self.augmenter.rng.randint(2, 4)
            for _ in range(n_aug):
                aug_type = self.augmenter.rng.choice(augmentation_types)
                aug_seq = self.augmenter.augment(original_seq, aug_type)
                augmented_sequences.append(aug_seq)
                augmented_labels.append(original_label)
        
        return augmented_sequences, np.vstack(augmented_labels)


def apply_mixup_augmentation(X1: np.ndarray, X2: np.ndarray, 
                             y1: np.ndarray, y2: np.ndarray,
                             alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply mixup augmentation on two samples.
    
    Args:
        X1, X2: Feature vectors
        y1, y2: Labels (multi-hot vectors)
        alpha: Beta distribution parameter for mixing ratio
        
    Returns:
        (mixed_X, mixed_y)
    """
    lam = np.random.beta(alpha, alpha)
    
    mixed_X = lam * X1 + (1 - lam) * X2
    mixed_y = np.maximum(y1, y2)  # Use logical OR for multi-label
    
    return mixed_X, mixed_y
