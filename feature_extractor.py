"""
Feature extraction methods for protein sequences.
"""
import numpy as np
from typing import Dict, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
import string


class ProteinFeatureExtractor:
    """Extract various features from protein sequences."""
    
    # Standard amino acid codes
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    
    # Biochemical properties
    PROPERTY_GROUPS = {
        'hydrophobic': set('AILMFVP'),
        'polar_uncharged': set('STNQ'),
        'polar_charged_positive': set('KR'),
        'polar_charged_negative': set('DE'),
        'aromatic': set('FWY'),
        'special': set('CGH'),
    }
    
    def __init__(self, k: int = 3):
        """
        Initialize feature extractor.
        
        Args:
            k: K-mer length
        """
        self.k = k
        self.kmer_vocab = set()
        self.vectorizer = None
    
    @staticmethod
    def normalize_sequence(seq: str) -> str:
        """Normalize protein sequence (uppercase, remove invalid chars)."""
        seq = seq.upper()
        return ''.join(c for c in seq if c in ProteinFeatureExtractor.AMINO_ACIDS)
    
    @staticmethod
    def get_kmer_features(seq: str, k: int = 3) -> List[str]:
        """
        Extract k-mer features from sequence.
        
        Args:
            seq: Protein sequence
            k: K-mer length
            
        Returns:
            List of k-mers
        """
        seq = ProteinFeatureExtractor.normalize_sequence(seq)
        kmers = []
        for i in range(len(seq) - k + 1):
            kmers.append(seq[i:i+k])
        return kmers
    
    @staticmethod
    def get_composition_features(seq: str) -> np.ndarray:
        """
        Get amino acid composition features.
        
        Returns:
            20-dimensional vector of amino acid percentages
        """
        seq = ProteinFeatureExtractor.normalize_sequence(seq)
        composition = np.zeros(20)
        
        for aa in seq:
            if aa in ProteinFeatureExtractor.AA_TO_IDX:
                composition[ProteinFeatureExtractor.AA_TO_IDX[aa]] += 1
        
        if len(seq) > 0:
            composition /= len(seq)
        
        return composition
    
    @staticmethod
    def get_property_features(seq: str) -> np.ndarray:
        """
        Get biochemical property features.
        
        Returns:
            6-dimensional vector of property ratios
        """
        seq = ProteinFeatureExtractor.normalize_sequence(seq)
        
        properties = np.zeros(len(ProteinFeatureExtractor.PROPERTY_GROUPS))
        
        for i, (prop_name, amino_acids) in enumerate(
            ProteinFeatureExtractor.PROPERTY_GROUPS.items()
        ):
            properties[i] = sum(1 for aa in seq if aa in amino_acids)
        
        if len(seq) > 0:
            properties /= len(seq)
        
        return properties
    
    @staticmethod
    def get_length_features(seq: str) -> np.ndarray:
        """
        Get length-based features.
        
        Returns:
            1-dimensional vector with sequence length (log scale)
        """
        seq = ProteinFeatureExtractor.normalize_sequence(seq)
        return np.array([np.log(len(seq) + 1)])
    
    @staticmethod
    def get_dipeptide_features(seq: str) -> np.ndarray:
        """
        Get dipeptide composition features.
        
        Returns:
            400-dimensional vector of dipeptide frequencies
        """
        seq = ProteinFeatureExtractor.normalize_sequence(seq)
        dipeptides = np.zeros(400)
        
        aa_set = set(ProteinFeatureExtractor.AMINO_ACIDS)
        
        for i in range(len(seq) - 1):
            aa1, aa2 = seq[i], seq[i+1]
            if aa1 in ProteinFeatureExtractor.AA_TO_IDX and aa2 in ProteinFeatureExtractor.AA_TO_IDX:
                idx = (ProteinFeatureExtractor.AA_TO_IDX[aa1] * 20 + 
                       ProteinFeatureExtractor.AA_TO_IDX[aa2])
                dipeptides[idx] += 1
        
        if len(seq) > 1:
            dipeptides /= (len(seq) - 1)
        
        return dipeptides
    
    def create_kmer_tfidf_features(self, sequences: Dict[str, str], 
                                   fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Create TF-IDF features from k-mers.
        
        Args:
            sequences: Dictionary of protein ID -> sequence
            fit: If True, fit the vectorizer; if False, use existing
            
        Returns:
            Tuple of (feature matrix, protein IDs)
        """
        protein_ids = list(sequences.keys())
        seq_list = [sequences[pid] for pid in protein_ids]
        kmer_seqs = [' '.join(self.get_kmer_features(seq, self.k)) for seq in seq_list]
        
        if fit or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=5000)
            features = self.vectorizer.fit_transform(kmer_seqs).toarray()
        else:
            features = self.vectorizer.transform(kmer_seqs).toarray()
        
        return features, protein_ids
    
    def create_composition_features(self, sequences: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """
        Create amino acid composition features.
        
        Returns:
            Tuple of (feature matrix, protein IDs)
        """
        protein_ids = list(sequences.keys())
        features = np.array([
            self.get_composition_features(sequences[pid]) 
            for pid in protein_ids
        ])
        
        return features, protein_ids
    
    def create_combined_features(self, sequences: Dict[str, str], 
                                fit_tfidf: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Create combined features from multiple feature types.
        
        Returns:
            Tuple of (combined feature matrix, protein IDs)
        """
        protein_ids = list(sequences.keys())
        
        # Get all feature types
        composition_feats, _ = self.create_composition_features(sequences)
        property_feats = np.array([
            self.get_property_features(sequences[pid]) 
            for pid in protein_ids
        ])
        length_feats = np.array([
            self.get_length_features(sequences[pid]) 
            for pid in protein_ids
        ])
        dipeptide_feats = np.array([
            self.get_dipeptide_features(sequences[pid]) 
            for pid in protein_ids
        ])
        
        # Combine all features
        combined_features = np.concatenate([
            composition_feats,
            property_feats,
            length_feats,
            dipeptide_feats
        ], axis=1)
        
        return combined_features, protein_ids


def main():
    """Test feature extraction."""
    from data_loader import CAFADataLoader
    
    # Load sample data
    data_dir = '/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/cafa-6-protein-function-prediction'
    loader = CAFADataLoader(data_dir)
    loader.load_train_data()
    
    # Sample a few sequences
    sample_ids = list(loader.train_sequences.keys())[:100]
    sample_seqs = {pid: loader.train_sequences[pid] for pid in sample_ids}
    
    extractor = ProteinFeatureExtractor(k=3)
    
    # Test different feature types
    print("Testing feature extraction...")
    
    comp_feats, ids = extractor.create_composition_features(sample_seqs)
    print(f"Composition features shape: {comp_feats.shape}")
    
    prop_feats = np.array([
        extractor.get_property_features(sample_seqs[pid]) 
        for pid in ids
    ])
    print(f"Property features shape: {prop_feats.shape}")
    
    dipep_feats = np.array([
        extractor.get_dipeptide_features(sample_seqs[pid]) 
        for pid in ids
    ])
    print(f"Dipeptide features shape: {dipep_feats.shape}")
    
    kmer_feats, ids = extractor.create_kmer_tfidf_features(sample_seqs, fit=True)
    print(f"K-mer TF-IDF features shape: {kmer_feats.shape}")
    
    combined_feats, ids = extractor.create_combined_features(sample_seqs, fit_tfidf=True)
    print(f"Combined features shape: {combined_feats.shape}")
    
    print("\nFeature extraction test completed successfully!")


if __name__ == '__main__':
    main()
