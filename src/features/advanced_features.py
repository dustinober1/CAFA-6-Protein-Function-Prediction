"""
Advanced feature engineering for protein function prediction.
Includes ESM embeddings, physicochemical properties, and taxonomy-aware features.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class PhysicochemicalFeatures:
    """Extract physicochemical properties from amino acid sequences."""
    
    # Amino acid properties
    HYDROPHOBICITY = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    }
    
    CHARGE = {
        'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
        'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
        'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
    }
    
    MOLECULAR_WEIGHT = {
        'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165,
        'G': 75, 'H': 155, 'I': 131, 'K': 146, 'L': 131,
        'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174,
        'S': 105, 'T': 119, 'V': 117, 'W': 204, 'Y': 181,
    }
    
    @staticmethod
    def get_features(sequence: str) -> Dict[str, float]:
        """
        Extract physicochemical features from a sequence.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        seq_len = len(sequence)
        
        if seq_len == 0:
            return {k: 0 for k in ['hydro_mean', 'hydro_std', 'hydro_max', 'hydro_min',
                                     'charge_mean', 'charge_std', 'mw_mean', 'mw_std',
                                     'aromatic_ratio', 'charged_ratio', 'polar_ratio']}
        
        # Hydrophobicity
        hydro_values = [PhysicochemicalFeatures.HYDROPHOBICITY.get(aa, 0) for aa in sequence]
        features['hydro_mean'] = np.mean(hydro_values)
        features['hydro_std'] = np.std(hydro_values)
        features['hydro_max'] = np.max(hydro_values) if hydro_values else 0
        features['hydro_min'] = np.min(hydro_values) if hydro_values else 0
        
        # Charge
        charge_values = [PhysicochemicalFeatures.CHARGE.get(aa, 0) for aa in sequence]
        features['charge_mean'] = np.mean(charge_values)
        features['charge_std'] = np.std(charge_values)
        
        # Molecular weight
        mw_values = [PhysicochemicalFeatures.MOLECULAR_WEIGHT.get(aa, 0) for aa in sequence]
        features['mw_mean'] = np.mean(mw_values)
        features['mw_std'] = np.std(mw_values)
        
        # Composition ratios
        aromatic = sequence.count('F') + sequence.count('W') + sequence.count('Y')
        features['aromatic_ratio'] = aromatic / seq_len
        
        charged = sum(1 for aa in sequence if aa in 'DER')
        features['charged_ratio'] = charged / seq_len
        
        polar = sum(1 for aa in sequence if aa in 'STNQC')
        features['polar_ratio'] = polar / seq_len
        
        return features


class AdvancedSequenceFeatures:
    """Extract advanced sequence-based features."""
    
    @staticmethod
    def get_kmer_features(sequence: str, k: int = 3) -> Dict[str, float]:
        """Extract k-mer frequency features."""
        features = {}
        seq_len = len(sequence)
        
        if seq_len < k:
            return {}
        
        kmers = defaultdict(int)
        for i in range(seq_len - k + 1):
            kmer = sequence[i:i+k]
            kmers[kmer] += 1
        
        # Normalize
        total = sum(kmers.values())
        for kmer, count in kmers.items():
            features[f'kmer_{kmer}'] = count / total if total > 0 else 0
        
        return features
    
    @staticmethod
    def get_secondary_structure_propensity(sequence: str) -> Dict[str, float]:
        """
        Estimate secondary structure propensities.
        """
        # Helix, sheet, coil propensities (Chou-Fasman scale)
        helix_prop = {'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.7, 'Q': 1.11,
                      'E': 1.51, 'G': 0.57, 'H': 1.0, 'I': 1.08, 'L': 1.21, 'K': 1.16,
                      'M': 1.45, 'F': 1.13, 'P': 0.57, 'S': 0.77, 'T': 0.83, 'W': 1.08,
                      'Y': 0.69, 'V': 1.06}
        
        sheet_prop = {'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19, 'Q': 1.1,
                      'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.6, 'L': 1.3, 'K': 0.74,
                      'M': 1.05, 'F': 1.38, 'P': 0.55, 'S': 0.75, 'T': 1.19, 'W': 1.37,
                      'Y': 1.47, 'V': 1.06}
        
        coil_prop = {'A': 0.66, 'R': 0.98, 'N': 1.56, 'D': 1.46, 'C': 1.02, 'Q': 0.98,
                     'E': 1.51, 'G': 1.57, 'H': 1.0, 'I': 0.47, 'L': 0.57, 'K': 1.16,
                     'M': 0.45, 'F': 0.6, 'P': 1.52, 'S': 0.75, 'T': 0.83, 'W': 0.37,
                     'Y': 0.69, 'V': 0.5}
        
        features = {}
        helix_avg = np.mean([helix_prop.get(aa, 1.0) for aa in sequence])
        sheet_avg = np.mean([sheet_prop.get(aa, 1.0) for aa in sequence])
        coil_avg = np.mean([coil_prop.get(aa, 1.0) for aa in sequence])
        
        features['helix_propensity'] = helix_avg
        features['sheet_propensity'] = sheet_avg
        features['coil_propensity'] = coil_avg
        
        return features
    
    @staticmethod
    def get_seq_complexity(sequence: str) -> Dict[str, float]:
        """Measure sequence complexity using entropy and composition."""
        from collections import Counter
        
        features = {}
        
        # Shannon entropy
        aa_counts = Counter(sequence)
        total = len(sequence)
        entropy = -sum((count / total) * np.log2(count / total + 1e-10) 
                       for count in aa_counts.values())
        features['seq_entropy'] = entropy
        
        # Amino acid diversity
        features['aa_diversity'] = len(aa_counts) / 20.0
        
        # Sequence length features
        features['log_seq_length'] = np.log1p(len(sequence))
        
        return features


class TaxonomyFeatures:
    """Taxonomy-aware features for proteins."""
    
    @staticmethod
    def get_taxon_features(taxon_id: str, taxonomy_data: Dict[str, any] = None) -> Dict[str, float]:
        """
        Extract taxonomy-based features.
        
        Args:
            taxon_id: NCBI taxonomy ID
            taxonomy_data: Optional dict with taxonomy hierarchy info
            
        Returns:
            Dictionary of taxonomy features
        """
        features = {}
        
        # Common taxon categories (this is a simplified encoding)
        major_kingdoms = {
            '9606': 1,  # Human
            '10090': 2,  # Mouse
            '6239': 3,  # C. elegans
            '7227': 4,  # Drosophila
            '4932': 5,  # Yeast
            '3702': 6,  # Arabidopsis
        }
        
        bacteria_ids = set([str(i) for i in range(511145, 511150)])  # Example range
        
        taxon_code = major_kingdoms.get(taxon_id, 0)
        features['taxon_kingdom'] = taxon_code
        features['is_bacteria'] = float(taxon_id in bacteria_ids)
        features['is_human'] = float(taxon_id == '9606')
        features['is_model_organism'] = float(taxon_id in major_kingdoms)
        
        return features


class AdvancedFeatureExtractor:
    """Combine all advanced features for protein function prediction."""
    
    def __init__(self, taxonomy_dict: Dict[str, str] = None):
        """
        Initialize extractor.
        
        Args:
            taxonomy_dict: Dict of protein_id -> taxon_id
        """
        self.taxonomy_dict = taxonomy_dict or {}
        self.physichem = PhysicochemicalFeatures()
        self.seq_features = AdvancedSequenceFeatures()
        self.taxon_features = TaxonomyFeatures()
    
    def extract_all_features(self, protein_id: str, sequence: str) -> np.ndarray:
        """
        Extract all advanced features for a protein.
        
        Args:
            protein_id: Protein ID
            sequence: Amino acid sequence
            
        Returns:
            Feature vector as numpy array
        """
        all_features = {}
        
        # Physicochemical
        pc_feats = self.physichem.get_features(sequence)
        all_features.update(pc_feats)
        
        # Sequence-based
        sstruct_feats = self.seq_features.get_secondary_structure_propensity(sequence)
        all_features.update(sstruct_feats)
        
        complexity_feats = self.seq_features.get_seq_complexity(sequence)
        all_features.update(complexity_feats)
        
        # Taxonomy
        taxon_id = self.taxonomy_dict.get(protein_id, '0')
        taxon_feats = self.taxon_features.get_taxon_features(taxon_id)
        all_features.update(taxon_feats)
        
        # Create feature vector in sorted order
        feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features[name] for name in feature_names])
        
        return feature_vector, feature_names
    
    def extract_batch(self, protein_seqs: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for a batch of proteins.
        
        Args:
            protein_seqs: Dict of protein_id -> sequence
            
        Returns:
            Feature matrix and protein IDs
        """
        features_list = []
        protein_ids = []
        feature_names = None
        
        for protein_id, sequence in protein_seqs.items():
            fv, fn = self.extract_all_features(protein_id, sequence)
            if feature_names is None:
                feature_names = fn
            features_list.append(fv)
            protein_ids.append(protein_id)
        
        feature_matrix = np.array(features_list)
        return feature_matrix, protein_ids, feature_names


def main():
    """Test advanced features."""
    seq = "MKVLWAALLVTFLAGCAKAKTEK"
    
    # Test physicochemical
    pc = PhysicochemicalFeatures.get_features(seq)
    print("Physicochemical features:", pc)
    
    # Test sequence features
    ss = AdvancedSequenceFeatures.get_secondary_structure_propensity(seq)
    print("Secondary structure:", ss)
    
    complexity = AdvancedSequenceFeatures.get_seq_complexity(seq)
    print("Sequence complexity:", complexity)
    
    # Test taxonomy
    tax = TaxonomyFeatures.get_taxon_features('9606')
    print("Taxonomy features:", tax)
    
    # Test full extractor
    extractor = AdvancedFeatureExtractor({'test_protein': '9606'})
    fv, fn = extractor.extract_all_features('test_protein', seq)
    print(f"\nFull feature vector shape: {fv.shape}")
    print(f"Feature names: {fn}")


if __name__ == '__main__':
    main()
