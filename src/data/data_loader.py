"""
Data loading and preprocessing for CAFA-6 protein function prediction.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, List
from collections import defaultdict
import pickle
from pathlib import Path


class CAFADataLoader:
    """Load and preprocess CAFA-6 protein function prediction data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to the data directory containing Train and Test folders
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'Train'
        self.test_dir = self.data_dir / 'Test'
        
        # Data storage
        self.train_sequences = {}
        self.train_terms = defaultdict(list)
        self.train_taxonomy = {}
        self.go_terms = set()
        self.test_sequences = {}
        self.ia_weights = {}
        
    def load_fasta(self, fasta_path: str) -> Dict[str, str]:
        """Load FASTA file and return dictionary of sequences."""
        sequences = {}
        current_id = None
        current_seq = []
        
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)
                    
                    # Parse header - extract protein ID (first part after >)
                    header = line[1:]  # Remove '>'
                    
                    # Handle different formats:
                    # Train: sp|ProteinID|GeneName...
                    # Test: ProteinID TaxonID
                    if '|' in header:
                        # Format: sp|ProteinID|GeneName...
                        parts = header.split('|')
                        current_id = parts[1] if len(parts) > 1 else parts[0]
                    else:
                        # Format: ProteinID TaxonID (test sequences)
                        # Extract just the protein ID before the space
                        current_id = header.split()[0]
                    
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # Save last sequence
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    def load_train_data(self):
        """Load all training data."""
        print("Loading training sequences...")
        fasta_path = self.train_dir / 'train_sequences.fasta'
        self.train_sequences = self.load_fasta(str(fasta_path))
        print(f"  Loaded {len(self.train_sequences)} sequences")
        
        print("Loading training terms...")
        terms_path = self.train_dir / 'train_terms.tsv'
        with open(terms_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    protein_id, go_term = parts[0], parts[1]
                    self.train_terms[protein_id].append(go_term)
                    self.go_terms.add(go_term)
        print(f"  Loaded {len(self.train_terms)} proteins with terms")
        print(f"  Total unique GO terms: {len(self.go_terms)}")
        
        print("Loading training taxonomy...")
        taxonomy_path = self.train_dir / 'train_taxonomy.tsv'
        with open(taxonomy_path, 'r') as f:
            next(f)  # Skip header if exists
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    protein_id, taxon_id = parts[0], parts[1]
                    self.train_taxonomy[protein_id] = taxon_id
        print(f"  Loaded taxonomy for {len(self.train_taxonomy)} proteins")
        
    def load_test_data(self):
        """Load test superset sequences."""
        print("Loading test sequences...")
        fasta_path = self.test_dir / 'testsuperset.fasta'
        self.test_sequences = self.load_fasta(str(fasta_path))
        print(f"  Loaded {len(self.test_sequences)} test sequences")
        
    def load_ia_weights(self):
        """Load information accretion weights."""
        print("Loading IA weights...")
        ia_path = self.data_dir / 'IA.tsv'
        with open(ia_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    go_term, weight = parts[0], float(parts[1])
                    self.ia_weights[go_term] = weight
        print(f"  Loaded weights for {len(self.ia_weights)} GO terms")
        
    def get_train_data_summary(self) -> Dict:
        """Get summary statistics of training data."""
        terms_per_protein = [len(terms) for terms in self.train_terms.values()]
        
        return {
            'n_proteins': len(self.train_sequences),
            'n_unique_terms': len(self.go_terms),
            'avg_terms_per_protein': np.mean(terms_per_protein),
            'min_terms': np.min(terms_per_protein),
            'max_terms': np.max(terms_per_protein),
            'n_test_proteins': len(self.test_sequences),
        }
    
    def split_train_test(self, test_size: float = 0.2, seed: int = 42) -> Tuple[
        Dict[str, str], Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]
    ]:
        """
        Split training data into train and validation sets.
        
        Args:
            test_size: Fraction of data to use for validation
            seed: Random seed
            
        Returns:
            Tuple of (train_seqs, val_seqs, train_terms, val_terms)
        """
        np.random.seed(seed)
        
        protein_ids = list(self.train_sequences.keys())
        np.random.shuffle(protein_ids)
        
        split_point = int(len(protein_ids) * (1 - test_size))
        
        train_ids = protein_ids[:split_point]
        val_ids = protein_ids[split_point:]
        
        train_seqs = {pid: self.train_sequences[pid] for pid in train_ids}
        val_seqs = {pid: self.train_sequences[pid] for pid in val_ids}
        
        train_terms_dict = {pid: self.train_terms[pid] for pid in train_ids}
        val_terms_dict = {pid: self.train_terms[pid] for pid in val_ids}
        
        return train_seqs, val_seqs, train_terms_dict, val_terms_dict
    
    def create_protein_to_terms_matrix(self, protein_ids: List[str], 
                                      all_terms: List[str] = None) -> np.ndarray:
        """
        Create a binary matrix of proteins x GO terms.
        
        Args:
            protein_ids: List of protein IDs
            all_terms: List of all GO terms (if None, uses self.go_terms)
            
        Returns:
            Binary matrix of shape (n_proteins, n_terms)
        """
        if all_terms is None:
            all_terms = sorted(self.go_terms)
        
        term_to_idx = {term: idx for idx, term in enumerate(all_terms)}
        
        matrix = np.zeros((len(protein_ids), len(all_terms)), dtype=np.int32)
        
        for i, protein_id in enumerate(protein_ids):
            if protein_id in self.train_terms:
                for term in self.train_terms[protein_id]:
                    if term in term_to_idx:
                        matrix[i, term_to_idx[term]] = 1
        
        return matrix, all_terms
    
    def save_processed_data(self, output_path: str):
        """Save processed data to pickle file."""
        data = {
            'train_sequences': self.train_sequences,
            'train_terms': dict(self.train_terms),
            'train_taxonomy': self.train_taxonomy,
            'go_terms': sorted(list(self.go_terms)),
            'test_sequences': self.test_sequences,
            'ia_weights': self.ia_weights,
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, input_path: str):
        """Load processed data from pickle file."""
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        self.train_sequences = data['train_sequences']
        self.train_terms = defaultdict(list, data['train_terms'])
        self.train_taxonomy = data['train_taxonomy']
        self.go_terms = set(data['go_terms'])
        self.test_sequences = data['test_sequences']
        self.ia_weights = data['ia_weights']
        print(f"Loaded processed data from {input_path}")


def main():
    """Test data loading."""
    data_dir = '/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/cafa-6-protein-function-prediction'
    
    loader = CAFADataLoader(data_dir)
    loader.load_train_data()
    loader.load_test_data()
    loader.load_ia_weights()
    
    summary = loader.get_train_data_summary()
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save processed data
    loader.save_processed_data('/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/processed_data.pkl')


if __name__ == '__main__':
    main()
