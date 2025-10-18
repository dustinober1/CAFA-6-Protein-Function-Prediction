"""
Advanced feature engineering techniques for CAFA-6 protein function prediction.
Implements cutting-edge feature extraction including protein language models,
graph-based features, and multi-scale representations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Some features will be disabled.")

try:
    from Bio import SeqIO
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("BioPython not available. Some features will be disabled.")


class ProteinLanguageModelFeatures:
    """Extract features from pre-trained protein language models."""
    
    def __init__(self, model_name: str = 'Rostlab/prot_bert', device: str = 'cpu'):
        """
        Initialize PLM feature extractor.
        
        Args:
            model_name: Name of pre-trained model
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load pre-trained model and tokenizer."""
        try:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.loaded = False
    
    def extract_embeddings(self, sequences: Dict[str, str], 
                          batch_size: int = 8, max_length: int = 512) -> Dict[str, np.ndarray]:
        """
        Extract protein embeddings from language model.
        
        Args:
            sequences: Dict of protein_id -> sequence
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Dict of protein_id -> embedding vector
        """
        if not self.loaded:
            print("Model not loaded, returning zeros")
            return {pid: np.zeros(1024) for pid in sequences.keys()}
        
        embeddings = {}
        protein_ids = list(sequences.keys())
        
        for i in range(0, len(protein_ids), batch_size):
            batch_ids = protein_ids[i:i+batch_size]
            batch_seqs = [sequences[pid] for pid in batch_ids]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use [CLS] token or mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output.cpu().numpy()
                else:
                    # Mean pooling over sequence length
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    
                    # Mask padding tokens
                    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
                    sum_embeddings = torch.sum(masked_embeddings, dim=1)
                    sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
                    batch_embeddings = sum_embeddings / sum_mask
                    batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Store embeddings
            for pid, embedding in zip(batch_ids, batch_embeddings):
                embeddings[pid] = embedding
        
        return embeddings
    
    def extract_attention_patterns(self, sequences: Dict[str, str],
                                 layers: List[int] = None) -> Dict[str, np.ndarray]:
        """
        Extract attention patterns for interpretable features.
        
        Args:
            sequences: Dict of protein_id -> sequence
            layers: List of layers to extract attention from
            
        Returns:
            Dict of protein_id -> attention features
        """
        if not self.loaded:
            return {}
        
        if layers is None:
            layers = [-1]  # Last layer
        
        attention_features = {}
        
        for pid, seq in sequences.items():
            # Tokenize single sequence
            inputs = self.tokenizer(
                seq,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                
                # Extract attention from specified layers
                layer_attentions = []
                for layer_idx in layers:
                    attention = outputs.attentions[layer_idx]  # (1, num_heads, seq_len, seq_len)
                    # Average over heads
                    avg_attention = attention.mean(dim=1).squeeze(0).cpu().numpy()
                    layer_attentions.append(avg_attention)
                
                # Concatenate layer attentions
                attention_matrix = np.concatenate(layer_attentions, axis=0)
                
                # Extract summary statistics
                attention_stats = [
                    np.mean(attention_matrix),
                    np.std(attention_matrix),
                    np.max(attention_matrix),
                    # Diagonal attention (self-attention)
                    np.mean(np.diag(attention_matrix)),
                    # Off-diagonal attention (cross-attention)
                    np.mean(attention_matrix[~np.eye(attention_matrix.shape[0], dtype=bool)])
                ]
                
                attention_features[pid] = np.array(attention_stats)
        
        return attention_features


class GraphBasedFeatures:
    """Extract graph-based features from protein contact maps and interaction networks."""
    
    def __init__(self):
        """Initialize graph feature extractor."""
        self.contact_map_cache = {}
    
    def predict_contact_map(self, sequence: str, window_size: int = 5) -> np.ndarray:
        """
        Predict contact map using simple co-evolutionary features.
        
        Args:
            sequence: Amino acid sequence
            window_size: Window size for local contacts
            
        Returns:
            Predicted contact map
        """
        seq_len = len(sequence)
        contact_map = np.zeros((seq_len, seq_len))
        
        # Simple contact prediction based on sequence distance and properties
        for i in range(seq_len):
            for j in range(i + window_size, min(i + 50, seq_len)):  # Limit range
                # Contact probability based on amino acid properties
                aa1, aa2 = sequence[i], sequence[j]
                
                # Hydrophobic-hydrophobic contacts
                if aa1 in 'AVLIMFWY' and aa2 in 'AVLIMFWY':
                    contact_prob = 0.3
                # Charged-charged contacts
                elif (aa1 in 'DEKR' and aa2 in 'DEKR'):
                    contact_prob = 0.2
                # Polar-polar contacts
                elif (aa1 in 'STNQ' and aa2 in 'STNQ'):
                    contact_prob = 0.15
                else:
                    contact_prob = 0.1
                
                # Distance decay
                distance_factor = np.exp(-(j - i) / 20.0)
                contact_map[i, j] = contact_prob * distance_factor
                contact_map[j, i] = contact_map[i, j]
        
        return contact_map
    
    def extract_graph_features(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Extract graph-based features from predicted contact maps.
        
        Args:
            sequences: Dict of protein_id -> sequence
            
        Returns:
            Dict of protein_id -> graph features
        """
        graph_features = {}
        
        for pid, seq in sequences.items():
            contact_map = self.predict_contact_map(seq)
            
            # Graph statistics
            features = []
            
            # Basic graph properties
            features.extend([
                np.mean(contact_map),  # Average contact strength
                np.std(contact_map),   # Contact variability
                np.max(contact_map),   # Strongest contact
                # Contact density
                np.sum(contact_map > 0.1) / (contact_map.shape[0] ** 2),
            ])
            
            # Node degree statistics
            node_degrees = np.sum(contact_map > 0.1, axis=1)
            features.extend([
                np.mean(node_degrees),  # Average degree
                np.std(node_degrees),   # Degree variability
                np.max(node_degrees),   # Hub nodes
            ])
            
            # Clustering coefficient (simplified)
            clustering_scores = []
            for i in range(min(len(seq), 100)):  # Limit computation
                neighbors = np.where(contact_map[i] > 0.1)[0]
                if len(neighbors) > 1:
                    # Calculate local clustering
                    subgraph = contact_map[np.ix_(neighbors, neighbors)]
                    clustering = np.sum(subgraph) / (len(neighbors) * (len(neighbors) - 1))
                    clustering_scores.append(clustering)
            
            if clustering_scores:
                features.extend([
                    np.mean(clustering_scores),
                    np.std(clustering_scores),
                ])
            else:
                features.extend([0, 0])
            
            graph_features[pid] = np.array(features)
        
        return graph_features


class MultiScaleSequenceFeatures:
    """Extract multi-scale sequence features at different resolution levels."""
    
    def __init__(self):
        """Initialize multi-scale feature extractor."""
        self.amino_acid_groups = {
            'hydrophobic': 'AVLIMFWY',
            'polar': 'STNQ',
            'charged': 'DEKR',
            'special': 'CGP'
        }
    
    def extract_multiscale_features(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Extract features at multiple sequence scales.
        
        Args:
            sequences: Dict of protein_id -> sequence
            
        Returns:
            Dict of protein_id -> multi-scale features
        """
        features = {}
        
        for pid, seq in sequences.items():
            all_features = []
            
            # Scale 1: Local (window-based) features
            local_features = self._extract_local_features(seq)
            all_features.extend(local_features)
            
            # Scale 2: Regional (domain-like) features
            regional_features = self._extract_regional_features(seq)
            all_features.extend(regional_features)
            
            # Scale 3: Global (whole protein) features
            global_features = self._extract_global_features(seq)
            all_features.extend(global_features)
            
            features[pid] = np.array(all_features)
        
        return features
    
    def _extract_local_features(self, sequence: str, window_sizes: List[int] = [3, 5, 7]) -> List[float]:
        """Extract local window-based features."""
        features = []
        seq_len = len(sequence)
        
        for window_size in window_sizes:
            if seq_len < window_size:
                continue
            
            # Sliding window statistics
            hydro_scores = []
            charge_scores = []
            
            for i in range(seq_len - window_size + 1):
                window = sequence[i:i+window_size]
                
                # Hydrophobicity
                hydro_score = sum(self.amino_acid_groups['hydrophobic'].count(aa) for aa in window) / window_size
                hydro_scores.append(hydro_score)
                
                # Charge
                charge_score = sum(self.amino_acid_groups['charged'].count(aa) for aa in window) / window_size
                charge_scores.append(charge_score)
            
            if hydro_scores:
                features.extend([
                    np.mean(hydro_scores),
                    np.std(hydro_scores),
                    np.max(hydro_scores),
                    np.min(hydro_scores),
                ])
                features.extend([
                    np.mean(charge_scores),
                    np.std(charge_scores),
                    np.max(charge_scores),
                    np.min(charge_scores),
                ])
        
        return features
    
    def _extract_regional_features(self, sequence: str, region_size: int = 20) -> List[float]:
        """Extract regional (domain-like) features."""
        features = []
        seq_len = len(sequence)
        
        if seq_len < region_size:
            return features
        
        # Split into regions
        n_regions = seq_len // region_size
        
        for i in range(n_regions):
            start = i * region_size
            end = min((i + 1) * region_size, seq_len)
            region = sequence[start:end]
            
            # Composition features
            for group_name, group_aas in self.amino_acid_groups.items():
                composition = sum(region.count(aa) for aa in group_aas) / len(region)
                features.append(composition)
            
            # Complexity features
            unique_aas = len(set(region))
            features.append(unique_aas / len(region))  # Diversity
            
            # Pattern features
            repeats = sum(1 for i in range(len(region)-1) if region[i] == region[i+1])
            features.append(repeats / len(region))
        
        return features
    
    def _extract_global_features(self, sequence: str) -> List[float]:
        """Extract global protein-level features."""
        features = []
        
        # Length-related features
        features.extend([
            len(sequence),
            np.log1p(len(sequence)),  # Log length
        ])
        
        # Global composition
        for group_name, group_aas in self.amino_acid_groups.items():
            composition = sum(sequence.count(aa) for aa in group_aas) / len(sequence)
            features.append(composition)
        
        # Sequence complexity
        aa_counts = Counter(sequence)
        entropy = -sum((count/len(sequence)) * np.log2(count/len(sequence)) 
                      for count in aa_counts.values())
        features.append(entropy)
        
        # Terminal features (N and C termini)
        if len(sequence) >= 10:
            n_term = sequence[:10]
            c_term = sequence[-10:]
            
            # Terminal composition
            n_hydro = sum(self.amino_acid_groups['hydrophobic'].count(aa) for aa in n_term) / 10
            c_hydro = sum(self.amino_acid_groups['hydrophobic'].count(aa) for aa in c_term) / 10
            features.extend([n_hydro, c_hydro])
        else:
            features.extend([0, 0])
        
        return features


class EvolutionaryFeatures:
    """Extract evolutionary features from multiple sequence alignments."""
    
    def __init__(self):
        """Initialize evolutionary feature extractor."""
        self.pssm_cache = {}
    
    def simulate_pssm(self, sequence: str) -> np.ndarray:
        """
        Simulate PSSM (Position-Specific Scoring Matrix) using simple heuristics.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Simulated PSSM matrix
        """
        aa_order = 'ACDEFGHIKLMNPQRSTVWY'
        seq_len = len(sequence)
        pssm = np.zeros((seq_len, 20))
        
        # Simple conservation scoring based on position and amino acid properties
        for i, aa in enumerate(sequence):
            aa_idx = aa_order.find(aa)
            if aa_idx == -1:
                continue
            
            # Base conservation score
            base_score = 1.0
            
            # Adjust based on position (N/C termini less conserved)
            if i < len(sequence) * 0.1 or i > len(sequence) * 0.9:
                position_factor = 0.7
            else:
                position_factor = 1.0
            
            # Adjust based on amino acid type
            if aa in 'C':  # Cysteines often conserved
                aa_factor = 1.3
            elif aa in 'P':  # Prolines often conserved
                aa_factor = 1.2
            elif aa in 'GAVL':  # Small hydrophobic
                aa_factor = 0.9
            else:
                aa_factor = 1.0
            
            final_score = base_score * position_factor * aa_factor
            
            # Set PSSM values
            pssm[i, aa_idx] = final_score
            
            # Add some noise to other positions
            for j in range(20):
                if j != aa_idx:
                    pssm[i, j] = np.random.exponential(0.1)
        
        return pssm
    
    def extract_evolutionary_features(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Extract evolutionary features from simulated PSSM.
        
        Args:
            sequences: Dict of protein_id -> sequence
            
        Returns:
            Dict of protein_id -> evolutionary features
        """
        features = {}
        
        for pid, seq in sequences.items():
            pssm = self.simulate_pssm(seq)
            
            # PSSM statistics
            pssm_features = []
            
            # Conservation scores
            conservation = np.max(pssm, axis=1)
            pssm_features.extend([
                np.mean(conservation),  # Average conservation
                np.std(conservation),   # Conservation variability
                np.max(conservation),   # Most conserved position
                np.min(conservation),   # Least conserved position
            ])
            
            # Position-specific patterns
            # N-terminal conservation
            n_term_len = min(10, len(conservation))
            n_term_conservation = np.mean(conservation[:n_term_len])
            pssm_features.append(n_term_conservation)
            
            # C-terminal conservation
            c_term_len = min(10, len(conservation))
            c_term_conservation = np.mean(conservation[-c_term_len:])
            pssm_features.append(c_term_conservation)
            
            # Central conservation
            if len(conservation) > 20:
                central_start = len(conservation) // 2 - 10
                central_end = len(conservation) // 2 + 10
                central_conservation = np.mean(conservation[central_start:central_end])
                pssm_features.append(central_conservation)
            else:
                pssm_features.append(np.mean(conservation))
            
            # Amino acid variability
            aa_variability = np.std(pssm, axis=0)
            pssm_features.extend([
                np.mean(aa_variability),
                np.std(aa_variability),
                np.max(aa_variability),
            ])
            
            features[pid] = np.array(pssm_features)
        
        return features


class AdvancedFeatureFusion:
    """Fuse multiple advanced feature types into unified representation."""
    
    def __init__(self, feature_types: List[str] = None):
        """
        Initialize feature fusion.
        
        Args:
            feature_types: List of feature types to extract
        """
        if feature_types is None:
            feature_types = ['multiscale', 'graph', 'evolutionary']
        
        self.feature_types = feature_types
        self.extractors = {}
        
        # Initialize extractors
        if 'multiscale' in feature_types:
            self.extractors['multiscale'] = MultiScaleSequenceFeatures()
        
        if 'graph' in feature_types:
            self.extractors['graph'] = GraphBasedFeatures()
        
        if 'evolutionary' in feature_types:
            self.extractors['evolutionary'] = EvolutionaryFeatures()
        
        if 'plm' in feature_types and TRANSFORMERS_AVAILABLE:
            self.extractors['plm'] = ProteinLanguageModelFeatures()
    
    def extract_all_features(self, sequences: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all specified feature types.
        
        Args:
            sequences: Dict of protein_id -> sequence
            
        Returns:
            (feature_matrix, feature_names)
        """
        all_features = {}
        feature_names = []
        
        for feature_type in self.feature_types:
            if feature_type in self.extractors:
                print(f"Extracting {feature_type} features...")
                
                if feature_type == 'plm':
                    # PLM features are high-dimensional
                    features = self.extractors[feature_type].extract_embeddings(sequences)
                    for pid, feat in features.items():
                        if pid not in all_features:
                            all_features[pid] = []
                        all_features[pid].extend(feat)
                    
                    # Add feature names for PLM
                    if features:
                        dim = len(list(features.values())[0])
                        feature_names.extend([f'plm_{i}' for i in range(dim)])
                
                else:
                    features = self.extractors[feature_type].extract_features(sequences)
                    for pid, feat in features.items():
                        if pid not in all_features:
                            all_features[pid] = []
                        all_features[pid].extend(feat)
                    
                    # Add feature names
                    if features:
                        dim = len(list(features.values())[0])
                        feature_names.extend([f'{feature_type}_{i}' for i in range(dim)])
        
        # Convert to matrix
        protein_ids = sorted(all_features.keys())
        feature_matrix = np.array([all_features[pid] for pid in protein_ids])
        
        return feature_matrix, protein_ids, feature_names
    
    def extract_with_metadata(self, sequences: Dict[str, str], 
                            metadata: Dict[str, Dict] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract features with optional metadata integration.
        
        Args:
            sequences: Dict of protein_id -> sequence
            metadata: Optional metadata dict
            
        Returns:
            (feature_matrix, protein_ids, feature_names)
        """
        feature_matrix, protein_ids, feature_names = self.extract_all_features(sequences)
        
        # Add metadata features if provided
        if metadata:
            metadata_features = []
            metadata_names = []
            
            for pid in protein_ids:
                if pid in metadata:
                    meta_feat = []
                    for key, value in metadata[pid].items():
                        if isinstance(value, (int, float)):
                            meta_feat.append(value)
                        elif isinstance(value, str):
                            # Simple encoding for categorical variables
                            meta_feat.append(hash(value) % 1000)
                    
                    metadata_features.append(meta_feat)
                else:
                    metadata_features.append([0] * len(metadata_names))
            
            if metadata_features:
                # Add metadata feature names
                if protein_ids and metadata_features[0]:
                    for i in range(len(metadata_features[0])):
                        metadata_names.append(f'meta_{i}')
                    
                    # Concatenate features
                    metadata_matrix = np.array(metadata_features)
                    feature_matrix = np.concatenate([feature_matrix, metadata_matrix], axis=1)
                    feature_names.extend(metadata_names)
        
        return feature_matrix, protein_ids, feature_names


def main():
    """Test advanced feature extraction methods."""
    print("Advanced feature extraction methods ready!")
    print("Available classes:")
    print("- ProteinLanguageModelFeatures")
    print("- GraphBasedFeatures")
    print("- MultiScaleSequenceFeatures")
    print("- EvolutionaryFeatures")
    print("- AdvancedFeatureFusion")
    
    # Test with sample data
    sample_sequences = {
        'protein_1': 'MKVLWAALLVTFLAGCAKAKTEK',
        'protein_2': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR'
    }
    
    # Test multi-scale features
    multiscale = MultiScaleSequenceFeatures()
    ms_features = multiscale.extract_multiscale_features(sample_sequences)
    print(f"Multi-scale features extracted: {list(ms_features.keys())}")
    
    # Test graph features
    graph = GraphBasedFeatures()
    graph_features = graph.extract_graph_features(sample_sequences)
    print(f"Graph features extracted: {list(graph_features.keys())}")
    
    # Test evolutionary features
    evo = EvolutionaryFeatures()
    evo_features = evo.extract_evolutionary_features(sample_sequences)
    print(f"Evolutionary features extracted: {list(evo_features.keys())}")


if __name__ == '__main__':
    main()
