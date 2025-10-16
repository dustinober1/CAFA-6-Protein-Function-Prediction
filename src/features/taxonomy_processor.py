"""
Taxonomic information processing and feature extraction.

Incorporates taxonomic lineage information to improve predictions by
leveraging evolutionary relationships between proteins.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TaxonomyProcessor:
    """Extract and process taxonomic information."""
    
    def __init__(self, taxonomy_path: str):
        """
        Initialize taxonomy processor.
        
        Args:
            taxonomy_path: Path to taxonomy TSV file (taxon_id, parent_id, name, rank)
        """
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = {}  # taxon_id -> info
        self.children = defaultdict(list)  # parent_id -> list of child_ids
        self.parents = {}  # taxon_id -> parent_id
        self.names = {}  # taxon_id -> name
        self.ranks = {}  # taxon_id -> rank
        
        self._load_taxonomy()
    
    def _load_taxonomy(self):
        """Load taxonomy from TSV file."""
        logger.info(f"Loading taxonomy from {self.taxonomy_path}...")
        
        try:
            df = pd.read_csv(self.taxonomy_path, sep='\t')
            
            # Expected columns: taxon_id, parent_id, name, rank
            if 'taxon_id' in df.columns:
                for _, row in df.iterrows():
                    taxon_id = int(row['taxon_id'])
                    parent_id = int(row.get('parent_id', 0))
                    name = row.get('name', '')
                    rank = row.get('rank', 'no rank')
                    
                    self.taxonomy[taxon_id] = {
                        'parent_id': parent_id,
                        'name': name,
                        'rank': rank,
                    }
                    self.names[taxon_id] = name
                    self.ranks[taxon_id] = rank
                    
                    if parent_id != 0:
                        self.parents[taxon_id] = parent_id
                        self.children[parent_id].append(taxon_id)
            
            logger.info(f"Loaded {len(self.taxonomy)} taxa")
        
        except Exception as e:
            logger.error(f"Error loading taxonomy: {e}")
    
    def get_lineage(self, taxon_id: int, include_self: bool = True) -> List[int]:
        """Get complete lineage from taxon to root."""
        lineage = []
        current = taxon_id
        
        if include_self:
            lineage.append(current)
        
        # Traverse up to root
        while current in self.parents:
            parent = self.parents[current]
            lineage.append(parent)
            current = parent
        
        return lineage
    
    def get_lineage_names(self, taxon_id: int) -> List[str]:
        """Get lineage with names."""
        lineage = self.get_lineage(taxon_id)
        return [self.names.get(tid, f"taxon_{tid}") for tid in lineage]
    
    def get_lineage_by_rank(self, taxon_id: int) -> Dict[str, int]:
        """Get lineage organized by rank."""
        lineage = self.get_lineage(taxon_id)
        by_rank = {}
        
        for tid in lineage:
            rank = self.ranks.get(tid, 'no rank')
            if rank not in by_rank:  # Keep the most specific (first) occurrence
                by_rank[rank] = tid
        
        return by_rank
    
    def get_siblings(self, taxon_id: int) -> List[int]:
        """Get sibling taxa."""
        if taxon_id not in self.parents:
            return []
        
        parent = self.parents[taxon_id]
        siblings = [t for t in self.children[parent] if t != taxon_id]
        return siblings
    
    def are_in_same_order(self, taxon1: int, taxon2: int) -> bool:
        """Check if two taxa are in the same taxonomic order."""
        lineage1_by_rank = self.get_lineage_by_rank(taxon1)
        lineage2_by_rank = self.get_lineage_by_rank(taxon2)
        
        if 'order' not in lineage1_by_rank or 'order' not in lineage2_by_rank:
            return False
        
        return lineage1_by_rank['order'] == lineage2_by_rank['order']
    
    def get_distance(self, taxon1: int, taxon2: int) -> int:
        """Get taxonomic distance between two taxa (levels to LCA)."""
        lineage1 = set(self.get_lineage(taxon1))
        lineage2 = set(self.get_lineage(taxon2))
        
        # Find lowest common ancestor
        common = lineage1 & lineage2
        if not common:
            return float('inf')
        
        # Get lineages as lists to find LCA
        lin1 = self.get_lineage(taxon1)
        lin2 = self.get_lineage(taxon2)
        
        # Find deepest common ancestor
        lca = None
        for t in lin1:
            if t in lin2:
                lca = t
                break
        
        if lca is None:
            return float('inf')
        
        return lin1.index(lca) + lin2.index(lca)


class TaxonomyFeatureExtractor:
    """Extract features from taxonomic information."""
    
    def __init__(self, taxonomy: TaxonomyProcessor, taxon_embeddings: Optional[Dict[int, np.ndarray]] = None):
        """
        Initialize taxonomy feature extractor.
        
        Args:
            taxonomy: TaxonomyProcessor instance
            taxon_embeddings: Optional pre-computed embeddings for taxa
        """
        self.taxonomy = taxonomy
        self.taxon_embeddings = taxon_embeddings or {}
    
    def extract_lineage_features(self, taxon_id: int, max_depth: int = 8) -> np.ndarray:
        """
        Extract lineage as one-hot encoded features.
        
        Args:
            taxon_id: Taxonomy ID
            max_depth: Maximum depth to encode
            
        Returns:
            One-hot encoded lineage features (max_depth * n_unique_ranks,)
        """
        lineage = self.taxonomy.get_lineage(taxon_id)[:max_depth]
        
        # Create feature vector with taxon IDs in fixed positions
        features = np.zeros(max_depth)
        for i, taxon in enumerate(lineage):
            # Normalize taxon ID to 0-1 range
            features[i] = min(taxon / 1e6, 1.0)
        
        return features
    
    def extract_rank_features(self, taxon_id: int) -> Dict[str, bool]:
        """
        Extract binary features for each major rank.
        
        Returns:
            Dict of rank -> is_in_lineage
        """
        lineage_by_rank = self.taxonomy.get_lineage_by_rank(taxon_id)
        
        major_ranks = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 
                       'family', 'genus', 'species']
        
        rank_features = {}
        for rank in major_ranks:
            rank_features[rank] = rank in lineage_by_rank
        
        return rank_features
    
    def extract_embedding_features(self, taxon_id: int) -> Optional[np.ndarray]:
        """Extract pre-computed embedding for taxon."""
        return self.taxon_embeddings.get(taxon_id)
    
    def extract_distance_features(self, taxon1: int, taxon2: int) -> Dict[str, float]:
        """
        Extract distance-based features between two taxa.
        
        Returns:
            Dict of distance features
        """
        distance = self.taxonomy.get_distance(taxon1, taxon2)
        same_order = self.taxonomy.are_in_same_order(taxon1, taxon2)
        
        return {
            'distance': float(min(distance, 100)),  # Cap at 100
            'distance_normalized': float(min(distance / 100, 1.0)),
            'same_order': float(same_order),
        }


class ProteinTaxonomyAssociator:
    """Associate proteins with taxonomic information."""
    
    def __init__(self, taxonomy_path: str):
        """
        Initialize protein-taxonomy associator.
        
        Args:
            taxonomy_path: Path to taxonomy file
        """
        self.taxonomy = TaxonomyProcessor(taxonomy_path)
        self.protein_taxa = {}  # protein_id -> taxon_id
        self.taxon_proteins = defaultdict(list)  # taxon_id -> list of protein_ids
    
    def load_associations(self, protein_taxonomy_path: str):
        """
        Load associations between proteins and taxa.
        
        Args:
            protein_taxonomy_path: Path to file mapping proteins to taxa
        """
        logger.info(f"Loading protein-taxonomy associations from {protein_taxonomy_path}...")
        
        try:
            df = pd.read_csv(protein_taxonomy_path, sep='\t', header=None)
            
            if len(df.columns) >= 2:
                for _, row in df.iterrows():
                    protein_id = str(row[0])
                    taxon_id = int(row[1])
                    
                    self.protein_taxa[protein_id] = taxon_id
                    self.taxon_proteins[taxon_id].append(protein_id)
            
            logger.info(f"Associated {len(self.protein_taxa)} proteins with taxa")
        
        except Exception as e:
            logger.error(f"Error loading associations: {e}")
    
    def get_taxon_for_protein(self, protein_id: str) -> Optional[int]:
        """Get taxon ID for a protein."""
        return self.protein_taxa.get(protein_id)
    
    def get_proteins_for_taxon(self, taxon_id: int) -> List[str]:
        """Get proteins associated with a taxon."""
        return self.taxon_proteins.get(taxon_id, [])
    
    def get_homologous_proteins(self, protein_id: str, max_distance: int = 20) -> List[Tuple[str, int]]:
        """
        Get homologous proteins (in similar taxa).
        
        Returns:
            List of (protein_id, distance) tuples
        """
        query_taxon = self.get_taxon_for_protein(protein_id)
        if query_taxon is None:
            return []
        
        homologs = []
        for other_taxon, proteins in self.taxon_proteins.items():
            distance = self.taxonomy.get_distance(query_taxon, other_taxon)
            if distance <= max_distance and distance < float('inf'):
                for p in proteins:
                    if p != protein_id:
                        homologs.append((p, distance))
        
        return sorted(homologs, key=lambda x: x[1])


def create_taxonomy_aware_features(processor: TaxonomyProcessor, 
                                   protein_taxonomy: ProteinTaxonomyAssociator,
                                   proteins: List[str],
                                   extract_embeddings: bool = False) -> Dict[str, np.ndarray]:
    """
    Create taxonomy-aware feature vectors for proteins.
    
    Args:
        processor: TaxonomyProcessor instance
        protein_taxonomy: ProteinTaxonomyAssociator instance
        proteins: List of protein IDs
        extract_embeddings: Whether to extract embedding features
        
    Returns:
        Dict of feature_name -> feature_matrix (n_proteins, n_features)
    """
    extractor = TaxonomyFeatureExtractor(processor)
    
    features = {}
    
    # Lineage features
    lineage_features = []
    rank_features_list = []
    
    for protein_id in proteins:
        taxon_id = protein_taxonomy.get_taxon_for_protein(protein_id)
        if taxon_id is None:
            # Use placeholder for missing taxa
            lineage_features.append(np.zeros(8))
            rank_features_list.append(np.zeros(8))
        else:
            lineage_features.append(extractor.extract_lineage_features(taxon_id))
            rank_dict = extractor.extract_rank_features(taxon_id)
            rank_features = np.array([rank_dict.get(rank, False) for rank in 
                                     ['superkingdom', 'kingdom', 'phylum', 'class', 
                                      'order', 'family', 'genus', 'species']], dtype=np.float32)
            rank_features_list.append(rank_features)
    
    features['taxonomy_lineage'] = np.vstack(lineage_features)
    features['taxonomy_ranks'] = np.vstack(rank_features_list)
    
    return features
