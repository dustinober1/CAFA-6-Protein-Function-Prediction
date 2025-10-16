"""
Gene Ontology (GO) hierarchy processing and hierarchical classification.

Implements:
- Parsing GO DAG from OBO format
- Term relationships (is_a, part_of, regulates)
- Hierarchical loss functions
- Term propagation and constraint enforcement
"""

import re
from typing import Dict, Set, List, Tuple, Optional, DefaultDict
from collections import defaultdict
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GOHierarchy:
    """Parse and manage GO ontology hierarchy."""
    
    def __init__(self, obo_path: str):
        """
        Initialize GO hierarchy from OBO file.
        
        Args:
            obo_path: Path to go-basic.obo file
        """
        self.obo_path = Path(obo_path)
        self.terms = {}  # term_id -> term_info
        self.parents = defaultdict(set)  # term_id -> set of parent term_ids
        self.children = defaultdict(set)  # term_id -> set of child term_ids
        self.relationships = defaultdict(lambda: defaultdict(set))  # term_id -> relation_type -> set of related_ids
        self.term_names = {}  # term_id -> name
        self.term_namespaces = {}  # term_id -> namespace (BP, CC, MF)
        self.obsolete_terms = set()
        self.root_terms = set()
        
        self._parse_obo()
    
    def _parse_obo(self):
        """Parse OBO file to extract ontology structure."""
        logger.info(f"Parsing GO ontology from {self.obo_path}...")
        
        with open(self.obo_path, 'r') as f:
            content = f.read()
        
        # Split by [Term] sections
        term_blocks = content.split('[Term]')[1:]  # Skip header
        
        for block in term_blocks:
            lines = block.strip().split('\n')
            term_id = None
            term_name = None
            namespace = None
            is_obsolete = False
            is_relations = []
            part_of_relations = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('id:'):
                    term_id = line.replace('id:', '').strip()
                elif line.startswith('name:'):
                    term_name = line.replace('name:', '').strip()
                elif line.startswith('namespace:'):
                    namespace = line.replace('namespace:', '').strip()
                elif line.startswith('is_a:'):
                    # Extract parent term ID
                    parent_id = line.replace('is_a:', '').strip().split('!')[0].strip()
                    is_relations.append(parent_id)
                elif line.startswith('relationship: part_of'):
                    # Extract part_of parent
                    parent_id = line.split('part_of')[1].strip().split('!')[0].strip()
                    part_of_relations.append(parent_id)
                elif line.startswith('is_obsolete:'):
                    is_obsolete = True
            
            if term_id and not is_obsolete:
                self.terms[term_id] = {
                    'name': term_name,
                    'namespace': namespace,
                    'is_a': is_relations,
                    'part_of': part_of_relations,
                }
                self.term_names[term_id] = term_name
                self.term_namespaces[term_id] = namespace
                
                # Build parent-child relationships
                for parent_id in is_relations + part_of_relations:
                    self.parents[term_id].add(parent_id)
                    self.children[parent_id].add(term_id)
                    
                    if 'is_a' in [line for line in lines if line.startswith('is_a:')]:
                        self.relationships[term_id]['is_a'].add(parent_id)
                    if 'part_of' in [line for line in lines if line.startswith('relationship:')]:
                        self.relationships[term_id]['part_of'].add(parent_id)
            elif is_obsolete:
                self.obsolete_terms.add(term_id)
        
        logger.info(f"Loaded {len(self.terms)} GO terms")
    
    def get_ancestors(self, term_id: str, include_self: bool = True) -> Set[str]:
        """Get all ancestor terms (transitive closure)."""
        ancestors = set()
        to_visit = [term_id] if include_self else list(self.parents[term_id])
        visited = set()
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            ancestors.add(current)
            
            for parent in self.parents[current]:
                if parent not in visited:
                    to_visit.append(parent)
        
        if not include_self:
            ancestors.discard(term_id)
        
        return ancestors
    
    def get_descendants(self, term_id: str, include_self: bool = True) -> Set[str]:
        """Get all descendant terms (transitive closure)."""
        descendants = set()
        to_visit = [term_id] if include_self else list(self.children[term_id])
        visited = set()
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            descendants.add(current)
            
            for child in self.children[current]:
                if child not in visited:
                    to_visit.append(child)
        
        if not include_self:
            descendants.discard(term_id)
        
        return descendants
    
    def get_level(self, term_id: str) -> int:
        """Get depth/level of term in hierarchy (0 for root terms)."""
        if not self.parents[term_id]:
            return 0
        return 1 + min(self.get_level(p) for p in self.parents[term_id])
    
    def get_depth(self, term_id: str) -> int:
        """Get height of term (0 for leaf terms)."""
        if not self.children[term_id]:
            return 0
        return 1 + max(self.get_depth(c) for c in self.children[term_id])
    
    def propagate_predictions(self, predictions: Dict[str, float],
                             strategy: str = 'max') -> Dict[str, float]:
        """
        Propagate predictions up the hierarchy to enforce constraints.
        
        Args:
            predictions: Dict of term_id -> prediction_score
            strategy: How to propagate - 'max', 'mean', or 'or'
                     'max': Parent gets max of children
                     'mean': Parent gets mean of children
                     'or': Parent gets 1 if any child is 1
        
        Returns:
            Propagated predictions
        """
        propagated = predictions.copy()
        
        # Sort terms by depth (process from leaves to roots)
        terms_by_depth = sorted(
            [t for t in self.terms if t in predictions],
            key=lambda t: -self.get_depth(t)
        )
        
        for term_id in terms_by_depth:
            for parent_id in self.parents[term_id]:
                if parent_id in propagated:
                    child_score = propagated[term_id]
                    parent_score = propagated.get(parent_id, 0.0)
                    
                    if strategy == 'max':
                        propagated[parent_id] = max(parent_score, child_score)
                    elif strategy == 'mean':
                        # Get average of all children
                        child_scores = [propagated.get(c, 0.0) for c in self.children[parent_id]]
                        propagated[parent_id] = np.mean(child_scores) if child_scores else parent_score
                    elif strategy == 'or':
                        propagated[parent_id] = max(parent_score, child_score)
        
        return propagated
    
    def get_term_pairs_by_level(self, max_levels: Optional[int] = None) -> Dict[int, List[Tuple[str, str]]]:
        """Get parent-child pairs grouped by level of parent term."""
        pairs_by_level = defaultdict(list)
        
        for child_id in self.terms:
            for parent_id in self.parents[child_id]:
                level = self.get_level(parent_id)
                if max_levels is None or level <= max_levels:
                    pairs_by_level[level].append((parent_id, child_id))
        
        return pairs_by_level


class HierarchicalBCELoss(nn.Module):
    """Hierarchical Binary Cross-Entropy Loss with parent-child constraints."""
    
    def __init__(self, hierarchy: GOHierarchy, go_terms: List[str],
                 constraint_weight: float = 0.1):
        """
        Initialize hierarchical loss.
        
        Args:
            hierarchy: GOHierarchy object
            go_terms: List of GO terms in order (matching prediction dimensions)
            constraint_weight: Weight for constraint violations (0-1)
        """
        super().__init__()
        self.hierarchy = hierarchy
        self.go_terms = go_terms
        self.constraint_weight = constraint_weight
        
        # Create term_id -> index mapping
        self.term_to_idx = {term: i for i, term in enumerate(go_terms)}
        self.idx_to_term = {i: term for term, i in self.term_to_idx.items()}
        
        # Build parent-child relationships in terms of indices
        self.parent_indices = [[] for _ in range(len(go_terms))]
        self.child_indices = [[] for _ in range(len(go_terms))]
        
        for child_id in go_terms:
            child_idx = self.term_to_idx[child_id]
            for parent_id in self.hierarchy.parents[child_id]:
                if parent_id in self.term_to_idx:
                    parent_idx = self.term_to_idx[parent_id]
                    self.parent_indices[child_idx].append(parent_idx)
                    self.child_indices[parent_idx].append(child_idx)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute hierarchical BCE loss.
        
        Args:
            predictions: Model predictions (batch_size, n_terms)
            targets: Ground truth labels (batch_size, n_terms)
            
        Returns:
            Hierarchical BCE loss
        """
        # Standard BCE loss
        bce_loss = torch.nn.functional.binary_cross_entropy(predictions, targets)
        
        if self.constraint_weight == 0:
            return bce_loss
        
        # Constraint loss: parent should be >= max(children)
        constraint_loss = 0.0
        n_constraints = 0
        
        for parent_idx, child_indices in enumerate(self.child_indices):
            if not child_indices:
                continue
            
            parent_pred = predictions[:, parent_idx]  # (batch_size,)
            max_child_pred = torch.stack([predictions[:, c_idx] for c_idx in child_indices], dim=1).max(dim=1)[0]
            
            # Violation when parent < max_child
            violation = torch.relu(max_child_pred - parent_pred)
            constraint_loss += violation.mean()
            n_constraints += 1
        
        if n_constraints > 0:
            constraint_loss = constraint_loss / n_constraints
        
        total_loss = bce_loss + self.constraint_weight * constraint_loss
        return total_loss


class ConstrainedMultiLabelLoss(nn.Module):
    """Multi-label loss with hierarchy constraints."""
    
    def __init__(self, hierarchy: GOHierarchy, go_terms: List[str],
                 weights: Optional[torch.Tensor] = None,
                 constraint_weight: float = 0.1):
        """
        Initialize constrained loss.
        
        Args:
            hierarchy: GOHierarchy object
            go_terms: List of GO terms
            weights: Per-class weights for BCE
            constraint_weight: Weight for hierarchy constraints
        """
        super().__init__()
        self.hierarchy = hierarchy
        self.go_terms = go_terms
        self.constraint_weight = constraint_weight
        
        # Create term_id -> index mapping
        self.term_to_idx = {term: i for i, term in enumerate(go_terms)}
        
        # BCE with weights if provided
        if weights is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        
        # Build constraints
        self.parent_child_pairs = []
        for child_id in go_terms:
            child_idx = self.term_to_idx[child_id]
            for parent_id in self.hierarchy.parents[child_id]:
                if parent_id in self.term_to_idx:
                    parent_idx = self.term_to_idx[parent_id]
                    self.parent_child_pairs.append((parent_idx, child_idx))
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with hierarchy constraints.
        
        Args:
            logits: Raw model outputs (batch_size, n_terms)
            targets: Ground truth labels (batch_size, n_terms)
            
        Returns:
            Loss value
        """
        # Main BCE loss
        bce_loss = self.bce(logits, targets)
        
        if self.constraint_weight == 0 or not self.parent_child_pairs:
            return bce_loss
        
        # Constraint loss: parent >= child (in probability space)
        predictions = torch.sigmoid(logits)
        constraint_loss = 0.0
        
        for parent_idx, child_idx in self.parent_child_pairs:
            parent_prob = predictions[:, parent_idx]
            child_prob = predictions[:, child_idx]
            
            # Violation when child > parent
            violation = torch.relu(child_prob - parent_prob)
            constraint_loss += violation.mean()
        
        constraint_loss = constraint_loss / max(len(self.parent_child_pairs), 1)
        
        total_loss = bce_loss + self.constraint_weight * constraint_loss
        return total_loss
