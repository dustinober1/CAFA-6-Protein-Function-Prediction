"""
Pre-trained protein embeddings using ESM-2 (Evolutionary Scale Modeling).

ESM-2 provides state-of-the-art protein sequence representations trained on
billions of protein sequences. This module extracts embeddings for use in
downstream tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ESM2Embedder:
    """Extract ESM-2 embeddings for protein sequences."""
    
    MODELS = {
        'esm2_t6': 'facebook/esm2_t6_8M',
        'esm2_t12': 'facebook/esm2_t12_35M',
        'esm2_t30': 'facebook/esm2_t30_150M',
        'esm2_t33': 'facebook/esm2_t33_650M',
    }
    
    def __init__(self, model_name: str = 'esm2_t33', device: Optional[str] = None):
        """
        Initialize ESM-2 embedder.
        
        Args:
            model_name: Which ESM-2 model to use. Options: esm2_t6, t12, t30, t33
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model {model_name}. Choose from {list(self.MODELS.keys())}")
    
    def load_model(self):
        """Load ESM-2 model and tokenizer."""
        if self._loaded:
            return
        
        try:
            from transformers import AutoTokenizer, EsmModel
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers")
        
        logger.info(f"Loading {self.model_name} model to {self.device}...")
        
        model_id = self.MODELS[self.model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = EsmModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self._loaded = True
        
        logger.info(f"Model loaded. Embedding dimension: {self.model.config.hidden_size}")
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if not self._loaded:
            self.load_model()
        return self.model.config.hidden_size
    
    def embed_sequences(self, sequences: List[str], batch_size: int = 4,
                       pool_type: str = 'mean') -> np.ndarray:
        """
        Extract embeddings for protein sequences.
        
        Args:
            sequences: List of amino acid sequences
            batch_size: Batch size for processing
            pool_type: How to pool token embeddings - 'mean', 'cls', or 'max'
            
        Returns:
            Embeddings of shape (n_sequences, embedding_dim)
        """
        if not self._loaded:
            self.load_model()
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                
                # Tokenize with padding
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True,
                                       truncation=True, max_length=1024)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Pool embeddings
                if pool_type == 'mean':
                    # Mean pooling with attention mask
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                    masked = token_embeddings * attention_mask
                    pooled = masked.sum(dim=1) / attention_mask.sum(dim=1)
                elif pool_type == 'cls':
                    # Use [CLS] token (first token)
                    pooled = outputs.last_hidden_state[:, 0, :]
                elif pool_type == 'max':
                    # Max pooling with attention mask
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    token_embeddings = token_embeddings.masked_fill(~attention_mask.bool(), float('-inf'))
                    pooled = token_embeddings.max(dim=1).values
                    pooled = torch.where(torch.isinf(pooled), torch.tensor(0.0, device=self.device), pooled)
                else:
                    raise ValueError(f"Unknown pool_type: {pool_type}")
                
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def embed_sequence_with_tokens(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both token-level and pooled embeddings for a single sequence.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Tuple of (token_embeddings, pooled_embedding)
            - token_embeddings: shape (seq_len, embedding_dim)
            - pooled_embedding: shape (embedding_dim,)
        """
        if not self._loaded:
            self.load_model()
        
        with torch.no_grad():
            inputs = self.tokenizer([sequence], return_tensors='pt', padding=True,
                                   truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get token embeddings (exclude special tokens)
            token_embs = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()  # Exclude [CLS] and [SEP]
            
            # Get pooled embedding
            pooled = token_embs.mean(axis=0)
            
            return token_embs, pooled
    
    def embed_with_cache(self, sequences: List[str], cache_path: Optional[str] = None,
                        batch_size: int = 4, pool_type: str = 'mean') -> np.ndarray:
        """
        Extract embeddings with optional caching.
        
        Args:
            sequences: List of sequences to embed
            cache_path: Path to cache file (numpy .npy format)
            batch_size: Batch size for processing
            pool_type: Pooling type
            
        Returns:
            Embeddings array
        """
        # Try to load from cache
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading embeddings from cache: {cache_path}")
            return np.load(cache_path)
        
        # Compute embeddings
        logger.info(f"Computing embeddings for {len(sequences)} sequences...")
        embeddings = self.embed_sequences(sequences, batch_size=batch_size, pool_type=pool_type)
        
        # Save to cache
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info(f"Cached embeddings to {cache_path}")
        
        return embeddings


class ProtBertEmbedder:
    """Extract embeddings using ProtBERT (BERT trained on protein sequences)."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize ProtBERT embedder.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load_model(self):
        """Load ProtBERT model and tokenizer."""
        if self._loaded:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers library required. Install with: pip install transformers")
        
        logger.info(f"Loading ProtBERT model to {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd",
                                                       do_lower_case=False)
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to(self.device)
        self.model.eval()
        self._loaded = True
        
        logger.info(f"ProtBERT loaded. Embedding dimension: {self.model.config.hidden_size}")
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if not self._loaded:
            self.load_model()
        return self.model.config.hidden_size
    
    def embed_sequences(self, sequences: List[str], batch_size: int = 4,
                       pool_type: str = 'mean') -> np.ndarray:
        """
        Extract embeddings for protein sequences.
        
        Args:
            sequences: List of amino acid sequences (space-separated)
            batch_size: Batch size for processing
            pool_type: How to pool - 'mean', 'cls', or 'max'
            
        Returns:
            Embeddings of shape (n_sequences, embedding_dim)
        """
        if not self._loaded:
            self.load_model()
        
        # Add spaces between amino acids if not already present
        processed = []
        for seq in sequences:
            if ' ' not in seq:
                seq = ' '.join(seq)
            processed.append(seq)
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(processed), batch_size):
                batch = processed[i:i + batch_size]
                
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True,
                                       truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Pool embeddings
                if pool_type == 'mean':
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                    masked = token_embeddings * attention_mask
                    pooled = masked.sum(dim=1) / attention_mask.sum(dim=1)
                elif pool_type == 'cls':
                    pooled = outputs.last_hidden_state[:, 0, :]
                elif pool_type == 'max':
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    token_embeddings = token_embeddings.masked_fill(~attention_mask.bool(), float('-inf'))
                    pooled = token_embeddings.max(dim=1).values
                    pooled = torch.where(torch.isinf(pooled), torch.tensor(0.0, device=self.device), pooled)
                else:
                    raise ValueError(f"Unknown pool_type: {pool_type}")
                
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
