"""
Simple embedding-based model for protein function prediction.
"""
import numpy as np
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinEmbedding(nn.Module):
    """Learn protein embeddings from amino acid sequences."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int = 1000):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
        """
        super(ProteinEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_encoding = nn.Embedding(max_length, embedding_dim)
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        embeddings = self.embedding(x) + self.position_encoding(positions)
        return embeddings


class SequenceToFunctionModel(nn.Module):
    """Direct sequence-to-function model with attention."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, output_dim: int,
                 hidden_dim: int = 128, max_length: int = 1000):
        """
        Initialize model.
        
        Args:
            vocab_size: Amino acid vocabulary size (21 for standard AAs + UNK + PAD)
            embedding_dim: Embedding dimension
            output_dim: Number of GO terms
            hidden_dim: Hidden dimension
            max_length: Maximum sequence length
        """
        super(SequenceToFunctionModel, self).__init__()
        
        self.embedding = ProteinEmbedding(vocab_size, embedding_dim, max_length)
        
        # Attention-based pooling
        self.attention_weights = nn.Linear(embedding_dim, 1)
        
        # FC layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input sequences (batch_size, max_length)
            lengths: Actual lengths of sequences (batch_size,)
            
        Returns:
            Output predictions (batch_size, output_dim)
        """
        # Embed sequences
        embeddings = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Attention-based pooling
        attention_logits = self.attention_weights(embeddings)  # (batch, seq_len, 1)
        
        # Mask padding positions if lengths provided
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            attention_logits = attention_logits.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        attention_weights = F.softmax(attention_logits, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled = (embeddings * attention_weights).sum(dim=1)  # (batch, embed_dim)
        
        # FC layers
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x


class SequenceEncoder:
    """Encode protein sequences to integers."""
    
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    AA_TO_ID = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}  # 1-indexed, 0 is padding
    
    @staticmethod
    def encode_sequence(seq: str, max_length: int = 1000) -> np.ndarray:
        """
        Encode sequence to integer array.
        
        Args:
            seq: Protein sequence
            max_length: Maximum length (pad/truncate to this)
            
        Returns:
            Integer array of shape (max_length,)
        """
        seq = seq.upper()
        
        # Convert to IDs
        encoded = np.zeros(max_length, dtype=np.int64)
        for i, aa in enumerate(seq[:max_length]):
            if aa in SequenceEncoder.AA_TO_ID:
                encoded[i] = SequenceEncoder.AA_TO_ID[aa]
        
        return encoded
    
    @staticmethod
    def encode_sequences(sequences: Dict[str, str], max_length: int = 1000) -> tuple:
        """
        Encode multiple sequences.
        
        Args:
            sequences: Dict of protein_id -> sequence
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (encoded sequences, sequence lengths, protein IDs)
        """
        protein_ids = list(sequences.keys())
        encoded = []
        lengths = []
        
        for pid in protein_ids:
            seq = sequences[pid].upper()
            clean_seq = ''.join(c for c in seq if c in SequenceEncoder.AMINO_ACIDS)
            lengths.append(min(len(clean_seq), max_length))
            encoded.append(SequenceEncoder.encode_sequence(clean_seq, max_length))
        
        return np.array(encoded), np.array(lengths), protein_ids


def main():
    """Test embedding-based model."""
    from data_loader import CAFADataLoader
    from torch.utils.data import DataLoader, TensorDataset
    
    # Load data
    print("Loading data...")
    loader = CAFADataLoader('/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/cafa-6-protein-function-prediction')
    loader.load_train_data()
    
    # Sample data
    sample_ids = list(loader.train_sequences.keys())[:300]
    train_seqs = {pid: loader.train_sequences[pid] for pid in sample_ids[:240]}
    val_seqs = {pid: loader.train_sequences[pid] for pid in sample_ids[240:]}
    
    print(f"Train set: {len(train_seqs)}")
    print(f"Val set: {len(val_seqs)}")
    
    # Encode sequences
    print("\nEncoding sequences...")
    encoder = SequenceEncoder()
    X_train, lengths_train, train_ids = encoder.encode_sequences(train_seqs, max_length=500)
    X_val, lengths_val, val_ids = encoder.encode_sequences(val_seqs, max_length=500)
    
    print(f"Encoded shape: {X_train.shape}")
    print(f"Sample lengths - min: {lengths_train.min()}, max: {lengths_train.max()}")
    
    # Create target matrices
    all_go_terms = sorted(loader.go_terms)
    y_train, _ = loader.create_protein_to_terms_matrix(train_ids, all_go_terms)
    y_val, _ = loader.create_protein_to_terms_matrix(val_ids, all_go_terms)
    
    # Use top terms
    term_frequencies = y_train.sum(axis=0)
    top_indices = np.argsort(term_frequencies)[-30:]
    
    y_train = y_train[:, top_indices]
    y_val = y_val[:, top_indices]
    
    print(f"Using {len(top_indices)} GO terms")
    
    # Create model
    print("\nTraining sequence-based model...")
    device = 'cpu'
    
    model = SequenceToFunctionModel(
        vocab_size=21,  # 20 AAs + PAD
        embedding_dim=64,
        output_dim=len(top_indices),
        hidden_dim=128,
        max_length=500
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training loop
    X_train_tensor = torch.LongTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.LongTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    for epoch in range(3):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor)
        
        print(f"Epoch {epoch+1}/3: Train loss={total_loss/len(train_loader):.4f}, "
              f"Val loss={val_loss:.4f}")
    
    print("\nEmbedding model test completed successfully!")


if __name__ == '__main__':
    main()
