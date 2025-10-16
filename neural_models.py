"""
Deep learning models for protein function prediction.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import time


class ProteinDataset(Dataset):
    """PyTorch Dataset for protein sequences."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, protein_ids: List[str] = None):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target matrix (n_samples, n_terms)
            protein_ids: Optional protein IDs
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.protein_ids = protein_ids or [str(i) for i in range(len(features))]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': self.targets[idx],
            'protein_id': self.protein_ids[idx]
        }


class DeepNeuralNetwork(nn.Module):
    """Deep neural network for multi-label classification."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None, 
                 dropout_rate: float = 0.3):
        """
        Initialize network.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output (number of GO terms)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
        """
        super(DeepNeuralNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Sigmoid for multi-label
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ConvolutionalProteinNet(nn.Module):
    """CNN-based network for sequence processing."""
    
    def __init__(self, input_dim: int, output_dim: int, embedding_dim: int = 64):
        """
        Initialize CNN network.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output (number of GO terms)
            embedding_dim: Embedding dimension
        """
        super(ConvolutionalProteinNet, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        
        # Conv layers expecting 2D input, reshape features to (batch, 1, features)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        
        # Global average pooling handled manually
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Embed and reshape for conv (batch, 1, embedding_dim)
        x = self.embedding(x)
        x = self.embedding_bn(x)
        x = self.relu(x)
        x = x.unsqueeze(1)  # (batch, 1, embedding_dim)
        
        # Conv layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        # Global average pooling
        x = x.mean(dim=2)  # (batch, 32)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output
        x = self.output(x)
        x = self.sigmoid(x)
        
        return x


class NeuralNetworkModel:
    """Wrapper for neural network training and prediction."""
    
    def __init__(self, network: nn.Module, name: str = "DNN", device: str = 'cpu'):
        """
        Initialize model wrapper.
        
        Args:
            network: PyTorch network module
            name: Model name
            device: Device to use (cpu or cuda)
        """
        self.network = network.to(device)
        self.name = name
        self.device = device
        self.go_terms = None
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, verbose: bool = False) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer
            criterion: Loss function
            verbose: Print info
            
        Returns:
            Average loss
        """
        self.network.train()
        total_loss = 0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            outputs = self.network(features)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Average loss
        """
        self.network.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.network(features)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_dataset: ProteinDataset, val_dataset: ProteinDataset,
             go_terms: List[str], epochs: int = 10, batch_size: int = 32,
             learning_rate: float = 0.001, verbose: bool = True):
        """
        Train the neural network.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            go_terms: List of GO terms
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print info
        """
        self.go_terms = go_terms
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label
        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        if verbose:
            print(f"Training {self.name}...")
            print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.evaluate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Train loss={train_loss:.4f}, "
                      f"Val loss={val_loss:.4f}")
    
    def predict(self, X_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Feature matrix
            threshold: Probability threshold
            
        Returns:
            Binary predictions
        """
        self.network.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_test_tensor)
        
        predictions = (outputs.cpu().numpy() >= threshold).astype(int)
        return predictions
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        self.network.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(X_test_tensor)
        
        return outputs.cpu().numpy()


def main():
    """Test neural network models."""
    from data_loader import CAFADataLoader
    from feature_extractor import ProteinFeatureExtractor
    
    # Set device
    device = 'cpu'  # Use CPU for compatibility
    
    print("Loading data...")
    loader = CAFADataLoader('/Users/dustinober/Kaggle/CAFA-6-Protein-Function-Prediction/cafa-6-protein-function-prediction')
    loader.load_train_data()
    
    # Use sample
    sample_ids = list(loader.train_sequences.keys())[:500]
    train_seqs = {pid: loader.train_sequences[pid] for pid in sample_ids[:400]}
    val_seqs = {pid: loader.train_sequences[pid] for pid in sample_ids[400:]}
    
    print(f"Train set: {len(train_seqs)} proteins")
    print(f"Val set: {len(val_seqs)} proteins")
    
    # Extract features
    print("\nExtracting features...")
    extractor = ProteinFeatureExtractor(k=3)
    X_train, train_ids = extractor.create_combined_features(train_seqs, fit_tfidf=True)
    X_val, val_ids = extractor.create_combined_features(val_seqs, fit_tfidf=False)
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Create target matrices
    all_go_terms = sorted(loader.go_terms)
    y_train, _ = loader.create_protein_to_terms_matrix(train_ids, all_go_terms)
    y_val, _ = loader.create_protein_to_terms_matrix(val_ids, all_go_terms)
    
    # Use top 50 terms for speed
    term_frequencies = y_train.sum(axis=0)
    top_indices = np.argsort(term_frequencies)[-50:]
    
    y_train = y_train[:, top_indices]
    y_val = y_val[:, top_indices]
    top_terms = [all_go_terms[i] for i in top_indices]
    
    print(f"Using {len(top_terms)} GO terms")
    
    # Create datasets
    train_dataset = ProteinDataset(X_train, y_train, train_ids)
    val_dataset = ProteinDataset(X_val, y_val, val_ids)
    
    # Create and train model
    print("\nTraining Deep Neural Network...")
    network = DeepNeuralNetwork(
        input_dim=X_train.shape[1],
        output_dim=len(top_terms),
        hidden_dims=[128, 64],
        dropout_rate=0.3
    )
    
    model = NeuralNetworkModel(network, name="DNN", device=device)
    model.train(
        train_dataset, val_dataset,
        top_terms,
        epochs=5,
        batch_size=16,
        learning_rate=0.001,
        verbose=True
    )
    
    # Evaluate
    print("\nMaking predictions...")
    predictions = model.predict(X_val, threshold=0.5)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y_val, predictions, average='micro', zero_division=0)
    print(f"Validation F1 Score: {f1:.4f}")
    
    print("\nNeural network test completed successfully!")


if __name__ == '__main__':
    main()
