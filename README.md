# CAFA-6 Protein Function Prediction

Advanced machine learning pipeline for predicting Gene Ontology (GO) terms for proteins based on amino acid sequences using state-of-the-art techniques.

## Overview

This project implements a comprehensive solution for the CAFA-6 (Critical Assessment of Functional Annotation) protein function prediction challenge. The system combines multiple approaches:

- **Pre-trained embeddings**: ESM-2 and ProtBERT models for sequence representation
- **Hierarchical classification**: GO ontology-aware training with constraint enforcement
- **Taxonomic information**: Leveraging evolutionary relationships
- **Advanced ensembles**: Stacking, weighted averaging, and voting strategies
- **Data augmentation**: Multiple sequence augmentation techniques
- **Hyperparameter optimization**: Bayesian search with Optuna
- **Cross-validation**: Stratified k-fold splitting for robust evaluation

## Project Structure

```
.
├── src/                          # Main source code
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py       # Data loading and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py # Hand-crafted features
│   │   ├── augmentation.py      # Data augmentation techniques
│   │   ├── go_ontology.py       # GO hierarchy processing
│   │   └── taxonomy_processor.py # Taxonomic information
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py   # Traditional ML (RF, SVM)
│   │   ├── neural_models.py     # Deep learning models
│   │   ├── embedding_model.py   # Sequence embeddings
│   │   ├── esm_embedder.py      # ESM-2 and ProtBERT
│   │   ├── enhanced_ensemble.py # Advanced ensembling
│   │   └── hpo.py               # Hyperparameter optimization
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluation.py        # Metrics and submission
│   │   └── cross_validation.py  # CV utilities
│   └── train_pipeline.py        # End-to-end pipeline
├── scripts/
│   ├── experiments.py           # Experimental runs
│   └── quick_test.py            # Quick validation
├── run_pipeline.py              # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── results/                     # Output directory
```

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dustinober1/CAFA-6-Protein-Function-Prediction.git
cd CAFA-6-Protein-Function-Prediction
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the CAFA-6 data from Kaggle and extract to the project root:
```
cafa-6-protein-function-prediction/
├── Train/
│   ├── go-basic.obo
│   ├── train_sequences.fasta
│   ├── train_taxonomy.tsv
│   └── train_terms.tsv
├── Test/
│   ├── testsuperset.fasta
│   └── testsuperset-taxon-list.tsv
├── IA.tsv
└── sample_submission.tsv
```

## Quick Start

### Run the Full Pipeline

```bash
python run_pipeline.py --data-dir cafa-6-protein-function-prediction --output-dir results
```

### Run Quick Test

```bash
python scripts/quick_test.py
```

### Run Experiments

```bash
python scripts/experiments.py features --n-proteins 500
python scripts/experiments.py threshold --n-proteins 500
python scripts/experiments.py sizes
```

## Usage Examples

### Feature Extraction

```python
from src.features.feature_extractor import ProteinFeatureExtractor

extractor = ProteinFeatureExtractor()
features = extractor.extract_combined_features(['MVHLTPEEKS', 'MVLSPADKTNV'])
# Returns (n_samples, 427) combined feature matrix
```

### Pre-trained Embeddings

```python
from src.models.esm_embedder import ESM2Embedder

embedder = ESM2Embedder('esm2_t33')  # Use largest model
embedder.load_model()

embeddings = embedder.embed_sequences(['MVHLTPEEKS'], batch_size=4)
# Returns (1, 1280) for t33 model
```

### GO Hierarchy Processing

```python
from src.features.go_ontology import GOHierarchy, HierarchicalBCELoss

hierarchy = GOHierarchy('cafa-6-protein-function-prediction/Train/go-basic.obo')

# Get ancestors of a GO term
ancestors = hierarchy.get_ancestors('GO:0001234')

# Enforce hierarchy constraints during training
loss_fn = HierarchicalBCELoss(hierarchy, go_terms, constraint_weight=0.1)
```

### Taxonomic Features

```python
from src.features.taxonomy_processor import TaxonomyProcessor, ProteinTaxonomyAssociator

taxonomy = TaxonomyProcessor('path/to/taxonomy.tsv')
associator = ProteinTaxonomyAssociator('path/to/taxonomy.tsv')
associator.load_associations('path/to/protein_taxonomy.tsv')

# Get lineage for a protein's taxon
taxon_id = associator.get_taxon_for_protein('protein_123')
lineage = taxonomy.get_lineage(taxon_id)
```

### Data Augmentation

```python
from src.features.augmentation import ProteinAugmenter, AugmentationPipeline

augmenter = ProteinAugmenter()
pipeline = AugmentationPipeline(augmenter)

# Create augmented dataset
augmented = pipeline.create_augmentation_mix(
    sequences=['MVHLTPEEKS'],
    augmentation_types=['similar', 'masking', 'deletion'],
    num_samples_per_seq=2
)
```

### Cross-Validation

```python
from src.evaluation.cross_validation import StratifiedMultiLabelKFold, CrossValidationEvaluator

cv = CrossValidationEvaluator(n_splits=5)
results = cv.cross_validate(model_factory, X, y, eval_fn=metric_fn)

print(f"CV F1: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
```

### Hyperparameter Optimization

```python
from src.models.hpo import BayesianHyperparameterOptimizer

optimizer = BayesianHyperparameterOptimizer(n_trials=100)
result = optimizer.optimize(objective_fn, param_space, direction='maximize')

print(f"Best parameters: {result['best_params']}")
print(f"Best score: {result['best_value']:.4f}")
```

### Advanced Ensembling

```python
from src.models.enhanced_ensemble import StackingEnsemble, EnsembleOptimizer

# Stacking ensemble
stacking = StackingEnsemble(base_models=[model1, model2, model3], meta_model=meta_model)
stacking.fit(X_train, y_train)
predictions = stacking.predict(X_test)

# Find best ensemble strategy
strategy, results = EnsembleOptimizer.find_best_ensemble_strategy(
    models=[model1, model2, model3],
    X_val=X_val,
    y_val=y_val,
    metric_fn=metric_fn
)
```

## Key Features

### 1. Pre-trained Protein Embeddings
- **ESM-2**: State-of-the-art protein language model (6M to 650M parameters)
- **ProtBERT**: BERT trained on protein sequences
- Token-level and sequence-level representations
- Optional caching for efficiency

### 2. Hierarchical GO Classification
- Automatic parsing of GO DAG from OBO format
- Parent-child relationship modeling
- Hierarchical BCE loss with constraints
- Term propagation strategies (max, mean, OR)

### 3. Taxonomic Integration
- Lineage extraction from NCBI taxonomy
- Taxonomy-aware feature vectors
- Homolog detection based on phylogenetic distance
- Rank-based features (kingdom, phylum, etc.)

### 4. Data Augmentation
- **Biochemically-aware substitution**: Replace with similar amino acids
- **Masking**: Random position masking
- **Deletion**: Random position deletion
- **Insertion**: Random amino acid insertion
- **Rotation**: Circular sequence shifts
- **Balanced augmentation**: Focus on minority classes

### 5. Cross-Validation
- **Stratified multi-label k-fold**: Respects class imbalance
- Automatic train/val/test splitting
- Comprehensive fold-wise evaluation
- Per-fold metrics and aggregation

### 6. Hyperparameter Optimization
- **Bayesian optimization** with Optuna
- **TPE sampler** for efficient search
- Per-model parameter spaces
- Parallel trial execution
- Model selection across architectures

### 7. Enhanced Ensembling
- **Weighted ensemble**: Per-model weight optimization
- **Voting ensemble**: Majority/plurality rules
- **Stacking**: Meta-learner on base predictions
- **Blending**: Holdout-set weight optimization
- Automatic strategy selection

## Model Architectures

### Traditional ML
- **Random Forest**: Multi-class with OneVsRest
- **SVM**: Linear SVM with class weighting

### Deep Learning
- **Deep Neural Network**: Multi-layer MLP with batch norm
- **Convolutional Network**: 1D convolutions for sequences
- **Sequence Encoder**: Learnable embeddings + attention

### Advanced Models
- **ESM-2 Embeddings**: Pre-trained transformer embeddings
- **ProtBERT Embeddings**: Domain-specific BERT model
- **Hierarchical Models**: With GO constraint enforcement

## Performance Metrics

The system evaluates using:
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Correct predictions / total positive predictions
- **Recall**: Correct predictions / total actual positives
- **Weighted IA Score**: Weighted by information accretion

## Configuration

Edit `src/train_pipeline.py` to customize:
- Feature types to use
- Model architectures and hyperparameters
- Ensemble strategy
- Training epochs and learning rates
- Batch sizes
- Validation strategy

## Output

The pipeline generates:
- `results/submission.tsv`: Kaggle competition submission format
  - Format: `protein_id \t go_term \t probability`
  - Includes top 1500 predictions per protein
  - Properly formatted scores

## Dependencies

Core dependencies:
- numpy >= 2.0
- pandas >= 2.0
- scikit-learn >= 1.0
- torch >= 2.0
- transformers >= 4.30
- biopython >= 1.80

Optional:
- optuna >= 3.0 (for Bayesian HPO)
- pytorch-lightning >= 2.0 (for advanced training)

See `requirements.txt` for complete list with versions.

## Advanced Usage

### Custom Feature Engineering

```python
from src.features.feature_extractor import ProteinFeatureExtractor

class CustomExtractor(ProteinFeatureExtractor):
    def extract_custom_features(self, sequence):
        # Your custom logic here
        pass
```

### Custom Loss Functions

```python
from src.features.go_ontology import ConstrainedMultiLabelLoss
import torch

hierarchy = GOHierarchy(obo_path)
loss_fn = ConstrainedMultiLabelLoss(
    hierarchy, 
    go_terms,
    weights=class_weights,
    constraint_weight=0.15
)

# Use in training loop
loss = loss_fn(logits, targets)
```

### Distributed Training

```python
import torch
from torch.nn.parallel import DataParallel

model = DataParallel(model, device_ids=[0, 1, 2, 3])
```

## Troubleshooting

### Memory Issues
- Reduce batch size in pipeline config
- Use smaller ESM model (t6 or t12)
- Enable gradient checkpointing for large models

### Slow Training
- Use CPU with Intel MKL-DNN
- Enable mixed precision training
- Use distributed training across GPUs

### Import Errors
- Ensure you're in the virtual environment
- Run `pip install -r requirements.txt` again
- Check PYTHONPATH includes src directory

## Competition Details

**Challenge**: CAFA-6 Protein Function Prediction
**Platform**: Kaggle
**Data**: 82,404 training proteins with 26,125 GO term annotations
**Evaluation**: Weighted F1-score based on information accretion
**Deadline**: Competition timeline

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cafa6_pipeline_2025,
  author = {Your Name},
  title = {CAFA-6 Protein Function Prediction Pipeline},
  year = {2025},
  url = {https://github.com/dustinober1/CAFA-6-Protein-Function-Prediction}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- CAFA organizers for the competition
- ESM-2 authors (Meta AI)
- ProtBERT authors (Rostlab)
- Kaggle community for discussions and insights

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Last Updated**: October 16, 2025
**Python Version**: 3.8+
**Status**: Active Development
