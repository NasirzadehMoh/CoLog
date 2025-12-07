# Training Module

This module provides the complete training and evaluation pipeline for CoLog's collaborative transformer model. It orchestrates model training, hyperparameter optimization, validation, and comprehensive evaluation of log anomaly detection performance.

## Overview

The training module implements end-to-end supervised learning for the CoLog neural network architecture. It handles data loading from preprocessed ground truth, model training with advanced optimization strategies, hyperparameter tuning with Ray Tune, and thorough performance evaluation. The module provides both standard training workflows and automated hyperparameter search capabilities.

### Key Features

- **Complete training pipeline**: End-to-end supervised learning with gradient-based optimization
- **Early stopping**: Automatic training termination when validation performance plateaus
- **Learning rate scheduling**: Adaptive learning rate decay for improved convergence
- **Gradient clipping**: Training stability through gradient norm constraints
- **Hyperparameter tuning**: Automated optimization with Ray Tune integration
- **Multi-GPU support**: Automatic data parallelism with PyTorch DataParallel
- **Comprehensive logging**: Detailed training metrics and progress tracking
- **Checkpoint management**: Best model saving based on validation accuracy
- **Data loading utilities**: Efficient ground truth loading with caching and preprocessing

## Directory Structure

```
train/
├── __init__.py                    # Package initialization
├── main.py                        # Main training pipeline and Trainer class
└── utils/                         # Training utilities
    ├── __init__.py
    ├── groundtruth_loader.py      # PyTorch Dataset for ground truth data
    ├── prediction_utils.py        # Prediction conversion utilities
    └── cmap.p                     # Cached colormap for visualization
```

## Core Components

### 1. Trainer Class (`main.py`)

The `Trainer` class orchestrates the complete training workflow including optimization, validation, and hyperparameter tuning.

**Key Methods:**
- `train_model()`: Execute standard training loop with early stopping
- `evaluate_model()`: Compute accuracy on validation/test sets
- `tune_model()`: Train with Ray Tune hyperparameter configuration
- `evaluate_tunable_model()`: Evaluate tunable model variant during optimization

**Training Features:**
- Iterative optimization over configurable epochs
- Gradient-based parameter updates with Adam optimizer
- Periodic validation evaluation
- Early stopping based on validation accuracy
- Learning rate decay when performance plateaus
- Gradient norm clipping for stability
- Best model checkpointing

### 2. GroundTruthLoader (`utils/groundtruth_loader.py`)

PyTorch Dataset implementation for loading preprocessed ground truth data with multi-modal features.

**Functionality:**
- Message preprocessing: Tokenization, stopword removal, normalization
- Token-to-index mapping: Vocabulary dictionary creation and caching
- Word embeddings: spaCy-based vector extraction for semantic features
- Sequence padding: Fixed-length sequence handling for batching
- Multi-modal data: Combined text and sequence feature retrieval

**Caching:**
- Token embeddings cached to disk after first extraction
- Token-to-index mappings stored for reuse
- Efficient loading across multiple training runs

### 3. Prediction Utilities (`utils/prediction_utils.py`)

Helper functions for converting model outputs to discrete class labels.

**Functions:**
- `predict_with_argmax()`: Convert logits to class predictions
- Batch processing for efficient prediction conversion
- Support for binary and multi-class classification

## Training Workflow

### Standard Training

The typical training workflow follows these steps:

1. **Data Loading**: Load preprocessed ground truth using GroundTruthLoader
2. **Model Initialization**: Create collaborative transformer architecture
3. **Training Loop**: Iterative optimization with early stopping
4. **Validation**: Periodic evaluation on validation set
5. **Checkpointing**: Save best model based on validation accuracy
6. **Final Evaluation**: Comprehensive testing on held-out test set

### Training Loop Details

Each training epoch performs:

1. **Forward Pass**: Process batches of log messages and sequences
2. **Loss Computation**: Cross-entropy loss between predictions and labels
3. **Backward Propagation**: Compute gradients via backpropagation
4. **Gradient Clipping**: Apply norm clipping if enabled
5. **Parameter Update**: Optimize model weights with Adam
6. **Progress Tracking**: Log metrics and display real-time progress

### Early Stopping Strategy

Training automatically stops when:
- Validation accuracy doesn't improve for `early_stop` consecutive epochs
- Maximum number of epochs (`max_epoch`) is reached
- Perfect validation accuracy (100%) is achieved

### Learning Rate Scheduling

The learning rate is adaptively reduced when:
- Validation accuracy plateaus for consecutive epochs
- Decay is applied by multiplying current LR by `lr_decay` factor
- Maximum `decay_times` decays have occurred

## Hyperparameter Tuning

The module integrates with Ray Tune for automated hyperparameter optimization.

**Tunable Hyperparameters:**
- Learning rate
- Batch size
- Embedding dimensions
- Number of attention heads
- Number of collaborative layers
- Dropout rates
- LSTM hidden dimensions
- And more...

**Tuning Process:**
1. Define hyperparameter search space in configuration
2. Initialize Ray Tune scheduler (e.g., ASHA, HyperBand)
3. Launch parallel training trials with different configurations
4. Evaluate each trial on validation set
5. Report accuracy metrics to Ray Tune
6. Select best configuration based on validation performance

**Tuning Features:**
- Parallel trial execution across multiple GPUs
- Early termination of poorly performing trials
- Automatic checkpoint management
- Best configuration selection

## Usage

### Basic Training

```python
from train.main import Trainer
from train.utils.groundtruth_loader import GroundTruthLoader
from neuralnetwork import CollaborativeTransformer
from torch.utils.data import DataLoader

# Load ground truth data
train_dataset = GroundTruthLoader(
    data_split='train_set',
    args=config,
    dataset_dir='datasets/hadoop',
    print_hints=True
)

val_dataset = GroundTruthLoader(
    data_split='val_set',
    args=config,
    dataset_dir='datasets/hadoop'
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = CollaborativeTransformer(config)
embedding_layer = model.get_embedding_layer()

# Train model
trainer = Trainer(config)
accuracies = trainer.train_model(model, embedding_layer, train_loader, val_loader)

print(f"Best validation accuracy: {max(accuracies):.2f}%")
```

### Hyperparameter Tuning with Ray Tune

```python
from train.main import Trainer
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Define search space
config = {
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "n_heads": tune.choice([4, 8, 16]),
    "n_collaborative_layers": tune.choice([2, 3, 4]),
    "dropout": tune.uniform(0.1, 0.5),
}

# Configure scheduler
scheduler = ASHAScheduler(
    metric="accuracy",
    mode="max",
    max_t=100,
    grace_period=10,
    reduction_factor=2
)

# Run hyperparameter search
analysis = tune.run(
    tune.with_parameters(Trainer(base_config).tune_model),
    config=config,
    num_samples=50,
    scheduler=scheduler,
    resources_per_trial={"cpu": 4, "gpu": 1}
)

# Get best configuration
best_config = analysis.get_best_config(metric="accuracy", mode="max")
print(f"Best hyperparameters: {best_config}")
```

### Model Evaluation

```python
from train.main import Trainer

# Load test dataset
test_dataset = GroundTruthLoader(
    data_split='test_set',
    args=config,
    dataset_dir='datasets/hadoop'
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate model
trainer = Trainer(config)
accuracy, predictions = trainer.evaluate_model(model, embedding_layer, test_loader)

print(f"Test accuracy: {accuracy:.2f}%")
```

## Configuration Parameters

### Training Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_epoch` | Maximum training epochs | 100 | 10-500 |
| `learning_rate` | Initial learning rate | 0.001 | 1e-5 to 1e-1 |
| `batch_size` | Training batch size | 32 | 8-256 |
| `early_stop` | Early stopping patience (epochs) | 10 | 5-50 |
| `lr_decay` | Learning rate decay factor | 0.5 | 0.1-0.9 |
| `decay_times` | Maximum LR decay applications | 3 | 0-10 |
| `grad_norm_clip` | Gradient clipping threshold | 1.0 | 0 (disabled) or 0.5-5.0 |
| `eval_start` | Epoch to start evaluation | 1 | 1-max_epoch |

### Data Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_dir` | Ground truth data directory | - |
| `data_split` | Dataset split (train_set/val_set/test_set) | train_set |
| `len_messages` | Maximum message sequence length | 100 |
| `len_sequences` | Maximum sequence feature length | 50 |
| `vocab_size` | Vocabulary size for embeddings | 10000 |

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `device` | Computation device (cuda/cpu) | cuda |
| `checkpoint_path` | Model checkpoint save directory | checkpoints/ |
| `result_path` | Training logs save directory | results/ |
| `seed` | Random seed for reproducibility | 42 |

## Output Files

### Training Metrics Log

`training_metrics.log` contains complete training history:

```
Arguments: Namespace(max_epoch=100, learning_rate=0.001, ...)

Epoch: 1, Loss: 0.523456, Lr: 1.000000e-03
Elapsed time: 45 sec, Speed: 0.234 sec/batch
Validation accuracy: 72.34%

Epoch: 2, Loss: 0.412345, Lr: 1.000000e-03
Elapsed time: 44 sec, Speed: 0.229 sec/batch
Validation accuracy: 78.56%

...

Training finished. Best validation accuracy: 94.23%
```

### Model Checkpoints

Best models are saved with naming convention:
- During training: `best<seed>.pkl`
- After training: `best<accuracy>_<seed>.pkl`

Example: `best94.23_42.pkl` (94.23% validation accuracy, seed 42)

### Progress Display

Real-time console output during training:

```
[ Epoch 15:  --Step: 234/500  --Loss: 0.3421  --Learning rate: 5.00e-04  --Remaining time: 3 (min) : 45 (sec) ] ---- 
Epoch 15 finished in 287 sec
```

## Data Format

### GroundTruthLoader Input

The loader expects preprocessed ground truth data with:

**Directory Structure:**
```
dataset_dir/
├── train_set.pkl           # Training split
├── val_set.pkl             # Validation split
└── test_set.pkl            # Test split
```

**Data Format (each .pkl file):**
```python
{
    'log_id': ['log_001', 'log_002', ...],           # Unique log identifiers
    'messages': [['msg1', 'msg2', ...], ...],        # Tokenized log messages
    'sequences': [[[0.1, 0.2, ...], ...], ...],      # Sequential features
    'labels': [0, 1, 0, 1, ...]                      # Binary labels (0=anomaly, 1=normal)
}
```

### GroundTruthLoader Output

Each batch from DataLoader contains:

```python
(
    log_id,           # Tuple of log IDs (batch_size,)
    input_messages,   # Tensor of token IDs (batch_size, len_messages)
    input_sequences,  # Tensor of sequence features (batch_size, len_sequences, feature_dim)
    target_labels     # Tensor of labels (batch_size,)
)
```

## Error Handling

The training pipeline includes comprehensive error handling:

- **Data Loading Errors**: Validates ground truth file existence and format
- **Training Errors**: Catches and logs exceptions during forward/backward passes
- **Evaluation Errors**: Gracefully handles evaluation failures, restores model state
- **Tuning Errors**: Reports Ray Tune errors with detailed stack traces

All errors are logged with full context for debugging.

## Performance Optimization

### Memory Optimization

- **Gradient Accumulation**: Supports effective batch size increase without memory overhead
- **Mixed Precision**: Compatible with PyTorch AMP for faster training
- **DataLoader Workers**: Multi-process data loading for I/O efficiency

### Computational Optimization

- **Multi-GPU Training**: Automatic DataParallel for distributed computation
- **Cached Embeddings**: Disk caching of token embeddings to avoid recomputation
- **Efficient Batching**: Fixed-length padding for optimal GPU utilization

### Training Speed Tips

1. **Use GPU**: Set `device='cuda'` for 10-50x speedup
2. **Increase batch size**: Larger batches improve GPU utilization
3. **Enable gradient clipping**: Prevents gradient explosion and wasted epochs
4. **Tune early stopping**: Avoid unnecessary epochs with proper patience setting
5. **Cache embeddings**: First run computes embeddings, subsequent runs load from cache

## Logging Configuration

The module uses Python's `logging` framework:

```python
import logging

# Configure logger for training module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
```

**Log Levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General training progress and milestones
- `WARNING`: Potential issues that don't halt training
- `ERROR`: Failures that prevent training completion

## Dependencies

### Core Dependencies

```
torch>=1.8.0
numpy>=1.19.0
```

### Tuning Dependencies

```
ray[tune]>=1.12.0
```

### Data Loading Dependencies

```
spacy>=3.0.0
en_core_web_lg>=3.0.0
nltk>=3.6.0
```

### Full Requirements

See `requirements.txt` in the project root for complete dependency list.

## Integration with Other Modules

### Ground Truth Module Integration

The training module consumes output from the ground truth module:

```python
# Ground truth generation (separate step)
from groundtruth import GroundTruth

gt = GroundTruth(dataset='hadoop', sequence_type='context', window_size=2)
gt.extract_ground_truth()  # Generates train/val/test splits

# Training (uses generated ground truth)
from train.utils.groundtruth_loader import GroundTruthLoader

train_data = GroundTruthLoader(data_split='train_set', dataset_dir='datasets/hadoop')
```

### Neural Network Module Integration

The training module works with neural network architectures:

```python
# Standard collaborative transformer
from neuralnetwork import CollaborativeTransformer

model = CollaborativeTransformer(config)
embedding_layer = model.get_embedding_layer()
trainer.train_model(model, embedding_layer, train_loader, val_loader)

# Tunable variant for hyperparameter search
from neuralnetwork import CollaborativeTransformerTunable

tunable_model = CollaborativeTransformerTunable(config)
trainer.tune_model(config, checkpoint_dir)
```

## Best Practices

### Training Workflow

1. **Start with small model**: Validate pipeline with minimal hyperparameters
2. **Monitor validation curve**: Check for overfitting (train-val gap)
3. **Use early stopping**: Prevent wasted computation on saturated models
4. **Save checkpoints frequently**: Preserve best models for later use
5. **Log all metrics**: Enable post-training analysis and debugging

### Hyperparameter Selection

1. **Learning rate**: Start with 1e-3, tune in range [1e-5, 1e-2]
2. **Batch size**: Use largest size that fits in GPU memory
3. **Early stopping patience**: 5-10 epochs for small datasets, 10-20 for large
4. **Gradient clipping**: Use 1.0-5.0 to prevent explosion
5. **LR decay**: Apply 0.5 decay factor after 2-3 epochs without improvement

### Data Preparation

1. **Balanced splits**: Ensure train/val/test have similar class distributions
2. **Sufficient data**: Aim for 1000+ samples per class for robust training
3. **Consistent preprocessing**: Use same tokenization and normalization across splits
4. **Cached embeddings**: Reuse embeddings across experiments for efficiency

## Troubleshooting

### Common Issues

**Issue**: Training loss increases or NaN values appear
- **Solution**: Reduce learning rate, enable gradient clipping, check data for corruption

**Issue**: Validation accuracy doesn't improve
- **Solution**: Increase model capacity, adjust hyperparameters, check for data leakage

**Issue**: Out of memory errors
- **Solution**: Reduce batch size, decrease sequence lengths, use gradient accumulation

**Issue**: Slow training speed
- **Solution**: Use GPU, increase batch size, enable multi-worker data loading

**Issue**: Ray Tune trials fail
- **Solution**: Check GPU availability, reduce resources_per_trial, validate config

### Debug Mode

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Remove warning filters in module imports:

```python
# Comment out in main.py and groundtruth_loader.py
# warnings.filterwarnings("ignore")
```

## Examples

### Complete Training Script

```python
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from train.main import Trainer
from train.utils.groundtruth_loader import GroundTruthLoader
from neuralnetwork import CollaborativeTransformer

# Configuration
class Config:
    dataset_dir = 'datasets/hadoop'
    max_epoch = 100
    learning_rate = 0.001
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result_path = 'results/'
    checkpoint_path = 'checkpoints/'
    eval_start = 1
    early_stop = 10
    lr_decay = 0.5
    decay_times = 3
    grad_norm_clip = 1.0
    seed = 42
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Model hyperparameters
    len_messages = 100
    len_sequences = 50
    vocab_size = 10000
    embed_size = 300
    n_heads = 8
    n_collaborative_layers = 3

config = Config()

# Set random seed
torch.manual_seed(config.seed)

# Load datasets
train_data = GroundTruthLoader('train_set', config, config.dataset_dir, print_hints=True)
val_data = GroundTruthLoader('val_set', config, config.dataset_dir)
test_data = GroundTruthLoader('test_set', config, config.dataset_dir)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

# Initialize model
model = CollaborativeTransformer(config).to(config.device)
embedding_layer = model.get_embedding_layer()

# Train model
trainer = Trainer(config)
print("Starting training...")
accuracies = trainer.train_model(model, embedding_layer, train_loader, val_loader)

# Evaluate on test set
print("\nEvaluating on test set...")
test_accuracy, predictions = trainer.evaluate_model(model, embedding_layer, test_loader)
print(f"Final test accuracy: {test_accuracy:.2f}%")

# Save final results
Path(config.result_path).mkdir(exist_ok=True)
with open(f"{config.result_path}/results.txt", 'w') as f:
    f.write(f"Best validation accuracy: {max(accuracies):.2f}%\n")
    f.write(f"Test accuracy: {test_accuracy:.2f}%\n")
```

## References

- PyTorch Documentation: https://pytorch.org/docs/
- Ray Tune Guide: https://docs.ray.io/en/latest/tune/
- CoLog Paper: [Include citation if available]

## License

Part of the CoLog project. See main project README for license information.

## Authors

Developed for the CoLog log anomaly detection system as part of thesis research.

## Version

Current version: 1.0.0

Last updated: December 2025
