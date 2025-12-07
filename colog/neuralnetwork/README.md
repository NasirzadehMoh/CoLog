# Neural Network Module

This module provides Transformer-based neural network architectures for CoLog's log anomaly detection system. It implements both single-modal and multi-modal transformer models designed to capture complex patterns in system log sequences through advanced attention mechanisms and collaborative learning.

## Overview

The neural network module contains state-of-the-art deep learning architectures that process log sequences to detect anomalies. The models combine LSTM-based temporal encoding with multi-head attention mechanisms, enabling both sequential pattern recognition and long-range dependency modeling. The collaborative variant introduces cross-modal attention for learning from multiple feature representations simultaneously.

### Key Features

- **Multiple architectures**: Vanilla and collaborative transformer variants
- **Multi-modal learning**: Process text and sequential features simultaneously
- **Collaborative attention**: Cross-modal information exchange and refinement
- **Hyperparameter tuning support**: Tunable variant for automated optimization
- **Pre-trained embeddings**: Support for loading and fine-tuning word embeddings
- **Modular design**: Reusable components for easy experimentation
- **Automatic masking**: Built-in padding mask computation
- **Regularization**: Dropout and layer normalization for stable training

## Directory Structure

```
neuralnetwork/
├── __init__.py                              # Package initialization and exports
├── vanilla_transformer.py                   # Single-modal transformer architecture
├── collaborative_transformer.py             # Multi-modal collaborative transformer
├── collaborative_transformer_tunable.py     # Tunable variant for hyperparameter optimization
└── components/                              # Reusable neural network components
    ├── __init__.py
    ├── layers.py                            # Fully connected and MLP layers
    └── layer_norm.py                        # Layer normalization implementation
```

## Model Architectures

### 1. Vanilla Transformer

A single-modal transformer architecture for processing log sequences with LSTM encoding and multi-head self-attention.

**Architecture Components:**
- **Embedding Layer**: Maps token IDs to dense vector representations
- **LSTM Encoder**: Captures sequential dependencies in embedded messages
- **Multi-Head Self-Attention**: Models relationships between sequence positions
- **Attention-Based Pooling**: Aggregates sequence into fixed-size representation
- **Classification Head**: Projects features to anomaly class logits

**Input:**
- Text token IDs: `(batch_size, seq_len)`

**Output:**
- Logits: `(batch_size, n_classes)` - typically 2 for binary anomaly detection

**Use Case:**
- Baseline model for log anomaly detection
- Single-feature learning scenarios
- Comparison benchmark for multi-modal approaches

### 2. Collaborative Transformer

A multi-modal transformer architecture with collaborative attention mechanisms for processing both textual content and sequential features.

**Architecture Components:**
- **Dual Embedding Pathways**: Separate processing for text and sequence features
  - Text pathway: Embedding → LSTM encoder
  - Sequence pathway: Linear projection
- **Collaborative Attention Blocks**: Cross-modal attention with multiple layers
  - Multi-head impressed attention (bidirectional cross-modal)
  - Modality adaptation layers (multi-glimpse attention)
  - Residual connections and layer normalization
- **Feature Fusion**: Combines adapted modalities
- **Classification Head**: Projects fused features to class logits

**Input:**
- Text token IDs: `(batch_size, len_messages)`
- Pre-computed embeddings: `(batch_size, len_messages, embed_size)`
- Sequence features: `(batch_size, len_sequences, sequences_fsize)`

**Output:**
- Logits: `(batch_size, n_classes)`

**Key Mechanisms:**
1. **Impressed Attention**: Bidirectional cross-modal attention
   - Text attends to sequences (text-impressed)
   - Sequences attend to text (sequence-impressed)
2. **Modality Adaptation**: Multi-glimpse attention for feature refinement
   - Creates multiple weighted views of each modality
   - Enables focusing on different aspects simultaneously
3. **Collaborative Fusion**: Merges adapted representations for classification

**Use Case:**
- Multi-modal log anomaly detection
- Scenarios with rich sequential features
- Production deployment with fixed hyperparameters

### 3. Collaborative Transformer Tunable

A hyperparameter-tunable variant of the collaborative transformer designed for automated hyperparameter optimization.

**Differences from Standard Collaborative Transformer:**
- **String-based configuration**: All hyperparameters specified as strings
- **Runtime evaluation**: Parameters evaluated using `eval()` during initialization
- **Optimization framework compatibility**: Seamless integration with Optuna, Ray Tune, Hyperopt
- **Dynamic architecture**: Model structure determined at instantiation time

**Tunable Parameters:**
- `hidden_size`: Hidden state dimension (e.g., `'256'`, `'512'`)
- `dropout_rate`: Dropout probability (e.g., `'0.1'`, `'0.3'`)
- `heads`: Number of attention heads (e.g., `'4'`, `'8'`)
- `layers`: Number of collaborative blocks (e.g., `'2'`, `'4'`)

**Use Case:**
- Hyperparameter search and optimization
- Automated architecture exploration
- Finding optimal configurations for new datasets

## Usage

### Basic Usage - Vanilla Transformer

```python
from neuralnetwork import VanillaTransformer
import torch

# Configuration
from argparse import Namespace
args = Namespace(
    embed_size=300,
    hidden_size=256,
    projection_size=2048,
    dropout_rate=0.1,
    heads=4,
    layers=2,
    len_messages=10,
    n_classes=2
)

# Initialize model
model = VanillaTransformer(
    args=args,
    vocab_size=50000,
    pretrained_embeddings=embeddings_matrix  # Shape: (vocab_size, embed_size)
)

# Forward pass
text_input = torch.randint(0, 50000, (32, 10))  # (batch, seq_len)
logits = model(text_input, None, None)
predictions = torch.argmax(logits, dim=-1)
```

### Basic Usage - Collaborative Transformer

```python
from neuralnetwork import CollaborativeTransformer
import torch

# Configuration
args = Namespace(
    embed_size=300,
    sequences_fsize=384,
    hidden_size=256,
    projection_size=2048,
    dropout_rate=0.1,
    heads=4,
    layers=2,
    len_messages=10,
    len_sequences=5,
    n_classes=2
)

# Initialize model
model = CollaborativeTransformer(
    args=args,
    vocab_size=50000,
    pretrained_embeddings=embeddings_matrix
)

# Forward pass
text_input = torch.randint(0, 50000, (32, 10))      # (batch, len_messages)
embeddings = torch.randn(32, 10, 300)                # (batch, len_messages, embed_size)
sequences = torch.randn(32, 5, 384)                  # (batch, len_sequences, sequences_fsize)

logits = model(text_input, embeddings, sequences)
predictions = torch.argmax(logits, dim=-1)
```

### Hyperparameter Tuning with Ray Tune

```python
from neuralnetwork import CollaborativeTransformerTunable
from ray import tune
from functools import partial

# Define tunable configuration
def create_model(config, vocab_size, embeddings):
    from argparse import Namespace
    args = Namespace(
        hidden_size=str(config['hidden_size']),
        dropout_rate=str(config['dropout_rate']),
        heads=str(config['heads']),
        layers=str(config['layers']),
        projection_size=2048,
        embed_size=300,
        sequences_fsize=384,
        len_messages=10,
        len_sequences=5,
        n_classes=2
    )
    
    return CollaborativeTransformerTunable(
        config=None,
        args=args,
        vocab_size=vocab_size,
        pretrained_embeddings=embeddings
    )

# Search space
config = {
    'hidden_size': tune.choice([128, 256, 512]),
    'dropout_rate': tune.uniform(0.1, 0.5),
    'heads': tune.choice([2, 4, 8]),
    'layers': tune.choice([1, 2, 4])
}

# Run tuning
analysis = tune.run(
    partial(train_function, create_model=create_model),
    config=config,
    num_samples=50
)
```

### Training Example

```python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Initialize model
model = CollaborativeTransformer(args, vocab_size, embeddings)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        text, embeddings, sequences, labels = batch
        
        # Forward pass
        logits = model(text, embeddings, sequences)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

### Inference Example

```python
# Load trained model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    for batch in test_loader:
        text, embeddings, sequences, _ = batch
        
        # Get predictions
        logits = model(text, embeddings, sequences)
        predictions = torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        
        # Anomaly scores (probability of anomaly class)
        anomaly_scores = probabilities[:, 0]  # Assuming class 0 is anomaly
```

## Configuration Parameters

### Required Parameters (All Models)

| Parameter | Type | Description | Typical Range |
|-----------|------|-------------|---------------|
| `embed_size` | int | Word embedding dimension | 100-768 |
| `hidden_size` | int | Hidden state dimension | 128-1024 |
| `projection_size` | int | Feedforward projection dimension | 512-4096 |
| `dropout_rate` | float | Dropout probability | 0.0-0.5 |
| `heads` | int | Number of attention heads | 2-16 |
| `layers` | int | Number of transformer blocks | 1-12 |
| `len_messages` | int | Sequence length for messages | Dataset-specific |
| `n_classes` | int | Number of output classes | 2 (binary) |

### Additional Parameters (Collaborative Models)

| Parameter | Type | Description | Typical Range |
|-----------|------|-------------|---------------|
| `sequences_fsize` | int | Sequence feature dimension | 128-512 |
| `len_sequences` | int | Number of sequence features | Dataset-specific |

### Initialization Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab_size` | int | Vocabulary size for embedding layer |
| `pretrained_embeddings` | tensor | Pre-trained word embeddings (optional) |

## Component Modules

### Layer Components (`components/layers.py`)

#### FC (Fully Connected Layer)
Single linear layer with optional ReLU and dropout.

```python
from neuralnetwork.components import FC

layer = FC(
    in_size=256,
    out_size=128,
    dropout_rate=0.1,
    use_relu=True
)
output = layer(input)  # (batch, 256) -> (batch, 128)
```

#### MLP (Multi-Layer Perceptron)
Two-layer feedforward network.

```python
from neuralnetwork.components import MLP

mlp = MLP(
    in_size=256,
    mid_size=1024,
    out_size=256,
    dropout_rate=0.1,
    use_relu=True
)
output = mlp(input)  # (batch, 256) -> (batch, 256)
```

### Layer Normalization (`components/layer_norm.py`)

```python
from neuralnetwork.components import LayerNorm

norm = LayerNorm(size=512, eps=1e-6)
output = norm(input)  # Normalizes across feature dimension
```

## Architecture Details

### Vanilla Transformer Architecture

```
Input (token IDs)
    ↓
Embedding Layer (vocab_size → embed_size)
    ↓
LSTM Encoder (embed_size → hidden_size)
    ↓
Multi-Head Self-Attention Block ×N
    ├── Multi-Head Attention
    ├── Residual + LayerNorm
    ├── Feedforward Network
    └── Residual + LayerNorm
    ↓
Attention-Based Flattening (seq → fixed size)
    ↓
Classification Head (2×hidden_size → n_classes)
    ↓
Logits
```

### Collaborative Transformer Architecture

```
Text Input                    Sequence Input
    ↓                              ↓
Embedding                    Linear Projection
    ↓                              ↓
LSTM Encoder                      │
    ↓                              ↓
    ─────────────────────────────────
              ↓
    Collaborative Block ×N
    ├── Multi-Head Impressed Attention
    │   ├── Text → Sequences (text-impressed)
    │   └── Sequences → Text (seq-impressed)
    ├── Residual + LayerNorm
    ├── Modality Adaptation Layers
    │   ├── Multi-Glimpse Attention (Text)
    │   └── Multi-Glimpse Attention (Sequences)
    ├── Feedforward Networks
    └── Residual + LayerNorm
              ↓
    Feature Fusion (Concatenation)
              ↓
    Classification Head
              ↓
          Logits
```

### Attention Mechanisms

#### Multi-Head Self-Attention
Standard scaled dot-product attention with multiple heads:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

- Parallel computation across multiple heads
- Each head learns different relationship patterns
- Outputs are concatenated and projected

#### Multi-Head Impressed Attention (Collaborative)
Bidirectional cross-modal attention:

```
Text-Impressed: Attention(Q_text, K_seq, V_seq)
Seq-Impressed:  Attention(Q_seq, K_text, V_text)
```

- Enables information flow between modalities
- Each modality attends to the other
- Residual connections preserve original features

#### Multi-Glimpse Attention (Modality Adaptation)
Multiple weighted views of sequence features:

```
For each glimpse i:
    α_i = softmax(MLP(x))
    glimpse_i = Σ(α_i × x)
Output = [glimpse_1, glimpse_2, ..., glimpse_k]
```

- Creates k different attention-weighted summaries
- Similar to multi-head but operates on sequence dimension
- Provides diverse perspectives on the same features

## Padding and Masking

All models automatically compute padding masks from input features:

```python
def create_padding_mask(feature):
    """
    Returns True for positions that are all zeros (padding).
    Shape: (batch, 1, 1, seq_len) for broadcasting with attention scores.
    """
    return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)
```

Padding positions are:
- Masked with large negative values (-1e9) before softmax
- Effectively excluded from attention computation
- Not included in gradient updates

## Model Comparison

| Feature | Vanilla | Collaborative | Tunable |
|---------|---------|---------------|---------|
| Input Modalities | Text only | Text + Sequences | Text + Sequences |
| Attention Type | Self-attention | Cross-modal + Self | Cross-modal + Self |
| Modality Adaptation | No | Yes (multi-glimpse) | Yes (multi-glimpse) |
| Hyperparameter Tuning | Manual | Manual | Automated |
| Parameters | ~1-10M | ~2-20M | ~2-20M (variable) |
| Training Speed | Fastest | Moderate | Moderate |
| Performance | Good | Better | Better (optimized) |
| Use Case | Baseline, single-modal | Production, multi-modal | Hyperparameter search |

## Performance Considerations

### Memory Usage
- **Vanilla**: O(L × H × N) where L=layers, H=hidden_size, N=seq_len
- **Collaborative**: ~2× Vanilla due to dual pathways
- **Batch size**: Reduce if OOM errors occur
- **Gradient accumulation**: Simulate larger batches with less memory

### Computational Complexity
- **Self-Attention**: O(N² × H) - quadratic in sequence length
- **Cross-Attention**: O(N × M × H) - product of sequence lengths
- **LSTM**: O(N × H²) - linear in sequence length
- **Total per forward pass**: Dominated by attention mechanisms

### Optimization Tips

1. **Use GPU acceleration**: 10-100× speedup over CPU
   ```python
   model = model.to('cuda')
   ```

2. **Enable mixed precision training**: ~2× speedup, 50% memory reduction
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       logits = model(text, embeddings, sequences)
   ```

3. **Gradient accumulation**: Larger effective batch size
   ```python
   for i, batch in enumerate(train_loader):
       loss = criterion(model(*batch[:-1]), batch[-1])
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **Freeze embeddings**: Reduce trainable parameters
   ```python
   model.embedding.weight.requires_grad = False
   ```

5. **Use DataParallel**: Multi-GPU training
   ```python
   model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
   ```

## Common Issues and Solutions

### Issue: Out of Memory (OOM)

**Solutions:**
- Reduce batch size
- Reduce `hidden_size` or `projection_size`
- Enable gradient checkpointing
- Use mixed precision training
- Reduce sequence length (`len_messages`, `len_sequences`)

### Issue: Slow Training

**Solutions:**
- Use GPU instead of CPU
- Enable mixed precision training
- Increase batch size (if memory allows)
- Use DataParallel for multi-GPU
- Profile code to identify bottlenecks

### Issue: Poor Convergence

**Solutions:**
- Adjust learning rate (try 1e-5 to 1e-3)
- Increase number of epochs
- Use learning rate scheduling
- Increase model capacity (more layers/hidden units)
- Check for label imbalance (use weighted loss)
- Verify data preprocessing

### Issue: Overfitting

**Solutions:**
- Increase dropout rate
- Add weight decay to optimizer
- Use early stopping
- Increase training data
- Apply data augmentation
- Reduce model capacity

### Issue: Dimension Mismatch Errors

**Solutions:**
- Verify `hidden_size` is divisible by `heads`
- Check input shapes match expected dimensions
- Ensure embeddings have correct `embed_size`
- Verify sequence features have correct `sequences_fsize`

## Integration with Training Pipeline

The models are designed to work with the CoLog training pipeline:

```python
# From train_colog.py
from neuralnetwork import CollaborativeTransformer
from train.utils.groundtruth_loader import GroundTruthLoader
from train.main import train

# Load data
train_loader = GroundTruthLoader(
    dataset='hadoop',
    batch_size=32,
    split='train'
)

# Initialize model
model = CollaborativeTransformer(args, vocab_size, embeddings)

# Train
train(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    num_epochs=50,
    device='cuda'
)
```

## Model Saving and Loading

### Save Model

```python
# Save entire model
torch.save(model.state_dict(), 'model_weights.pth')

# Save with optimizer state
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

### Load Model

```python
# Load weights only
model.load_state_dict(torch.load('model_weights.pth'))

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Extending the Architecture

### Adding Custom Layers

```python
from neuralnetwork.components import MLP, LayerNorm

class CustomTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.norm1 = LayerNorm(args.hidden_size)
        self.ffn = MLP(args.hidden_size, args.projection_size, args.hidden_size)
        self.norm2 = LayerNorm(args.hidden_size)
        
    def forward(self, x, mask):
        # Self-attention with residual
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + attended)
        
        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

### Custom Model Architecture

```python
from neuralnetwork import CollaborativeTransformer

class CustomCoLog(CollaborativeTransformer):
    def __init__(self, args, vocab_size, pretrained_embeddings):
        super().__init__(args, vocab_size, pretrained_embeddings)
        
        # Add custom components
        self.custom_layer = nn.Linear(args.hidden_size, args.hidden_size)
    
    def forward(self, text, embeddings, sequences):
        # Use parent forward pass
        logits = super().forward(text, embeddings, sequences)
        
        # Apply custom processing
        # ...
        
        return logits
```

## Dependencies

- **PyTorch** (`torch`): Deep learning framework
- **NumPy** (`numpy`): Numerical operations
- **Logging**: Built-in Python logging

Install dependencies:
```bash
pip install torch numpy
```

For hyperparameter tuning:
```bash
pip install ray[tune] optuna
```

## Testing

### Unit Testing Example

```python
import unittest
import torch
from neuralnetwork import VanillaTransformer

class TestVanillaTransformer(unittest.TestCase):
    def setUp(self):
        from argparse import Namespace
        self.args = Namespace(
            embed_size=128,
            hidden_size=64,
            projection_size=256,
            dropout_rate=0.1,
            heads=4,
            layers=2,
            len_messages=5,
            n_classes=2
        )
        self.vocab_size = 1000
        self.embeddings = torch.randn(self.vocab_size, self.args.embed_size)
    
    def test_forward_pass(self):
        model = VanillaTransformer(self.args, self.vocab_size, self.embeddings)
        model.eval()
        
        text = torch.randint(0, self.vocab_size, (8, self.args.len_messages))
        
        with torch.no_grad():
            logits = model(text, None, None)
        
        self.assertEqual(logits.shape, (8, self.args.n_classes))
```

## Related Modules

- **`groundtruth/`**: Data preprocessing and ground truth generation
- **`train/`**: Training pipeline and utilities
- **`utils/`**: Metrics and evaluation utilities

## References

### Papers

1. **Transformer Architecture**
   - Vaswani et al. (2017). "Attention Is All You Need". NeurIPS.

2. **Layer Normalization**
   - Ba et al. (2016). "Layer Normalization". arXiv:1607.06450.

3. **LSTM Networks**
   - Hochreiter & Schmidhuber (1997). "Long Short-Term Memory". Neural Computation.

4. **Multi-Modal Learning**
   - Baltrusaitis et al. (2019). "Multimodal Machine Learning: A Survey and Taxonomy". IEEE TPAMI.

### Related Work

- Log anomaly detection with deep learning
- Cross-modal attention mechanisms
- Transformer-based sequence classification

## License

Part of the CoLog project. See main project README for license information.

## Authors

Developed for the CoLog log anomaly detection system as part of thesis research.

## Version

Current version: 1.0.0

Last updated: December 2025
