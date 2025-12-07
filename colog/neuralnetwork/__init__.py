"""
neuralnetwork package — Neural network architectures for CoLog log anomaly detection

This package provides Transformer-based neural network architectures designed for
detecting anomalies in system log sequences. It includes both vanilla and collaborative
transformer implementations, along with modular layer components.

The package supports multi-modal learning, combining textual log messages with
sequential features to capture complex patterns in system logs.

Modules
-------
vanilla_transformer
    Single-modal Transformer architecture using self-attention mechanisms
    for processing log sequences with LSTM-based feature extraction.

collaborative_transformer
    Multi-modal Transformer architecture with collaborative attention
    mechanisms for processing both text and sequence features.

collaborative_transformer_tunable
    Hyperparameter-tunable version of the collaborative transformer,
    designed for integration with optimization frameworks like Optuna.

components
    Subpackage containing reusable neural network layer components
    including layer normalization, fully connected layers, and MLPs.

Classes
-------
VanillaTransformer
    Transformer-based model with LSTM encoding and multi-head self-attention
    for single-modal log anomaly detection.

CollaborativeTransformer
    Multi-modal transformer with cross-modal attention, enabling collaborative
    learning between textual and sequential features.

CollaborativeTransformerTunable
    Tunable version of the collaborative transformer with string-based
    hyperparameter configuration for automated hyperparameter search.

Usage
-----
Import model architectures for training or inference:

    from neuralnetwork import VanillaTransformer, CollaborativeTransformer
    
    # Initialize vanilla transformer
    model = VanillaTransformer(
        args=config,
        vocab_size=50000,
        pretrained_embeddings=embeddings_matrix
    )
    
    # Initialize collaborative transformer
    model = CollaborativeTransformer(
        args=config,
        vocab_size=50000,
        pretrained_embeddings=embeddings_matrix
    )

For hyperparameter tuning:

    from neuralnetwork import CollaborativeTransformerTunable
    
    # Configuration with tunable parameters (as strings)
    config.hidden_size = '512'
    config.dropout_rate = '0.1'
    
    model = CollaborativeTransformerTunable(
        args=config,
        vocab_size=50000,
        pretrained_embeddings=embeddings_matrix
    )

Notes
-----
- All models are PyTorch nn.Module subclasses
- Pre-trained embeddings can be loaded and optionally frozen
- Models support automatic padding mask computation
- Dropout and layer normalization are applied for regularization

See Also
--------
components : Neural network layer building blocks
torch.nn.Module : Base class for all neural network modules
"""

from .vanilla_transformer import VanillaTransformer
from .collaborative_transformer import CollaborativeTransformer, TextEmbedding
from .collaborative_transformer_tunable import CollaborativeTransformerTunable
from . import constants

__all__ = [
    'VanillaTransformer',
    'CollaborativeTransformer',
    'TextEmbedding',
    'CollaborativeTransformerTunable',
    'constants',
]

__version__ = '1.0.0'
