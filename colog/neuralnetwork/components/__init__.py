"""
layers package — Neural network layer components for CoLog

This package contains custom neural network layer implementations used in the
CoLog transformer architecture. These modular, reusable components form the
foundation for building deep learning models for log anomaly detection.

Modules
-------
layer_norm
    Layer normalization implementation with learnable affine transformation.
layers
    Fully connected layers and multi-layer perceptron implementations.

Classes
-------
LayerNorm
    Layer normalization with learnable scale (gamma) and shift (beta) parameters.
    Normalizes activations across features for improved training stability.

FC
    Fully connected (linear) layer with optional ReLU activation and dropout
    regularization. Provides flexible building blocks for feedforward networks.

MLP
    Multi-layer perceptron (two-layer feedforward network). Combines FC layers
    to create deeper representations with increased model capacity.

Usage
-----
Import these layers to construct custom neural network architectures:

    from neuralnetwork.layers import LayerNorm, FC, MLP
    
    # Create a simple feedforward block
    fc_layer = FC(in_size=512, out_size=256, dropout_rate=0.1)
    mlp_layer = MLP(in_size=512, mid_size=2048, out_size=512, dropout_rate=0.1)
    layer_norm = LayerNorm(size=512)

Notes
-----
All layer classes are subclasses of torch.nn.Module and integrate seamlessly
with PyTorch's automatic differentiation and optimization frameworks. They
support standard training/evaluation modes and GPU acceleration.

See Also
--------
torch.nn.Module : Base class for all neural network modules in PyTorch.
"""

from .layer_norm import LayerNorm
from .layers import FC, MLP

__all__ = [
    'LayerNorm',
    'FC',
    'MLP',
]

__version__ = '1.0.0'
