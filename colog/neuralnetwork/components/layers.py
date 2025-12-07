"""
layers.py — Neural network layer components for CoLog

This module contains custom neural network layer implementations used in the
CoLog transformer architecture. These layers form the building blocks for
feedforward networks and multi-layer perceptrons in the model.

The module provides two main components:
- FC (Fully Connected): A single fully connected layer with optional ReLU
  activation and dropout regularization.
- MLP (Multi-Layer Perceptron): A two-layer feedforward network combining
  FC layers for increased representation capacity.

These layers are designed to be modular, configurable, and compatible with
PyTorch's nn.Module ecosystem.

Classes
-------
FC
    Fully connected layer with optional activation and dropout.
MLP
    Two-layer feedforward network (multi-layer perceptron).

Notes
-----
All layers support gradient-based optimization and can be integrated into
larger neural network architectures. Dropout and activation functions are
configurable to accommodate different training scenarios.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FC(nn.Module):
    """
    Fully connected (linear) layer with optional ReLU activation and dropout.

    This class implements a single fully connected layer that can optionally
    include ReLU activation and dropout regularization. It provides flexibility
    for constructing feedforward networks with different configurations.

    The forward pass computation is:
        x = Linear(x)
        x = ReLU(x)      [if use_relu=True]
        x = Dropout(x)   [if dropout_rate > 0]

    Parameters
    ----------
    in_size : int
        Input feature dimensionality. Must be positive.
    out_size : int
        Output feature dimensionality. Must be positive.
    dropout_rate : float, optional
        Dropout probability in range [0, 1). Default is 0.0 (no dropout).
    use_relu : bool, optional
        Whether to apply ReLU activation after linear transformation.
        Default is True.

    Attributes
    ----------
    dropout_rate : float
        Stored dropout rate for conditional dropout application.
    use_relu : bool
        Stored flag for conditional ReLU application.
    linear : nn.Linear
        Linear transformation layer.
    relu : nn.ReLU, optional
        ReLU activation function (only if use_relu=True).
    dropout : nn.Dropout, optional
        Dropout layer (only if dropout_rate > 0).

    Examples
    --------
    >>> import torch
    >>> fc = FC(in_size=256, out_size=128, dropout_rate=0.1, use_relu=True)
    >>> x = torch.randn(32, 256)
    >>> output = fc(x)
    >>> output.shape
    torch.Size([32, 128])

    >>> # Without ReLU and dropout
    >>> fc_linear = FC(in_size=128, out_size=64, dropout_rate=0.0, use_relu=False)
    >>> x = torch.randn(16, 128)
    >>> output = fc_linear(x)
    >>> output.shape
    torch.Size([16, 64])

    Notes
    -----
    - ReLU activation is applied in-place for memory efficiency.
    - Dropout is applied during training and automatically disabled during
      evaluation mode.
    - This layer is commonly used as a building block in transformer
      feedforward networks and MLPs.

    Raises
    ------
    ValueError
        If in_size or out_size is not positive.
        If dropout_rate is not in range [0, 1).
    """

    def __init__(self, in_size: int, out_size: int, dropout_rate: float = 0.0,
                 use_relu: bool = True) -> None:
        """
        Initialize FC layer.

        Parameters
        ----------
        in_size : int
            Input feature dimensionality. Must be positive.
        out_size : int
            Output feature dimensionality. Must be positive.
        dropout_rate : float, optional
            Dropout probability. Default is 0.0.
        use_relu : bool, optional
            Whether to use ReLU activation. Default is True.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        super(FC, self).__init__()

        # Validate inputs
        if not isinstance(in_size, int) or in_size <= 0:
            raise ValueError(f"in_size must be a positive integer, got {in_size}")
        if not isinstance(out_size, int) or out_size <= 0:
            raise ValueError(f"out_size must be a positive integer, got {out_size}")
        if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in range [0, 1), got {dropout_rate}")
        if not isinstance(use_relu, bool):
            raise ValueError(f"use_relu must be a boolean, got {type(use_relu)}")

        self.dropout_rate = dropout_rate
        self.use_relu = use_relu

        # Linear transformation
        self.linear = nn.Linear(in_size, out_size)

        # Optional ReLU activation
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        # Optional dropout regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        logger.debug('Initialized FC layer: in_size=%d, out_size=%d, dropout=%.2f, relu=%s',
                     in_size, out_size, dropout_rate, use_relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fully connected layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., in_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, ..., out_size).

        Notes
        -----
        The forward pass applies transformations in this order:
        1. Linear transformation
        2. ReLU activation (if enabled)
        3. Dropout (if enabled and in training mode)
        """
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron (two-layer feedforward network).

    This class implements a two-layer feedforward neural network, commonly
    used in transformer architectures and other deep learning models. The MLP
    consists of an FC layer with activation/dropout, followed by a linear
    projection to the output dimensionality.

    The forward pass computation is:
        x = FC(x)           [first layer with optional ReLU and dropout]
        x = Linear(x)       [second layer projection]

    Parameters
    ----------
    in_size : int
        Input feature dimensionality. Must be positive.
    mid_size : int
        Hidden layer dimensionality. Must be positive.
    out_size : int
        Output feature dimensionality. Must be positive.
    dropout_rate : float, optional
        Dropout probability applied in the first layer. Default is 0.0.
    use_relu : bool, optional
        Whether to apply ReLU activation in the first layer. Default is True.

    Attributes
    ----------
    fc : FC
        First fully connected layer with optional ReLU and dropout.
    linear : nn.Linear
        Second linear layer for output projection.

    Examples
    --------
    >>> import torch
    >>> mlp = MLP(in_size=512, mid_size=2048, out_size=512, dropout_rate=0.1)
    >>> x = torch.randn(32, 10, 512)
    >>> output = mlp(x)
    >>> output.shape
    torch.Size([32, 10, 512])

    >>> # MLP without dropout or ReLU
    >>> mlp_linear = MLP(in_size=256, mid_size=1024, out_size=256,
    ...                  dropout_rate=0.0, use_relu=False)
    >>> x = torch.randn(16, 256)
    >>> output = mlp_linear(x)
    >>> output.shape
    torch.Size([16, 256])

    Notes
    -----
    - This is a standard feedforward network structure used in transformer
      architectures (e.g., BERT, GPT) after self-attention layers.
    - The hidden dimension (mid_size) is typically larger than input/output
      dimensions to increase model capacity.
    - Dropout is only applied in the first layer, not the output layer.

    Raises
    ------
    ValueError
        If any size parameter is not positive.
        If dropout_rate is not in valid range [0, 1).
    """

    def __init__(self, in_size: int, mid_size: int, out_size: int,
                 dropout_rate: float = 0.0, use_relu: bool = True) -> None:
        """
        Initialize MLP.

        Parameters
        ----------
        in_size : int
            Input feature dimensionality. Must be positive.
        mid_size : int
            Hidden layer dimensionality. Must be positive.
        out_size : int
            Output feature dimensionality. Must be positive.
        dropout_rate : float, optional
            Dropout probability. Default is 0.0.
        use_relu : bool, optional
            Whether to use ReLU activation. Default is True.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        super(MLP, self).__init__()

        # Validate inputs (FC layer will validate in_size, mid_size, dropout_rate)
        if not isinstance(out_size, int) or out_size <= 0:
            raise ValueError(f"out_size must be a positive integer, got {out_size}")

        # First layer: FC with optional ReLU and dropout
        self.fc = FC(in_size, mid_size, dropout_rate=dropout_rate, use_relu=use_relu)

        # Second layer: Linear projection to output size
        self.linear = nn.Linear(mid_size, out_size)

        logger.debug('Initialized MLP: in_size=%d, mid_size=%d, out_size=%d, dropout=%.2f, relu=%s',
                     in_size, mid_size, out_size, dropout_rate, use_relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-layer perceptron.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, ..., in_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, ..., out_size).

        Notes
        -----
        The computation flow:
        1. First FC layer (with ReLU and dropout if configured)
        2. Second linear layer (output projection)
        """
        return self.linear(self.fc(x))
