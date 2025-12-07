"""
layer_norm.py — Layer Normalization for Neural Networks

This module provides a LayerNorm implementation for normalizing activations
across features. Layer normalization is a technique to normalize the inputs
across the features for each sample in a batch, which helps stabilize training
and improve convergence in deep neural networks.

The implementation follows the layer normalization technique described in:
Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization.
arXiv preprint arXiv:1607.06450.

Classes
-------
LayerNorm
    A PyTorch module implementing layer normalization with learnable affine
    parameters (scale and shift).
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """
    Layer Normalization module with learnable affine transformation.

    This module applies layer normalization to the input tensor, normalizing
    across the feature dimension. It includes learnable scale (gamma) and
    shift (beta) parameters for affine transformation after normalization.

    The normalization is computed as:
        y = gamma * (x - mean) / sqrt(variance + eps) + beta

    where mean and variance are computed across the last dimension (features).

    Parameters
    ----------
    size : int
        The size of the feature dimension to normalize. Must be positive.
    eps : float, optional
        A small constant added to the variance for numerical stability.
        Default is 1e-6.

    Attributes
    ----------
    eps : float
        Epsilon value for numerical stability.
    gamma : torch.nn.Parameter
        Learnable scale parameter, initialized to ones.
    beta : torch.nn.Parameter
        Learnable shift parameter, initialized to zeros.

    Examples
    --------
    >>> import torch
    >>> layer_norm = LayerNorm(size=512)
    >>> x = torch.randn(32, 10, 512)  # (batch, sequence, features)
    >>> output = layer_norm(x)
    >>> output.shape
    torch.Size([32, 10, 512])

    Notes
    -----
    - Unlike Batch Normalization, Layer Normalization does not maintain
      running statistics and performs the same computation at training and
      inference time.
    - The normalization is computed independently for each sample in the batch.
    - This implementation is suitable for sequence models like Transformers
      where batch size may vary.

    References
    ----------
    .. [1] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization.
           arXiv preprint arXiv:1607.06450.
    """

    def __init__(self, size: int, eps: Optional[float] = 1e-6) -> None:
        """
        Initialize LayerNorm module.

        Parameters
        ----------
        size : int
            The size of the feature dimension to normalize. Must be positive.
        eps : Optional[float], optional
            A small constant for numerical stability. Default is 1e-6.

        Raises
        ------
        ValueError
            If size is not a positive integer.
        TypeError
            If size is not an integer or eps is not a float.
        """
        super(LayerNorm, self).__init__()

        # Validate inputs
        if not isinstance(size, int):
            raise TypeError(f"size must be an integer, got {type(size).__name__}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be a number, got {type(eps).__name__}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.eps = eps

        # Learnable affine parameters
        # gamma (scale parameter) initialized to ones
        self.gamma = nn.Parameter(torch.ones(size))
        # beta (shift parameter) initialized to zeros
        self.beta = nn.Parameter(torch.zeros(size))

        logger.debug(f"Initialized LayerNorm with size={size}, eps={eps}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., size) where the last dimension
            matches the size parameter specified during initialization.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as input, after applying
            learnable affine transformation.

        Raises
        ------
        ValueError
            If the last dimension of input does not match the expected size.
        TypeError
            If input is not a torch.Tensor.

        Notes
        -----
        The normalization is performed as follows:
        1. Compute mean and standard deviation across the last dimension
        2. Normalize: (x - mean) / (std + eps)
        3. Apply affine transformation: gamma * normalized + beta
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x).__name__}")

        if x.shape[-1] != self.gamma.shape[0]:
            raise ValueError(
                f"Input last dimension ({x.shape[-1]}) does not match "
                f"LayerNorm size ({self.gamma.shape[0]})"
            )

        # Compute statistics across the last dimension (features)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Normalize and apply affine transformation
        normalized = (x - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta

        return output

    def extra_repr(self) -> str:
        """
        Return extra representation string for the module.

        Returns
        -------
        str
            String representation containing size and eps parameters.
        """
        return f"size={self.gamma.shape[0]}, eps={self.eps}"

    def reset_parameters(self) -> None:
        """
        Reset learnable parameters to their initial values.

        This method resets gamma to ones and beta to zeros, which is the
        default initialization. Useful for re-initializing the layer during
        experiments or transfer learning scenarios.

        Returns
        -------
        None
        """
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        logger.debug("Reset LayerNorm parameters to initial values")
