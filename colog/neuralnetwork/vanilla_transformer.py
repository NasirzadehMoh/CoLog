"""
vanilla_transformer.py — Vanilla Transformer model for CoLog log anomaly detection

This module implements a Transformer-based neural network architecture for
detecting anomalies in system log sequences. The model combines LSTM-based
feature extraction with multi-head self-attention mechanisms to capture both
sequential dependencies and long-range relationships in log data.

Key architectural components:
    - Embedding layer: Maps log token IDs to dense vector representations
    - LSTM encoder: Extracts sequential features from embedded log messages
    - Multi-head self-attention: Captures relationships between different
      positions in the log sequence
    - Attention-based flattening: Aggregates sequence representations into
      fixed-size vectors for classification
    - Classification head: Projects flattened features to anomaly class logits

The model architecture follows the Transformer encoder design with modifications
specific to log anomaly detection:
    1. Pre-trained embeddings are loaded and frozen/fine-tuned based on config
    2. LSTM processes embedded sequences to capture temporal dependencies
    3. Multiple self-attention layers refine representations
    4. Attention-based pooling creates sequence-level representations
    5. Classification head produces final anomaly predictions

Usage
-----
    from neuralnetwork.vanilla_transformer import VanillaTransformer
    
    # Initialize model
    model = VanillaTransformer(
        args=config,
        vocab_size=50000,
        pretrained_embeddings=embeddings_matrix
    )
    
    # Forward pass
    logits = model(input_ids, None, None)
    predictions = torch.argmax(logits, dim=-1)

Notes
-----
- The model expects pre-tokenized log sequences as input
- Padding masks are computed automatically from input features
- Dropout is applied throughout for regularization
- Layer normalization improves training stability

See Also
--------
collaborative_transformer.py : Multi-modal transformer with additional features
components.layers : Building blocks for neural network layers
"""

import logging
from typing import Optional
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import MLP, LayerNorm
from . import constants

# Ignore all warnings (comment out during debugging if needed)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def create_padding_mask(feature: torch.Tensor) -> torch.Tensor:
    """
    Create padding mask for input sequences.

    This function generates a boolean mask indicating which positions in the
    input sequence are padding tokens (all zeros). The mask is used to prevent
    attention mechanisms from attending to padding positions.

    Parameters
    ----------
    feature : torch.Tensor
        Input feature tensor of shape (batch_size, seq_len, feature_dim).

    Returns
    -------
    torch.Tensor
        Boolean mask of shape (batch_size, 1, 1, seq_len) where True indicates
        padding positions.

    Notes
    -----
    The mask is unsqueezed to be broadcast-compatible with attention score
    tensors of shape (batch_size, heads, seq_len, seq_len).
    """
    try:
        if feature.dim() < 2:
            raise ValueError(f"Feature tensor must have at least 2 dimensions, got {feature.dim()}")
        
        # Compute the mask by checking which positions are all zeros
        mask = (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
        
        # Edge case: warn if entire batch consists of padding
        if mask.all():
            logger.warning("Entire batch consists of padding (all zero features). "
                         "This may indicate a data loading or preprocessing issue.")
        
        return mask
    except Exception as e:
        logger.error(f"Error creating padding mask: {e}")
        raise


class AttentionFlatten(nn.Module):
    """
    Attention-based sequence flattening module.

    This module reduces a variable-length sequence to a fixed-size vector
    representation using learned attention weights. It computes attention
    scores for each position in the sequence and creates a weighted sum of
    the sequence features.

    The flattening operation:
        1. Compute attention scores for each position using MLP
        2. Mask padding positions with large negative values
        3. Apply softmax to get normalized attention weights
        4. Compute weighted sum of sequence features
        5. Project to output dimensionality

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (int): Hidden dimension size
        - projection_size (int): Intermediate projection dimension
        - dropout_rate (float): Dropout probability

    Attributes
    ----------
    args : argparse.Namespace
        Stored configuration.
    mlp : MLP
        Multi-layer perceptron for computing attention scores.
    output_projection : nn.Linear
        Linear projection to output dimension (2 * hidden_size).

    Notes
    -----
    The output dimension is doubled (2 * hidden_size) to provide additional
    capacity for the classification head.
    """

    def __init__(self, args):
        """Initialize AttentionFlatten module."""
        super(AttentionFlatten, self).__init__()
        try:
            if not hasattr(args, 'hidden_size') or args.hidden_size <= 0:
                raise ValueError(f"Invalid hidden_size: {getattr(args, 'hidden_size', None)}")
            if not hasattr(args, 'projection_size') or args.projection_size <= 0:
                raise ValueError(f"Invalid projection_size: {getattr(args, 'projection_size', None)}")
            if not hasattr(args, 'dropout_rate') or not (0 <= args.dropout_rate < 1):
                raise ValueError(f"Invalid dropout_rate: {getattr(args, 'dropout_rate', None)}")
            
            self.args = args
            logger.debug(f"Initialized AttentionFlatten with hidden_size={args.hidden_size}, projection_size={args.projection_size}")

            self.mlp = MLP(
                in_size=args.hidden_size,
                mid_size=args.projection_size,
                out_size=1,
                dropout_rate=args.dropout_rate,
                use_relu=True
            )

            self.output_projection = nn.Linear(
                args.hidden_size,
                args.hidden_size * 2
            )
        except Exception as e:
            logger.error(f"Error initializing AttentionFlatten: {e}")
            raise

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-based flattening to sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor of shape (batch_size, seq_len, hidden_size).
        x_mask : torch.Tensor
            Padding mask of shape (batch_size, 1, 1, seq_len).

        Returns
        -------
        torch.Tensor
            Flattened features of shape (batch_size, 2 * hidden_size).
        """
        try:
            logger.debug(f"AttentionFlatten forward: input shape={x.shape}")
            
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
            if x.size(-1) != self.args.hidden_size:
                raise ValueError(f"Expected hidden_size={self.args.hidden_size}, got {x.size(-1)}")
            
            attention_weights = self.mlp(x)
            attention_weights = attention_weights.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                constants.ATTENTION_MASK_VALUE
            )
            attention_weights = F.softmax(attention_weights, dim=1)

            # Check for NaN or Inf in attention weights
            if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
                logger.warning("NaN or Inf detected in attention weights")
                attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=1.0, neginf=0.0)

            attended_features_list = []
            for i in range(1):
                attended_features_list.append(
                    torch.sum(attention_weights[:, :, i: i + 1] * x, dim=1)
                )

            attended_features = torch.cat(attended_features_list, dim=1)
            attended_features = self.output_projection(attended_features)
            logger.debug(f"AttentionFlatten output shape={attended_features.shape}")

            return attended_features
        except Exception as e:
            logger.error(f"Error in AttentionFlatten forward pass: {e}")
            raise


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    This module implements the multi-head attention mechanism from "Attention
    is All You Need" (Vaswani et al., 2017). It allows the model to jointly
    attend to information from different representation subspaces at different
    positions.

    The multi-head attention computation:
        1. Project inputs to query, key, and value representations
        2. Split projections into multiple heads
        3. Compute scaled dot-product attention for each head
        4. Concatenate head outputs
        5. Apply final output projection

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (int): Hidden dimension size (must be divisible by heads)
        - heads (int): Number of attention heads
        - dropout_rate (float): Dropout probability

    Attributes
    ----------
    args : argparse.Namespace
        Stored configuration.
    value_projection : nn.Linear
        Linear projection for values.
    key_projection : nn.Linear
        Linear projection for keys.
    query_projection : nn.Linear
        Linear projection for queries.
    output_projection : nn.Linear
        Final linear projection after concatenating heads.
    dropout : nn.Dropout
        Dropout layer applied to attention weights.

    Notes
    -----
    The hidden_size must be divisible by the number of heads to ensure
    equal distribution across attention heads.
    """

    def __init__(self, args):
        """Initialize MultiHeadAttention module."""
        super(MultiHeadAttention, self).__init__()
        try:
            if not hasattr(args, 'hidden_size') or args.hidden_size <= 0:
                raise ValueError(f"Invalid hidden_size: {getattr(args, 'hidden_size', None)}")
            if not hasattr(args, 'heads') or args.heads <= 0:
                raise ValueError(f"Invalid heads: {getattr(args, 'heads', None)}")
            if args.hidden_size % args.heads != 0:
                raise ValueError(f"hidden_size ({args.hidden_size}) must be divisible by heads ({args.heads})")
            
            self.args = args
            logger.debug(f"Initialized MultiHeadAttention with hidden_size={args.hidden_size}, heads={args.heads}")

            self.value_projection = nn.Linear(args.hidden_size, args.hidden_size)
            self.key_projection = nn.Linear(args.hidden_size, args.hidden_size)
            self.query_projection = nn.Linear(args.hidden_size, args.hidden_size)
            self.output_projection = nn.Linear(args.hidden_size, args.hidden_size)

            self.dropout = nn.Dropout(args.dropout_rate)
        except Exception as e:
            logger.error(f"Error initializing MultiHeadAttention: {e}")
            raise

    def forward(self, v: torch.Tensor, k: torch.Tensor, q: torch.Tensor, 
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply multi-head attention.

        Parameters
        ----------
        v : torch.Tensor
            Value tensor of shape (batch_size, seq_len, hidden_size).
        k : torch.Tensor
            Key tensor of shape (batch_size, seq_len, hidden_size).
        q : torch.Tensor
            Query tensor of shape (batch_size, seq_len, hidden_size).
        mask : torch.Tensor, optional
            Attention mask of shape (batch_size, 1, 1, seq_len).

        Returns
        -------
        torch.Tensor
            Attended output of shape (batch_size, seq_len, hidden_size).
        """
        try:
            logger.debug(f"MultiHeadAttention forward: q shape={q.shape}, k shape={k.shape}, v shape={v.shape}")
            
            # Validate input shapes
            if v.shape != k.shape or k.shape != q.shape:
                raise ValueError(f"Shape mismatch: v={v.shape}, k={k.shape}, q={q.shape}")
            if q.size(-1) != self.args.hidden_size:
                raise ValueError(f"Expected hidden_size={self.args.hidden_size}, got {q.size(-1)}")
            
            batch_size = q.size(0)
            v = self.value_projection(v).view(
                batch_size,
                -1,
                self.args.heads,
                int(self.args.hidden_size / self.args.heads)
            ).transpose(1, 2)

            k = self.key_projection(k).view(
                batch_size,
                -1,
                self.args.heads,
                int(self.args.hidden_size / self.args.heads)
            ).transpose(1, 2)

            q = self.query_projection(q).view(
                batch_size,
                -1,
                self.args.heads,
                int(self.args.hidden_size / self.args.heads)
            ).transpose(1, 2)

            attended_output = self.compute_attention(v, k, q, mask)

            attended_output = attended_output.transpose(1, 2).contiguous().view(
                batch_size,
                -1,
                self.args.hidden_size
            )
            attended_output = self.output_projection(attended_output)
            logger.debug(f"MultiHeadAttention output shape={attended_output.shape}")

            return attended_output
        except Exception as e:
            logger.error(f"Error in MultiHeadAttention forward pass: {e}")
            raise

    def compute_attention(self, value: torch.Tensor, key: torch.Tensor, 
                         query: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        This method implements the core attention mechanism:
            Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

        Parameters
        ----------
        value : torch.Tensor
            Value tensor of shape (batch_size, heads, seq_len, head_dim).
        key : torch.Tensor
            Key tensor of shape (batch_size, heads, seq_len, head_dim).
        query : torch.Tensor
            Query tensor of shape (batch_size, heads, seq_len, head_dim).
        mask : torch.Tensor, optional
            Attention mask of shape (batch_size, 1, 1, seq_len).

        Returns
        -------
        torch.Tensor
            Attention output of shape (batch_size, heads, seq_len, head_dim).

        Notes
        -----
        Scaling by sqrt(head_dim) prevents dot products from growing too large,
        which would push softmax into regions with small gradients.
        """
        try:
            head_dim = query.size(-1)
            
            if head_dim == 0:
                raise ValueError("Head dimension cannot be zero")

            attention_scores = torch.matmul(
                query, key.transpose(-2, -1)
            ) / math.sqrt(head_dim)
            
            # Check for NaN or Inf in attention scores
            if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
                logger.warning("NaN or Inf detected in attention scores before masking")
                attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)

            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask, constants.ATTENTION_MASK_VALUE)

            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Check for NaN in attention weights after softmax
            if torch.isnan(attention_weights).any():
                logger.warning("NaN detected in attention weights after softmax")
                attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
            
            attention_weights = self.dropout(attention_weights)

            return torch.matmul(attention_weights, value)
        except Exception as e:
            logger.error(f"Error in compute_attention: {e}")
            raise


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network.

    This module implements the position-wise feed-forward network used in
    Transformer architectures. It consists of two linear transformations
    with a ReLU activation in between:
        FFN(x) = max(0, xW1 + b1)W2 + b2

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (int): Input/output dimension
        - projection_size (int): Intermediate dimension
        - dropout_rate (float): Dropout probability

    Attributes
    ----------
    mlp : MLP
        Two-layer feed-forward network with ReLU activation.

    Notes
    -----
    The intermediate dimension (projection_size) is typically larger than
    the hidden_size to increase model capacity.
    """

    def __init__(self, args):
        """Initialize FeedForwardNetwork module."""
        super(FeedForwardNetwork, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.projection_size,
            out_size=args.hidden_size,
            dropout_rate=args.dropout_rate,
            use_relu=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        return self.mlp(x)


class SelfAttention(nn.Module):
    """
    Transformer encoder block with self-attention and feed-forward network.

    This module implements a single Transformer encoder layer consisting of:
        1. Multi-head self-attention sub-layer
        2. Position-wise feed-forward network sub-layer
        Each sub-layer is followed by residual connection and layer normalization.

    The forward computation:
        y = LayerNorm(x + Dropout(MultiHeadAttention(x, x, x)))
        y = LayerNorm(y + Dropout(FeedForward(y)))

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (int): Hidden dimension size
        - heads (int): Number of attention heads
        - projection_size (int): Feed-forward intermediate dimension
        - dropout_rate (float): Dropout probability

    Attributes
    ----------
    multi_head_attention : MultiHeadAttention
        Multi-head attention mechanism.
    feed_forward : FeedForwardNetwork
        Position-wise feed-forward network.
    attention_dropout : nn.Dropout
        Dropout applied after attention.
    attention_norm : LayerNorm
        Layer normalization after attention sub-layer.
    feed_forward_dropout : nn.Dropout
        Dropout applied after feed-forward network.
    feed_forward_norm : LayerNorm
        Layer normalization after feed-forward sub-layer.

    Notes
    -----
    The residual connections help gradient flow and enable training of
    deeper networks. Layer normalization stabilizes training.
    """

    def __init__(self, args):
        """Initialize SelfAttention encoder block."""
        super(SelfAttention, self).__init__()
        logger.debug(f"Initializing SelfAttention block with hidden_size={args.hidden_size}")

        self.multi_head_attention = MultiHeadAttention(args)
        self.feed_forward = FeedForwardNetwork(args)

        self.attention_dropout = nn.Dropout(args.dropout_rate)
        self.attention_norm = LayerNorm(args.hidden_size)

        self.feed_forward_dropout = nn.Dropout(args.dropout_rate)
        self.feed_forward_norm = LayerNorm(args.hidden_size)

    def forward(self, y: torch.Tensor, y_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply self-attention encoder block.

        Parameters
        ----------
        y : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size).
        y_mask : torch.Tensor, optional
            Padding mask of shape (batch_size, 1, 1, seq_len).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        y = self.attention_norm(y + self.attention_dropout(
            self.multi_head_attention(y, y, y, y_mask)
        ))

        y = self.feed_forward_norm(y + self.feed_forward_dropout(
            self.feed_forward(y)
        ))

        return y


class VanillaTransformer(nn.Module):
    """
    Vanilla Transformer model for log anomaly detection.

    This model implements a Transformer-based architecture for detecting
    anomalies in log sequences. It combines LSTM-based sequential encoding
    with multi-head self-attention to capture both local and global patterns
    in log data.

    Architecture overview:
        1. Embedding layer: Maps token IDs to dense vectors
        2. LSTM encoder: Extracts sequential features
        3. Transformer encoder: Multiple self-attention layers
        4. Attention flattening: Aggregates sequence to fixed vector
        5. Classification head: Projects to anomaly class logits

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing:
        - embed_size (int): Embedding dimension
        - hidden_size (int): Hidden state dimension
        - projection_size (int): Feed-forward intermediate dimension
        - heads (int): Number of attention heads
        - layers (int): Number of Transformer encoder layers
        - dropout_rate (float): Dropout probability
        - n_classes (int): Number of output classes
    vocab_size : int
        Size of the vocabulary (number of unique tokens).
    pretrained_embeddings : numpy.ndarray
        Pre-trained embedding matrix of shape (vocab_size, embed_size).

    Attributes
    ----------
    args : argparse.Namespace
        Stored configuration.
    embedding : nn.Embedding
        Token embedding layer initialized with pretrained weights.
    lstm : nn.LSTM
        Bidirectional LSTM for sequential feature extraction.
    encoder_blocks : nn.ModuleList
        List of Transformer encoder blocks (SelfAttention modules).
    text_attention_flatten : AttentionFlatten
        Attention-based sequence flattening module.
    projection_norm : LayerNorm
        Layer normalization before classification.
    classification_head : nn.Linear
        Final linear layer for classification.
    activation_func : nn.PReLU
        Parametric ReLU activation for output.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from argparse import Namespace
    >>> 
    >>> # Configuration
    >>> args = Namespace(
    ...     embed_size=300,
    ...     hidden_size=512,
    ...     projection_size=2048,
    ...     heads=8,
    ...     layers=6,
    ...     dropout_rate=0.1,
    ...     n_classes=2
    ... )
    >>> 
    >>> # Initialize model
    >>> vocab_size = 10000
    >>> embeddings = np.random.randn(vocab_size, 300)
    >>> model = VanillaTransformer(args, vocab_size, embeddings)
    >>> 
    >>> # Forward pass
    >>> batch_size, seq_len = 16, 128
    >>> input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    >>> logits = model(input_ids, None, None)
    >>> logits.shape
    torch.Size([16, 2])

    Notes
    -----
    - The model expects pre-tokenized sequences as input
    - Padding masks are computed automatically from input features
    - The LSTM is single-directional (can be made bidirectional if needed)
    - Pre-trained embeddings are copied during initialization
    """

    def __init__(self, args, vocab_size: int, pretrained_embeddings):
        """Initialize VanillaTransformer model."""
        super(VanillaTransformer, self).__init__()
        try:
            # Validate arguments
            if vocab_size <= 0:
                raise ValueError(f"vocab_size must be positive, got {vocab_size}")
            if not hasattr(args, 'embed_size') or args.embed_size <= 0:
                raise ValueError(f"Invalid embed_size: {getattr(args, 'embed_size', None)}")
            if not hasattr(args, 'hidden_size') or args.hidden_size <= 0:
                raise ValueError(f"Invalid hidden_size: {getattr(args, 'hidden_size', None)}")
            if not hasattr(args, 'heads') or args.heads <= 0:
                raise ValueError(f"Invalid heads: {getattr(args, 'heads', None)}")
            if not hasattr(args, 'layers') or args.layers <= 0:
                raise ValueError(f"Invalid layers: {getattr(args, 'layers', None)}")
            if not hasattr(args, 'n_classes') or args.n_classes <= 0:
                raise ValueError(f"Invalid n_classes: {getattr(args, 'n_classes', None)}")
            
            logger.info(f"Initializing VanillaTransformer with vocab_size={vocab_size}, embed_size={args.embed_size}, "
                        f"hidden_size={args.hidden_size}, heads={args.heads}, layers={args.layers}, n_classes={args.n_classes}")

            self.args = args

            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=args.embed_size
            )

            # Load pre-trained embedding weights
            logger.info(f"Loading pre-trained embeddings with shape {pretrained_embeddings.shape}")
            if pretrained_embeddings.shape[0] != vocab_size:
                raise ValueError(f"Embedding vocab size mismatch: expected {vocab_size}, got {pretrained_embeddings.shape[0]}")
            if pretrained_embeddings.shape[1] != args.embed_size:
                raise ValueError(f"Embedding dimension mismatch: expected {args.embed_size}, got {pretrained_embeddings.shape[1]}")
            
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

            self.lstm = nn.LSTM(
                input_size=args.embed_size,
                hidden_size=args.hidden_size,
                num_layers=1,
                batch_first=True
            )

            logger.info(f"Creating {args.layers} Transformer encoder blocks")
            self.encoder_blocks = nn.ModuleList([
                SelfAttention(args) for _ in range(args.layers)
            ])

            # Attention-based flattening for sequence aggregation
            self.text_attention_flatten = AttentionFlatten(args)

            # Normalization and classification layers
            self.projection_norm = LayerNorm(2 * args.hidden_size)
            self.classification_head = nn.Linear(2 * args.hidden_size, args.n_classes)
            self.activation_func = nn.PReLU()
            logger.info("VanillaTransformer initialization completed")
        except Exception as e:
            logger.error(f"Error initializing VanillaTransformer: {e}")
            raise

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input sequences.

        This method processes input token IDs through the embedding layer,
        LSTM encoder, and Transformer encoder blocks to produce sequence-level
        feature representations.

        Parameters
        ----------
        x : torch.Tensor
            Input token IDs of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Extracted features of shape (batch_size, 2 * hidden_size).
        """
        try:
            logger.debug(f"forward_features: input shape={x.shape}")
            
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor (batch_size, seq_len), got {x.dim()}D")
            if x.size(0) == 0:
                raise ValueError("Batch size cannot be zero")
            
            text_mask = create_padding_mask(x.unsqueeze(2))

            text_embeddings = self.embedding(x)
            logger.debug(f"Embedding output shape={text_embeddings.shape}")
            
            # Check for invalid token IDs
            if (x < 0).any():
                raise ValueError("Input contains negative token IDs")
            if (x >= self.embedding.num_embeddings).any():
                raise ValueError(f"Input contains token IDs >= vocab_size ({self.embedding.num_embeddings})")

            text_features, _ = self.lstm(text_embeddings)
            logger.debug(f"LSTM output shape={text_features.shape}")
            
            # Check for NaN or Inf after LSTM
            if torch.isnan(text_features).any() or torch.isinf(text_features).any():
                logger.warning("NaN or Inf detected in LSTM output")
                text_features = torch.nan_to_num(text_features, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)

            for idx, encoder in enumerate(self.encoder_blocks):
                logger.debug(f"Processing encoder block {idx + 1}/{len(self.encoder_blocks)}")
                text_features = encoder(text_features, text_mask)
                
                # Check for NaN or Inf after each encoder block
                if torch.isnan(text_features).any() or torch.isinf(text_features).any():
                    logger.warning(f"NaN or Inf detected after encoder block {idx + 1}")
                    text_features = torch.nan_to_num(text_features, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)

            # Aggregate sequence to fixed-size vector
            text_features = self.text_attention_flatten(
                text_features,
                text_mask
            )

            # Apply layer normalization
            text_features = self.projection_norm(text_features)
            logger.debug(f"forward_features output shape={text_features.shape}")

            return text_features
        except Exception as e:
            logger.error(f"Error in forward_features: {e}")
            raise

    def forward(self, x: torch.Tensor, _unused1, _unused2) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input token IDs of shape (batch_size, seq_len).
        _unused1 : Any
            Unused parameter for compatibility with collaborative transformer.
        _unused2 : Any
            Unused parameter for compatibility with collaborative transformer.

        Returns
        -------
        torch.Tensor
            Classification logits of shape (batch_size, n_classes).

        Notes
        -----
        The unused parameters maintain API compatibility with other model
        variants that may use additional input modalities.
        """
        try:
            logger.debug(f"VanillaTransformer forward: input shape={x.shape}")
            
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor input, got {type(x)}")
            
            text_features = self.forward_features(x)

            # Apply classification head
            output = self.classification_head(text_features)
            output = self.activation_func(output)
            
            # Final check for NaN or Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.error("NaN or Inf detected in final output")
                output = torch.nan_to_num(output, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)
            
            logger.debug(f"VanillaTransformer forward output shape={output.shape}")

            return output
        except Exception as e:
            logger.error(f"Error in VanillaTransformer forward pass: {e}")
            raise
