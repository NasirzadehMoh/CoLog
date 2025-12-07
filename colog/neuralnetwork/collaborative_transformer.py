"""
collaborative_transformer.py — Collaborative Transformer model for CoLog log anomaly detection

This module implements a multi-modal Transformer-based neural network architecture
for detecting anomalies in system log sequences. The model processes both textual
log messages and sequential features through collaborative attention mechanisms,
enabling cross-modal information exchange and refinement.

Key architectural components:
    - Dual embedding pathways: Separate processing for text and sequence features
    - LSTM encoder: Extracts temporal dependencies from embedded log messages
    - Collaborative attention blocks: Cross-modal attention with modality adaptation
    - Multi-head impressed attention: Bidirectional information flow between modalities
    - Modality adaptation layers: Dynamic feature re-weighting and glimpse mechanisms
    - Classification head: Projects fused multi-modal features to anomaly class logits

The collaborative architecture enables:
    1. Text and sequence features are processed through separate pathways
    2. Cross-modal attention allows each modality to attend to the other
    3. Modality adaptation layers create multiple glimpses of each feature space
    4. Residual connections and layer normalization ensure stable training
    5. Final fusion combines both modalities for classification

Usage
-----
    from neuralnetwork.collaborative_transformer import CollaborativeTransformer
    
    # Initialize model
    model = CollaborativeTransformer(
        args=config,
        vocab_size=50000,
        pretrained_embeddings=embeddings_matrix
    )
    
    # Forward pass
    logits = model(text_input, embeddings, sequence_features)
    predictions = torch.argmax(logits, dim=-1)

Notes
-----
- The model expects both text token IDs and pre-computed sequence features
- Padding masks are computed automatically for both modalities
- Dropout and layer normalization improve regularization and stability
- The collaborative blocks enable rich multi-modal interactions

See Also
--------
vanilla_transformer.py : Single-modal transformer without collaborative features
components.layers : Building blocks for neural network layers
"""

import logging
from typing import Optional, Tuple
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



class ModalityAdaptationLayer(nn.Module):
    """
    Modality adaptation layer with multi-glimpse attention mechanism.

    This module creates multiple weighted views (glimpses) of input features
    using learned attention weights. It enables the model to focus on different
    aspects of the input simultaneously, similar to multi-head attention but
    applied across the sequence dimension for feature aggregation.

    The adaptation process:
        1. Compute attention scores for each glimpse using MLP
        2. Mask padding positions with large negative values
        3. Apply softmax to get normalized attention weights per glimpse
        4. For each glimpse, compute weighted sum of sequence features
        5. Either stack glimpses or merge them into a single vector

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (int): Hidden dimension size
        - projection_size (int): Intermediate projection dimension
        - dropout_rate (float): Dropout probability
    num_glimpses : int
        Number of attention glimpses to create.
    merge : bool, optional
        If True, concatenate and project all glimpses into a single vector.
        If False, return stacked glimpses. Default is False.

    Attributes
    ----------
    args : argparse.Namespace
        Stored configuration.
    merge : bool
        Whether to merge glimpses.
    num_glimpses : int
        Number of attention glimpses.
    mlp : MLP
        Multi-layer perceptron for computing attention scores.
    output_projection : nn.Linear, optional
        Linear projection when merge=True, projects concatenated glimpses
        to output dimension (2 * hidden_size).

    Notes
    -----
    When merge=True, the output dimension is 2 * hidden_size to provide
    additional capacity for downstream processing. When merge=False, the
    output has shape (batch_size, num_glimpses, hidden_size).
    """

    def __init__(self, args, num_glimpses: int, merge: bool = False):
        """Initialize ModalityAdaptationLayer module."""
        super(ModalityAdaptationLayer, self).__init__()
        try:
            if not hasattr(args, 'hidden_size') or args.hidden_size <= 0:
                raise ValueError(f"Invalid hidden_size: {getattr(args, 'hidden_size', None)}")
            if not hasattr(args, 'projection_size') or args.projection_size <= 0:
                raise ValueError(f"Invalid projection_size: {getattr(args, 'projection_size', None)}")
            if not hasattr(args, 'dropout_rate') or not (0 <= args.dropout_rate < 1):
                raise ValueError(f"Invalid dropout_rate: {getattr(args, 'dropout_rate', None)}")
            if num_glimpses <= 0:
                raise ValueError(f"num_glimpses must be positive, got {num_glimpses}")
            
            self.args = args
            self.merge = merge
            self.num_glimpses = num_glimpses
            logger.debug(f"Initialized ModalityAdaptationLayer with num_glimpses={num_glimpses}, merge={merge}")

            self.mlp = MLP(
                in_size=args.hidden_size,
                mid_size=args.projection_size,
                out_size=num_glimpses,
                dropout_rate=args.dropout_rate,
                use_relu=True
            )

            if self.merge:
                self.output_projection = nn.Linear(
                    args.hidden_size * num_glimpses,
                    args.hidden_size * 2
                )
        except Exception as e:
            logger.error(f"Error initializing ModalityAdaptationLayer: {e}")
            raise

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply modality adaptation to input features.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, seq_len, hidden_size).
        x_mask : torch.Tensor, optional
            Padding mask of shape (batch_size, 1, 1, seq_len).

        Returns
        -------
        torch.Tensor
            If merge=True: Merged features of shape (batch_size, 2 * hidden_size).
            If merge=False: Stacked glimpses of shape (batch_size, num_glimpses, hidden_size).
        """
        try:
            logger.debug(f"ModalityAdaptationLayer forward: input shape={x.shape}, merge={self.merge}")
            
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
            if x.size(-1) != self.args.hidden_size:
                raise ValueError(f"Expected hidden_size={self.args.hidden_size}, got {x.size(-1)}")
            
            attention_weights = self.mlp(x)
            
            if x_mask is not None:
                attention_weights = attention_weights.masked_fill(
                    x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                    constants.ATTENTION_MASK_VALUE
                )
            
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Check for NaN or Inf in attention weights
            if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
                logger.warning("NaN or Inf detected in modality adaptation attention weights")
                attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=1.0, neginf=0.0)

            attended_features_list = []
            for i in range(self.num_glimpses):
                attended_features_list.append(
                    torch.sum(attention_weights[:, :, i: i + 1] * x, dim=1)
                )

            if self.merge:
                attended_features = torch.cat(attended_features_list, dim=1)
                attended_features = self.output_projection(attended_features)
                logger.debug(f"ModalityAdaptationLayer merged output shape={attended_features.shape}")
                return attended_features

            stacked_features = torch.stack(attended_features_list).transpose_(0, 1)
            logger.debug(f"ModalityAdaptationLayer stacked output shape={stacked_features.shape}")
            return stacked_features
        except Exception as e:
            logger.error(f"Error in ModalityAdaptationLayer forward pass: {e}")
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




class MultiHeadImpressedAttention(nn.Module):
    """
    Multi-head impressed attention for cross-modal interaction.

    This module implements a dual-attention mechanism that enables cross-modal
    information exchange between two different feature spaces (e.g., text and
    sequence features). It first applies self-attention within the source
    modality, then cross-attention to the target modality, followed by a
    feed-forward network.

    The impressed attention computation:
        1. Self-attention: Refine source features using self-attention
        2. Cross-attention: Attend to target modality features
        3. Feed-forward: Apply position-wise transformation
        Each sub-layer uses residual connections and layer normalization

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
    multi_head_attention_1 : MultiHeadAttention
        Self-attention mechanism for the source modality.
    multi_head_attention_2 : MultiHeadAttention
        Cross-attention mechanism to target modality.
    feed_forward : FeedForwardNetwork
        Position-wise feed-forward network.
    self_attention_dropout : nn.Dropout
        Dropout applied after self-attention.
    self_attention_norm : LayerNorm
        Layer normalization after self-attention sub-layer.
    cross_attention_dropout : nn.Dropout
        Dropout applied after cross-attention.
    cross_attention_norm : LayerNorm
        Layer normalization after cross-attention sub-layer.
    feed_forward_dropout : nn.Dropout
        Dropout applied after feed-forward network.
    feed_forward_norm : LayerNorm
        Layer normalization after feed-forward sub-layer.

    Notes
    -----
    This module enables bidirectional information flow between modalities when
    used symmetrically (e.g., text->sequence and sequence->text).
    """

    def __init__(self, args):
        """Initialize MultiHeadImpressedAttention module."""
        super(MultiHeadImpressedAttention, self).__init__()
        logger.debug(f"Initializing MultiHeadImpressedAttention with hidden_size={args.hidden_size}")

        self.multi_head_attention_1 = MultiHeadAttention(args)
        self.multi_head_attention_2 = MultiHeadAttention(args)
        self.feed_forward = FeedForwardNetwork(args)

        self.self_attention_dropout = nn.Dropout(args.dropout_rate)
        self.self_attention_norm = LayerNorm(args.hidden_size)

        self.cross_attention_dropout = nn.Dropout(args.dropout_rate)
        self.cross_attention_norm = LayerNorm(args.hidden_size)

        self.feed_forward_dropout = nn.Dropout(args.dropout_rate)
        self.feed_forward_norm = LayerNorm(args.hidden_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor, 
                x_mask: Optional[torch.Tensor], y_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Apply impressed attention mechanism.

        Parameters
        ----------
        x : torch.Tensor
            Source modality features of shape (batch_size, seq_len_x, hidden_size).
        y : torch.Tensor
            Target modality features of shape (batch_size, seq_len_y, hidden_size).
        x_mask : torch.Tensor, optional
            Padding mask for source modality of shape (batch_size, 1, 1, seq_len_x).
        y_mask : torch.Tensor, optional
            Padding mask for target modality of shape (batch_size, 1, 1, seq_len_y).

        Returns
        -------
        torch.Tensor
            Impressed features of shape (batch_size, seq_len_x, hidden_size).

        Notes
        -----
        The output maintains the sequence length of the source modality (x) while
        incorporating information from the target modality (y) via cross-attention.
        """
        try:
            logger.debug(f"MultiHeadImpressedAttention forward: x shape={x.shape}, y shape={y.shape}")
            
            # Self-attention on source modality
            x = self.self_attention_norm(x + self.self_attention_dropout(
                self.multi_head_attention_1(v=x, k=x, q=x, mask=x_mask)
            ))

            # Cross-attention to target modality
            x = self.cross_attention_norm(x + self.cross_attention_dropout(
                self.multi_head_attention_2(v=y, k=y, q=x, mask=y_mask)
            ))

            # Feed-forward transformation
            x = self.feed_forward_norm(x + self.feed_forward_dropout(
                self.feed_forward(x)
            ))
            
            logger.debug(f"MultiHeadImpressedAttention output shape={x.shape}")
            return x
        except Exception as e:
            logger.error(f"Error in MultiHeadImpressedAttention forward pass: {e}")
            raise




class CollaborativeBlock(nn.Module):
    """
    Collaborative attention block for multi-modal feature refinement.

    This module implements a bidirectional cross-modal attention mechanism
    where two modalities (text and sequence features) exchange information
    through impressed attention. Each modality attends to the other while
    maintaining its own representation through residual connections.

    Architecture:
        1. Apply impressed attention: text->sequence and sequence->text
        2. Add residual connections to preserve original features
        3. If not the last layer, apply modality adaptation and normalization
        4. Return refined features for both modalities

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (int): Hidden dimension size
        - len_messages (int): Number of glimpses for text modality adaptation
        - len_sequences (int): Number of glimpses for sequence modality adaptation
        - layers (int): Total number of collaborative blocks
        - dropout_rate (float): Dropout probability
    i : int
        Block index (0-based), used to determine if this is the last layer.

    Attributes
    ----------
    args : argparse.Namespace
        Stored configuration.
    text_impressed_attention : MultiHeadImpressedAttention
        Impressed attention for text modality.
    sequence_impressed_attention : MultiHeadImpressedAttention
        Impressed attention for sequence modality.
    is_last_layer : bool
        Whether this is the final collaborative block.
    text_modality_adaptation : ModalityAdaptationLayer, optional
        Adaptation layer for text features (not present in last layer).
    sequence_modality_adaptation : ModalityAdaptationLayer, optional
        Adaptation layer for sequence features (not present in last layer).
    text_norm : LayerNorm, optional
        Layer normalization for text features (not present in last layer).
    sequence_norm : LayerNorm, optional
        Layer normalization for sequence features (not present in last layer).
    residual_dropout : nn.Dropout, optional
        Dropout for residual connections (not present in last layer).

    Notes
    -----
    The last layer omits modality adaptation to produce final representations
    suitable for downstream classification tasks.
    """

    def __init__(self, args, i: int):
        """Initialize CollaborativeBlock module."""
        super(CollaborativeBlock, self).__init__()
        try:
            if not hasattr(args, 'layers') or args.layers <= 0:
                raise ValueError(f"Invalid layers: {getattr(args, 'layers', None)}")
            if i < 0 or i >= args.layers:
                raise ValueError(f"Block index {i} out of range [0, {args.layers - 1}]")
            
            self.args = args
            self.is_last_layer = (i == args.layers - 1)
            logger.debug(f"Initializing CollaborativeBlock {i}, is_last_layer={self.is_last_layer}")

            self.text_impressed_attention = MultiHeadImpressedAttention(args)
            self.sequence_impressed_attention = MultiHeadImpressedAttention(args)

            if not self.is_last_layer:
                if not hasattr(args, 'len_messages') or args.len_messages <= 0:
                    raise ValueError(f"Invalid len_messages: {getattr(args, 'len_messages', None)}")
                if not hasattr(args, 'len_sequences') or args.len_sequences <= 0:
                    raise ValueError(f"Invalid len_sequences: {getattr(args, 'len_sequences', None)}")
                
                self.text_modality_adaptation = ModalityAdaptationLayer(
                    args, args.len_messages, merge=False
                )
                self.sequence_modality_adaptation = ModalityAdaptationLayer(
                    args, args.len_sequences, merge=False
                )
                self.text_norm = LayerNorm(args.hidden_size)
                self.sequence_norm = LayerNorm(args.hidden_size)
                self.residual_dropout = nn.Dropout(args.dropout_rate)
        except Exception as e:
            logger.error(f"Error initializing CollaborativeBlock: {e}")
            raise

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor], 
                y: torch.Tensor, y_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply collaborative attention block.

        Parameters
        ----------
        x : torch.Tensor
            Text features of shape (batch_size, seq_len_text, hidden_size).
        x_mask : torch.Tensor, optional
            Padding mask for text of shape (batch_size, 1, 1, seq_len_text).
        y : torch.Tensor
            Sequence features of shape (batch_size, seq_len_seq, hidden_size).
        y_mask : torch.Tensor, optional
            Padding mask for sequences of shape (batch_size, 1, 1, seq_len_seq).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Refined text and sequence features, both of shape 
            (batch_size, seq_len, hidden_size).

        Notes
        -----
        In the last layer, only impressed attention and residual addition are
        performed. In intermediate layers, modality adaptation and layer
        normalization are also applied.
        """
        try:
            logger.debug(f"CollaborativeBlock forward: x shape={x.shape}, y shape={y.shape}")
            
            # Apply impressed attention bidirectionally
            text_impressed = self.text_impressed_attention(x, y, x_mask, y_mask)
            sequence_impressed = self.sequence_impressed_attention(y, x, y_mask, x_mask)

            # Add residual connections
            x = text_impressed + x
            y = sequence_impressed + y

            if self.is_last_layer:
                logger.debug(f"CollaborativeBlock (last layer) output: x shape={x.shape}, y shape={y.shape}")
                return x, y

            # Apply modality adaptation for intermediate layers
            text_adapted = self.text_modality_adaptation(x, x_mask)
            sequence_adapted = self.sequence_modality_adaptation(y, y_mask)

            # Apply normalization and residual dropout
            x = self.text_norm(x + self.residual_dropout(text_adapted))
            y = self.sequence_norm(y + self.residual_dropout(sequence_adapted))
            
            logger.debug(f"CollaborativeBlock output: x shape={x.shape}, y shape={y.shape}")
            return x, y
        except Exception as e:
            logger.error(f"Error in CollaborativeBlock forward pass: {e}")
            raise


class TextEmbedding(nn.Module):
    """
    Text embedding layer with pre-trained weights.

    This module creates dense vector representations of log message tokens
    using pre-trained word embeddings. The embeddings are loaded from a
    pre-trained matrix and can be frozen or fine-tuned based on configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing:
        - embedding_size (int): Embedding dimension size
    vocab_size : int
        Size of the vocabulary (number of unique tokens).
    pretrained_embeddings : numpy.ndarray
        Pre-trained embedding matrix of shape (vocab_size, embedding_size).

    Attributes
    ----------
    embedding : nn.Embedding
        PyTorch embedding layer initialized with pre-trained weights.

    Notes
    -----
    By default, the embedding weights are frozen (requires_grad=False) to
    preserve pre-trained representations. This can be changed by setting
    embedding.weight.requires_grad = True.
    """

    def __init__(self, args, vocab_size: int, pretrained_embeddings):
        """Initialize TextEmbedding module."""
        super(TextEmbedding, self).__init__()
        try:
            if vocab_size <= 0:
                raise ValueError(f"vocab_size must be positive, got {vocab_size}")
            if not hasattr(args, 'embedding_size') or args.embedding_size <= 0:
                raise ValueError(f"Invalid embedding size: {getattr(args, 'embedding_size', None)}")
            if pretrained_embeddings.shape[0] != vocab_size:
                raise ValueError(f"Embedding vocab size mismatch: expected {vocab_size}, got {pretrained_embeddings.shape[0]}")
            if pretrained_embeddings.shape[1] != args.embedding_size:
                raise ValueError(f"Embedding dimension mismatch: expected {args.embedding_size}, got {pretrained_embeddings.shape[1]}")
            
            logger.info(f"Initializing TextEmbedding with vocab_size={vocab_size}, embedding_size={args.embedding_size}")

            # Create embedding layer
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=args.embedding_size
            )

            # Load pre-trained embedding weights
            logger.info(f"Loading pre-trained embeddings with shape {pretrained_embeddings.shape}")
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

            # Freeze embeddings by default
            self.embedding.weight.requires_grad = False
            logger.info("Embedding weights frozen (requires_grad=False)")
        except Exception as e:
            logger.error(f"Error initializing TextEmbedding: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input token IDs.

        Parameters
        ----------
        x : torch.Tensor
            Input token IDs of shape (batch_size, seq_len).
        
        Returns
        -------
        torch.Tensor
            Embedded representations of shape (batch_size, seq_len, embedding_size).
        """
        try:
            if x.dim() != 2:
                raise ValueError(f"Expected 2D input tensor (batch_size, seq_len), got {x.dim()}D")
            if (x < 0).any():
                raise ValueError("Input contains negative token IDs")
            if (x >= self.embedding.num_embeddings).any():
                raise ValueError(f"Input contains token IDs >= vocab_size ({self.embedding.num_embeddings})")
            
            embedding = self.embedding(x)
            return embedding
        except Exception as e:
            logger.error(f"Error in TextEmbedding forward pass: {e}")
            raise




class CollaborativeTransformer(nn.Module):
    """
    Collaborative Transformer model for multi-modal log anomaly detection.

    This model implements a collaborative multi-modal architecture that processes
    both textual log messages and sequential features through bidirectional
    cross-modal attention mechanisms. The collaborative blocks enable rich
    information exchange between modalities, allowing each to benefit from the
    other's representations.

    Architecture overview:
        1. Text embedding: Maps token IDs to dense vectors using pre-trained embeddings
        2. Text LSTM: Extracts temporal features from embedded log messages
        3. Sequence adapter: Projects raw sequence features to hidden dimension
        4. Collaborative blocks: Multiple layers of cross-modal attention
        5. Modality adaptation: Aggregates each modality to fixed-size vectors
        6. Feature fusion: Combines text and sequence representations
        7. Classification head: Projects fused features to anomaly class logits

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing:
        - embedding_size (int): Embedding dimension
        - hidden_size (int): Hidden state dimension
        - sequences_fsize (int): Input dimension for sequence features
        - projection_size (int): Feed-forward intermediate dimension
        - heads (int): Number of attention heads
        - layers (int): Number of collaborative blocks
        - len_messages (int): Number of glimpses for text modality
        - len_sequences (int): Number of glimpses for sequence modality
        - dropout_rate (float): Dropout probability
        - n_classes (int): Number of output classes
    vocab_size : int
        Size of the vocabulary (number of unique tokens).
    pretrained_embeddings : numpy.ndarray
        Pre-trained embedding matrix of shape (vocab_size, embedding_size).

    Attributes
    ----------
    args : argparse.Namespace
        Stored configuration.
    text_embedding : TextEmbedding
        Embedding layer for text tokens.
    text_lstm : nn.LSTM
        LSTM for sequential encoding of text.
    sequence_adapter : nn.Linear
        Linear projection from sequence feature size to hidden size.
    encoder_blocks : nn.ModuleList
        List of collaborative attention blocks.
    sequence_modality_adaptation : ModalityAdaptationLayer
        Aggregation layer for sequence features.
    text_modality_adaptation : ModalityAdaptationLayer
        Aggregation layer for text features.
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
    ...     embedding_size=300,
    ...     hidden_size=512,
    ...     sequences_fsize=256,
    ...     projection_size=2048,
    ...     heads=8,
    ...     layers=4,
    ...     len_messages=10,
    ...     len_sequences=5,
    ...     dropout_rate=0.1,
    ...     n_classes=2
    ... )
    >>> 
    >>> # Initialize model
    >>> vocab_size = 10000
    >>> embeddings = np.random.randn(vocab_size, 300)
    >>> model = CollaborativeTransformer(args, vocab_size, embeddings)
    >>> 
    >>> # Forward pass
    >>> batch_size, seq_len = 16, 128
    >>> text_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    >>> seq_features = torch.randn(batch_size, seq_len, 256)
    >>> logits = model(text_input, None, seq_features)
    >>> logits.shape
    torch.Size([16, 2])

    Notes
    -----
    - The model requires both text and sequence features as input
    - Embeddings are passed separately to support external embedding computation
    - Padding masks are computed automatically from input features
    - The collaborative architecture enables bidirectional information flow
    """

    def __init__(self, args, vocab_size: int, pretrained_embeddings, num_classes: int = None):
        """Initialize CollaborativeTransformer model."""
        super(CollaborativeTransformer, self).__init__()
        try:
            # Validate arguments
            if vocab_size <= 0:
                raise ValueError(f"vocab_size must be positive, got {vocab_size}")
            if not hasattr(args, 'embedding_size') or args.embedding_size <= 0:
                raise ValueError(f"Invalid embedding_size: {getattr(args, 'embedding_size', None)}")
            if not hasattr(args, 'hidden_size') or args.hidden_size <= 0:
                raise ValueError(f"Invalid hidden_size: {getattr(args, 'hidden_size', None)}")
            if not hasattr(args, 'sequences_fsize') or args.sequences_fsize <= 0:
                raise ValueError(f"Invalid sequences_fsize: {getattr(args, 'sequences_fsize', None)}")
            if not hasattr(args, 'layers') or args.layers <= 0:
                raise ValueError(f"Invalid layers: {getattr(args, 'layers', None)}")
            
            # Use num_classes from parameter if provided, otherwise fall back to args
            if num_classes is not None:
                self.num_classes = num_classes
            elif hasattr(args, 'num_classes'):
                self.num_classes = args.num_classes
            else:
                raise ValueError("num_classes must be provided either as parameter or in args")
            
            if self.num_classes <= 0:
                raise ValueError(f"Invalid number of classes: {self.num_classes}")
            
            logger.info(f"Initializing CollaborativeTransformer with vocab_size={vocab_size}, "
                        f"embedding_size={args.embedding_size}, hidden_size={args.hidden_size}, "
                        f"sequences_fsize={args.sequences_fsize}, layers={args.layers}, "
                        f"num_classes={self.num_classes}")

            self.args = args

            # Text embedding layer
            self.text_embedding = TextEmbedding(args, vocab_size, pretrained_embeddings)

            # Text LSTM encoder
            self.text_lstm = nn.LSTM(
                input_size=args.embedding_size,
                hidden_size=args.hidden_size,
                num_layers=1,
                batch_first=True
            )

            # Sequence feature adapter
            self.sequence_adapter = nn.Linear(args.sequences_fsize, args.hidden_size)

            # Collaborative encoder blocks
            logger.info(f"Creating {args.layers} collaborative encoder blocks")
            self.encoder_blocks = nn.ModuleList([
                CollaborativeBlock(args, i) for i in range(args.layers)
            ])

            # Modality adaptation layers for final aggregation
            self.sequence_modality_adaptation = ModalityAdaptationLayer(args, 1, merge=True)
            self.text_modality_adaptation = ModalityAdaptationLayer(args, 1, merge=True)

            # Normalization and classification layers
            self.projection_norm = LayerNorm(2 * args.hidden_size)
            self.classification_head = nn.Linear(2 * args.hidden_size, self.num_classes)
            self.activation_func = nn.PReLU()
            logger.info("CollaborativeTransformer initialization completed")
        except Exception as e:
            logger.error(f"Error initializing CollaborativeTransformer: {e}")
            raise

    def forward_features(self, x: torch.Tensor, embedding: torch.Tensor, 
                        y: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-modal features from inputs.

        This method processes text and sequence features through the collaborative
        encoder blocks, applies modality adaptation, and fuses the representations.

        Parameters
        ----------
        x : torch.Tensor
            Text token IDs of shape (batch_size, seq_len).
        embedding : torch.Tensor
            Pre-computed text embeddings of shape (batch_size, seq_len, embedding_size).
            If None, embeddings are computed from x using the text_embedding layer.
        y : torch.Tensor
            Sequence features of shape (batch_size, seq_len, sequences_fsize).

        Returns
        -------
        torch.Tensor
            Fused multi-modal features of shape (batch_size, 2 * hidden_size).

        Notes
        -----
        Masks are only applied in the first collaborative block to reduce
        computational overhead. Subsequent blocks process full sequences.
        """
        try:
            logger.debug(f"forward_features: x shape={x.shape}, y shape={y.shape}")
            
            if x.dim() != 2:
                raise ValueError(f"Expected 2D text input (batch_size, seq_len), got {x.dim()}D")
            if y.dim() != 3:
                raise ValueError(f"Expected 3D sequence input (batch_size, seq_len, fsize), got {y.dim()}D")
            if x.size(0) != y.size(0):
                raise ValueError(f"Batch size mismatch: x={x.size(0)}, y={y.size(0)}")
            
            # Create padding masks
            text_mask = create_padding_mask(x.unsqueeze(2))
            sequence_mask = create_padding_mask(y)

            # Compute text embeddings if not provided
            if embedding is None:
                embedding = self.text_embedding(x)
                logger.debug(f"Computed text embeddings shape={embedding.shape}")

            # Encode text through LSTM
            text_features, _ = self.text_lstm(embedding)
            logger.debug(f"Text LSTM output shape={text_features.shape}")
            
            # Check for NaN or Inf after LSTM
            if torch.isnan(text_features).any() or torch.isinf(text_features).any():
                logger.warning("NaN or Inf detected in text LSTM output")
                text_features = torch.nan_to_num(text_features, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)

            # Adapt sequence features to hidden dimension
            sequence_features = self.sequence_adapter(y)
            logger.debug(f"Sequence adapter output shape={sequence_features.shape}")

            # Process through collaborative encoder blocks
            for i, encoder in enumerate(self.encoder_blocks):
                # Only apply masks in the first block
                current_text_mask, current_sequence_mask = None, None
                if i == 0:
                    current_text_mask, current_sequence_mask = text_mask, sequence_mask
                
                logger.debug(f"Processing collaborative block {i + 1}/{len(self.encoder_blocks)}")
                text_features, sequence_features = encoder(
                    text_features, current_text_mask, 
                    sequence_features, current_sequence_mask
                )
                
                # Check for NaN or Inf after each block
                if torch.isnan(text_features).any() or torch.isinf(text_features).any():
                    logger.warning(f"NaN or Inf detected in text features after block {i + 1}")
                    text_features = torch.nan_to_num(text_features, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)
                if torch.isnan(sequence_features).any() or torch.isinf(sequence_features).any():
                    logger.warning(f"NaN or Inf detected in sequence features after block {i + 1}")
                    sequence_features = torch.nan_to_num(sequence_features, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)

            # Aggregate modalities to fixed-size vectors
            text_features = self.text_modality_adaptation(
                text_features,
                None
            )
            logger.debug(f"Text modality adaptation output shape={text_features.shape}")

            sequence_features = self.sequence_modality_adaptation(
                sequence_features,
                None
            )
            logger.debug(f"Sequence modality adaptation output shape={sequence_features.shape}")

            # Fuse multi-modal features
            combined_features = text_features + sequence_features
            combined_features = self.projection_norm(combined_features)
            logger.debug(f"Combined features shape={combined_features.shape}")

            return combined_features
        except Exception as e:
            logger.error(f"Error in forward_features: {e}")
            raise

    def forward(self, x: torch.Tensor, embedding: Optional[torch.Tensor], 
                y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the collaborative transformer.

        Parameters
        ----------
        x : torch.Tensor
            Text token IDs of shape (batch_size, seq_len).
        embedding : torch.Tensor, optional
            Pre-computed text embeddings of shape (batch_size, seq_len, embedding_size).
            If None, embeddings are computed from x.
        y : torch.Tensor
            Sequence features of shape (batch_size, seq_len, sequences_fsize).

        Returns
        -------
        torch.Tensor
            Classification logits of shape (batch_size, n_classes).

        Notes
        -----
        The embedding parameter allows for flexibility in how embeddings are
        computed, supporting both internal embedding lookup and external
        pre-computation.
        """
        try:
            logger.debug(f"CollaborativeTransformer forward: x shape={x.shape}, y shape={y.shape}")
            
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for x, got {type(x)}")
            if not isinstance(y, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for y, got {type(y)}")
            
            # Extract multi-modal features
            combined_features = self.forward_features(x, embedding, y)

            # Apply classification head
            output = self.classification_head(combined_features)
            output = self.activation_func(output)
            
            # Final check for NaN or Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.error("NaN or Inf detected in final output")
                output = torch.nan_to_num(output, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)
            
            logger.debug(f"CollaborativeTransformer output shape={output.shape}")

            return output
        except Exception as e:
            logger.error(f"Error in CollaborativeTransformer forward pass: {e}")
            raise

