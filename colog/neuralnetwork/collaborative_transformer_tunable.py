"""
collaborative_transformer_tunable.py — Hyperparameter-tunable Collaborative Transformer for CoLog

This module implements a tunable version of the Collaborative Transformer architecture
for detecting anomalies in system log sequences. Unlike the standard collaborative
transformer, this version uses string-based configuration values that can be evaluated
at runtime, enabling seamless integration with hyperparameter optimization frameworks
such as Optuna, Ray Tune, or Hyperopt.

The tunable architecture processes both textual log messages and sequential features
through collaborative attention mechanisms, with all architectural parameters (hidden
dimensions, number of layers, attention heads, etc.) specified as evaluable strings
to support dynamic hyperparameter search.

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

Tunable Parameters:
    All architectural dimensions are specified as strings that are evaluated at
    runtime using eval(), enabling hyperparameter optimization frameworks to
    search over discrete and continuous parameter spaces:
        - hidden_size: Hidden state dimension (as string)
        - dropout_rate: Dropout probability (as string)
        - heads: Number of attention heads (as string)
        - layers: Number of collaborative blocks (as string)

Usage
-----
    from neuralnetwork.collaborative_transformer_tunable import CollaborativeTransformerTunable
    from argparse import Namespace
    
    # Configuration with string-valued hyperparameters
    args = Namespace(
        hidden_size='512',
        dropout_rate='0.1',
        heads='8',
        layers='4',
        projection_size=2048,
        embed_size=300,
        sequences_fsize=256,
        len_messages=10,
        len_sequences=5,
        n_classes=2
    )
    
    # Initialize model
    model = CollaborativeTransformerTunable(None, args, vocab_size=50000, 
                                           pretrained_embeddings=embeddings_matrix)
    
    # Forward pass
    logits = model(text_input, sequence_features)
    predictions = torch.argmax(logits, dim=-1)

Notes
-----
- The config parameter is maintained for API compatibility but can be None
- String parameters are evaluated using eval() during initialization
- Pre-trained embeddings are frozen by default (requires_grad=False)
- Padding masks are computed automatically from input features
- The model expects both text token IDs and pre-computed sequence features

See Also
--------
collaborative_transformer.py : Standard collaborative transformer with fixed parameters
vanilla_transformer.py : Single-modal transformer without collaborative features
components.layers : Building blocks for neural network layers
"""

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
    # Compute the mask by checking which positions are all zeros
    mask = (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)
    
    return mask


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
    config : object, optional
        Configuration object (maintained for compatibility, can be None).
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (str): Hidden dimension size (evaluated at runtime)
        - projection_size (int): Intermediate projection dimension
        - dropout_rate (str): Dropout probability (evaluated at runtime)
    num_glimpses : int
        Number of attention glimpses to create.
    merge : bool, optional
        If True, concatenate and project all glimpses into a single vector.
        If False, return stacked glimpses. Default is False.

    Attributes
    ----------
    config : object
        Stored configuration (may be None).
    args : argparse.Namespace
        Stored argument configuration.
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

    def __init__(self, config, args, num_glimpses: int, merge: bool = False):
        """Initialize ModalityAdaptationLayer module."""
        super(ModalityAdaptationLayer, self).__init__()
     
        self.config = config
        self.args = args
        self.merge = merge
        self.num_glimpses = num_glimpses
        
        # Evaluate string parameters with config in scope
        hidden_size = eval(args.hidden_size, {'config': config}) if isinstance(args.hidden_size, str) else args.hidden_size
        dropout_rate = eval(args.dropout_rate, {'config': config}) if isinstance(args.dropout_rate, str) else args.dropout_rate

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=args.projection_size,
            out_size=num_glimpses,
            dropout_rate=dropout_rate,
            use_relu=True
        )

        if self.merge:
            self.output_projection = nn.Linear(
                hidden_size * num_glimpses,
                hidden_size * 2
            )

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
        if x.dim() != 3:
            pass
        
        attention_weights = self.mlp(x)
        
        if x_mask is not None:
            attention_weights = attention_weights.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                constants.ATTENTION_MASK_VALUE
            )
        
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Check for NaN or Inf in attention weights
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=1.0, neginf=0.0)

        attended_features_list = []
        for i in range(self.num_glimpses):
            attended_features_list.append(
                torch.sum(attention_weights[:, :, i: i + 1] * x, dim=1)
            )

        if self.merge:
            attended_features = torch.cat(attended_features_list, dim=1)
            attended_features = self.output_projection(attended_features)
            return attended_features

        stacked_features = torch.stack(attended_features_list).transpose_(0, 1)
        return stacked_features


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
    config : object, optional
        Configuration object (maintained for compatibility, can be None).
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (str): Hidden dimension size (must be divisible by heads)
        - heads (str): Number of attention heads
        - dropout_rate (str): Dropout probability

    Attributes
    ----------
    config : object
        Stored configuration.
    args : argparse.Namespace
        Stored argument configuration.
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

    def __init__(self, config, args):
        """Initialize MultiHeadAttention module."""
        super(MultiHeadAttention, self).__init__()

        # Evaluate string parameters with config in scope
        hidden_size = eval(args.hidden_size, {'config': config}) if isinstance(args.hidden_size, str) else args.hidden_size
        heads = eval(args.heads, {'config': config}) if isinstance(args.heads, str) else args.heads
        dropout_rate = eval(args.dropout_rate, {'config': config}) if isinstance(args.dropout_rate, str) else args.dropout_rate
        
        if hidden_size % heads != 0:
            pass
        
        self.config = config
        self.args = args

        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

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
        # Evaluate string parameters with config in scope
        hidden_size = eval(self.args.hidden_size, {'config': self.config}) if isinstance(self.args.hidden_size, str) else self.args.hidden_size
        heads = eval(self.args.heads, {'config': self.config}) if isinstance(self.args.heads, str) else self.args.heads
        batch_size = q.size(0)
        
        v = self.value_projection(v).view(
            batch_size,
            -1,
            heads,
            int(hidden_size / heads)
        ).transpose(1, 2)

        k = self.key_projection(k).view(
            batch_size,
            -1,
            heads,
            int(hidden_size / heads)
        ).transpose(1, 2)

        q = self.query_projection(q).view(
            batch_size,
            -1,
            heads,
            int(hidden_size / heads)
        ).transpose(1, 2)

        attended_output = self.compute_attention(v, k, q, mask)

        attended_output = attended_output.transpose(1, 2).contiguous().view(
            batch_size,
            -1,
            hidden_size
        )
        attended_output = self.output_projection(attended_output)

        return attended_output

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
        head_dim = query.size(-1)
        
        if head_dim == 0:
            pass

        attention_scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(head_dim)
        
        # Check for NaN or Inf in attention scores
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, constants.ATTENTION_MASK_VALUE)

        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Check for NaN in attention weights after softmax
        if torch.isnan(attention_weights).any():
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, value)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network.

    This module implements the position-wise feed-forward network used in
    Transformer architectures. It consists of two linear transformations
    with a ReLU activation in between:
        FFN(x) = max(0, xW1 + b1)W2 + b2

    Parameters
    ----------
    config : object, optional
        Configuration object (maintained for compatibility, can be None).
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (str): Input/output dimension
        - projection_size (int): Intermediate dimension
        - dropout_rate (str): Dropout probability

    Attributes
    ----------
    mlp : MLP
        Two-layer feed-forward network with ReLU activation.

    Notes
    -----
    The intermediate dimension (projection_size) is typically larger than
    the hidden_size to increase model capacity.
    """

    def __init__(self, config, args):
        """Initialize FeedForwardNetwork module."""
        super(FeedForwardNetwork, self).__init__()

        # Evaluate string parameters with config in scope
        hidden_size = eval(args.hidden_size, {'config': config}) if isinstance(args.hidden_size, str) else args.hidden_size
        dropout_rate = eval(args.dropout_rate, {'config': config}) if isinstance(args.dropout_rate, str) else args.dropout_rate

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=args.projection_size,
            out_size=hidden_size,
            dropout_rate=dropout_rate,
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
    config : object, optional
        Configuration object (maintained for compatibility, can be None).
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (str): Hidden dimension size
        - heads (str): Number of attention heads
        - projection_size (int): Feed-forward intermediate dimension
        - dropout_rate (str): Dropout probability

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

    def __init__(self, config, args):
        """Initialize SelfAttention encoder block."""
        super(SelfAttention, self).__init__()
        
        # Evaluate string parameters with config in scope
        hidden_size = eval(args.hidden_size, {'config': config}) if isinstance(args.hidden_size, str) else args.hidden_size
        dropout_rate = eval(args.dropout_rate, {'config': config}) if isinstance(args.dropout_rate, str) else args.dropout_rate

        self.multi_head_attention = MultiHeadAttention(config, args)
        self.feed_forward = FeedForwardNetwork(config, args)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attention_norm = LayerNorm(hidden_size)

        self.feed_forward_dropout = nn.Dropout(dropout_rate)
        self.feed_forward_norm = LayerNorm(hidden_size)

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
    config : object, optional
        Configuration object (maintained for compatibility, can be None).
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (str): Hidden dimension size
        - heads (str): Number of attention heads
        - projection_size (int): Feed-forward intermediate dimension
        - dropout_rate (str): Dropout probability

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

    def __init__(self, config, args):
        """Initialize MultiHeadImpressedAttention module."""
        super(MultiHeadImpressedAttention, self).__init__()
        
        # Evaluate string parameters with config in scope
        hidden_size = eval(args.hidden_size, {'config': config}) if isinstance(args.hidden_size, str) else args.hidden_size
        dropout_rate = eval(args.dropout_rate, {'config': config}) if isinstance(args.dropout_rate, str) else args.dropout_rate

        self.multi_head_attention_1 = MultiHeadAttention(config, args)
        self.multi_head_attention_2 = MultiHeadAttention(config, args)
        self.feed_forward = FeedForwardNetwork(config, args)

        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.self_attention_norm = LayerNorm(hidden_size)

        self.cross_attention_dropout = nn.Dropout(dropout_rate)
        self.cross_attention_norm = LayerNorm(hidden_size)

        self.feed_forward_dropout = nn.Dropout(dropout_rate)
        self.feed_forward_norm = LayerNorm(hidden_size)

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
        
        return x


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
    config : object, optional
        Configuration object (maintained for compatibility, can be None).
    args : argparse.Namespace
        Configuration object containing:
        - hidden_size (str): Hidden dimension size
        - len_messages (int): Number of glimpses for text modality adaptation
        - len_sequences (int): Number of glimpses for sequence modality adaptation
        - layers (str): Total number of collaborative blocks
        - dropout_rate (str): Dropout probability
    i : int
        Block index (0-based), used to determine if this is the last layer.

    Attributes
    ----------
    config : object
        Stored configuration.
    args : argparse.Namespace
        Stored argument configuration.
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

    def __init__(self, config, args, i: int):
        """Initialize CollaborativeBlock module."""
        super(CollaborativeBlock, self).__init__()
        
        # Evaluate string parameters with config in scope
        layers = eval(args.layers, {'config': config}) if isinstance(args.layers, str) else args.layers
        hidden_size = eval(args.hidden_size, {'config': config}) if isinstance(args.hidden_size, str) else args.hidden_size
        dropout_rate = eval(args.dropout_rate, {'config': config}) if isinstance(args.dropout_rate, str) else args.dropout_rate
        
        self.config = config
        self.args = args
        self.is_last_layer = (i == layers - 1)

        self.text_impressed_attention = MultiHeadImpressedAttention(config, args)
        self.sequence_impressed_attention = MultiHeadImpressedAttention(config, args)

        if not self.is_last_layer:
            self.text_modality_adaptation = ModalityAdaptationLayer(
                config, args, args.len_messages, merge=False
            )
            self.sequence_modality_adaptation = ModalityAdaptationLayer(
                config, args, args.len_sequences, merge=False
            )
            self.text_norm = LayerNorm(hidden_size)
            self.sequence_norm = LayerNorm(hidden_size)
            self.residual_dropout = nn.Dropout(dropout_rate)

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
        # Apply impressed attention bidirectionally
        text_impressed = self.text_impressed_attention(x, y, x_mask, y_mask)
        sequence_impressed = self.sequence_impressed_attention(y, x, y_mask, x_mask)

        # Add residual connections
        x = text_impressed + x
        y = sequence_impressed + y

        if self.is_last_layer:
            return x, y

        # Apply modality adaptation for intermediate layers
        text_adapted = self.text_modality_adaptation(x, x_mask)
        sequence_adapted = self.sequence_modality_adaptation(y, y_mask)

        # Apply normalization and residual dropout
        x = self.text_norm(x + self.residual_dropout(text_adapted))
        y = self.sequence_norm(y + self.residual_dropout(sequence_adapted))
        
        return x, y


class CollaborativeTransformerTunable(nn.Module):
    """
    Hyperparameter-tunable Collaborative Transformer for log anomaly detection.

    This model implements a collaborative multi-modal architecture with tunable
    hyperparameters specified as strings that are evaluated at runtime. This design
    enables seamless integration with hyperparameter optimization frameworks while
    maintaining the rich cross-modal attention capabilities of the collaborative
    transformer architecture.

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
    config : object, optional
        Configuration object (maintained for compatibility, can be None).
    args : argparse.Namespace
        Configuration object containing:
        - embed_size (int): Embedding dimension
        - hidden_size (str): Hidden state dimension (evaluated at runtime)
        - sequences_fsize (int): Input dimension for sequence features
        - projection_size (int): Feed-forward intermediate dimension
        - heads (str): Number of attention heads (evaluated at runtime)
        - layers (str): Number of collaborative blocks (evaluated at runtime)
        - len_messages (int): Number of glimpses for text modality
        - len_sequences (int): Number of glimpses for sequence modality
        - dropout_rate (str): Dropout probability (evaluated at runtime)
        - n_classes (int): Number of output classes
    vocab_size : int
        Size of the vocabulary (number of unique tokens).
    pretrained_embeddings : numpy.ndarray
        Pre-trained embedding matrix of shape (vocab_size, embed_size).

    Attributes
    ----------
    config : object
        Stored configuration.
    args : argparse.Namespace
        Stored argument configuration.
    embedding : nn.Embedding
        Embedding layer for text tokens (frozen by default).
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
    >>> # Configuration with string-valued hyperparameters
    >>> args = Namespace(
    ...     embed_size=300,
    ...     hidden_size='512',
    ...     sequences_fsize=256,
    ...     projection_size=2048,
    ...     heads='8',
    ...     layers='4',
    ...     len_messages=10,
    ...     len_sequences=5,
    ...     dropout_rate='0.1',
    ...     n_classes=2
    ... )
    >>> 
    >>> # Initialize model
    >>> vocab_size = 10000
    >>> embeddings = np.random.randn(vocab_size, 300)
    >>> model = CollaborativeTransformerTunable(None, args, vocab_size, embeddings)
    >>> 
    >>> # Forward pass
    >>> batch_size, seq_len = 16, 128
    >>> text_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    >>> seq_features = torch.randn(batch_size, seq_len, 256)
    >>> logits = model(text_input, seq_features)
    >>> logits.shape
    torch.Size([16, 2])

    Notes
    -----
    - The model requires both text and sequence features as input
    - Embeddings are frozen by default (requires_grad=False)
    - Padding masks are computed automatically from input features
    - All string parameters are evaluated using eval() during initialization
    - The config parameter can be None for hyperparameter tuning scenarios
    """

    def __init__(self, config, args, vocab_size: int, pretrained_embeddings, num_classes: int = None):
        """Initialize CollaborativeTransformerTunable model."""
        super(CollaborativeTransformerTunable, self).__init__()

        # Use num_classes from parameter if provided, otherwise fall back to args.n_classes
        if num_classes is not None:
            self.num_classes = num_classes
        
        # Evaluate string parameters with config in scope
        hidden_size = eval(args.hidden_size, {'config': config}) if isinstance(args.hidden_size, str) else args.hidden_size
        layers = eval(args.layers, {'config': config}) if isinstance(args.layers, str) else args.layers

        # self.config = config
        self.args = args

        # Text embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.embedding_size
        )
        
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # Freeze embeddings by default
        self.embedding.weight.requires_grad = False

        # Text LSTM encoder
        self.text_lstm = nn.LSTM(
            input_size=args.embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Sequence feature adapter
        self.sequence_adapter = nn.Linear(args.sequences_fsize, hidden_size)

        # Collaborative encoder blocks
        self.encoder_blocks = nn.ModuleList([
            CollaborativeBlock(config, args, i) for i in range(layers)
        ])

        # Modality adaptation layers for final aggregation
        self.sequence_modality_adaptation = ModalityAdaptationLayer(config, args, 1, merge=True)
        self.text_modality_adaptation = ModalityAdaptationLayer(config, args, 1, merge=True)

        # Normalization and classification layers
        self.projection_norm = LayerNorm(2 * hidden_size)
        self.classification_head = nn.Linear(2 * hidden_size, self.num_classes)
        self.activation_func = nn.PReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the tunable collaborative transformer.

        Parameters
        ----------
        x : torch.Tensor
            Text token IDs of shape (batch_size, seq_len).
        y : torch.Tensor
            Sequence features of shape (batch_size, seq_len, sequences_fsize).

        Returns
        -------
        torch.Tensor
            Classification logits of shape (batch_size, n_classes).

        Notes
        -----
        Masks are only applied in the first collaborative block to reduce
        computational overhead. Subsequent blocks process full sequences.
        """
        # Create padding masks
        text_mask = create_padding_mask(x.unsqueeze(2))
        sequence_mask = create_padding_mask(y)

        # Compute text embeddings
        text_embeddings = self.embedding(x)

        # Encode text through LSTM
        text_features, _ = self.text_lstm(text_embeddings)
        
        # Check for NaN or Inf after LSTM
        if torch.isnan(text_features).any() or torch.isinf(text_features).any():
            text_features = torch.nan_to_num(text_features, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)

        # Adapt sequence features to hidden dimension
        sequence_features = self.sequence_adapter(y)

        # Process through collaborative encoder blocks
        for i, encoder in enumerate(self.encoder_blocks):
            # Only apply masks in the first block
            current_text_mask, current_sequence_mask = None, None
            if i == 0:
                current_text_mask, current_sequence_mask = text_mask, sequence_mask
            
            text_features, sequence_features = encoder(
                text_features, current_text_mask, 
                sequence_features, current_sequence_mask
            )
            
            # Check for NaN or Inf after each block
            if torch.isnan(text_features).any() or torch.isinf(text_features).any():
                text_features = torch.nan_to_num(text_features, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)
            if torch.isnan(sequence_features).any() or torch.isinf(sequence_features).any():
                sequence_features = torch.nan_to_num(sequence_features, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)

        # Aggregate modalities to fixed-size vectors
        text_features = self.text_modality_adaptation(
            text_features,
            None
        )

        sequence_features = self.sequence_modality_adaptation(
            sequence_features,
            None
        )

        # Fuse multi-modal features
        combined_features = text_features + sequence_features
        combined_features = self.projection_norm(combined_features)

        # Apply classification head
        output = self.classification_head(combined_features)
        output = self.activation_func(output)
        
        # Final check for NaN or Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=constants.LARGE_POSITIVE_VALUE, neginf=constants.LARGE_NEGATIVE_VALUE)

        return output
    