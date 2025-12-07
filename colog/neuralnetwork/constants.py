"""
constants.py — Constants used across the CoLog neural network module

This module contains all constant values, configuration defaults, and magic strings
used throughout the neural network architectures and components.
"""

# ============================================================================
# Numerical Stability
# ============================================================================

LAYER_NORM_EPS = 1e-6
ATTENTION_MASK_VALUE = -1e9  # Large negative value for masking in attention
LARGE_POSITIVE_VALUE = 1e9   # Large positive value for numerical stability
LARGE_NEGATIVE_VALUE = -1e9  # Large negative value for numerical stability

# ============================================================================
# Device Types
# ============================================================================

DEVICE_AUTO = 'auto'
DEVICE_CPU = 'cpu'
DEVICE_CUDA = 'cuda'

# ============================================================================
# Model Architecture Defaults
# ============================================================================

DEFAULT_HIDDEN_SIZE = 256
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_PROJECTION_SIZE = 2048

# ============================================================================
# Embedding Configuration
# ============================================================================

DEFAULT_EMBEDDING_DIM = 384
PADDING_IDX = 0

# ============================================================================
# Validation Ranges
# ============================================================================

MIN_DROPOUT_RATE = 0.0
MAX_DROPOUT_RATE = 1.0
MIN_HIDDEN_SIZE = 1
MIN_NUM_HEADS = 1
MIN_NUM_LAYERS = 1
MIN_PROJECTION_SIZE = 1

# ============================================================================
# Tensor Dimensions
# ============================================================================

EXPECTED_EMBEDDING_DIMS = 3  # (batch, sequence, features)
EXPECTED_SEQUENCE_DIMS = 2   # (batch, features)
