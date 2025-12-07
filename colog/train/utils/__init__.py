"""
utils package for CoLog training utilities

This package contains utility modules for the CoLog training system:
- groundtruth_loader: PyTorch Dataset loader for ground truth data
- groundtruth_detector: Auto-detection of groundtruth configurations
- prediction_utils: Prediction conversion and threshold utilities
- constants: System-wide constants and configuration defaults

Classes
-------
GroundTruthLoader
    PyTorch Dataset for loading preprocessed ground truth data.

Functions
---------
detect_groundtruth_config
    Automatically detect groundtruth configuration for a dataset.
predict_with_argmax
    Convert prediction probabilities to class labels using argmax.
predict_with_threshold
    Convert binary classification probabilities using a custom threshold.
"""

from .groundtruth_loader import GroundTruthLoader
from .groundtruth_detector import detect_groundtruth_config
from .prediction_utils import predict_with_argmax, predict_with_threshold
from . import constants

__all__ = [
    'GroundTruthLoader',
    'detect_groundtruth_config',
    'predict_with_argmax',
    'predict_with_threshold',
    'constants',
]
