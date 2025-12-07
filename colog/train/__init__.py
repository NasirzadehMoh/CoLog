"""
train package for CoLog model training

This package contains the training pipeline for the CoLog collaborative
transformer model, including training orchestration, evaluation, and
hyperparameter tuning capabilities.

Modules
-------
main
    Core training pipeline with Trainer class for model training and evaluation.
utils
    Training utilities including data loading and prediction functions.

Classes
-------
Trainer
    Main training orchestrator for CoLog models.

Subpackages
-----------
utils
    Contains GroundTruthLoader, prediction utilities, and helper functions.
"""

from .main import Trainer
from .main import tune_model

__all__ = [
    'Trainer',
    'tune_model'
]
