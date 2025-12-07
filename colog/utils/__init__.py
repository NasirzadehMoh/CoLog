"""
utils — Utility modules for CoLog training and testing system

This package provides essential utilities for the CoLog log anomaly detection
training and testing pipeline, including CLI argument parsing, constants, and evaluation metrics.

Modules:
    cli: Command-line interface with argument parsing and validation for both training and testing
    constants: System-wide constants and default configuration values
    metrics: Binary classification metrics and visualization tools

Public API:
    parse_train_arguments: CLI argument parsing for training mode
    parse_test_arguments: CLI argument parsing for testing/evaluation mode
    BinaryClassification: Comprehensive binary classification metrics and plotting
    constants: All configuration constants and defaults
"""

from .cli import parse_train_arguments, parse_test_arguments
from .metrics import ClassificationMetrics
from . import constants

__all__ = [
    'parse_train_arguments',
    'parse_test_arguments',
    'ClassificationMetrics',
    'constants',
]
